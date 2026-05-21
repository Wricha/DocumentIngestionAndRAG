from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import redis.asyncio as aioredis
import uuid
import json
import os

from app.core.config import settings

router = APIRouter()

gemini_client = genai.Client(api_key=settings.gemini_api_key)

MCP_SERVER_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_server.py")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=20)
    model: str = Field(default="gemini-2.5-flash")
    system_prompt: str = Field(
        default=(
            "You are AuraRAG, a highly capable AI assistant. "
            "You have access to two tools: search_documents (searches the user's "
            "uploaded document base) and web_search (searches the internet in "
            "real-time via Google). Use them whenever needed to give accurate, "
            "cited answers. Always mention the source of your information."
        )
    )


class SourceItem(BaseModel):
    type: str
    title: str
    url: Optional[str] = None
    score: Optional[float] = None
    preview: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceItem]
    reasoning_trace: List[str]


# ── Redis helpers ─────────────────────────────────────────────────────────────
async def get_redis():
    r = aioredis.from_url(settings.redis_url)
    try:
        yield r
    finally:
        await r.close()


async def redis_append(redis, session_id: str, role: str, text: str):
    key = f"chat:{session_id}"
    await redis.rpush(key, json.dumps({"role": role, "text": text}))
    await redis.ltrim(key, -50, -1)   # keep last 50 messages


async def redis_history(redis, session_id: str) -> List[Dict]:
    key = f"chat:{session_id}"
    vals = await redis.lrange(key, 0, -1)
    return [json.loads(v) for v in vals]


# ── MCP ↔ Gemini bridge helpers ───────────────────────────────────────────────
def mcp_tool_to_gemini(tool) -> types.FunctionDeclaration:
    """Convert an MCP tool definition → Gemini FunctionDeclaration."""
    schema = tool.inputSchema or {}
    properties = {}
    required = schema.get("required", [])

    for name, prop in schema.get("properties", {}).items():
        prop_type = prop.get("type", "string").upper()
        gemini_type = getattr(types.Type, prop_type, types.Type.STRING)
        properties[name] = types.Schema(
            type=gemini_type,
            description=prop.get("description", "")
        )

    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description or "",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties=properties,
            required=required,
        )
    )


def parse_sources(tool_name: str, tool_result: str) -> List[SourceItem]:
    """Extract structured sources from MCP tool result text."""
    sources = []
    if tool_name == "web_search":
        for block in tool_result.split("\n\n"):
            lines = {
                k.strip(): v.strip()
                for line in block.splitlines()
                if ": " in line
                for k, v in [line.split(": ", 1)]
            }
            if lines.get("Title"):
                sources.append(SourceItem(
                    type="web",
                    title=lines.get("Title", ""),
                    url=lines.get("Link"),
                    preview=lines.get("Snippet"),
                ))
    elif tool_name == "search_documents":
        for block in tool_result.split("\n\n"):
            if block.startswith("[Segment"):
                first_line = block.splitlines()[0]
                sources.append(SourceItem(
                    type="document",
                    title=first_line,
                    preview=block[:200],
                ))
    return sources


# ── Core: MCP session + Gemini agentic loop ───────────────────────────────────
async def run_agentic_chat(
    query: str,
    history: List[Dict],
    session_id: str,
    top_k: int,
    model: str,
    system_prompt: str,
) -> Dict:
    """
    1. Open MCP session → discover tools
    2. Convert tools → Gemini Function Declarations
    3. Send query + history + tools to Gemini
    4. If Gemini calls a tool → execute via MCP → feed result back
    5. Repeat until Gemini returns a final text answer
    """

    reasoning = []
    all_sources: List[SourceItem] = []

    server_params = StdioServerParameters(
        command="python",
        args=[MCP_SERVER_PATH]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            reasoning.append("✅ MCP server connected.")

            # 1. Discover tools
            tools_response = await mcp_session.list_tools()
            mcp_tools = tools_response.tools
            reasoning.append(
                f"🔧 Tools available: {[t.name for t in mcp_tools]}"
            )

            # 2. Convert to Gemini format
            gemini_declarations = [mcp_tool_to_gemini(t) for t in mcp_tools]
            gemini_tool = types.Tool(function_declarations=gemini_declarations)

            # 3. Build conversation contents
            #    Include history + current query
            history_text = "\n".join(
                f"{m['role']}: {m['text']}" for m in history[:-1]  # exclude current
            )
            full_query = (
                f"Conversation so far:\n{history_text}\n\nUser: {query}"
                if history_text else query
            )

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=full_query)]
                )
            ]

            # 4. Agentic loop — Gemini can call multiple tools
            MAX_TOOL_ROUNDS = 5
            for round_num in range(MAX_TOOL_ROUNDS):
                reasoning.append(f"🤖 Gemini thinking... (round {round_num + 1})")

                response = gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=[gemini_tool],
                        temperature=0.2,
                        max_output_tokens=1000,
                    )
                )

                candidate = response.candidates[0]
                parts = candidate.content.parts

                # Check for tool calls in this response
                function_calls = [p for p in parts if p.function_call]
                text_parts = [p for p in parts if p.text]

                if not function_calls:
                    # No tool calls — Gemini has a final answer
                    reasoning.append("✅ Gemini produced final answer.")
                    final_answer = "\n".join(p.text for p in text_parts).strip()
                    return {
                        "answer": final_answer,
                        "sources": all_sources,
                        "reasoning": reasoning,
                    }

                # Append Gemini's response (with function calls) to contents
                contents.append(candidate.content)

                # Execute each tool call via MCP
                tool_response_parts = []
                for fc_part in function_calls:
                    tool_name = fc_part.function_call.name
                    tool_args = dict(fc_part.function_call.args)

                    # Inject top_k / session_id where relevant
                    if tool_name == "search_documents":
                        tool_args.setdefault("top_k", top_k)
                        tool_args.setdefault("session_id", session_id)

                    reasoning.append(
                        f"🔨 Calling MCP tool: `{tool_name}` with args: {tool_args}"
                    )

                    try:
                        mcp_result = await mcp_session.call_tool(tool_name, tool_args)
                        result_text = mcp_result.content[0].text
                        reasoning.append(
                            f"📥 `{tool_name}` returned {len(result_text)} chars."
                        )
                        # Parse sources
                        all_sources.extend(parse_sources(tool_name, result_text))

                    except Exception as e:
                        result_text = f"Tool error: {str(e)}"
                        reasoning.append(f"❌ `{tool_name}` failed: {str(e)}")

                    tool_response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={"result": result_text}
                            )
                        )
                    )

                # Feed all tool results back to Gemini
                contents.append(
                    types.Content(role="tool", parts=tool_response_parts)
                )

            # Exceeded MAX_TOOL_ROUNDS — return whatever Gemini last said
            reasoning.append("⚠️ Max tool rounds reached. Returning best answer.")
            last_text = next(
                (p.text for p in reversed(parts) if p.text), "No answer generated."
            )
            return {
                "answer": last_text,
                "sources": all_sources,
                "reasoning": reasoning,
            }


# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    redis=Depends(get_redis)
):
    session_id = request.session_id or str(uuid.uuid4())

    # Save user message to Redis
    await redis_append(redis, session_id, "user", request.query)

    # Fetch full history
    history = await redis_history(redis, session_id)

    try:
        result = await run_agentic_chat(
            query=request.query,
            history=history,
            session_id=session_id,
            top_k=request.top_k,
            model=request.model,
            system_prompt=request.system_prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save assistant reply to Redis
    await redis_append(redis, session_id, "assistant", result["answer"])

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        sources=result["sources"],
        reasoning_trace=result["reasoning"],
    )


@router.get("/health")
async def health():
    return {"status": "ok"}