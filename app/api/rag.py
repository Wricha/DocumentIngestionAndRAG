from fastapi import APIRouter, Body, Depends, HTTPException
from typing import List, Optional, Dict, Any
from app.core.embeddings import HFEmbeddingProvider
from app.core.pineconeAdapter import PineconeVectorAdapter
from app.core.config import settings
from app.core.db import AsyncSessionLocal, Booking
from pydantic import BaseModel, Field
import uuid
import json
import asyncio
import redis.asyncio as aioredis
from groq import Groq

router = APIRouter()

async def get_redis():
    r = aioredis.from_url(settings.redis_url)
    try:
        yield r
    finally:
        await r.close()

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    top_k: int = Field(default=3, ge=1, le=20)

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[Dict[str, Any]]

class BookingRequest(BaseModel):
    name: str
    email: str
    date: str
    time: str

def get_embedding_provider():
    return HFEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_adapter():
    return PineconeVectorAdapter()

groq_client = Groq(api_key=settings.groq_api_key)

async def append_to_redis_history(redis, session_id: str, role:str, text: str):
    key = f"chat:{session_id}"
    entry = json.dumps({"role": role, "text": text})
    await redis.rpush(key, entry)
    await redis.ltrim(key, -50, -1)

async def get_redis_history(redis, session_id: str):
    key = f"chat:{session_id}"
    vals = await redis.lrange(key, 0, -1)
    return [json.loads(v) for v in vals]

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    redis=Depends(get_redis),
    embedding: HFEmbeddingProvider = Depends(get_embedding_provider),
    vector_adapter: PineconeVectorAdapter = Depends(get_vector_adapter)
):
    session_id = request.session_id or str(uuid.uuid4())

    await append_to_redis_history(redis, session_id, "user", request.query)

    q_vec = (await embedding.embed([request.query]))[0]
    results = await vector_adapter.query(q_vec, top_k=request.top_k)
    context = [r["metadata"].get("text_preview", "") for r in results]

    memory = await get_redis_history(redis, session_id)
    context_str = "\n---\n".join(context)
    history_str = "\n".join([f"{m['role']}: {m['text']}" for m in memory])

    prompt = f"""You are an assistant that answers only based on CONTEXT and user's question.

CONTEXT:
{context_str}

CONVERSATION HISTORY:
{history_str}

QUESTION:
{request.query}

Answer concisely:
"""
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful RAG assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=400
    )

    answer = completion.choices[0].message.content.strip()

    await append_to_redis_history(redis, session_id, "assistant", answer)

    return ChatResponse(session_id=session_id, answer=answer, sources=[{"id": r["id"], "score": r["score"], "metadata": r["metadata"]} for r in results])

@router.post("/book")
async def book_interview(request: BookingRequest):
    async with AsyncSessionLocal() as session:
        b = Booking(
            name=request.name,
            email=request.email,
            date=request.date,
            time=request.time,
            metadata={}
        )
        session.add(b)
        await session.commit()

    return {"status": "booked", "name": request.name, "email": request.email, "date": request.date, "time": request.time}