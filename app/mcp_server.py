from fastmcp import FastMCP
from googleapiclient.discovery import build
from app.core.pineconeAdapter import PineconeVectorAdapter
from app.core.embeddings import HFEmbeddingProvider
from app.core.config import settings

# ── MCP Server ────────────────────────────────────────────────────────────────
mcp = FastMCP("AuraRAG Server")

# ── Lazy initialization ───────────────────────────────────────────────────────
_embeddings = None
_vector_adapter = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HFEmbeddingProvider()
    return _embeddings


def get_vector_adapter():
    global _vector_adapter
    if _vector_adapter is None:
        _vector_adapter = PineconeVectorAdapter(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
        )
    return _vector_adapter


# ── Tool 1: Document Search (Pinecone RAG) ────────────────────────────────────
@mcp.tool
def search_documents(query: str, session_id: str = None, top_k: int = 3) -> str:
    """
    Search the securely ingested document base for a user query.

    Args:
        query: The search term or user question.
        session_id: Optional ID to restrict retrieval to a specific user session.
        top_k: Number of relevant document chunks to return (default 3).

    Returns:
        Formatted string of relevant document segments with match scores.
    """
    try:
        embeddings = get_embeddings()
        vector_adapter = get_vector_adapter()

        query_vector = embeddings.embed_query(query)
        results = vector_adapter.query(
            vector=query_vector,
            top_k=top_k,
            session_id=session_id,
        )

        if not results:
            return "No matching document segments found in the ingested base."

        chunks = []
        for i, match in enumerate(results):
            text = match.get("metadata", {}).get("text", "")
            source = match.get("metadata", {}).get("source", "Unknown Document")
            score = match.get("score", 0.0)
            chunks.append(
                f"[Segment {i+1}] Source: {source} (Score: {score:.2f})\n---\n{text}\n---"
            )

        return "\n\n".join(chunks)

    except Exception as e:
        return f"Error searching documents: {str(e)}"


# ── Tool 2: Web Search (Google Custom Search API) ─────────────────────────────
@mcp.tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Perform a real-time web search using the Google Custom Search API.

    Args:
        query: The search keywords or question.
        max_results: Max number of results to retrieve (default 3, max 10).

    Returns:
        Formatted string containing title, URL, and snippet of each result.
    """
    try:
        service = build(
            "customsearch", "v1",
            developerKey=settings.google_api_key
        )
        res = service.cse().list(
            q=query,
            cx=settings.google_cse_id,
            num=min(max_results, 10),
        ).execute()

        items = res.get("items", [])
        if not items:
            return f"No web results found for: {query}"

        lines = []
        for i, item in enumerate(items):
            lines.append(
                f"[Web Result {i+1}]\n"
                f"Title: {item['title']}\n"
                f"Link: {item['link']}\n"
                f"Snippet: {item.get('snippet', 'No snippet available.')}\n"
            )

        return "\n\n".join(lines)

    except Exception as e:
        return f"Error executing web search: {str(e)}"


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()