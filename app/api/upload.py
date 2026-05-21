from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from typing import List
import io

from app.core.config import settings
from app.core.embeddings import HFEmbeddingProvider
from app.core.pineconeAdapter import PineconeVectorAdapter

router = APIRouter()

# ── Lazy singletons ───────────────────────────────────────────────────────────
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


# ── Response schema ───────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    filename: str
    chunks: int
    session_id: str
    message: str


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from PDF or plain text files."""
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="pypdf not installed. Run: pip install pypdf"
            )

    elif ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Supported: pdf, txt, md"
        )


# ── Chunking ──────────────────────────────────────────────────────────────────
def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Upload endpoint ───────────────────────────────────────────────────────────
@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Query(..., description="Session ID to associate uploaded doc with"),
):
    """
    Upload a PDF or text file. Extracts text, splits into chunks,
    embeds via HuggingFace, and stores in Pinecone under the session_id.
    """
    # 1. Read file bytes
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 2. Extract text
    try:
        raw_text = extract_text(file_bytes, file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the file.")

    # 3. Split into chunks
    chunks = split_into_chunks(raw_text, chunk_size=500, overlap=50)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no text chunks.")

    # 4. Embed chunks
    try:
        embeddings = get_embeddings()
        vectors = embeddings.embed_documents(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # 5. Upsert into Pinecone with metadata
    try:
        vector_adapter = get_vector_adapter()
        records = [
            {
                "id": f"{session_id}_{file.filename}_chunk{i}",
                "values": vector,
                "metadata": {
                    "text": chunk,
                    "source": file.filename,
                    "session_id": session_id,
                    "chunk_index": i,
                }
            }
            for i, (vector, chunk) in enumerate(zip(vectors, chunks))
        ]
        vector_adapter.upsert(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone upsert failed: {str(e)}")

    return UploadResponse(
        filename=file.filename,
        chunks=len(chunks),
        session_id=session_id,
        message=f"Successfully ingested {len(chunks)} chunks from {file.filename}",
    )