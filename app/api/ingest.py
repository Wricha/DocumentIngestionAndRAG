from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from io import BytesIO
import uuid

from app.core.extract import extract_text_from_pdf, extract_text_from_txt
from app.core.utils import chunk_sentences, chunk_sliding
from app.core.embeddings import HFEmbeddingProvider
from app.core.pineconeAdapter import PineconeVectorAdapter
from app.core.db import AsyncSessionLocal, Documents

router = APIRouter()

def get_embedding_provider():
    return HFEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_adapter():
    return PineconeVectorAdapter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    source: str = Form(None),
    chunking_strategy: str = Form("sliding"),
    chunk_size: int = Form(500),
    overlap: int = Form(50),
    embedding_provider: HFEmbeddingProvider = Depends(get_embedding_provider),
    vector_adapter: PineconeVectorAdapter = Depends(get_vector_adapter),
):

    data = await file.read()

    if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(BytesIO(data))
    elif file.content_type == "text/plain" or file.filename.lower().endswith(".txt"):
        text = extract_text_from_txt(BytesIO(data))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if chunking_strategy == "sliding":
        chunks = chunk_sliding(text, chunk_size=chunk_size, overlap=overlap)
    elif chunking_strategy == "sentences":
        chunks = chunk_sentences(text, max_chunk_size=chunk_size)
    else:
        raise HTTPException(status_code=400, detail="Invalid chunking strategy")

    session_id = str(uuid.uuid4())

    embeddings_raw = await embedding_provider.embed(chunks)

    embeddings_clean = []
    for emb in embeddings_raw:
        if hasattr(emb, "tolist"):
            embeddings_clean.append([float(x) for x in emb.tolist()])
        else:
            embeddings_clean.append([float(x) for x in emb])

    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"text_preview": chunk, "source": file.filename, "session_id": session_id} for chunk in chunks]

    await vector_adapter.upsert(ids=ids, vectors=embeddings_clean, metadatas=metadatas)

    async with AsyncSessionLocal() as session:
        doc = Documents(
            source=source or file.filename,
            metadata={"num_chunks": len(chunks), "session_id": session_id, "text_preview": chunks[0]}
        )
        session.add(doc)
        await session.commit()

    return {"message": f"Uploaded and processed {file.filename} with {len(chunks)} chunks.", "session_id": session_id}
