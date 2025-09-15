from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from typing import List
from app.core.extract import extract_text_from_pdf, extract_text_from_txt
from app.core.utils import chunk_sentences, chunk_sliding
from app.core.embeddings import HFEmbeddingProvider
from app.core.pineconeAdapter import PineconeVectorAdapter
from app.core.config import get_settings, Settings
from app.core.db import AsyncSessionLocal, Documents

router = APIRouter()

def get_embedding_provider():
    return HFEmbeddingProvider(model_name=get_settings().embedding_model_name)

def get_vector_adapter():
    return PineconeVectorAdapter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    source: str = Form(...),
    metadata: str = Form(None),
    chunking_strategy: str = Form("sliding"),
    chunk_size: int = Form(500),
    overlap: int = Form(50),
    embedding_provider: HFEmbeddingProvider = Depends(get_embedding_provider),
    vector_adapter: PineconeVectorAdapter = Depends(get_vector_adapter),
):
    data = await file.read()
    if file.content_type == "application/pdf":
        text = await extract_text_from_pdf(file)
    elif file.content_type == "text/plain":
        text = await extract_text_from_txt(file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if chunking_strategy == "sliding":
        chunks = chunk_sliding(text, chunk_size=chunk_size, overlap=overlap)
    elif chunking_strategy == "sentences":
        chunks = chunk_sentences(text, max_chunk_size=chunk_size)
    else:
        raise HTTPException(status_code=400, detail="Invalid chunking strategy")

    embeddings = await embedding_provider.embed(chunks)

    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "metadata": metadata, "text": chunk} for chunk in chunks]

    await vector_adapter.upsert(ids=ids, vectors=embeddings, metadatas=metadatas)

    async with AsyncSessionLocal() as session:
        doc = Documents(source=source or file.filename, metadata={"num_chunks":len(chunks)})
        session.add(doc)
        await session.commit()

    return {"message": f"Uploaded and processed {file.filename} with {len(chunks)} chunks."}
