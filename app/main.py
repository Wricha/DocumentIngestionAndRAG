from fastapi import FastAPI
from app.api.ingest import router as ingest_router
from app.api.rag import router as rag_router
from app.core.config import settings

app = FastAPI(title="FastAPI RAG Backend")

app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
app.include_router(rag_router, prefix="/rag", tags=["rag"])

@app.get("/")
async def root():
    return {"status": "ok", "service": "fastapi-rag-backend", "env": settings.env}
