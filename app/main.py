from fastapi import FastAPI
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router

app = FastAPI(title="AuraRAG Chatbot Backend")

app.include_router(upload_router, prefix="/rag", tags=["upload"])
app.include_router(chat_router, prefix="/rag", tags=["chat"])
