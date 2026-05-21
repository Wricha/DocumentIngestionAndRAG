# app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str

    # Google Custom Search (optional — only needed if using Google CSE web search)
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None

    # Pinecone
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "document-embeddings"
    pinecone_cloud: Optional[str] = None
    pinecone_region: Optional[str] = None
    embedding_dimension: int = 768

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Database (optional legacy)
    database_url: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()