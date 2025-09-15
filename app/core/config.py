from pydantic import BaseSettings

class Settings(BaseSettings):
    env: str = "development"
    groq_api_key: str
    redis_url: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str | None = None
    pinecone_region: str | None = None
    embedding_dimension: int = 384
    database_url: str

    class Config:
        env_file = ".env"

settings= Settings()