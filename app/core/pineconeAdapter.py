import asyncio
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings

class PineconeVectorAdapter:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key, environment=settings.pinecone_region)

        if settings.pinecone_index_name not in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec (cloud=settings.pinecone_cloud, region=settings.pinecone_region)
            )

        self.index = self.pc.Index(settings.pinecone_index_name)

    async def upsert(self,
                 ids: List[str],
                 vectors: List[List[float]],
                 metadatas: List[Dict[str, Any]]):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.index.upsert(vectors=[
                {"id": vid, "values": vectors[i], "metadata": metadatas[i]}
                for i, vid in enumerate(ids)
            ])
    )


    async def query(
            self,
            vector: List[float],
            top_k: int
    ) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            lambda: self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        )
        results=[
            {"id": m.id, "score": m.score, "metadata": m.metadata}
            for m in res.matches
        ]
        return results    
