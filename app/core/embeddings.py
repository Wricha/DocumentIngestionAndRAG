from sentence_transformers import SentenceTransformer
import asyncio

class HFEmbeddingProvider:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    async def embed(self, texts):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, texts)

    def _embed_sync(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().tolist()
