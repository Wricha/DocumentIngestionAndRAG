from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import asyncio

class HFEmbeddingProvider:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, texts)
    
    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded)

        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings.cpu().tolist()
