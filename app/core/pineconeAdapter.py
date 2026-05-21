import sqlite3
import json
import numpy as np
import asyncio
from typing import List, Dict, Any

class PineconeVectorAdapter:
    """
    Local Vector Adapter that implements the exact same interface as PineconeVectorAdapter
    but uses a fully local SQLite database and numpy cosine similarity computations.
    """
    def __init__(self):
        # Store index db locally in the app/core folder
        self.db_path = "app/core/local_vector_db.db"
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                source TEXT,
                text_preview TEXT,
                embedding TEXT,
                metadata TEXT
            )
        """)
        # Index on session_id for lightning-fast lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON document_chunks (session_id)")
        conn.commit()
        conn.close()

    async def upsert(self,
                  ids: List[str],
                  vectors: List[List[float]],
                  metadatas: List[Dict[str, Any]]):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_upsert,
            ids, vectors, metadatas
        )

    def _sync_upsert(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for i, vid in enumerate(ids):
            meta = metadatas[i]
            session_id = meta.get("session_id")
            source = meta.get("source", "Uploaded Document")
            text_preview = meta.get("text_preview", "")
            
            cursor.execute("""
                INSERT OR REPLACE INTO document_chunks (id, session_id, source, text_preview, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                vid,
                session_id,
                source,
                text_preview,
                json.dumps(vectors[i]),
                json.dumps(meta)
            ))
        conn.commit()
        conn.close()

    async def query(
            self,
            vector: List[float],
            top_k: int,
            session_id: str = None
    ) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._sync_query,
            vector, top_k, session_id
        )

    def _sync_query(self, query_vector: List[float], top_k: int, session_id: str = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Restrict querying to active session documents
        if session_id:
            cursor.execute("""
                SELECT id, embedding, metadata FROM document_chunks 
                WHERE session_id = ?
            """, (session_id,))
        else:
            cursor.execute("SELECT id, embedding, metadata FROM document_chunks")
            
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
            
        # Perform highly optimized local numpy vector computations
        query_np = np.array(query_vector)
        query_norm = np.linalg.norm(query_np)
        
        matches = []
        for vid, emb_str, meta_str in rows:
            try:
                emb = np.array(json.loads(emb_str))
                meta = json.loads(meta_str)
                
                emb_norm = np.linalg.norm(emb)
                if query_norm > 0 and emb_norm > 0:
                    score = float(np.dot(query_np, emb) / (query_norm * emb_norm))
                else:
                    score = 0.0
                    
                matches.append({
                    "id": vid,
                    "score": score,
                    "metadata": meta
                })
            except Exception:
                continue
            
        # Sort scores by descending order and return top K results
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_k]
