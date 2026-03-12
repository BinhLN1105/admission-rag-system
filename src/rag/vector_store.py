import os
import chromadb
from src.rag.embedder import Embedder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_DIR = os.path.join(BASE_DIR, "models", "vector_db", "chroma_db")

class VectorStore:
    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.embedder = Embedder()
        self.collection = self.client.get_or_create_collection(
            name="university_info"
        )

    def add_documents(self, docs: list, metadatas: list, ids: list):
        embeddings = self.embedder.embed_documents(docs)
        self.collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, top_k: int = 3, ma_nganh: str = "", ma_truong: str = "", to_hop: str = ""):
        query_embedding = self.embedder.embed_text(query)
        
        conditions = []
        if ma_nganh:
            conditions.append({"ma_nganh": {"$eq": ma_nganh}})
        if ma_truong:
            conditions.append({"ma_truong": {"$eq": ma_truong}})
        if to_hop:
            conditions.append({"ma_to_hop": {"$eq": to_hop}})
            
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}
        else:
            where_filter = None
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        return results
