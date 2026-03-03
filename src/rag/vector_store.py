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

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.embedder.embed_text(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
