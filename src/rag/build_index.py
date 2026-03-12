import os
import json
from src.rag.vector_store import VectorStore
from src.rag.embedder import Embedder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAG_JSON_PATH = os.path.join(BASE_DIR, "data", "rag_processed_data.json")

def build_index():
    print("Dang xay dung RAG Index vao ChromaDB tu rag_processed_data.json...")
    
    if not os.path.exists(RAG_JSON_PATH):
        print(f"Khong tim thay file: {RAG_JSON_PATH}")
        return
    
    with open(RAG_JSON_PATH, 'r', encoding='utf-8') as f:
        rag_documents = json.load(f)
    
    print(f"Da doc {len(rag_documents)} documents tu JSON.")
    
    store = VectorStore()
    embedder = Embedder()
    
    # Xoa collection cu (neu co) de tranh trung lap khi re-index
    try:
        store.client.delete_collection("university_info")
        print("Da xoa collection cu.")
    except Exception:
        pass
    
    # Tao lai collection
    store.collection = store.client.get_or_create_collection(name="university_info")
    
    docs = []
    metadatas = []
    ids = []
    total = len(rag_documents)
    
    for idx, item in enumerate(rag_documents):
        page_content = item.get("page_content", "").strip()
        if not page_content:
            continue
        
        meta = item.get("metadata", {})
        # Chi luu metadata dang string/int de ChromaDB chap nhan
        metadata_clean = {
            "ma_nganh": str(meta.get("ma_nganh", "")),
            "ma_truong": str(meta.get("ma_truong", "")),
            "ma_to_hop": str(meta.get("ma_to_hop", "")),
            "nam": int(meta.get("nam", 0)),
            "ten_truong": str(meta.get("ten_truong", ""))
        }
        
        docs.append(page_content)
        metadatas.append(metadata_clean)
        ids.append(f"doc_{idx}")
        
        # Batch insert moi 500 docs de tranh OOM
        if len(docs) >= 500:
            embeddings = embedder.embed_documents(docs)
            store.collection.add(embeddings=embeddings, documents=docs, metadatas=metadatas, ids=ids)
            print(f"Da index {idx+1} / {total} documents...")
            docs, metadatas, ids = [], [], []
    
    # Flush phan con lai
    if docs:
        embeddings = embedder.embed_documents(docs)
        store.collection.add(embeddings=embeddings, documents=docs, metadatas=metadatas, ids=ids)
    
    print(f"Hoan tat! Da index tong cong {total} documents voi metadata ma_nganh/ma_truong.")

if __name__ == "__main__":
    build_index()
