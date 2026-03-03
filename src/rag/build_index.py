import os
from src.rag.vector_store import VectorStore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAG_DOCS_DIR = os.path.join(BASE_DIR, "data", "rag_documents")

def build_index():
    print("🚀 Đang xây dựng RAG Index vào ChromaDB...")
    store = VectorStore()
    
    docs = []
    metadatas = []
    ids = []
    
    idx = 0
    for filename in os.listdir(RAG_DOCS_DIR):
        if not filename.endswith(".txt"):
            continue
        
        filepath = os.path.join(RAG_DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Chia nhỏ text theo block
        blocks = content.split("\n\n")
        source = filename
        
        for block in blocks:
            text = block.strip()
            if not text:
                continue
            
            docs.append(text)
            metadatas.append({"source": source})
            ids.append(f"doc_{idx}")
            idx += 1

    if docs:
        store.add_documents(docs, metadatas, ids)
        print(f"✅ Đã index {len(docs)} đoạn văn bản thành công!")
    else:
        print("Không tìm thấy dữ liệu để index.")

if __name__ == "__main__":
    build_index()
