from src.rag.vector_store import VectorStore

class Retriever:
    def __init__(self):
        self.store = VectorStore()

    def retrieve(self, query: str, ma_nganh: str = "", top_k: int = 2) -> str:
        # Tự động gộp ma_nganh vào query để search vector cho chuẩn
        search_query = f"{query} {ma_nganh}".strip()
        results = self.store.search(search_query, top_k=15)
        
        if not results['documents'] or not results['documents'][0]:
            return "Không có thông tin liên quan trong cơ sở dữ liệu."
            
        retrieved_docs = results['documents'][0]
        
        # Simple Re-ranking bằng keyword matching
        query_words = search_query.lower().split()
        
        scored_docs = []
        for doc in retrieved_docs:
            doc_lower = doc.lower()
            score: int = 0
            # CỘNG ĐIỂM NẶNG cho các từ khóa cốt lõi (Mã trường, Tên trường chính xác)
            if "bka" in query_words and "bka" in doc_lower: score += 10
            if "bách khoa" in query.lower() and "bách khoa" in doc_lower: score += 10
            if "uit" in query_words and "uit" in doc_lower: score += 10
            
            # Boost trực tiếp cho đúng mô tả của ngành
            if ma_nganh and ma_nganh in doc_lower: 
                score += 20
            
            for word in query_words:
                if len(word) > 2 and word in doc_lower:
                    score += 1
            scored_docs.append((score, doc))
            
        # Sắp xếp lại theo điểm keyword (giảm dần)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Lấy top_k văn bản sau khi rerank
        best_docs = [doc for doc_score, doc in scored_docs[0:top_k]]
        
        context = "\n\n".join(best_docs)
        return context

if __name__ == "__main__":
    r = Retriever()
    print("Kết quả tìm kiếm cho query 'Bách Khoa Hà Nội':")
    print(r.retrieve("Thông tin về trường Đại học Bách Khoa Hà Nội"))
