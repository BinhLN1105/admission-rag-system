from src.rag.vector_store import VectorStore

class Retriever:
    def __init__(self):
        self.store = VectorStore()

    def retrieve(self, query: str, top_k: int = 8) -> str:
        # Đôi khi mô hình lấy thiếu kết quả nếu top_k ban đầu nhỏ. Ta sẽ query lấy 15 kết quả, vì n_results giới hạn theo size của collection (=17).
        results = self.store.search(query, top_k=15)
        
        if not results['documents'] or not results['documents'][0]:
            return "Không có thông tin liên quan trong cơ sở dữ liệu."
            
        retrieved_docs = results['documents'][0]
        
        # Simple Re-ranking bằng keyword matching
        query_words = query.lower().split()
        
        scored_docs = []
        for doc in retrieved_docs:
            doc_lower = doc.lower()
            score = 0
            # CỘNG ĐIỂM NẶNG cho các từ khóa cốt lõi (Mã trường, Tên trường chính xác)
            if "bka" in query_words and "bka" in doc_lower: score += 10
            if "bách khoa" in query.lower() and "bách khoa" in doc_lower: score += 10
            if "uit" in query_words and "uit" in doc_lower: score += 10
            if "7480107" in query_words and "7480107" in doc_lower: score += 10
            
            for word in query_words:
                if len(word) > 2 and word in doc_lower:
                    score += 1
            scored_docs.append((score, doc))
            
        # Sắp xếp lại theo điểm keyword (giảm dần)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Lấy top_k văn bản sau khi rerank
        best_docs = [doc for score, doc in scored_docs[:top_k]]
        
        context = "\n\n".join(best_docs)
        return context

if __name__ == "__main__":
    r = Retriever()
    print("Kết quả tìm kiếm cho query 'Bách Khoa Hà Nội':")
    print(r.retrieve("Thông tin về trường Đại học Bách Khoa Hà Nội"))
