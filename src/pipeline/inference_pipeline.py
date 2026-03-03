import os
from src.rag.retriever import Retriever
from src.ml_model.predict import predict_probability
import requests

# Trong thực tế, bạn có thể gọi thẳng thư viện của LLM (ví dụ google-generativeai)
# Ở đây dùng một API ảo hoặc code demo

class InferencePipeline:
    def __init__(self):
        self.retriever = Retriever()

    def run(self, query: str, diem: float, khu_vuc: str, dc_2023: float, dc_2024: float, dc_2025: float) -> str:
        # Bước 1: RAG
        # Tương tự, giảm số top_k lấy cuối cùng xuống còn 2 để prompt sinh ra được trong, không nhiễu.
        context = self.retriever.retrieve(query, top_k=2)

        # Bước 2: ML Dự đoán
        ml_res = predict_probability(
            diem_thi_sinh=diem,
            khu_vuc=khu_vuc,
            diem_chuan_2023=dc_2023,
            diem_chuan_2024=dc_2024,
            diem_chuan_2025=dc_2025
        )

        prob_str = ml_res["phan_tram"]
        danh_gia = ml_res["danh_gia"]

        # Bước 3: Tổng hợp (LLM prompt)
        prompt = f"""Bạn là một chuyên gia tư vấn tuyển sinh đại học nhiệt tình và thân thiện.
        Dựa vào kết quả tìm kiếm (RAG):
        {context}
        
        Và kết quả dự đoán (Machine Learning):
        - Điểm thi thí sinh (đã cộng ưu tiên khu vực): {ml_res['diem_co_uu_tien']}
        - Cơ hội trúng tuyển: {prob_str}
        - Lời khuyên: {danh_gia}
        
        Câu hỏi của thí sinh: {query}
        
        Hãy viết câu trả lời kết hợp thông tin trên một cách tự nhiên, súc tích (khoảng 3-4 câu).
        """

        # Trả về prompt cho frontend hoặc gọi thẳng Gemini/OpenAI tại đây
        # Ví dụ gọi giả lập:
        
        response = f"Theo thông tin tìm thấy:\n{context}\n\nVới điểm số {diem} (cộng ưu tiên KV {khu_vuc} thành {ml_res['diem_co_uu_tien']}), dự đoán khả năng trúng tuyển của bạn là {prob_str} ({danh_gia}).\n\n[PROMPT LLM MẪU]:\n{prompt}"
        return response

if __name__ == "__main__":
    pipeline = InferencePipeline()
    ket_qua = pipeline.run(
        query="Khoa học máy tính Bách Khoa",
        diem=27.5,
        khu_vuc="KV2",
        dc_2023=27.5,
        dc_2024=28.0,
        dc_2025=27.8
    )
    print(ket_qua)
