import os
import re
import pandas as pd
from src.rag.retriever import Retriever
from src.ml_model.predict import predict_probability

# Trong thực tế, bạn có thể gọi thẳng thư viện của LLM (ví dụ google-generativeai)
# Ở đây dùng một API ảo hoặc code demo

class InferencePipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.df_diem_chuan = pd.read_csv(os.path.join(self.base_dir, "data", "processed", "diem_chuan_full.csv"))

    def run(self, query: str, ma_nganh: str, to_hop: str, diem: float, khu_vuc: str) -> str:
        # Nếu người dùng để trống câu hỏi -> Kích hoạt chế độ tìm kiếm Top 3 trường
        if not query or query.strip() == "":
            return self._auto_recommend_top_3(ma_nganh, to_hop, diem, khu_vuc)

        # Bước 1: RAG
        # Tương tự, giảm số top_k lấy cuối cùng xuống còn 2 để prompt sinh ra được trong, không nhiễu.
        context = self.retriever.retrieve(query=query, ma_nganh=ma_nganh, top_k=2)

        # Trích xuất mã trường từ context RAG (Phụ thuộc vào format "[Tên Trường - Mã: BKA]")
        # Lọc bỏ các mã chứa toàn chữ số vì đó là mã ngành (ví dụ: 7480201)
        raw_matches = re.findall(r'- Mã:\s*([A-Z0-9]+)\]', context)
        truong_matches = [m for m in raw_matches if not m.isdigit()]
        
        if not truong_matches:
            return f"Theo thông tin tìm thấy:\n{context}\n\n⚠️ **Lưu ý:** Không nhận diện được bạn đang hỏi trường nào trong câu hỏi. Vui lòng ghi lại câu hỏi có tên trường rõ ràng (VD: Bách Khoa, BKA, Khoa học Tự nhiên...) hoặc để trống câu hỏi để AI tự động tìm trường cho bạn."
        
        # Giả định lấy trường đầu tiên xuất hiện trong top K
        ma_truong = truong_matches[0]
        
        # Tra cứu điểm chuẩn thật của Trường + Ngành + Tổ hợp đó
        school_majors = self.df_diem_chuan[
            (self.df_diem_chuan["ma_truong"] == ma_truong) & 
            (self.df_diem_chuan["ma_nganh"].astype(str) == str(ma_nganh)) &
            (self.df_diem_chuan["ma_to_hop"].astype(str) == str(to_hop))
        ]
        
        if school_majors.empty:
             return f"⚠️ **Thông báo quan trọng:** Trường **{ma_truong}** hiện **không áp dụng xét tuyển** môn thi **{to_hop}** cho ngành có mã **{ma_nganh}** mà bạn chọn. Bạn có thể nộp bằng khối khác, trường khác hoặc để trống ô câu hỏi để hệ thống tự tìm trường phù hợp nhé!\n\nThông tin tham khảo:\n{context}"
             
        # Lấy điểm chuẩn 3 năm của đúng tổ hợp điểm cao nhất (hoặc lấy max/mean)
        dc_2023 = school_majors[school_majors["nam"] == 2023]["diem_chuan"].max()
        dc_2024 = school_majors[school_majors["nam"] == 2024]["diem_chuan"].max()
        dc_2025 = school_majors[school_majors["nam"] == 2025]["diem_chuan"].max()
        
        # Điền khuyết nếu thiếu năm
        if pd.isna(dc_2023): dc_2023 = dc_2025 if not pd.isna(dc_2025) else 25.0
        if pd.isna(dc_2024): dc_2024 = dc_2025 if not pd.isna(dc_2025) else 25.0
        if pd.isna(dc_2025): dc_2025 = dc_2024 if not pd.isna(dc_2024) else 25.0

        # Bước 2: ML Dự đoán
        ml_res = predict_probability(
            diem_thi_sinh=diem,
            khu_vuc=khu_vuc,
            diem_chuan_2023=float(dc_2023),
            diem_chuan_2024=float(dc_2024),
            diem_chuan_2025=float(dc_2025)
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

    def _auto_recommend_top_3(self, ma_nganh: str, to_hop: str, diem: float, khu_vuc: str) -> str:
        # Lọc tất cả các trường có đào tạo ngành này BẰNG tổ hợp này
        df_nganh = self.df_diem_chuan[
            (self.df_diem_chuan["ma_nganh"].astype(str) == str(ma_nganh)) & 
            (self.df_diem_chuan["ma_to_hop"].astype(str) == str(to_hop))
        ]
        
        if df_nganh.empty:
            return f"⚠️ **Xin lỗi:** Hiện tại không có trường nào xét tuyển ngành mã **{ma_nganh}** bằng mức tổ hợp **{to_hop}**. Vui lòng thao tác lại với ngành khác hoặc khối thi khác nhé!"
            
        truong_list = df_nganh["ma_truong"].unique()
        results: list[dict[str, str | float]] = []
        
        for ma_truong in truong_list:
            school_data = df_nganh[df_nganh["ma_truong"] == ma_truong]
            ten_truong = school_data["ten_truong"].iloc[0]
            ten_nganh = school_data["ten_nganh"].iloc[0]
            
            dc_2023 = school_data[school_data["nam"] == 2023]["diem_chuan"].max()
            dc_2024 = school_data[school_data["nam"] == 2024]["diem_chuan"].max()
            dc_2025 = school_data[school_data["nam"] == 2025]["diem_chuan"].max()
            
            if pd.isna(dc_2023): dc_2023 = dc_2025 if not pd.isna(dc_2025) else 25.0
            if pd.isna(dc_2024): dc_2024 = dc_2025 if not pd.isna(dc_2025) else 25.0
            if pd.isna(dc_2025): dc_2025 = dc_2024 if not pd.isna(dc_2024) else 25.0
            
            ml_res = predict_probability(
                diem_thi_sinh=diem,
                khu_vuc=khu_vuc,
                diem_chuan_2023=float(dc_2023),
                diem_chuan_2024=float(dc_2024),
                diem_chuan_2025=float(dc_2025)
            )
            
            # Lưu lại cả raw probability để sort
            prob_raw = float(ml_res["phan_tram"].replace("%", ""))
            
            results.append({
                "ma_truong": ma_truong,
                "ten_truong": ten_truong,
                "ten_nganh": ten_nganh,
                "prob_raw": prob_raw,
                "prob_str": ml_res["phan_tram"],
                "danh_gia": ml_res["danh_gia"],
                "diem_co_uu_tien": ml_res["diem_co_uu_tien"]
            })
            
        # Sắp xếp giảm dần theo xác suất đỗ
        results.sort(key=lambda x: x["prob_raw"], reverse=True)
        
        # Lấy top 3
        top_k = min(3, len(results))
        top_results = results[:top_k]
        
        # Build response
        nganh_name = top_results[0]["ten_nganh"] if top_results else ma_nganh
        diem_uutien = top_results[0]["diem_co_uu_tien"] if top_results else diem
        
        res_str = f"🔍 **GỢI Ý TỰ ĐỘNG TOP {top_k} TRƯỜNG DÀNH CHO BẠN**\n\n"
        res_str += f"Dựa trên mức điểm **{diem}** (Cộng ưu tiên {khu_vuc} thành **{diem_uutien}**) và nguyện vọng học ngành **{nganh_name} ({ma_nganh})**, hệ thống AI đề xuất các trường sau:\n\n"
        
        for i, res in enumerate(top_results):
            res_str += f"**{i+1}. {res['ten_truong']} ({res['ma_truong']})**\n"
            res_str += f"- Khả năng trúng tuyển: **{res['prob_str']}**\n"
            res_str += f"- Đánh giá AI: *{res['danh_gia']}*\n\n"
            
        return res_str.strip()

if __name__ == "__main__":
    pipeline = InferencePipeline()
    ket_qua = pipeline.run(
        query="Khoa học máy tính Bách Khoa",
        ma_nganh="IT1",
        to_hop="A00",
        diem=27.5,
        khu_vuc="KV2"
    )
    print(ket_qua)
