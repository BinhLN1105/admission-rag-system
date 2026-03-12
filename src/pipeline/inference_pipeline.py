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
        self.df_diem_chuan = pd.read_csv(os.path.join(self.base_dir, "data", "ml_processed_data.csv"))

    def run(self, query: str, ma_nganh: str, to_hop: str, diem: float, khu_vuc: str) -> str:
        # Nếu người dùng để trống câu hỏi -> Kích hoạt chế độ tìm kiếm Top 3 trường
        if not query or query.strip() == "":
            return self._auto_recommend_top_3(ma_nganh, to_hop, diem, khu_vuc)

        # Bước 1: Trích xuất sớm mã trường từ query để ép RAG tìm đúng
        # Ánh xạ các tên gọi phổ biến (như UIT) sang mã trường chính thức trong hệ thống (QSC)
        COMMON_ALIASES = {
            "UIT": "QSC", "HCMUT": "QSB", "KHTN": "QST", "USSH": "QSX",
            "UEH": "KSA", "NEU": "KHA", "FTU": "NTH", "BK": "QSB"
        }
        
        query_upper = query.upper()
        # Thay thế alias trong query để engine nhận diện (vd UIT -> QSC)
        for alias, real_code in COMMON_ALIASES.items():
            if f"\\b{alias}\\b" in query_upper or f" {alias} " in f" {query_upper} " or f"^{alias} " in f"{query_upper} " or f" {alias}$" in f" {query_upper}":
                query_upper = query_upper.replace(alias, real_code)
                
        query_caps = set(re.findall(r'\b[A-Z]{2,4}\b', query_upper))
        all_school_codes = set(self.df_diem_chuan['ma_truong'].unique())
        valid_codes = query_caps.intersection(all_school_codes)
        ma_truong_query = list(valid_codes)[0] if valid_codes else ""
        
        # Bước 2: RAG (Yêu cầu Retriever ưu tiên lọc theo mã ngành, mã trường, khối)
        context = self.retriever.retrieve(query=query, ma_nganh=ma_nganh, to_hop=to_hop, ma_truong=ma_truong_query, top_k=2)

        # Trích xuất mã trường từ context RAG (nếu người dùng ko gõ mã trường)
        # RAG sinh dữ liệu dạng: "Vào năm 2024, trường Đại học Bách khoa Hà Nội (mã trường: BKA)..."
        raw_matches = re.findall(r'(?i)mã\s*(?:trường)?\s*:\s*([A-Za-z0-9]+)', context)
        # Loại bỏ các mã toàn bộ là chữ số (bởi đó thường là mã ngành như 7480201)
        truong_matches = [m.upper() for m in raw_matches if not m.isdigit()]
        
        # Nếu RAG vẫn có, ta ghi đè ma_truong_query
        # Nếu không có, xài ma_truong_query trích xuất từ câu hỏi ban đầu
        if truong_matches:
            ma_truong_query = truong_matches[0]
            
        if not ma_truong_query:
            return f"Theo thông tin tìm kiếm:\n{context}\n\n⚠️ **Lưu ý:** Không nhận diện được bạn đang hỏi trường nào trong câu hỏi. Vui lòng ghi lại câu hỏi có tên trường rõ ràng (VD: Bách Khoa, BKA, Khoa học Tự nhiên...) hoặc để trống câu hỏi để AI tự động tìm trường cho bạn."
        
        ma_truong = ma_truong_query
        
        # Tra cứu điểm chuẩn thật của Trường + Ngành + Tổ hợp đó
        school_majors = self.df_diem_chuan[
            (self.df_diem_chuan["ma_truong"] == ma_truong) & 
            (self.df_diem_chuan["ma_nganh_chuan"].astype(str) == str(ma_nganh)) &
            (self.df_diem_chuan["ma_to_hop"].astype(str) == str(to_hop))
        ]
        
        # Fallback: nếu không có tổ hợp yêu cầu, tìm tổ hợp khác cùng trường+ngành
        fallback_note = ""
        if school_majors.empty:
            school_any_tohop = self.df_diem_chuan[
                (self.df_diem_chuan["ma_truong"] == ma_truong) & 
                (self.df_diem_chuan["ma_nganh_chuan"].astype(str) == str(ma_nganh))
            ]
            if not school_any_tohop.empty:
                available_tohops = school_any_tohop["ma_to_hop"].unique().tolist()
                fallback_note = f"\n\n⚠️ **Lưu ý:** Trường **{ma_truong}** không xét khối **{to_hop}** cho ngành này. Các khối có sẵn: **{', '.join(available_tohops)}**. Dự đoán dưới đây dựa trên dữ liệu tổng hợp các khối có sẵn."
                school_majors = school_any_tohop
            else:
                return f"⚠️ **Thông báo:** Không tìm thấy ngành mã **{ma_nganh}** tại trường **{ma_truong}**. Hãy để trống câu hỏi để hệ thống tự tìm trường phù hợp!\n\nThông tin tham khảo:\n{context}"
             
        # Kiểm tra xem ngành này có chia làm nhiều hệ (VD: đại trà, chất lượng cao) không
        sub_majors_count = school_majors["ma_nganh"].nunique()
        disclaimer_note = ""
        if sub_majors_count > 1:
            disclaimer_note = f"\n\n⚠️ **Lưu ý phụ:** Ngành này tại trường {ma_truong} có {sub_majors_count} chương trình/hệ đào tạo khác nhau. Điểm chuẩn dưới đây dùng để dự đoán ML đã được lấy trung bình từ các hệ đó."
            
        # Lấy điểm chuẩn 3 năm
        # CHÚ Ý: Sử dụng .mean() thay vì .max() hay .min() để lấy mức trung bình hợp lý nhất cho toàn bộ các hệ/nhánh của ngành này.
        dc_2023 = school_majors["diem_chuan_2023"].mean()
        dc_2024 = school_majors["diem_chuan_2024"].mean()
        dc_2025 = school_majors["diem_chuan_2025"].mean()
        
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

        response = f"Theo thông tin tìm thấy:\n{context}\n\nVới điểm số {diem} (cộng ưu tiên KV {khu_vuc} thành {ml_res['diem_co_uu_tien']}), dự đoán khả năng trúng tuyển của bạn là {prob_str} ({danh_gia}).{fallback_note}{disclaimer_note}\n\n[PROMPT LLM MẪU]:\n{prompt}"
        return response

    def _auto_recommend_top_3(self, ma_nganh: str, to_hop: str, diem: float, khu_vuc: str) -> str:
        # Lọc tất cả các trường có đào tạo ngành này BẰNG tổ hợp này
        df_nganh = self.df_diem_chuan[
            (self.df_diem_chuan["ma_nganh_chuan"].astype(str) == str(ma_nganh)) & 
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
            
            dc_2023 = school_data["diem_chuan_2023"].mean()
            dc_2024 = school_data["diem_chuan_2024"].mean()
            dc_2025 = school_data["diem_chuan_2025"].mean()
            
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
                "ten_nganh_chuan": school_data["ten_nganh_chuan"].iloc[0] if "ten_nganh_chuan" in school_data.columns else ten_nganh,
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
        nganh_name = top_results[0]["ten_nganh_chuan"] if top_results else ma_nganh
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
