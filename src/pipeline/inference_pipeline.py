import os
import re
import difflib
import pandas as pd
from src.rag.retriever import Retriever
from src.ml_model.predict import predict_probability

import json

class InferencePipeline:
    def __init__(self):
        # Load data paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(current_dir, "../../data"))
        
        # 1. Load Mapping Configuration (Decoupled from code)
        config_path = os.path.join(data_dir, "school_config.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.abbreviations = config.get("abbreviations", {})
                self.token_weights = config.get("token_weights", {})
                self.noise_pattern = re.compile(config.get("noise_pattern", ""))
        except Exception as e:
            print(f"Warning: Could not load school_config.json: {e}")
            self.abbreviations = {}
            self.token_weights = {}
            self.noise_pattern = re.compile(r'')

        # 2. Load Score Database
        data_path = os.path.join(data_dir, "ml_processed_data.csv")
        try:
            self.df_diem_chuan = pd.read_csv(data_path)
            # Pre-compute normalized names for fuzzy matching
            self._school_data = []
            for ma_truong in self.df_diem_chuan['ma_truong'].unique():
                name = self.df_diem_chuan[self.df_diem_chuan['ma_truong'] == ma_truong]['ten_truong'].iloc[0]
                norm_name = self._normalize_school_name(name)
                tokens = set(norm_name.split())
                self._school_data.append({
                    'ma_truong': ma_truong,
                    'name_norm': norm_name,
                    'tokens': tokens,
                    'is_hcm': any(k in norm_name for k in ['hồ chí minh', 'hcm', 'tphcm']),
                    'is_hn': any(k in norm_name for k in ['hà nội', 'hn']),
                    'is_dn': any(k in norm_name for k in ['đà nẵng', 'đn']),
                    'is_ct': any(k in norm_name for k in ['cần thơ', 'ct'])
                })
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df_diem_chuan = pd.DataFrame()
            self._school_data = []
        
        # Init components
        self.retriever = Retriever()

    def _normalize_school_name(self, name: str) -> str:
        name = str(name).lower()
        name = re.sub(r'[.,]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def _expand_abbreviations(self, query: str) -> str:
        query = query.lower()
        # Handle dot-separated abbreviations like T.P.H.C.M
        query = re.sub(r'([a-z])\.([a-z])', r'\1\2', query)
        for abbr, full in self.abbreviations.items():
            query = re.sub(rf'\b{abbr}\b', full, query)
        return query

    def _fuzzy_find_school(self, query: str) -> str:
        query_expanded = self._expand_abbreviations(query)
        query_clean = self.noise_pattern.sub('', query_expanded).strip()
        query_tokens = set(query_clean.split())
        
        has_hcm_keyword = any(k in query_expanded for k in ['hồ chí minh', 'hcm', 'tphcm'])
        has_hn_keyword = any(k in query_expanded for k in ['hà nội', 'hn'])
        has_dn_keyword = any(k in query_expanded for k in ['đà nẵng', 'đn'])
        has_ct_keyword = any(k in query_expanded for k in ['cần thơ', 'ct'])

        best_score = -1
        best_ma = ""

        for school in self._school_data:
            # Substring match (High priority)
            if query_clean in school['name_norm'] or school['name_norm'] in query_clean:
                score = 0.9
            else:
                # Weighted token score
                common_tokens = query_tokens.intersection(school['tokens'])
                if not common_tokens:
                    continue
                
                score = 0.0
                for token in common_tokens:
                    score += self.token_weights.get(token, 1.0)
                
                # Normalize score
                score = score / (len(query_tokens) + 1)

            # Location Penalty/Bonus (More strict)
            if has_hcm_keyword:
                if school['is_hcm']: score += 0.4
                else: score -= 0.5
            if has_hn_keyword:
                if school['is_hn']: score += 0.4
                else: score -= 0.5
            if has_dn_keyword:
                if school['is_dn']: score += 0.4
                else: score -= 0.5
            if has_ct_keyword:
                if school['is_ct']: score += 0.4
                else: score -= 0.5

            if score > best_score:
                best_score = score
                best_ma = school['ma_truong']

        return best_ma if best_score > 0.4 else ""

    def run(self, query: str, ma_nganh: str, to_hop: str, diem: float, khu_vuc: str) -> str:
        if diem > 31:
            return "Xin thứ lỗi vì hệ thống hiện tại chỉ có thể đưa ra kết quả dự đoán trên thang điểm 30 (tổ hợp môn thi thpt quốc gia). Điểm bạn nhập vượt quá 30 điểm nên chúng tôi rất xin lỗi vì sự bất tiện này. Bạn hãy sử dụng tính năng tính toán điểm tự động để có thể đưa ra kết quả chính xác nhất."
            
        if not query or len(query.strip()) < 2:
            return self._auto_recommend_top_3(ma_nganh, to_hop, diem, khu_vuc)

        # Bước 1: Trích xuất mã trường từ query
        query_upper = query.upper()
        query_caps = set(re.findall(r'\b[A-Z]{2,4}\b', query_upper))
        all_school_codes = set(self.df_diem_chuan['ma_truong'].unique())
        valid_codes = query_caps.intersection(all_school_codes)
        ma_truong_query = list(valid_codes)[0] if valid_codes else ""
        
        if not ma_truong_query:
            ma_truong_query = self._fuzzy_find_school(query)

        if not ma_truong_query:
            context = self.retriever.retrieve(query=query, ma_nganh=ma_nganh, to_hop=to_hop, top_k=2)
            return f"Theo thông tin tìm kiếm:\n{context}\n\n⚠️ **Lưu ý:** Không nhận diện được bạn đang hỏi trường nào. Vui lòng ghi rõ tên trường (VD: Bách Khoa, BKA...) hoặc để trống câu hỏi để AI tự động tìm trường."
            
        ma_truong = ma_truong_query
        
        # Bước 2: Tra cứu điểm chuẩn và Logic Mapping Ngành
        IT_CODES = ["7480201", "7480101", "7480103", "7480106", "7480202", "7480104", "7480109", "7480107", "7460108"]
        BIZ_CODES = ["7340101", "7340201", "7340301", "7340115", "7310101", "7340120", "7340122"]
        
        # Kiểm tra xem trường có ngành này không (bất kể tổ hợp môn)
        school_any_tohop = self.df_diem_chuan[
            (self.df_diem_chuan["ma_truong"] == ma_truong) & 
            (self.df_diem_chuan["ma_nganh_chuan"].astype(str) == str(ma_nganh))
        ]
        
        # 1. Nếu hoàn toàn không có ngành này tại trường -> Báo lỗi & gợi ý ngành cùng nhóm
        if school_any_tohop.empty:
            similar_majors_msg = ""
            all_school_majors = self.df_diem_chuan[self.df_diem_chuan["ma_truong"] == ma_truong]
            if str(ma_nganh) in IT_CODES:
                similar = all_school_majors[all_school_majors["ma_nganh_chuan"].astype(str).isin(IT_CODES)]
                if not similar.empty:
                    names = similar["ten_nganh_chuan"].unique()
                    similar_majors_msg = f"\n\n💡 **Gợi ý:** Tuy nhiên, trường có các ngành tương đương thuộc nhóm **Công nghệ thông tin**: **{', '.join(names)}**. Bạn có thể thử hỏi lại với tên các ngành này."
            elif str(ma_nganh) in BIZ_CODES:
                similar = all_school_majors[all_school_majors["ma_nganh_chuan"].astype(str).isin(BIZ_CODES)]
                if not similar.empty:
                    names = similar["ten_nganh_chuan"].unique()
                    similar_majors_msg = f"\n\n💡 **Gợi ý:** Tuy nhiên, trường có các ngành tương đương thuộc nhóm **Kinh tế - Quản lý**: **{', '.join(names)}**. Bạn có thể thử hỏi lại với tên các ngành này."
            
            # RAG theo tên trường + query thay vì dùng mã ngành không tồn tại
            context = self.retriever.retrieve(query=f"{query} {ma_truong}", top_k=2)
            return f"⚠️ **Thông báo:** Rất tiếc, trường **{ma_truong}** không tuyển sinh ngành có mã **{ma_nganh}** cho bất kỳ tổ hợp môn nào.{similar_majors_msg}\n\nThông tin tham khảo từ RAG:\n{context}"

        # 2. Nếu có ngành nhưng không có tổ hợp môn yêu cầu
        school_majors = school_any_tohop[school_any_tohop["ma_to_hop"].astype(str) == str(to_hop)]
        fallback_note = ""
        if school_majors.empty:
            available_tohops = school_any_tohop["ma_to_hop"].unique().tolist()
            fallback_note = f"\n\n⚠️ **Lưu ý:** Trường **{ma_truong}** không xét khối **{to_hop}** cho ngành này. Các khối có sẵn: **{', '.join(available_tohops)}**. Dự đoán dưới đây dựa trên dữ liệu tổng hợp các khối có sẵn."
            school_majors = school_any_tohop

        # Bước 3: RAG
        context = self.retriever.retrieve(query=query, ma_nganh=ma_nganh, to_hop=to_hop, ma_truong=ma_truong, top_k=2)

        # Chi tiết hệ đào tạo
        program_details = []
        for idx, row in school_majors.drop_duplicates(subset=['ma_nganh']).iterrows():
            bench_24 = row['diem_chuan_2024'] if not pd.isna(row['diem_chuan_2024']) else "N/A"
            program_details.append(f"{row['ten_nganh']}: {bench_24}")
        
        sub_majors_count = len(program_details)
        disclaimer_note = ""
        if sub_majors_count > 1:
            details_str = "\n   - " + "\n   - ".join(program_details[:5])
            if sub_majors_count > 5: details_str += "\n   - ..."
            disclaimer_note = f"\n\n⚠️ **Lưu ý về hệ đào tạo:** Ngành này tại trường {ma_truong} có {sub_majors_count} hệ/chương trình đào tạo. Điểm chuẩn 2024 tham khảo:{details_str}\n   => Dự đoán ML đang tính dựa trên mức **điểm trung bình** của các hệ này."
            
        # Lấy điểm chuẩn 3 năm
        dc_2023 = school_majors["diem_chuan_2023"].mean()
        dc_2024 = school_majors["diem_chuan_2024"].mean()
        dc_2025 = school_majors["diem_chuan_2025"].mean()
        
        if pd.isna(dc_2023): dc_2023 = dc_2025 if not pd.isna(dc_2025) else 25.0
        if pd.isna(dc_2024): dc_2024 = dc_2025 if not pd.isna(dc_2025) else 25.0
        if pd.isna(dc_2025): dc_2025 = dc_2024 if not pd.isna(dc_2024) else 25.0

        max_dc = max([dc_2023, dc_2024, dc_2025])
        if diem <= 31 and max_dc > 35:
            return f"Xin thứ lỗi vì hệ thống hiện tại chỉ có thể đưa ra kết quả dự đoán trên thang điểm 30 (tổ hợp môn thi thpt quốc gia) mà trường **{ma_truong}** với mã ngành **{ma_nganh}** mà bạn tìm kiếm trong các năm gần đây được xét tuyển trên thang điểm riêng (có thể là điểm hệ số 40, ĐGNL 100, 1200...) nên chúng tôi rất xin lỗi vì sự bất tiện này.\n\nThông tin tham khảo từ hệ thống:\n{context}"

        # Bước 4: ML Dự đoán
        ml_res = predict_probability(
            diem_thi_sinh=diem,
            khu_vuc=khu_vuc,
            diem_chuan_2023=float(dc_2023),
            diem_chuan_2024=float(dc_2024),
            diem_chuan_2025=float(dc_2025)
        )

        prob_str = ml_res["phan_tram"]
        danh_gia = ml_res["danh_gia"]

        # Bước 5: Tổng hợp response
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

        response = f"Theo thông tin tìm thấy:\n{context}\n\nVới điểm số {diem} (cộng ưu tiên KV {khu_vuc} thành {ml_res['diem_co_uu_tien']}), xác suất trúng tuyển tính toán bởi **Mô hình Ensemble (AI)** là **{prob_str}** ({danh_gia}).{fallback_note}{disclaimer_note}\n\n[PROMPT LLM MẪU]:\n{prompt}"
        return response

    def _auto_recommend_top_3(self, ma_nganh: str, to_hop: str, diem: float, khu_vuc: str) -> str:
        df_nganh = self.df_diem_chuan[
            (self.df_diem_chuan["ma_nganh_chuan"].astype(str) == str(ma_nganh)) & 
            (self.df_diem_chuan["ma_to_hop"].astype(str) == str(to_hop))
        ]
        
        if df_nganh.empty:
            return f"⚠️ **Xin lỗi:** Hiện tại không có trường nào xét tuyển ngành mã **{ma_nganh}** bằng mức tổ hợp **{to_hop}**."
            
        truong_list = df_nganh["ma_truong"].unique()
        results = []
        
        for ma_truong in truong_list:
            school_data = df_nganh[df_nganh["ma_truong"] == ma_truong]
            ten_truong = school_data["ten_truong"].iloc[0]
            
            dc_2023 = school_data["diem_chuan_2023"].mean()
            dc_2024 = school_data["diem_chuan_2024"].mean()
            dc_2025 = school_data["diem_chuan_2025"].mean()
            
            if pd.isna(dc_2023): dc_2023 = dc_2025 if not pd.isna(dc_2025) else 25.0
            if pd.isna(dc_2024): dc_2024 = dc_2025 if not pd.isna(dc_2025) else 25.0
            if pd.isna(dc_2025): dc_2025 = dc_2024 if not pd.isna(dc_2024) else 25.0
            
            ml_res = predict_probability(diem, khu_vuc, float(dc_2023), float(dc_2024), float(dc_2025))
            results.append({
                "ten_truong": ten_truong,
                "ma_truong": ma_truong,
                "prob": ml_res["xac_suat_do"],
                "phan_tram": ml_res["phan_tram"],
                "danh_gia": ml_res["danh_gia"]
            })
        
        top_results = sorted(results, key=lambda x: x["prob"], reverse=True)[:3]
        res_str = "Hệ thống tự động tìm thấy 3 trường phù hợp nhất cho bạn:\n\n"
        for i, res in enumerate(top_results):
            res_str += f"**{i+1}. {res['ten_truong']} ({res['ma_truong']})**\n"
            res_str += f"- Khả năng trúng tuyển: **{res['phan_tram']}**\n"
            res_str += f"- Đánh giá AI: *{res['danh_gia']}*\n\n"
        return res_str.strip()

if __name__ == "__main__":
    pipeline = InferencePipeline()
    print("Pipeline ready.")
