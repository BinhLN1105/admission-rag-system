import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from app.schema import InferenceRequest, InferenceResponse, ChatRequest, ChatResponse

# ==================== LAZY LOADING PIPELINE ====================
pipeline = None


def get_pipeline():
    """Khởi tạo InferencePipeline chỉ khi cần (Lazy Loading)"""
    global pipeline
    if pipeline is None:
        try:
            from src.pipeline.inference_pipeline import InferencePipeline
            print("🚀 Đang khởi tạo InferencePipeline (SentenceTransformer)...")
            pipeline = InferencePipeline()
            print("✅ InferencePipeline đã load thành công!")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo InferencePipeline: {e}")
            pipeline = None
            raise HTTPException(
                status_code=503,
                detail="Hệ thống AI đang khởi tạo. Vui lòng chờ vài giây và thử lại."
            )
    return pipeline


# ==================== ROUTER ====================
router = APIRouter()



@router.post("/tu-van", response_model=InferenceResponse)
async def tu_van_tuyen_sinh(request: InferenceRequest):
    """Endpoint tư vấn tuyển sinh bằng AI — yêu cầu đầy đủ thông tin từ form"""
    if not request.diem_thi or not request.ma_nganh or not request.to_hop:
        raise HTTPException(status_code=422, detail="Thiếu thông tin bắt buộc")

    try:
        pipe = get_pipeline()
        result = pipe.run(
            query=request.query or "",
            ma_nganh=request.ma_nganh,
            to_hop=request.to_hop,
            diem=request.diem_thi,
            khu_vuc=request.khu_vuc
        )
        return InferenceResponse(ket_qua=result)

    except Exception as e:
        print(f"Lỗi khi chạy pipeline: {e}")
        raise HTTPException(
            status_code=500,
            detail="Đã xảy ra lỗi trong quá trình tư vấn. Vui lòng thử lại sau."
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_tuyen_sinh(request: ChatRequest):
    """
    Chatbot tự do — trích xuất trường + ngành từ câu hỏi, lọc RAG chính xác.
    """
    import re
    import difflib

    msg = request.message.strip()
    if not msg:
        raise HTTPException(status_code=422, detail="Tin nhắn không được để trống.")

    try:
        pipe = get_pipeline()

        # ===== BƯỚC 1: Trích xuất mã trường =====
        query_upper = msg.upper()
        query_caps = set(re.findall(r'\b[A-Z]{2,4}\b', query_upper))
        all_school_codes = set(pipe.df_diem_chuan['ma_truong'].unique())
        valid_codes = query_caps.intersection(all_school_codes)
        ma_truong = list(valid_codes)[0] if valid_codes else ""

        if not ma_truong:
            ma_truong = pipe._fuzzy_find_school(msg)

        # ===== BƯỚC 2: Trích xuất tên ngành → tìm ma_nganh =====
        ma_nganh = ""
        ten_nganh_found = ""
        msg_lower = msg.lower()

        # Lấy danh sách tên ngành unique từ database
        col_ten = 'ten_nganh_chuan' if 'ten_nganh_chuan' in pipe.df_diem_chuan.columns else 'ten_nganh'
        col_ma = 'ma_nganh_chuan' if 'ma_nganh_chuan' in pipe.df_diem_chuan.columns else 'ma_nganh'

        # Nếu có ma_truong, chỉ tìm trong các ngành của trường đó
        if ma_truong:
            scope_df = pipe.df_diem_chuan[pipe.df_diem_chuan['ma_truong'] == ma_truong]
        else:
            scope_df = pipe.df_diem_chuan

        major_names = scope_df[[col_ma, col_ten]].drop_duplicates()

        # Fuzzy match tên ngành từ câu hỏi
        best_ratio = 0
        for _, row in major_names.iterrows():
            ten_nganh = str(row[col_ten]).lower()
            # Kiểm tra substring trực tiếp
            if ten_nganh in msg_lower:
                ma_nganh = str(row[col_ma])
                ten_nganh_found = str(row[col_ten])
                best_ratio = 1.0
                break
            # Fuzzy match từng từ khóa ngành trong câu hỏi
            ratio = difflib.SequenceMatcher(None, ten_nganh, msg_lower).ratio()
            if ratio > best_ratio and ratio > 0.35:
                # Xác minh thêm: ít nhất 1 từ khóa cốt lõi của tên ngành phải có trong câu hỏi
                core_words = [w for w in ten_nganh.split() if len(w) > 2]
                matched_core = sum(1 for w in core_words if w in msg_lower)
                if matched_core >= 1:
                    best_ratio = ratio
                    ma_nganh = str(row[col_ma])
                    ten_nganh_found = str(row[col_ten])

        # Trích xuất tổ hợp môn từ câu hỏi (nếu có)
        to_hop_match = re.search(r'\b([ABCD]\d{2})\b', query_upper)
        to_hop = to_hop_match.group(1) if to_hop_match else ""

        # ===== BƯỚC 3: RAG có lọc trường + ngành =====
        school_header = ""
        if ma_truong:
            school_rows = pipe.df_diem_chuan[pipe.df_diem_chuan["ma_truong"] == ma_truong]
            ten_truong = school_rows["ten_truong"].iloc[0] if not school_rows.empty else ma_truong
            school_header = f"🏫 **{ten_truong} ({ma_truong})**"
            if ten_nganh_found:
                school_header += f" — Ngành: **{ten_nganh_found}**"
            school_header += "\n\n"

        # Gọi RAG với cả ma_truong VÀ ma_nganh
        context = pipe.retriever.retrieve(
            query=msg,
            ma_truong=ma_truong,
            ma_nganh=ma_nganh,
            to_hop=to_hop,
            top_k=3
        )

        if not context or context == "Không có thông tin liên quan trong cơ sở dữ liệu.":
            return ChatResponse(
                reply=(
                    "😔 Mình chưa tìm thấy thông tin phù hợp trong cơ sở dữ liệu.\n\n"
                    "Bạn thử hỏi cụ thể hơn, ví dụ:\n"
                    "- *\"Điểm chuẩn ngành CNTT trường Bách Khoa Hà Nội\"*\n"
                    "- *\"Học viện Kỹ thuật Mật mã tuyển sinh ngành nào?\"*"
                )
            )

        # ===== BƯỚC 4: Trích xuất điểm và nhận định =====
        score_match = re.search(
            r'(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:điểm|đ\b|d\b)',
            msg, re.IGNORECASE
        )
        user_score = float(score_match.group(1).replace(',', '.')) if score_match else None
        
        # Trích xuất Khu Vực từ văn bản
        khu_vuc = "KV3"
        kv_match = re.search(r'\b(KV1|KV2-NT|KV2NT|KV2|KV3)\b', query_upper)
        if kv_match:
            khu_vuc = kv_match.group(1)
            if khu_vuc == "KV2NT": khu_vuc = "KV2-NT"

        verdict = ""
        # 4.1 Thử dự đoán bằng Mô hình Học Máy (ML) nếu có đủ thông tin
        import pandas as pd
        if user_score and ma_truong and ma_nganh:
            row = scope_df[scope_df[col_ma].astype(str) == ma_nganh]
            if not row.empty:
                from src.ml_model.predict import predict_probability
                dc23 = row['diem_chuan_2023'].iloc[0]
                dc24 = row['diem_chuan_2024'].iloc[0]
                dc25 = row['diem_chuan_2025'].iloc[0]
                
                v_23 = float(dc23) if pd.notna(dc23) else 0.0
                v_24 = float(dc24) if pd.notna(dc24) else 0.0
                v_25 = float(dc25) if pd.notna(dc25) else 0.0

                if v_25 <= 0: v_25 = v_24 if v_24 > 0 else 25.0
                if v_24 <= 0: v_24 = v_25 if v_25 > 0 else 25.0
                if v_23 <= 0: v_23 = v_24 if v_24 > 0 else 25.0
                
                try:
                    p_res = predict_probability(
                        diem_thi_sinh=user_score,
                        khu_vuc=khu_vuc,
                        diem_chuan_2023=v_23,
                        diem_chuan_2024=v_24,
                        diem_chuan_2025=v_25
                    )
                    
                    verdict = (
                        f"\n\n🤖 **Dự đoán từ AI Model:**\n"
                        f"- Điểm của bạn (sau cộng ưu tiên {khu_vuc}): **{p_res['diem_co_uu_tien']}**\n"
                        f"- Xác suất trúng tuyển: **{p_res['phan_tram']}**\n"
                        f"- Đánh giá: **{p_res['danh_gia']}**\n"
                    )
                except Exception as e:
                    print(f"ML Prediction Error: {e}")

        # 4.2 Fallback thuật toán tính trung bình văn bản (nếu ML không kích hoạt do thiếu ngành cụ thể)
        if not verdict and user_score:
            dc_matches = re.findall(r'(\d{2}(?:[.,]\d+)?)\s*điểm', context)
            dc_scores = []
            for m in dc_matches:
                try:
                    val = float(m.replace(',', '.'))
                    if 10.0 <= val <= 40.0:
                        dc_scores.append(val)
                except Exception:
                    pass

            if dc_scores:
                avg_dc = sum(dc_scores) / len(dc_scores)
                chenh_lech = round(user_score - avg_dc, 2)
                if chenh_lech >= 2:
                    verdict = (
                        f"\n\n✅ **Nhận định:** Với **{user_score} điểm**, bạn cao hơn điểm chuẩn tham khảo "
                        f"({avg_dc:.1f} điểm) khoảng **+{chenh_lech} điểm** — **Khả năng đỗ khá cao!**"
                    )
                elif chenh_lech >= 0:
                    verdict = (
                        f"\n\n🟡 **Nhận định:** Với **{user_score} điểm**, bạn xấp xỉ điểm chuẩn tham khảo "
                        f"({avg_dc:.1f} điểm) — **Có cơ hội nhưng cạnh tranh cao.**"
                    )
                elif chenh_lech >= -1.5:
                    verdict = (
                        f"\n\n🟠 **Nhận định:** Với **{user_score} điểm**, bạn thấp hơn điểm chuẩn tham khảo "
                        f"({avg_dc:.1f} điểm) khoảng **{abs(chenh_lech)} điểm** — **Cơ hội thấp, nên cân nhắc nguyện vọng phụ.**"
                    )
                else:
                    verdict = (
                        f"\n\n🔴 **Nhận định:** Với **{user_score} điểm**, bạn thấp hơn điểm chuẩn tham khảo "
                        f"({avg_dc:.1f} điểm) đến **{abs(chenh_lech)} điểm** — **Khó đỗ, nên chọn trường/ngành khác.**"
                    )

        reply = (
            f"{school_header}"
            f"{context}"
            f"{verdict}\n\n"
            f"💡 *Để tính xác suất trúng tuyển chính xác hơn, hãy điền thông tin vào **form tư vấn** bên trên nhé!*"
        )
        return ChatResponse(reply=reply)

    except Exception as e:
        print(f"Lỗi chatbot: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Lỗi khi xử lý câu hỏi. Vui lòng thử lại."
        )



@router.get("/majors")
async def get_majors():
    """
    Trả về danh sách tất cả các Mã ngành và Tên ngành duy nhất đã được chuẩn hóa.
    Đảm bảo mỗi tên ngành chỉ xuất hiện 1 lần với mã chuẩn nhất (đầu 7).
    """
    print("DEBUG: Calling /api/majors")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "data", "ml_processed_data.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Linh hoạt với tên cột
        ma_col = 'ma_nganh_chuan' if 'ma_nganh_chuan' in df.columns else 'ma_nganh'
        ten_col = 'ten_nganh_chuan' if 'ten_nganh_chuan' in df.columns else 'ten_nganh'

        if ma_col not in df.columns or ten_col not in df.columns:
            return {"majors": []}

        # Lấy danh sách ngành duy nhất
        majors = (df[[ma_col, ten_col]]
                  .drop_duplicates(subset=[ma_col])
                  .rename(columns={ma_col: 'ma_nganh', ten_col: 'ten_nganh'})
                  .sort_values(by='ten_nganh'))

        # Làm sạch dữ liệu
        majors['ma_nganh'] = majors['ma_nganh'].astype(str).str.strip()
        majors = majors[majors['ten_nganh'].str.strip() != '']
        
        # Sử dụng cột chuẩn hóa
        col_ma = 'ma_nganh_chuan' if 'ma_nganh_chuan' in df.columns else 'ma_nganh'
        col_ten = 'ten_nganh_chuan' if 'ten_nganh_chuan' in df.columns else 'ten_nganh'
        
        # 1. Lọc mã ngành 7 chữ số (ưu tiên đầu số 7)
        df[col_ma] = df[col_ma].astype(str)
        df = df[df[col_ma].str.fullmatch(r'\d{7}')]
        
        # 2. Đếm tần suất cặp (Tên, Mã) để tìm tên chuẩn cho mỗi mã
        counts = df.groupby([col_ma, col_ten]).size().reset_index(name='count')
        
        # 3. Với mỗi Mã ngành (col_ma), chọn Tên ngành (col_ten) chuẩn nhất
        counts['name_len'] = counts[col_ten].str.len()
        counts = counts.sort_values(by=['count', 'name_len'], ascending=[False, True])
        
        # Mỗi mã ngành chỉ lấy 1 tên đại diện duy nhất
        unique_majors = counts.drop_duplicates(subset=[col_ma], keep='first')
        
        # 4. Gọt dũa kết quả cho Frontend
        result = []
        for _, row in unique_majors.iterrows():
            if str(row[col_ten]).strip():
                result.append({
                    "ma_nganh": str(row[col_ma]),
                    "ten_nganh": str(row[col_ten])
                })
        
        # Sắp xếp theo tên cho dễ tìm
        result = sorted(result, key=lambda x: x['ten_nganh'])
        
        return {"majors": result}
    
    return {"majors": []}



