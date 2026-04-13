import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from app.schema import InferenceRequest, InferenceResponse

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
    """Endpoint tư vấn tuyển sinh bằng AI"""
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


@router.get("/majors")
async def get_majors():
    """
    Trả về danh sách ngành học để frontend dùng cho dropdown tìm kiếm
    """
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(BASE_DIR, "data", "ml_processed_data.csv")

        if not os.path.exists(csv_path):
            print(f"⚠️ Không tìm thấy file dữ liệu: {csv_path}")
            return {"majors": []}

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
        
        return {"majors": majors.to_dict('records')}

    except Exception as e:
        print(f"❌ Lỗi khi lấy danh sách majors: {e}")
        return {"majors": []}