import os
import pandas as pd
from fastapi import APIRouter
from app.schema import InferenceRequest, InferenceResponse
from src.pipeline.inference_pipeline import InferencePipeline

router = APIRouter()
pipeline = InferencePipeline()

@router.post("/tu-van", response_model=InferenceResponse)
async def tu_van_tuyen_sinh(request: InferenceRequest):
    result = pipeline.run(
        query=request.query,
        ma_nganh=request.ma_nganh,
        to_hop=request.to_hop,
        diem=request.diem_thi,
        khu_vuc=request.khu_vuc
    )
    
    return InferenceResponse(ket_qua=result)

@router.get("/majors")
async def get_majors():
    """
    Trả về danh sách tất cả các Mã ngành và Tên ngành duy nhất có trong dữ liệu huấn luyện.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "data", "processed", "diem_chuan_full.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Chỉ lấy mã ngành và tên ngành
        unique_majors = df[['ma_nganh', 'ten_nganh']].drop_duplicates()
        # Chuyển đổi thành string nếu mã ngành đang là Int để hiển thị đồng nhất
        unique_majors['ma_nganh'] = unique_majors['ma_nganh'].astype(str)
        # Sắp xếp theo tên ngành cho dễ tìm
        unique_majors = unique_majors.sort_values(by="ten_nganh").to_dict('records')
        return {"majors": unique_majors}
    
    return {"majors": []}
