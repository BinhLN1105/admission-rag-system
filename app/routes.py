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
    csv_path = os.path.join(BASE_DIR, "data", "ml_processed_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        col_ma = 'ma_nganh_chuan' if 'ma_nganh_chuan' in df.columns else 'ma_nganh'
        col_ten = 'ten_nganh_chuan' if 'ten_nganh_chuan' in df.columns else 'ten_nganh'
        
        # Đếm tần suất xuất hiện của mỗi cặp (mã, tên) trong toàn bộ dataset
        # Tên nào xuất hiện nhiều nhất cho 1 mã ngành = tên chuẩn nhất
        name_counts = df.groupby([col_ma, col_ten]).size().reset_index(name='count')
        best_names = name_counts.sort_values([col_ma, 'count'], ascending=[True, False])
        unique_majors = best_names.drop_duplicates(subset=[col_ma], keep='first')[[col_ma, col_ten]]
        
        # Rename cho frontend
        unique_majors = unique_majors.rename(columns={col_ma: 'ma_nganh', col_ten: 'ten_nganh'})
        
        # Chuyển đổi thành string và lọc bỏ các dòng tên rỗng
        unique_majors['ma_nganh'] = unique_majors['ma_nganh'].astype(str)
        unique_majors = unique_majors.fillna('')
        unique_majors = unique_majors[unique_majors['ten_nganh'].str.strip() != '']
        
        # Sắp xếp theo tên ngành cho dễ tìm
        unique_majors = unique_majors.sort_values(by="ten_nganh").to_dict('records')
        return {"majors": unique_majors}
    
    return {"majors": []}
