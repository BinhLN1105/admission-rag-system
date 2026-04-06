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
    Trả về danh sách tất cả các Mã ngành và Tên ngành duy nhất đã được chuẩn hóa.
    Đảm bảo mỗi tên ngành chỉ xuất hiện 1 lần với mã chuẩn nhất (đầu 7).
    """
    print("DEBUG: Calling /api/majors")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "data", "ml_processed_data.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Sử dụng cột chuẩn hóa
        col_ma = 'ma_nganh_chuan' if 'ma_nganh_chuan' in df.columns else 'ma_nganh'
        col_ten = 'ten_nganh_chuan' if 'ten_nganh_chuan' in df.columns else 'ten_nganh'
        
        # 1. Lọc mã ngành 7 chữ số (ưu tiên đầu số 7)
        df[col_ma] = df[col_ma].astype(str)
        df = df[df[col_ma].str.fullmatch(r'\d{7}')]
        
        # 2. Đếm tần suất cặp (Tên, Mã) để tìm tên chuẩn cho mỗi mã
        counts = df.groupby([col_ma, col_ten]).size().reset_index(name='count')
        
        # 3. Với mỗi Mã ngành (col_ma), chọn Tên ngành (col_ten) chuẩn nhất
        # Ưu tiên các tên ngắn hơn (thường là tên gốc) và xuất hiện nhiều nhất
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
