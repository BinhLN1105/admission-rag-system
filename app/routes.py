from fastapi import APIRouter
from app.schema import InferenceRequest, InferenceResponse
from src.pipeline.inference_pipeline import InferencePipeline

router = APIRouter()
pipeline = InferencePipeline()

@router.post("/tu-van", response_model=InferenceResponse)
async def tu_van_tuyen_sinh(request: InferenceRequest):
    # Lấy thông tin điểm chuẩn mô phỏng (trong thực tế sẽ query từ DB)
    # Ở đây chúng ta cho giả định vài con số
    dc_2023 = 26.5
    dc_2024 = 27.0
    dc_2025 = 27.5
    
    result = pipeline.run(
        query=request.query,
        diem=request.diem_thi,
        khu_vuc=request.khu_vuc,
        dc_2023=dc_2023,
        dc_2024=dc_2024,
        dc_2025=dc_2025
    )
    
    return InferenceResponse(ket_qua=result)
