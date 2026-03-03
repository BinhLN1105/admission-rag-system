from pydantic import BaseModel

class InferenceRequest(BaseModel):
    diem_thi: float
    khu_vuc: str
    ma_nganh: str
    query: str

class InferenceResponse(BaseModel):
    ket_qua: str
