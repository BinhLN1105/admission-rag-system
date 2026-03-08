from pydantic import BaseModel
from typing import Optional

class InferenceRequest(BaseModel):
    diem_thi: float
    khu_vuc: str
    ma_nganh: str
    to_hop: str
    query: Optional[str] = ""

class InferenceResponse(BaseModel):
    ket_qua: str
