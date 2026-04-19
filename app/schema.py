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
    facts: Optional[dict] = None
    ket_qua_llm: Optional[str] = None

# Schema riêng cho Chatbot (không cần form data, chỉ cần câu hỏi tự do)
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    reply_llm: Optional[str] = None
