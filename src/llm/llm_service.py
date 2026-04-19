import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "your_gemini_api_key_here":
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.has_llm = True
            except Exception as e:
                print(f"❌ Lỗi cấu hình Gemini: {e}")
                self.has_llm = False
        else:
            self.has_llm = False
            print("⚠️ Cảnh báo: GEMINI_API_KEY chưa được thiết lập. LLM sẽ bị vô hiệu hóa.")

    async def format_form_response(self, facts: dict) -> Optional[str]:
        """
        Diễn đạt lại kết quả tư vấn từ Form bằng ngôn ngữ tự nhiên.
        """
        if not self.has_llm:
            return None

        prompt = f"""
        Bạn là một chuyên gia tư vấn tuyển sinh đại học tại Việt Nam. 
        Dưới đây là các dữ liệu thực tế và kết quả dự đoán từ mô hình Máy học (ML) cho một thí sinh:

        --- DỮ LIỆU THỰC TẾ ---
        - Trường: {facts.get('ten_truong')} ({facts.get('ma_truong')})
        - Ngành: {facts.get('ten_nganh')}
        - Tổ hợp: {facts.get('to_hop')}
        - Điểm chuẩn 2023: {facts.get('dc_2023')}
        - Điểm chuẩn 2024: {facts.get('dc_2024')}
        - Điểm chuẩn 2025: {facts.get('dc_2025')}

        --- KẾT QUẢ DỰ ĐOÁN TỪ ML ---
        - Điểm của thí sinh (đã cộng ưu tiên {facts.get('khu_vuc')}): {facts.get('diem_uu_tien')}
        - Xác suất trúng tuyển: {facts.get('phan_tram')}
        - Đánh giá của AI: {facts.get('danh_gia')}

        --- THÔNG TIN RAG (BỔ SUNG) ---
        {facts.get('context_rag')}

        --- YÊU CẦU ---
        Hãy viết lại kết quả này dưới dạng một lời tư vấn chuyên nghiệp, thân thiện và cá nhân hóa.
        1. TRỰC DIỆN & NGẮN GỌN: Đi thẳng vào phân tích, tuyệt đối KHÔNG có câu chào hỏi (VD: "Chào bạn...", "Với tư cách là...")
        2. Tuyệt đối KHÔNG được tự ý thay đổi các con số: điểm chuẩn, điểm thí sinh, và đặc biệt là % xác suất trúng tuyển.
        3. Tóm tắt ngắn gọn xu hướng điểm chuẩn qua các năm.
        4. Đưa ra nhận định dựa trên "Đánh giá của AI" đã cung cấp.
        5. Trình bày bằng Markdown, sử dụng emoji phù hợp.
        6. Nếu có thông tin bổ sung từ RAG hữu ích (như học phí, chương trình đào tạo), hãy đưa vào một cách khéo léo.
        7. Không được tự ý bịa thêm thông tin không có trong dữ liệu trên.
        8. Tuyệt đối không được thay đổi hoặc bịa đặt điểm chuẩn của các năm.
        9. Với các năm mà có điểm chuẩn là 0.0 thì có nghĩa là trường đó chưa tuyển sinh ngành đó trong năm đó hoặc tuyển sinh với thang điểm khác thang điểm 30 (thi thpt quốc gia).

        Ngôn ngữ: Tiếng Việt.
        """
        
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Lỗi gọi Gemini (Form): {e}")
            return None

    async def format_chat_response(self, user_msg: str, rag_context: str, ml_verdict: str) -> Optional[str]:
        """
        Diễn đạt lại câu trả lời chatbot.
        """
        if not self.has_llm:
            return None

        prompt = f"""
        Bạn là chatbot tư vấn tuyển sinh thông minh. 
        Người dùng hỏi: "{user_msg}"
        
        Thông tin tìm kiếm được (RAG context):
        {rag_context}
        
        Nhận định từ hệ thống (ML verdict nếu có):
        {ml_verdict}
        
        --- YÊU CẦU ---
        1. Trả lời câu hỏi của người dùng một cách tự nhiên, súc tích (3-6 câu).
        2. Dựa hoàn toàn vào thông tin đã cung cấp ở trên. KHÔNG bịa thêm số liệu.
        3. Sử dụng Markdown để trình bày các điểm chính cho rõ ràng (dùng bullet points nếu cần).
        4. Thân thiện, chuyên nghiệp, dùng emoji phù hợp.
        5. Nếu thông tin không có trong ngữ cảnh, hãy lịch sự từ chối hoặc hướng dẫn họ dùng Form tư vấn để có kết quả chính xác hơn.

        Ngôn ngữ: Tiếng Việt.
        """
        
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Lỗi gọi Gemini (Chat): {e}")
            return None
