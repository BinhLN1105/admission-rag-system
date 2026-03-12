# Hệ Thống Tư Vấn Tuyển Sinh Thông Minh (RAG + ML)

Dự án phát triển hệ thống hỗ trợ tư vấn tuyển sinh đại học, sử dụng sức mạnh của **Machine Learning** để dự đoán khả năng trúng tuyển và **RAG (Retrieval-Augmented Generation)** để cung cấp thông tin chi tiết về các trường và ngành học. Dữ liệu được thu thập thực tế từ VnExpress.

---

## 🚀 Các Tính Năng Chính
- **Dự đoán trúng tuyển**: Phân tích lịch sử điểm chuẩn phối hợp với tổ hợp môn và khu vực ưu tiên để tính toán xác suất.
- **Tư vấn thông minh (RAG)**: Tìm kiếm và trả lời các câu hỏi về mô tả ngành nghề, học phí, và thông tin trường dựa trên cơ sở dữ liệu văn bản.
- **Dữ liệu thực tế**: Hệ thống sử dụng dữ liệu điểm chuẩn giai đoạn 2023-2025 được thu thập từ nguồn VnExpress.
- **Giao diện thân thiện**: Web app xây dựng trên FastAPI với giao diện hiện đại, phản hồi nhanh.

## 📂 Cấu Trúc Dự Án

```text
Project_AI/
├── app/                  # Ứng dụng Web (FastAPI)
│   ├── main.py           # Entry point của ứng dụng
│   ├── routes.py         # Xử lý các luồng API
│   └── static/           # Giao diện người dùng (HTML/CSS/JS)
├── data/                 # Quản lý dữ liệu
│   ├── raw/              # Dữ liệu điểm chuẩn gốc (Crawl từ VnExpress)
│   ├── processed/        # Dữ liệu đã qua làm sạch và xử lý đặc trưng
│   └── rag_documents/    # Văn bản tri thức cho hệ thống RAG
├── models/               # Lưu trữ các Model đã huấn luyện (RF, Logistic)
├── notebooks/            # Notebooks nghiên cứu, crawl dữ liệu và thử nghiệm
├── src/                  # Mã nguồn xử lý cốt lõi
│   ├── data_processing/  # Tiền xử lý và Feature Engineering
│   ├── ml_model/         # Huấn luyện và dự đoán Machine Learning
│   ├── rag/              # Xây dựng Index và truy vấn Vector DB
│   └── pipeline/         # Kết nối ML và RAG
└── requirements.txt      # Các thư viện cần thiết
```

## 🛠️ Hướng Dẫn Cài Đặt

**1. Cài đặt môi trường**
```bash
pip install -r requirements.txt
```

**2. Tiền xử lý dữ liệu & Huấn luyện**
Hệ thống cần xử lý dữ liệu thô và xây dựng chỉ mục vector trước khi hoạt động:
```bash
# Gộp dữ liệu và tạo đặc trưng
python src/data_processing/merge_data.py
# Huấn luyện mô hình ML
python src/ml_model/train.py
# Xây dựng cơ sở dữ liệu vector
python src/rag/build_index.py
```

**3. Khởi chạy ứng dụng**
```bash
python app/main.py
```
Truy cập: `http://localhost:8000`

## 📊 Quy Trình Xử Lý
1. **Dữ liệu**: Crawl dữ liệu từ VnExpress -> Lưu vào `raw/`.
2. **Xử lý**: Chuẩn hóa mã trường/ngành -> Feature Engineering (xu hướng điểm, biến động) -> Lưu vào `processed/`.
3. **ML Pipeline**: Huấn luyện Ensemble Model (Random Forest + Logistic Regression).
4. **RAG Pipeline**: Nhúng văn bản bằng `sentence-transformers` -> Lưu vào `ChromaDB`.
5. **Inference**: Nhận input -> ML dự đoán % -> RAG tìm thông tin bổ trợ -> LLM tổng hợp câu trả lời.

---
*Dự án đang trong quá trình hoàn thiện và nâng cấp các thuật toán nhúng tiếng Việt.*
