# Hệ Thống Tư Vấn Tuyển Sinh RAG + ML

Dự án xây dựng hệ thống tư vấn tuyển sinh đại học thông minh, kết hợp giữa mô hình học máy (Machine Learning) để dự đoán xác suất trúng tuyển và hệ thống RAG (Retrieval-Augmented Generation) để cung cấp thông tin chi tiết về trường/ngành học.

## Tổng quan Kiến Trúc

Dự án được triển khai theo 5 giai đoạn:
1. **Dữ liệu & Cấu trúc**: Khởi tạo thư mục và dữ liệu giả lập (Synthetic Data).
2. **Machine Learning Model**: Huấn luyện Random Forest và Logistic Regression để dự đoán xác suất đỗ đại học dựa trên lịch sử điểm chuẩn. Sử dụng Ensemble Learning tính trung bình xác suất.
3. **RAG System**: Sử dụng `sentence-transformers` và `ChromaDB` để nhúng (embed) và lưu trữ thông tin về các trường đại học, ngành học. Cải tiến Retrieval bằng thuật toán kết hợp Vector Similarity và Keyword Matching (Hybrid Search).
4. **Pipeline Kết nối**: Tổng hợp kết quả từ ML và RAG để tạo ra prompt chất lượng cho LLM (Large Language Model) sinh câu trả lời tự nhiên.
5. **Web Application**: Giao diện Web xây dựng bằng FastAPI, HTML/CSS/JS thuần tư vấn phản hồi trực tiếp cho người dùng.

## Cấu trúc thư mục

```text
Project_AI/
├── app/                  # Web app (FastAPI)
│   ├── static/           # HTML, CSS, JS
│   ├── main.py           # Khởi chạy server
│   ├── routes.py         # API Endpoints
│   └── schema.py         # Pydantic schemas
├── data/                 # Dữ liệu
│   ├── raw/              # Dữ liệu điểm chuẩn các năm (.csv)
│   ├── processed/        # Dữ liệu sau feature engineering
│   ├── rag_documents/    # Văn bản dành cho RAG (.txt)
│   └── generate_data.py  # Script tạo dữ liệu giả lập
├── models/               # Nơi lưu Models
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── vector_db/        # Database ChromaDB
├── notebooks/            # Jupyter notebooks thử nghiệm
├── src/                  # Mã nguồn chính
│   ├── data_processing/  # Tiền xử lý dữ liệu
│   ├── ml_model/         # Script Train, Predict, Evaluate ML
│   ├── rag/              # Script Embedder, Retriever, VectorStore
│   └── pipeline/         # Kết nối RAG và ML
├── .env                  # Biến môi trường
├── requirements.txt      # Thư viện Python
└── README.md             # Tệp hướng dẫn
```

## Cài đặt & Khởi chạy

**Bước 1: Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

**Bước 2: Khởi tạo Dữ liệu & Huấn luyện Model**
Chạy lệnh sau để tạo dummy data (hoặc load data thật), huấn luyện ML và Index văn bản cho RAG:
```bash
python data/generate_data.py
python src/ml_model/train.py
python src/rag/build_index.py
```

**Bước 3: Khởi chạy Web Server**
```bash
python app/main.py
```
*(Hoặc chạy lệnh `uvicorn app.main:app --reload`)*

Truy cập địa chỉ `http://localhost:8000` trên trình duyệt để sử dụng hệ thống!

## Nâng cấp tương lai
- Kết nối pipeline với API thật của OpenAI (`gpt-4`) hoặc Google Gemini để lấy đoạn sinh văn bản tự nhiên thay vì trả về Prompt mẫu.
- Thay thế Synthetic Data bằng dữ liệu thật được Crawl từ các trang điểm chuẩn.
- Tinh chỉnh RAG (BGE-M3 / PhoBERT) để embedding tiếng Việt chuẩn xác hơn.
