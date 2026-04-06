import os

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Đường dẫn dữ liệu
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAG_DOCUMENTS_DIR = os.path.join(DATA_DIR, "rag_documents")

# Đường dẫn mô hình
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Đường dẫn source code
SRC_DIR = os.path.join(BASE_DIR, "src")

# Cấu hình mô hình ML
ML_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "models": ["logistic_regression", "random_forest"],
    "feature_cols": [
        "diem_thi_sinh", "diem_cong_kv",
        "diem_chuan_2023", "diem_chuan_2024", "diem_chuan_2025",
        "trung_binh_3nam", "xu_huong_24_25", "chenh_lech"
    ],
    "label_col": "ket_qua"
}

# Cấu hình RAG
RAG_CONFIG = {
    "collection_name": "university_info",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200
}

# Cấu hình ưu tiên khu vực
def load_priority_scores():
    """Tải điểm ưu tiên từ file csv"""
    import pandas as pd
    csv_path = os.path.join(RAW_DATA_DIR, "uu_tien.csv")
    if not os.path.exists(csv_path):
        # Fallback values if file is missing
        return {
            "KV1": 0.75,
            "KV2": 0.5,
            "KV2NT": 0.25,
            "KV3": 0.0
        }
    
    try:
        df = pd.read_csv(csv_path)
        # Tạo map từ cột khu_vuc và diem_cong_kv
        priority_map = dict(zip(df['khu_vuc'], df['diem_cong_kv']))
        return priority_map
    except Exception as e:
        print(f"Error loading priority scores: {e}")
        return {}
