"""
feature_engineering.py
Chuẩn bị features đầu vào cho ML model.
"""

import pandas as pd
import numpy as np


# Các features dùng để train
FEATURE_COLS = [
    "diem_thi_sinh",
    "diem_cong_kv",
    "diem_chuan_2023",
    "diem_chuan_2024",
    "diem_chuan_2025",
    "trung_binh_3nam",
    "xu_huong_24_25",
    "chenh_lech",
]

LABEL_COL = "ket_qua"


def load_features(csv_path: str):
    """Đọc training_data.csv và trả về X, y."""
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]
    return X, y


from src.utils.config import load_priority_scores

# Tải điểm ưu tiên
PRIORITY_MAP = load_priority_scores()

def build_input_vector(
    diem_thi_sinh: float,
    khu_vuc: str,
    diem_chuan_2023: float,
    diem_chuan_2024: float,
    diem_chuan_2025: float,
) -> pd.DataFrame:
    """
    Tạo vector features từ input của người dùng.
    Dùng khi gọi predict lúc runtime.
    """
    diem_cong = float(PRIORITY_MAP.get(str(khu_vuc).upper().replace("-", ""), 0.0))
    diem_co_uu_tien = diem_thi_sinh + diem_cong

    trung_binh = float(round(float((diem_chuan_2023 + diem_chuan_2024 + diem_chuan_2025) / 3), 2))
    xu_huong   = float(round(float(diem_chuan_2025 - diem_chuan_2024), 2))
    chenh_lech = float(round(float(diem_co_uu_tien - diem_chuan_2025), 2))

    return pd.DataFrame([{
        "diem_thi_sinh":   diem_thi_sinh,
        "diem_cong_kv":    diem_cong,
        "diem_chuan_2023": diem_chuan_2023,
        "diem_chuan_2024": diem_chuan_2024,
        "diem_chuan_2025": diem_chuan_2025,
        "trung_binh_3nam": trung_binh,
        "xu_huong_24_25":  xu_huong,
        "chenh_lech":      chenh_lech,
    }])