"""
predict.py
Load model đã train và dự đoán xác suất đỗ cho thí sinh.
"""

import os
import joblib
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

KV_MAP = {"KV1": 0.75, "KV2": 0.5, "KV2NT": 0.25, "KV3": 0.0}


def _load_models():
    rf     = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    lr     = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return rf, lr, scaler


def predict_probability(
    diem_thi_sinh: float,
    khu_vuc: str,
    diem_chuan_2023: float,
    diem_chuan_2024: float,
    diem_chuan_2025: float,
) -> dict:
    """
    Dự đoán xác suất đỗ của thí sinh (kết hợp Random Forest & Logistic Regression).

    Returns:
        {
            "xac_suat_do": 0.85,        # 0.0 - 1.0
            "phan_tram": "85%",
            "danh_gia": "Khá cao",
            "chenh_lech": -1.0,         # so với điểm chuẩn 2025
            "diem_co_uu_tien": 27.0
        }
    """
    rf, lr, scaler = _load_models()

    diem_cong        = KV_MAP.get(khu_vuc.upper(), 0.0)
    diem_co_uu_tien  = float(round(float(diem_thi_sinh + diem_cong), 2))
    trung_binh       = float(round(float((diem_chuan_2023 + diem_chuan_2024 + diem_chuan_2025) / 3), 2))
    xu_huong         = float(round(float(diem_chuan_2025 - diem_chuan_2024), 2))
    chenh_lech       = float(round(float(diem_co_uu_tien - diem_chuan_2025), 2))

    X = pd.DataFrame([{
        "diem_thi_sinh":   diem_thi_sinh,
        "diem_cong_kv":    diem_cong,
        "diem_chuan_2023": diem_chuan_2023,
        "diem_chuan_2024": diem_chuan_2024,
        "diem_chuan_2025": diem_chuan_2025,
        "trung_binh_3nam": trung_binh,
        "xu_huong_24_25":  xu_huong,
        "chenh_lech":      chenh_lech,
    }])

    # Dự đoán bằng Random Forest
    prob_rf = rf.predict_proba(X)[0][1]
    
    # Dự đoán bằng Logistic Regression (cần chuẩn hóa)
    X_sc = scaler.transform(X)
    prob_lr = lr.predict_proba(X_sc)[0][1]

    # Kết hợp (Ensemble) - Average
    prob = (prob_rf + prob_lr) / 2

    # Đánh giá định tính
    if prob >= 0.80:
        danh_gia = "Rất cao — nên đăng ký nguyện vọng 1"
    elif prob >= 0.60:
        danh_gia = "Khá cao — có thể đăng ký nguyện vọng 1"
    elif prob >= 0.40:
        danh_gia = "Trung bình — nên cân nhắc nguyện vọng 2"
    elif prob >= 0.20:
        danh_gia = "Thấp — nên đăng ký làm nguyện vọng dự phòng"
    else:
        danh_gia = "Rất thấp — nên xem xét trường/ngành khác"

    return {
        "xac_suat_do":      float(round(float(prob), 4)),
        "phan_tram":        f"{prob*100:.1f}%",
        "danh_gia":         danh_gia,
        "chenh_lech":       chenh_lech,
        "diem_co_uu_tien":  diem_co_uu_tien,
        "diem_cong_kv":     diem_cong,
    }


if __name__ == "__main__":
    # Test thử
    result = predict_probability(
        diem_thi_sinh=26.0,
        khu_vuc="KV3",
        diem_chuan_2023=27.0,
        diem_chuan_2024=27.5,
        diem_chuan_2025=28.0,
    )
    print("Kết quả dự đoán:")
    for k, v in result.items():
        print(f"  {k}: {v}")