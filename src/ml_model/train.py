"""
train.py
Train Logistic Regression + Random Forest, lưu model ra file .pkl
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)

# ---------- Đường dẫn ----------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "training_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "diem_thi_sinh", "diem_cong_kv",
    "diem_chuan_2023", "diem_chuan_2024", "diem_chuan_2025",
    "trung_binh_3nam", "xu_huong_24_25", "chenh_lech",
]
LABEL_COL = "ket_qua"


def train():
    # 1. Load data
    print("📂 Đang load dữ liệu...")
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURE_COLS]
    y  = df[LABEL_COL]
    print(f"   Tổng: {len(df)} mẫu | Đỗ: {y.sum()} | Rớt: {(y==0).sum()}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── 4a. Logistic Regression ──────────────────────────────────
    print("\n🔵 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)

    y_pred_lr  = lr.predict(X_test_sc)
    y_prob_lr  = lr.predict_proba(X_test_sc)[:, 1]
    acc_lr     = accuracy_score(y_test, y_pred_lr)
    auc_lr     = roc_auc_score(y_test, y_prob_lr)
    cv_lr      = cross_val_score(lr, scaler.transform(X), y, cv=5, scoring="roc_auc").mean()

    print(f"   Accuracy : {acc_lr:.4f}")
    print(f"   ROC-AUC  : {auc_lr:.4f}")
    print(f"   CV AUC   : {cv_lr:.4f}")
    results["logistic"] = {"acc": acc_lr, "auc": auc_lr, "cv_auc": cv_lr}

    # ── 4b. Random Forest ────────────────────────────────────────
    print("\n🟢 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        min_samples_leaf=10, random_state=42
    )
    rf.fit(X_train, y_train)   # RF không cần scale

    y_pred_rf  = rf.predict(X_test)
    y_prob_rf  = rf.predict_proba(X_test)[:, 1]
    acc_rf     = accuracy_score(y_test, y_pred_rf)
    auc_rf     = roc_auc_score(y_test, y_prob_rf)
    cv_rf      = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()

    print(f"   Accuracy : {acc_rf:.4f}")
    print(f"   ROC-AUC  : {auc_rf:.4f}")
    print(f"   CV AUC   : {cv_rf:.4f}")
    results["random_forest"] = {"acc": acc_rf, "auc": auc_rf, "cv_auc": cv_rf}

    # Feature importance
    fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n📊 Feature Importance (Random Forest):")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 40)
        print(f"   {feat:<22} {bar} {imp:.4f}")

    # 5. So sánh & chọn model tốt hơn
    print("\n🏆 So sánh:")
    print(f"   Logistic Regression — AUC: {auc_lr:.4f} | CV: {cv_lr:.4f}")
    print(f"   Random Forest       — AUC: {auc_rf:.4f} | CV: {cv_rf:.4f}")
    best = "Random Forest" if auc_rf >= auc_lr else "Logistic Regression"
    print(f"   ✅ Model tốt hơn: {best}")

    # 6. Lưu model
    joblib.dump(lr,     os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(rf,     os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"\n💾 Đã lưu model vào {MODELS_DIR}/")

    return lr, rf, scaler, results


if __name__ == "__main__":
    train()