"""
evaluate.py
Đánh giá chi tiết model sau khi train.
Xuất biểu đồ vào reports/figures/
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "processed", "training_data.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
FIGURES_DIR  = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURE_COLS = [
    "diem_thi_sinh", "diem_cong_kv",
    "diem_chuan_2023", "diem_chuan_2024", "diem_chuan_2025",
    "trung_binh_3nam", "xu_huong_24_25", "chenh_lech",
]


def evaluate():
    # Load data & models
    df     = pd.read_csv(DATA_PATH)
    X      = df[FEATURE_COLS]
    y      = df["ket_qua"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    lr     = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    rf     = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    X_test_sc = scaler.transform(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Đánh giá ML Model — Dự đoán Xác suất Đỗ Đại học", fontsize=14, fontweight="bold")

    # ── 1. ROC Curve ──────────────────────────────────────────────
    ax = axes[0]
    for name, model, X_ev in [
        ("Logistic Regression", lr, X_test_sc),
        ("Random Forest",       rf, X_test),
    ]:
        prob = model.predict_proba(X_ev)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)

    ax.plot([0,1],[0,1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 2. Confusion Matrix (Random Forest) ───────────────────────
    ax = axes[1]
    y_pred_rf = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_rf)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Rớt","Đỗ"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix\n(Random Forest)")

    # ── 3. Feature Importance ─────────────────────────────────────
    ax = axes[2]
    fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    colors = ["#e74c3c" if f == "chenh_lech" else "#3498db" for f in fi.index]
    fi.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Feature Importance\n(Random Forest)")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "ml_evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✅ Biểu đồ đã lưu: {out_path}")

    # In classification report
    print("\n📋 Classification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, target_names=["Rớt","Đỗ"]))

    return out_path


if __name__ == "__main__":
    evaluate()