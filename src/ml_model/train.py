"""
train.py
Train Logistic Regression + Random Forest, lưu model ra file .pkl
Tự động xuất báo cáo và biểu đồ vào thư mục reports/
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# ---------- Đường dẫn ----------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "training_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURE_COLS = [
    "diem_thi_sinh", "diem_cong_kv",
    "diem_chuan_2023", "diem_chuan_2024", "diem_chuan_2025",
    "trung_binh_3nam", "xu_huong_24_25", "chenh_lech",
]
LABEL_COL = "ket_qua"


def train():
    # 1. Load data
    print("📂 Đang load dữ liệu...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Không tìm thấy dữ liệu tại {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURE_COLS]
    y  = df[LABEL_COL]
    print(f"   Tổng: {len(df)} mẫu | Đỗ: {int(y.sum())} | Rớt: {len(y[y==0])}")

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

    # 5. Xuất báo cáo & Vẽ biểu đồ (Tự động phục vụ Documentation)
    print(f"   Đang tạo báo cáo trong {REPORTS_DIR}...")
    
    # Thiết lập phong cách báo cáo (Gu thẩm mỹ chuyên nghiệp)
    sns.set_theme(style="white") # Bỏ gridlines rối rắm
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=14, labelsize=12)

    # ── 5a. Feature Importance Plot ──────────────────────────────
    plt.figure(figsize=(10, 6))
    
    # Custom Palette: Đỏ đậm cho top 1, Xám nhạt cho các ô còn lại
    custom_palette = ["#B22222"] + ["#A9A9A9"] * (len(fi) - 1)
    ax = sns.barplot(x=fi.values, y=fi.index, palette=custom_palette)
    
    plt.title("Mức Độ Quan Trọng Của Các Yếu Tố (Feature Importance)", pad=15, fontweight='bold')
    plt.xlabel("Trọng số ảnh hưởng (Importance Score)")
    plt.ylabel("")
    
    # Bỏ viền thừa (spines)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ── 5b. Confusion Matrix Plot (Best Model) ────────────────────
    best_model_name = "Random Forest" if auc_rf >= auc_lr else "Logistic Regression"
    y_test_best = y_test
    y_pred_best = y_pred_rf if best_model_name == "Random Forest" else y_pred_lr
    
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(y_test_best, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Rớt', 'Đỗ'], yticklabels=['Rớt', 'Đỗ'],
                annot_kws={"size": 16, "weight": "bold"})
    plt.title(f"Ma Trận Nhầm Lẫn (Confusion Matrix) - {best_model_name}", pad=15, fontweight='bold')
    plt.ylabel('Thực tế (Actual)')
    plt.xlabel('Dự đoán (Predicted)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ── 5c. ROC Curve Plot ───────────────────────────────────────
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(7, 6))
    
    # Tính toán ROC cho cả 2 model
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    
    plt.plot(fpr_rf, tpr_rf, color='#B22222', lw=2.5, label=f'Random Forest (AUC = {auc_rf:.3f})')
    plt.plot(fpr_lr, tpr_lr, color='#4682B4', lw=2, linestyle='--', label=f'Logistic Regression (AUC = {auc_lr:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle=':')
    
    plt.title("Đường Cong ROC (Receiver Operating Characteristic)", pad=15, fontweight='bold')
    plt.xlabel('Tỷ lệ Dương tính Giả (False Positive Rate)')
    plt.ylabel('Tỷ lệ Dương tính Thật (True Positive Rate)')
    plt.legend(loc="lower right", frameon=False)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    export_report = True
    if export_report:

        # ── 5c. Export Metrics Text ──────────────────────────────────
        with open(os.path.join(REPORTS_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write("=== BÁO CÁO ĐÁNH GIÁ MÔ HÌNH ===\n")
            f.write(f"Thời gian: {pd.Timestamp.now()}\n\n")
            
            f.write("1. LOGISTIC REGRESSION:\n")
            f.write(f"   Accuracy: {acc_lr:.4f}\n")
            f.write(f"   ROC-AUC : {auc_lr:.4f}\n")
            f.write(f"   CV AUC  : {cv_lr:.4f}\n\n")
            f.write(classification_report(y_test, y_pred_lr))
            f.write("\n" + "-"*30 + "\n\n")
            
            f.write("2. RANDOM FOREST:\n")
            f.write(f"   Accuracy: {acc_rf:.4f}\n")
            f.write(f"   ROC-AUC : {auc_rf:.4f}\n")
            f.write(f"   CV AUC  : {cv_rf:.4f}\n\n")
            f.write(classification_report(y_test, y_pred_rf))
            f.write("\n" + "-"*30 + "\n\n")
            f.write(f"🏆 Best Model: {best_model_name}\n")
        
        print(f"📈 Đã lưu biểu đồ vào {FIGURES_DIR}/")
    else:
        print("   Bỏ qua giai đoạn xuất báo cáo hình ảnh.")

    # 6. So sánh & chọn model tốt hơn
    print("\n🏆 So sánh:")
    print(f"   Logistic Regression — AUC: {auc_lr:.4f} | CV: {cv_lr:.4f}")
    print(f"   Random Forest       — AUC: {auc_rf:.4f} | CV: {cv_rf:.4f}")
    best_model_name = "Random Forest" if auc_rf >= auc_lr else "Logistic Regression"
    print(f"   ✅ Model tốt hơn: {best_model_name}")

    # 7. Lưu model
    joblib.dump(lr,     os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(rf,     os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"\n💾 Đã lưu model vào {MODELS_DIR}/")

    return lr, rf, scaler, results


if __name__ == "__main__":
    train()
