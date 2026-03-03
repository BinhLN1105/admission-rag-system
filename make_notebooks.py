import json
import os

NOTEBOOK_TEMPLATE = {
 "cells": [],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def create_notebook(filename, cells_sources):
    nb = dict(NOTEBOOK_TEMPLATE)
    nb["cells"] = []
    for source in cells_sources:
        nb["cells"].append({
            "cell_type": "code" if not source.startswith("#") else "markdown",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\\n" for line in source.split("\\n")] if not source.startswith("#") else [source]
        })
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

os.makedirs("d:/Project_AI/notebooks", exist_ok=True)

# 02_feature_engineering.ipynb
create_notebook("d:/Project_AI/notebooks/02_feature_engineering.ipynb", [
    "# Khám phá dữ liệu feature, xem trước data chuẩn bị cho ML",
    "import pandas as pd\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns",
    "df = pd.read_csv('../data/processed/feature_engineered.csv')\\ndf.head()",
    "df_train = pd.read_csv('../data/processed/training_data.csv')\\ndf_train.head()"
])

# 03_train_ml_model.ipynb
create_notebook("d:/Project_AI/notebooks/03_train_ml_model.ipynb", [
    "# Thử nghiệm train mô hình ML",
    "import pandas as pd\\nfrom sklearn.model_selection import train_test_split\\nfrom sklearn.ensemble import RandomForestClassifier",
    "df = pd.read_csv('../data/processed/training_data.csv')",
    "X = df[['diem_thi_sinh', 'diem_cong_kv', 'diem_chuan_2023', 'diem_chuan_2024', 'diem_chuan_2025', 'trung_binh_3nam', 'xu_huong_24_25', 'chenh_lech']]\\ny = df['ket_qua']",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
    "rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)\\nrf.fit(X_train, y_train)",
    "from sklearn.metrics import accuracy_score, roc_auc_score\\npreds = rf.predict(X_test)\\nprobs = rf.predict_proba(X_test)[:, 1]\\nprint('Accuracy:', accuracy_score(y_test, preds))\\nprint('AUC:', roc_auc_score(y_test, probs))"
])

# 04_evaluation.ipynb
create_notebook("d:/Project_AI/notebooks/04_evaluation.ipynb", [
    "# Đánh giá mô hình chi tiết",
    "import joblib\\nimport pandas as pd\\nimport matplotlib.pyplot as plt\\nfrom sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report",
    "rf = joblib.load('../models/random_forest.pkl')\\ndf = pd.read_csv('../data/processed/training_data.csv')\\nX = df[['diem_thi_sinh', 'diem_cong_kv', 'diem_chuan_2023', 'diem_chuan_2024', 'diem_chuan_2025', 'trung_binh_3nam', 'xu_huong_24_25', 'chenh_lech']]\\ny = df['ket_qua']",
    "cm = confusion_matrix(y, rf.predict(X))\\ndisp = ConfusionMatrixDisplay(cm)\\ndisp.plot()\\nplt.show()",
    "print(classification_report(y, rf.predict(X)))"
])

print("Tạo Notebook thành công!")
