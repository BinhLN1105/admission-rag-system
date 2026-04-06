import os
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "ml_processed_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "rag_processed_data.json")


def format_page_content(row):
    nam = int(row["nam"])
    ten_truong = str(row.get("ten_truong_chuan") or row.get("ten_truong") or "").strip()
    ma_truong = str(row.get("ma_truong") or "").strip()
    ten_nganh = str(row.get("ten_nganh_chuan") or row.get("ten_nganh") or "").strip()
    ma_nganh = str(row.get("ma_nganh_chuan") or row.get("ma_nganh") or "").strip()
    ma_to_hop = str(row.get("ma_to_hop") or "").strip()
    diem = row.get("diem_chuan")
    phuong_thuc = str(row.get("phuong_thuc") or "").strip()
    chi_tieu = row.get("chi_tieu")

    content = f"Vào năm {nam}, trường {ten_truong} (mã trường: {ma_truong}) lấy mức điểm chuẩn là {diem} điểm cho ngành {ten_nganh} (mã ngành: {ma_nganh})"
    if ma_to_hop and ma_to_hop.lower() != "nan":
        content += f", xét tuyển theo tổ hợp môn {ma_to_hop}"
    if chi_tieu not in [None, "", "nan"]:
        content += f" với chỉ tiêu {chi_tieu}"
    if phuong_thuc and phuong_thuc.lower() != "nan":
        content += f"; phương thức xét tuyển: {phuong_thuc}"
    content += "."
    return content


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Không tìm thấy file data nguồn: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH, dtype=str)
    if df.empty:
        print("File dữ liệu RAG nguồn rỗng.")
        return

    # Sử dụng phiên bản chuẩn nếu có
    if "ma_nganh_chuan" in df.columns:
        df["ma_nganh"] = df["ma_nganh_chuan"].fillna(df["ma_nganh"])
    if "ten_nganh_chuan" in df.columns:
        df["ten_nganh"] = df["ten_nganh_chuan"].fillna(df["ten_nganh"])
    if "ten_truong_chuan" in df.columns:
        df["ten_truong"] = df["ten_truong_chuan"].fillna(df["ten_truong"])

    docs = []
    for _, row in df.iterrows():
        try:
            nam = int(float(str(row["nam"]).strip()))
        except Exception:
            continue

        page_content = format_page_content(row)
        metadata = {
            "nam": nam,
            "ma_truong": str(row.get("ma_truong", "")).strip(),
            "ten_truong": str(row.get("ten_truong", "")).strip(),
            "ma_nganh": str(row.get("ma_nganh", "")).strip(),
            "ten_nganh": str(row.get("ten_nganh", "")).strip(),
            "ma_to_hop": str(row.get("ma_to_hop", "")).strip(),
        }
        docs.append({"page_content": page_content, "metadata": metadata})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Đã tạo {len(docs)} document RAG tại: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
