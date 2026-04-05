"""
process.py
----------
Đọc input_raw.xlsx → tra mã ngành (fuzzy) + thông tin trường
→ Upsert vào diem_chuan_master.xlsx (2 sheet: diem_chuan, truong)
"""

import os
import sys
import unicodedata
import re
import pandas as pd
from thefuzz import process as fuzz_process
from openpyxl import load_workbook, Workbook

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THUTHAPDATADIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MA_NGANH_FILE = os.path.join(BASE_DIR, "bang-ma-nganh-dai-hoc-2022.xlsx")
INPUT_FILE    = os.path.join(THUTHAPDATADIR, "input", "input_raw.xlsx")
MASTER_FILE   = os.path.join(THUTHAPDATADIR, "output", "diem_chuan_master.xlsx")

# ── Danh sách trường nội bộ (mở rộng dần) ────────────────────────────────────
TRUONG_DB = {
    "BKA":  {"ten_truong": "ĐH Bách Khoa Hà Nội",          "khu_vuc": "Hà Nội",   "loai_truong": "Công lập",  "hoc_phi": 22000000},
    "UET":  {"ten_truong": "ĐH Công nghệ - ĐHQGHN",         "khu_vuc": "Hà Nội",   "loai_truong": "Công lập",  "hoc_phi": 20000000},
    "NEU":  {"ten_truong": "ĐH Kinh tế Quốc dân",           "khu_vuc": "Hà Nội",   "loai_truong": "Công lập",  "hoc_phi": 16000000},
    "PTIT": {"ten_truong": "ĐH Bưu chính Viễn thông",       "khu_vuc": "Hà Nội",   "loai_truong": "Công lập",  "hoc_phi": 14000000},
    "HUST": {"ten_truong": "ĐH Khoa học Tự nhiên HN",       "khu_vuc": "Hà Nội",   "loai_truong": "Công lập",  "hoc_phi": 15000000},
    "UIT":  {"ten_truong": "ĐH Công nghệ Thông tin TP.HCM", "khu_vuc": "TP.HCM",   "loai_truong": "Công lập",  "hoc_phi": 22000000},
    "FTU":  {"ten_truong": "ĐH Ngoại thương",               "khu_vuc": "Hà Nội",   "loai_truong": "Công lập",  "hoc_phi": 18000000},
    "RMIT": {"ten_truong": "ĐH RMIT Việt Nam",              "khu_vuc": "TP.HCM",   "loai_truong": "Quốc tế",   "hoc_phi": 280000000},
    "DUT":  {"ten_truong": "ĐH Bách Khoa Đà Nẵng",          "khu_vuc": "Đà Nẵng",  "loai_truong": "Công lập",  "hoc_phi": 13000000},
    "CTU":  {"ten_truong": "ĐH Cần Thơ",                    "khu_vuc": "Cần Thơ",  "loai_truong": "Công lập",  "hoc_phi": 12000000},
    "VNU":  {"ten_truong": "ĐH Quốc gia Hà Nội",            "khu_vuc": "Hà Nội",   "loai_truong": "Công lập",  "hoc_phi": 18000000},
    "HCMUS":{"ten_truong": "ĐH Khoa học Tự nhiên TP.HCM",   "khu_vuc": "TP.HCM",   "loai_truong": "Công lập",  "hoc_phi": 17000000},
    "HCMUT":{"ten_truong": "ĐH Bách Khoa TP.HCM",           "khu_vuc": "TP.HCM",   "loai_truong": "Công lập",  "hoc_phi": 23000000},
    "DAI":  {"ten_truong": "ĐH Đại Nam",                    "khu_vuc": "Hà Nội",   "loai_truong": "Tư thục",   "hoc_phi": 35000000},
    "FPT":  {"ten_truong": "ĐH FPT",                        "khu_vuc": "Hà Nội",   "loai_truong": "Tư thục",   "hoc_phi": 60000000},
    "UTH":  {"ten_truong": "ĐH Giao thông Vận tải TP.HCM",  "khu_vuc": "TP.HCM",   "loai_truong": "Công lập",  "hoc_phi": 13000000},
    "HUI":  {"ten_truong": "ĐH Công nghiệp TP.HCM",         "khu_vuc": "TP.HCM",   "loai_truong": "Công lập",  "hoc_phi": 18000000},
    "TDMU": {"ten_truong": "ĐH Thủ Dầu Một",                "khu_vuc": "Bình Dương","loai_truong": "Công lập",  "hoc_phi": 14000000},
    "IUH":  {"ten_truong": "ĐH Công nghiệp Thực phẩm HCM",  "khu_vuc": "TP.HCM",   "loai_truong": "Công lập",  "hoc_phi": 15000000},
    "NLS":  {"ten_truong": "ĐH Nông Lâm TP.HCM",            "khu_vuc": "TP.HCM",   "loai_truong": "Công lập",  "hoc_phi": 14000000},
    "TSN":  {"ten_truong": "ĐH Nha Trang",    "khu_vuc": "Khánh Hoà",   "loai_truong": "Công lập",  "hoc_phi": 16000000},
    "DDT":  {"ten_truong": "ĐH Duy Tân",    "khu_vuc": "Đà Nẵng",   "loai_truong": "Tư thục",  "hoc_phi": 16000000},
}


def log(msg: str):
    print(msg, flush=True)


def load_ma_nganh() -> pd.DataFrame:
    """Đọc bảng mã ngành, chỉ lấy dòng có mã ngành 7 chữ số (đại học)."""
    df = pd.read_excel(MA_NGANH_FILE, sheet_name="Sheet5", header=1)
    df.columns = ["stt", "ma_nganh", "ten_nganh", "hieu_luc", "ghi_chu"]
    df = df.dropna(subset=["ma_nganh"])
    # Chỉ lấy mã 7 chữ số (bậc đại học)
    df["ma_nganh"] = df["ma_nganh"].astype(str).str.strip()
    df = df[df["ma_nganh"].str.match(r"^\d{7}$")]
    df["ten_nganh"] = df["ten_nganh"].astype(str).str.strip()
    return df[["ma_nganh", "ten_nganh"]].reset_index(drop=True)


def _normalize(text: str) -> str:
    """Bỏ dấu tiếng Việt, lowercase để so sánh fuzzy."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_str.lower().strip()

def clean_ten_nganh(text: str) -> str:
    """Loại bỏ phần nội dung trong ngoặc đơn như (Chương trình tiên tiến), (Phân hiệu...) để khớp tên ngành gốc."""
    text = str(text).strip()
    text = re.sub(r'\(.*?\)', '', text)
    return " ".join(text.split())

def fuzzy_match_nganh(ten_nganh_input: str, ma_nganh_df: pd.DataFrame, threshold: int = 60):
    """Trả về (ma_nganh, ten_nganh_chuan, score).
    So sánh dạng không dấu để khớp dù người dùng nhập không dấu.
    """
    input_norm = _normalize(ten_nganh_input)
    # Tạo dict: tên không dấu → index gốc
    norm_choices = {_normalize(t): t for t in ma_nganh_df["ten_nganh"].tolist()}
    match_norm, score = fuzz_process.extractOne(input_norm, list(norm_choices.keys()))
    if score >= threshold:
        ten_goc = norm_choices[match_norm]
        row = ma_nganh_df[ma_nganh_df["ten_nganh"] == ten_goc].iloc[0]
        return row["ma_nganh"], row["ten_nganh"], score
    return None, None, score



def load_or_create_master():
    """Đọc master hoặc tạo mới nếu chưa có."""
    if os.path.exists(MASTER_FILE):
        try:
            df_dc = pd.read_excel(MASTER_FILE, sheet_name="diem_chuan")
            df_tr = pd.read_excel(MASTER_FILE, sheet_name="truong")
            return df_dc, df_tr
        except Exception:
            pass

    df_dc = pd.DataFrame(columns=[
        "ma_truong", "ten_truong", "ma_nganh", "ten_nganh",
        "ma_to_hop", "nam", "diem_chuan", "chi_tieu"
    ])
    df_tr = pd.DataFrame(columns=[
        "ma_truong", "ten_truong", "khu_vuc", "loai_truong", "hoc_phi"
    ])
    return df_dc, df_tr


def save_master(df_dc: pd.DataFrame, df_tr: pd.DataFrame):
    """Lưu master Excel 2 sheet."""
    os.makedirs(os.path.dirname(MASTER_FILE), exist_ok=True)
    with pd.ExcelWriter(MASTER_FILE, engine="openpyxl") as writer:
        df_dc.to_excel(writer, sheet_name="diem_chuan", index=False)
        df_tr.to_excel(writer, sheet_name="truong", index=False)


def process(input_path: str = INPUT_FILE):
    results = {
        "success": [],
        "warnings": [],
        "errors": [],
        "total": 0,
        "added": 0,
        "skipped": 0,
    }

    # 1. Kiểm tra file input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    log("📂 Đang đọc file input...")
    df_input = pd.read_excel(input_path)
    df_input.columns = [c.strip().lower().replace(" ", "_") for c in df_input.columns]

    required_cols = {"ma_tuyen_sinh", "ten_nganh", "to_hop", "nam", "diem_chuan"}
    missing = required_cols - set(df_input.columns)
    if missing:
        raise ValueError(f"File thiếu cột: {missing}. Cần có: {required_cols}")

    df_input = df_input.dropna(subset=["ma_tuyen_sinh", "ten_nganh", "diem_chuan"])
    results["total"] = len(df_input)
    log(f"   → {len(df_input)} dòng dữ liệu")

    # 2. Load bảng mã ngành
    log("📖 Đang load bảng mã ngành...")
    ma_nganh_df = load_ma_nganh()
    log(f"   → {len(ma_nganh_df)} ngành đại học")

    # 3. Load master
    log("📋 Đang load file master...")
    df_dc, df_tr = load_or_create_master()

    # 4. Xử lý từng dòng
    log("⚙️  Đang xử lý...")
    new_dc_rows = []
    new_tr_keys = set(df_tr["ma_truong"].tolist()) if len(df_tr) > 0 else set()

    for idx, row in df_input.iterrows():
        ma_ts   = str(row["ma_tuyen_sinh"]).strip().upper()
        ten_raw = str(row["ten_nganh"])
        ten_ng  = clean_ten_nganh(ten_raw)
        to_hop  = str(row["to_hop"]).strip().upper()
        nam     = int(row["nam"])
        diem    = float(row["diem_chuan"])
        chi_tieu = int(row["chi_tieu"]) if "chi_tieu" in df_input.columns and pd.notna(row.get("chi_tieu")) else 0

        # Tra thông tin trường
        if ma_ts not in TRUONG_DB:
            results["warnings"].append(f"Dòng {idx+2}: Mã '{ma_ts}' chưa có trong danh sách trường. Bỏ qua.")
            results["skipped"] += 1
            continue

        truong_info = TRUONG_DB[ma_ts]

        # Fuzzy match tên ngành → mã ngành
        ma_nganh, ten_nganh_chuan, score = fuzzy_match_nganh(ten_ng, ma_nganh_df)
        if ma_nganh is None:
            results["warnings"].append(f"Dòng {idx+2}: Không khớp ngành '{ten_ng}' (score={score}). Bỏ qua.")
            results["skipped"] += 1
            continue

        if score < 80:
            results["warnings"].append(f"Dòng {idx+2}: '{ten_ng}' → '{ten_nganh_chuan}' (score={score}, thấp)")

        # Kiểm tra trùng trong master
        key_mask = (
            (df_dc["ma_truong"] == ma_ts) &
            (df_dc["ma_nganh"]  == ma_nganh) &
            (df_dc["ma_to_hop"] == to_hop) &
            (df_dc["nam"]       == nam)
        ) if len(df_dc) > 0 else pd.Series([], dtype=bool)

        if len(df_dc) > 0 and key_mask.any():
            # Cập nhật nếu điểm thay đổi
            df_dc.loc[key_mask, "diem_chuan"] = diem
            df_dc.loc[key_mask, "chi_tieu"]   = chi_tieu
            results["success"].append(f"✏️  Cập nhật: {ma_ts} | {ten_nganh_chuan} | {to_hop} | {nam}")
        else:
            new_dc_rows.append({
                "ma_truong":  ma_ts,
                "ten_truong": truong_info["ten_truong"],
                "ma_nganh":   ma_nganh,
                "ten_nganh":  ten_nganh_chuan,
                "ma_to_hop":  to_hop,
                "nam":        nam,
                "diem_chuan": diem,
                "chi_tieu":   chi_tieu,
            })
            results["success"].append(f"✅ Thêm mới: {ma_ts} | {ten_nganh_chuan} | {to_hop} | {nam} | {diem}")
            results["added"] += 1

        # Cập nhật sheet truong nếu chưa có
        if ma_ts not in new_tr_keys:
            new_tr_row = {
                "ma_truong":   ma_ts,
                "ten_truong":  truong_info["ten_truong"],
                "khu_vuc":     truong_info["khu_vuc"],
                "loai_truong": truong_info["loai_truong"],
                "hoc_phi":     truong_info["hoc_phi"],
            }
            if len(df_tr) == 0 or not (df_tr["ma_truong"] == ma_ts).any():
                df_tr = pd.concat([df_tr, pd.DataFrame([new_tr_row])], ignore_index=True)
            new_tr_keys.add(ma_ts)

    # Ghép dòng mới vào master
    if new_dc_rows:
        df_dc = pd.concat([df_dc, pd.DataFrame(new_dc_rows)], ignore_index=True)

    # Sắp xếp
    df_dc = df_dc.sort_values(["ma_truong", "ma_nganh", "ma_to_hop", "nam"]).reset_index(drop=True)
    df_tr = df_tr.sort_values("ma_truong").reset_index(drop=True)

    # Lưu
    save_master(df_dc, df_tr)
    log(f"💾 Đã lưu vào: {MASTER_FILE}")
    log(f"   Sheet diem_chuan: {len(df_dc)} dòng | Sheet truong: {len(df_tr)} trường")

    results["skipped"] = results["total"] - results["added"] - (len(results["warnings"]) - sum(1 for w in results["warnings"] if "thấp" in w))
    return results, df_dc.to_dict(orient="records")


if __name__ == "__main__":
    results, preview = process()
    print("\n📊 KẾT QUẢ:")
    for msg in results["success"]:
        print(" ", msg)
    for msg in results["warnings"]:
        print("  ⚠️ ", msg)
    print(f"\nTổng: {results['total']} | Thêm: {results['added']} | Bỏ qua: {results['skipped']}")
