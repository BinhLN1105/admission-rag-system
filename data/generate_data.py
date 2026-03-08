import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ================================================================
# 1. ĐỊNH NGHĨA DỮ LIỆU GỐC
# ================================================================

TRUONG = [
    {"ma_truong": "BKA",  "ten_truong": "ĐH Bách Khoa Hà Nội",           "khu_vuc": "Hà Nội",      "loai_truong": "Công lập", "hoc_phi": 22000000},
    {"ma_truong": "UET",  "ten_truong": "ĐH Công nghệ - ĐHQGHN",          "khu_vuc": "Hà Nội",      "loai_truong": "Công lập", "hoc_phi": 20000000},
    {"ma_truong": "NEU",  "ten_truong": "ĐH Kinh tế Quốc dân",            "khu_vuc": "Hà Nội",      "loai_truong": "Công lập", "hoc_phi": 16000000},
    {"ma_truong": "PTIT", "ten_truong": "ĐH Bưu chính Viễn thông",        "khu_vuc": "Hà Nội",      "loai_truong": "Công lập", "hoc_phi": 14000000},
    {"ma_truong": "HUST", "ten_truong": "ĐH Khoa học Tự nhiên HN",        "khu_vuc": "Hà Nội",      "loai_truong": "Công lập", "hoc_phi": 15000000},
    {"ma_truong": "UIT",  "ten_truong": "ĐH Công nghệ Thông tin TP.HCM",  "khu_vuc": "TP.HCM",     "loai_truong": "Công lập", "hoc_phi": 22000000},
    {"ma_truong": "FTU",  "ten_truong": "ĐH Ngoại thương",                "khu_vuc": "Hà Nội",      "loai_truong": "Công lập", "hoc_phi": 18000000},
    {"ma_truong": "RMIT", "ten_truong": "ĐH RMIT Việt Nam",               "khu_vuc": "TP.HCM",     "loai_truong": "Quốc tế",  "hoc_phi": 280000000},
    {"ma_truong": "DUT",  "ten_truong": "ĐH Bách Khoa Đà Nẵng",           "khu_vuc": "Đà Nẵng",    "loai_truong": "Công lập", "hoc_phi": 13000000},
    {"ma_truong": "CTU",  "ten_truong": "ĐH Cần Thơ",                     "khu_vuc": "Cần Thơ",    "loai_truong": "Công lập", "hoc_phi": 12000000},
]

NGANH = [
    {"ma_nganh": "7480201", "ten_nganh": "Khoa học Máy tính",           "to_hop": ["A00","A01"],      "nhom": "CNTT"},
    {"ma_nganh": "7480202", "ten_nganh": "Mạng máy tính và TT",         "to_hop": ["A00","A01"],      "nhom": "CNTT"},
    {"ma_nganh": "7480101", "ten_nganh": "Công nghệ Thông tin",         "to_hop": ["A00","A01","D01"], "nhom": "CNTT"},
    {"ma_nganh": "7480107", "ten_nganh": "Trí tuệ Nhân tạo",            "to_hop": ["A00","A01"],      "nhom": "CNTT"},
    {"ma_nganh": "7520201", "ten_nganh": "Kỹ thuật Điện tử - Viễn thông","to_hop": ["A00","A01"],     "nhom": "KT"},
    {"ma_nganh": "7340101", "ten_nganh": "Quản trị Kinh doanh",         "to_hop": ["A00","D01","D07"],"nhom": "KT"},
    {"ma_nganh": "7340201", "ten_nganh": "Tài chính - Ngân hàng",       "to_hop": ["A00","D01","D07"],"nhom": "KT"},
    {"ma_nganh": "7220201", "ten_nganh": "Ngôn ngữ Anh",                "to_hop": ["D01"],            "nhom": "NN"},
    {"ma_nganh": "7440112", "ten_nganh": "Khoa học Dữ liệu",            "to_hop": ["A00","A01"],      "nhom": "CNTT"},
    {"ma_nganh": "7520110", "ten_nganh": "Kỹ thuật Máy tính",           "to_hop": ["A00","A01"],      "nhom": "KT"},
]

# Điểm chuẩn base theo trường + ngành (dùng để tạo dữ liệu thực tế hơn)
DIEM_BASE = {
    ("BKA",  "7480201"): [27.0, 27.5, 28.0],
    ("BKA",  "7480107"): [28.0, 28.5, 29.0],
    ("BKA",  "7520201"): [25.5, 26.0, 26.5],
    ("UET",  "7480201"): [27.5, 27.0, 27.5],
    ("UET",  "7480101"): [26.0, 26.5, 27.0],
    ("UET",  "7480107"): [28.0, 28.5, 28.0],
    ("UIT",  "7480101"): [25.0, 25.5, 26.0],
    ("UIT",  "7480201"): [26.5, 27.0, 27.5],
    ("UIT",  "7440112"): [26.0, 26.5, 27.0],
    ("NEU",  "7340101"): [26.5, 27.0, 27.25],
    ("NEU",  "7340201"): [25.5, 26.0, 26.5],
    ("FTU",  "7340101"): [27.0, 27.25, 27.5],
    ("FTU",  "7220201"): [28.0, 28.25, 28.5],
    ("PTIT", "7480101"): [22.0, 22.5, 23.0],
    ("PTIT", "7480202"): [21.5, 22.0, 22.5],
    ("HUST", "7480201"): [25.0, 25.5, 26.0],
    ("HUST", "7440112"): [24.5, 25.0, 25.5],
    ("DUT",  "7480101"): [21.0, 21.5, 22.0],
    ("DUT",  "7520201"): [20.5, 21.0, 21.5],
    ("CTU",  "7480101"): [18.0, 18.5, 19.0],
    ("CTU",  "7340101"): [17.5, 18.0, 18.5],
}

YEARS = [2023, 2024, 2025]

# ================================================================
# 2. TẠO FILE ĐIỂM CHUẨN TỪNG NĂM
# ================================================================

def make_diem_chuan(year_idx, year):
    rows = []
    for (ma_truong, ma_nganh), diem_list in DIEM_BASE.items():
        nganh = next(n for n in NGANH if n["ma_nganh"] == ma_nganh)
        truong = next(t for t in TRUONG if t["ma_truong"] == ma_truong)
        for to_hop in nganh["to_hop"]:
            diem = diem_list[year_idx] + np.random.uniform(-0.1, 0.1)
            diem = round(diem * 4) / 4  # làm tròn 0.25
            rows.append({
                "ma_truong":  ma_truong,
                "ten_truong": truong["ten_truong"],
                "ma_nganh":   ma_nganh,
                "ten_nganh":  nganh["ten_nganh"],
                "ma_to_hop":  to_hop,
                "nam":        year,
                "diem_chuan": diem,
                "chi_tieu":   np.random.choice([60, 80, 100, 120, 150]),
                "phuong_thuc":"THPT",
            })
    return pd.DataFrame(rows)

for i, yr in enumerate(YEARS):
    df = make_diem_chuan(i, yr)
    path = f"./data/raw/diem_chuan_{yr}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"✅ diem_chuan_{yr}.csv — {len(df)} dòng")

# ================================================================
# 3. MERGE → diem_chuan_full.csv
# ================================================================

dfs = [pd.read_csv(f"./data/raw/diem_chuan_{yr}.csv") for yr in YEARS]
full = pd.concat(dfs, ignore_index=True)
full.to_csv("./data/processed/diem_chuan_full.csv", index=False, encoding="utf-8-sig")
print(f"✅ diem_chuan_full.csv — {len(full)} dòng")

# ================================================================
# 4. FEATURE ENGINEERING → feature_engineered.csv
# ================================================================

pivot = full.pivot_table(
    index=["ma_truong","ma_nganh","ma_to_hop"],
    columns="nam",
    values="diem_chuan"
).reset_index()
pivot.columns = ["ma_truong","ma_nganh","ma_to_hop","dc_2023","dc_2024","dc_2025"]

pivot["xu_huong_24_25"] = pivot["dc_2025"] - pivot["dc_2024"]
pivot["xu_huong_23_24"] = pivot["dc_2024"] - pivot["dc_2023"]
pivot["trung_binh_3nam"] = pivot[["dc_2023","dc_2024","dc_2025"]].mean(axis=1).round(2)
pivot["do_bien_dong"] = pivot[["dc_2023","dc_2024","dc_2025"]].std(axis=1).round(3)

pivot.to_csv("./data/processed/feature_engineered.csv", index=False, encoding="utf-8-sig")
print(f"✅ feature_engineered.csv — {len(pivot)} dòng")

# ================================================================
# 5. SYNTHETIC TRAINING DATA → training_data.csv
# ================================================================

records = []
KHU_VUC = {"KV1":0.75, "KV2":0.5, "KV2NT":0.25, "KV3":0.0}

for _, row in pivot.iterrows():
    dc_latest = row["dc_2025"]
    dc_prev   = row["dc_2024"]
    xu_huong  = row["xu_huong_24_25"]
    tb        = row["trung_binh_3nam"]

    for _ in range(80):  # 80 thí sinh giả lập mỗi ngành/trường/tổ hợp
        kv   = np.random.choice(list(KHU_VUC.keys()), p=[0.15,0.25,0.35,0.25])
        cong = KHU_VUC[kv]

        diem_raw = round(np.random.normal(loc=dc_latest, scale=2.5), 2)
        diem_raw = max(0.0, min(30.0, diem_raw))
        diem_co_uu_tien = round(diem_raw + cong, 2)

        chenh_lech = round(diem_co_uu_tien - dc_latest, 2)

        # Xác suất đỗ dùng sigmoid (thực tế hơn hard threshold)
        prob = 1 / (1 + np.exp(-2.5 * chenh_lech))
        ket_qua = int(np.random.random() < prob)

        records.append({
            "ma_truong":          row["ma_truong"],
            "ma_nganh":           row["ma_nganh"],
            "ma_to_hop":          row["ma_to_hop"],
            "diem_thi_sinh":      diem_raw,
            "khu_vuc":            kv,
            "diem_cong_kv":       cong,
            "diem_co_uu_tien":    diem_co_uu_tien,
            "diem_chuan_2023":    row["dc_2023"],
            "diem_chuan_2024":    row["dc_2024"],
            "diem_chuan_2025":    row["dc_2025"],
            "trung_binh_3nam":    tb,
            "xu_huong_24_25":     xu_huong,
            "chenh_lech":         chenh_lech,
            "ket_qua":            ket_qua,  # 1=đỗ, 0=rớt
        })

train_df = pd.DataFrame(records)
train_df.to_csv("./data/processed/training_data.csv", index=False, encoding="utf-8-sig")

do_rate = train_df["ket_qua"].mean()
print(f"✅ training_data.csv — {len(train_df)} dòng | tỉ lệ đỗ: {do_rate:.1%}")

# ================================================================
# 6. RAG DOCUMENTS
# ================================================================

mo_ta = ""
for n in NGANH:
    mo_ta += f"""
Ngành: {n['ten_nganh']}
Mã ngành: {n['ma_nganh']}
Tổ hợp xét tuyển: {', '.join(n['to_hop'])}
Nhóm ngành: {n['nhom']}

"""

mo_ta_chi_tiet = {
    "7480201": "Khoa học Máy tính đào tạo kỹ sư có nền tảng lý thuyết vững chắc về thuật toán, cấu trúc dữ liệu, trí tuệ nhân tạo và phát triển phần mềm. Cơ hội việc làm: Lập trình viên, Kỹ sư AI, Nhà khoa học dữ liệu, Nghiên cứu viên.",
    "7480202": "Mạng máy tính và TT cung cấp kiến thức trọng tâm về nền tảng hạ tầng công nghệ, vận hành hệ thống mạng diện rộng, điện toán đám mây và an toàn thông tin cơ bản. Cơ hội việc làm: Kỹ sư thiết kế mạng, Chuyên viên quản trị hệ thống ISP.",
    "7480101": "Công nghệ Thông tin là ngành đa dạng nhất trong lĩnh vực CNTT, bao gồm lập trình, mạng, cơ sở dữ liệu, bảo mật. Cơ hội việc làm: Lập trình viên full-stack, Quản trị hệ thống, PM.",
    "7480107": "Trí tuệ Nhân tạo là ngành mới, đào tạo chuyên sâu về Machine Learning, Deep Learning, Computer Vision, NLP. Cơ hội việc làm: AI Engineer, ML Engineer, Research Scientist tại các công ty công nghệ lớn.",
    "7520201": "Kỹ thuật Điện tử - Viễn thông kết hợp giữa phần mềm và phần cứng, vi mạch, truyền tải tín hiệu quang, thiết kế mạch điện tử. Cơ hội việc làm: Kỹ sư vi mạch, Thiết kế phần cứng IoT, Chuyên gia viễn thông.",
    "7340101": "Quản trị Kinh doanh đào tạo kỹ năng quản lý, lãnh đạo, chiến lược kinh doanh. Cơ hội việc làm: Quản lý doanh nghiệp, Tư vấn kinh doanh, Entrepreneur, Chuyên viên phòng ban nhân sự kinh doanh.",
    "7340201": "Tài chính - Ngân hàng cung cấp nền tảng vững về kinh tế mô hình, đầu tư, tín dụng, chứng khoán và quản lý rủi ro tài chính. Cơ hội việc làm: Chuyên viên phân tích tài chính, nhân viên tín dụng, quản lý quỹ.",
    "7220201": "Ngôn ngữ Anh đào tạo ngôn ngữ học, dịch thuật, giao tiếp thương mại, đáp ứng xu thế hội nhập toàn cầu hóa. Cơ hội việc làm: Phiên dịch viên, Biên dịch viên, Giảng viên khối ngoại ngữ, truyền thông quốc tế.",
    "7440112": "Khoa học Dữ liệu kết hợp thống kê, lập trình và kinh doanh để xử lý lượng cực lớn Big Data. Cơ hội việc làm: Data Analyst, Data Scientist, Data Engineer, Chuyên viên phân tích doanh nghiệp.",
    "7520110": "Kỹ thuật Máy tính học sâu về kiến trúc máy tính, nhúng, thiết kế chip, hệ điều hành. Kỹ sư là những người giỏi cả lõi C/C++ và điện tử. Cơ hội việc làm: Kỹ sư nhúng Embedded System, Hệ điều hành, Robotic.",
}

full_mo_ta = "=== MÔ TẢ CÁC NGÀNH HỌC ===\n"
for ma, mo in mo_ta_chi_tiet.items():
    nganh = next((n for n in NGANH if n["ma_nganh"] == ma), None)
    if nganh:
        full_mo_ta += f"\n[{nganh['ten_nganh']} - Mã: {ma}]\n{mo}\nTổ hợp: {', '.join(nganh['to_hop'])}\n"

with open("./data/rag_documents/mo_ta_nganh.txt", "w", encoding="utf-8") as f:
    f.write(full_mo_ta)

truong_txt = "=== THÔNG TIN CÁC TRƯỜNG ĐẠI HỌC ===\n"
for t in TRUONG:
    truong_txt += f"""
[{t['ten_truong']} - Mã: {t['ma_truong']}]
Khu vực: {t['khu_vuc']}
Loại trường: {t['loai_truong']}
Học phí: {t['hoc_phi']:,} VNĐ/năm
"""

with open("./data/rag_documents/thong_tin_truong.txt", "w", encoding="utf-8") as f:
    f.write(truong_txt)

print("✅ mo_ta_nganh.txt")
print("✅ thong_tin_truong.txt")
print("\n🎉 Tất cả dữ liệu đã được tạo thành công!")