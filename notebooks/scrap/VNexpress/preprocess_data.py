import pandas as pd
import json
import os
import glob
import numpy as np
import re

# Tu dien anh xa cac nhom nganh pho bien ve ma chuan cua Bo GD (MOET)
# Bo sung them nhieu nganh theo yeu cau "dong bo them nhieu truong, nganh"
COMMON_MAJOR_MAP = {
    # === CNTT & Khoa hoc may tinh ===
    "công nghệ thông tin": "7480201",
    "khoa học máy tính": "7480101",
    "kỹ thuật phần mềm": "7480103",
    "an toàn thông tin": "7480202",
    "an ninh mạng": "7480202",
    "an toàn dữ liệu": "7480202",
    "trí tuệ nhân tạo": "7480107",
    "hệ thống thông tin": "7480104",
    "mạng máy tính": "7480102",
    "công nghệ đa phương tiện": "7480110",
    "truyền thông đa phương tiện": "7480110",
    "kỹ thuật máy tính": "7480106",
    "khoa học dữ liệu": "7460108",
    "thiết kế vi mạch": "7520207",
    
    # === Kinh te & Quan tri ===
    "kinh tế": "7310101",
    "quản trị kinh doanh": "7340101",
    "tài chính - ngân hàng": "7340201",
    "tài chính": "7340201",
    "ngân hàng": "7340201",
    "kế toán": "7340301",
    "kiểm toán": "7340302",
    "marketing": "7340115",
    "kinh doanh quốc tế": "7340120",
    "kinh tế quốc tế": "7310106",
    "logistics": "7510605",
    "thương mại điện tử": "7340122",
    "quản trị du lịch": "7810103",
    "quản trị khách sạn": "7810201",
    "quản lý công nghiệp": "7510601",
    
    # === Luat ===
    "luật": "7380101",
    "luật kinh tế": "7380107",
    
    # === Y - Duoc ===
    "y khoa": "7720101",
    "dược học": "7720201",
    "dược": "7720201",
    "điều dưỡng": "7720301",
    "răng hàm mặt": "7720501",
    "kỹ thuật xét nghiệm y học": "7720601",
    
    # === Ngon ngu ===
    "ngôn ngữ anh": "7220201",
    "tiếng anh": "7220201",
    "ngôn ngữ trung": "7220204",
    "ngôn ngữ nhật": "7220209",
    "ngôn ngữ hàn": "7220210",
    "ngôn ngữ pháp": "7220203",
    
    # === Su pham ===
    "sư phạm toán": "7140209",
    "sư phạm ngữ văn": "7140217",
    "sư phạm tiếng anh": "7140231",
    "giáo dục mầm non": "7140201",
    
    # === Ky thuat ===
    "kỹ thuật điện": "7520201",
    "kỹ thuật điện tử": "7520203",
    "kỹ thuật điện tử - viễn thông": "7520207",
    "kỹ thuật điều khiển": "7520216",
    "kỹ thuật điều khiển và tự động hóa": "7520216",
    "kỹ thuật cơ khí": "7520103",
    "kỹ thuật xây dựng": "7580201",
    "kiến trúc": "7580101",
    "kỹ thuật ô tô": "7520130",
    "kỹ thuật cơ điện tử": "7520114",
    "cơ điện tử": "7520114",
    "kỹ thuật hóa học": "7520301",
    "kỹ thuật môi trường": "7520320",
    "kỹ thuật sinh học": "7420202",
    "công nghệ sinh học": "7420201",
    "công nghệ thực phẩm": "7540101",
    "kỹ thuật y sinh": "7520212",
    "kỹ thuật hàng không": "7520120",
    "cơ khí hàng không": "7520120",
    "kỹ thuật tàu thủy": "7520122",
    "kỹ thuật vật liệu": "7520309",
    "công nghệ vật liệu": "7510402",
    
    # === Bao chi & Truyen thong ===
    "báo chí": "7320101",
    "quan hệ công chúng": "7320108",
    "quan hệ quốc tế": "7310206",
    "chính trị học": "7310201",
    
    # === Khoa hoc co ban ===
    "toán ứng dụng": "7460112",
    "vật lý": "7440102",
    "hóa học": "7440112",
}

def normalize_ten_nganh(ten):
    ten = str(ten).strip()
    # Loai bo cac doan trong ngoac don, sau dau gach ngang, hoac cac tu du thua
    ten = re.sub(r'(?i)(\s*\(.*\)\s*|-.*|chất lượng cao|chuyên ngành.*|cttt.*|chương trình.*|tiên tiến.*|liên kết.*|định hướng.*|tăng cường.*|học bằng.*|song bằng.*|lớp chọn.*)', '', ten).strip()
    # Clean up trailing punctuation
    ten = re.sub(r'[\*\s:;,]+$', '', ten).strip()
    # Viet hoa chu cai dau
    return ten.capitalize()

def get_dynamic_major_map(df):
    """
    Hoc hoi ma chuan tu cac truong co gan ma 7 so chuan trong du lieu.
    Xay dung 2 chieu: ten_nganh_chuan -> ma_7digit VÀ ten_nganh_goc -> ma_7digit
    """
    d_map = {}
    for _, row in df.iterrows():
        ma = str(row['ma_nganh']).strip()
        match = re.search(r'(\d{7})', ma)
        if match:
            code = match.group(1)
            # Map tên chuẩn hóa
            ten_chuan = normalize_ten_nganh(row['ten_nganh'])
            if ten_chuan and ten_chuan not in d_map:
                d_map[ten_chuan] = code
            # Map tên gốc (lowercase) để tăng cơ hội khớp
            ten_goc = str(row['ten_nganh']).strip().lower()
            ten_goc_clean = re.sub(r'\s*\(.*\)\s*', '', ten_goc).strip()
            if ten_goc_clean and ten_goc_clean not in d_map:
                d_map[ten_goc_clean] = code
    return d_map

def normalize_ma_nganh(ma, ten_chuan, dynamic_map):
    ma = str(ma).strip()
    
    # Level 1: Nếu mã gốc đã chứa 7 chữ số (chuẩn Bộ GD) thì trích xuất luôn
    match = re.search(r'(\d{7})', ma)
    if match:
        return match.group(1)
    
    ten_lower = str(ten_chuan).lower().strip()
    # Clean thêm ký tự thừa
    ten_lower = re.sub(r'[\*\s:;,]+$', '', ten_lower).strip()
    
    # Level 2: Khớp CHÍNH XÁC với COMMON_MAJOR_MAP
    if ten_lower in COMMON_MAJOR_MAP:
        return COMMON_MAJOR_MAP[ten_lower]
    
    # Level 3: Khớp CHÍNH XÁC với dynamic_map (học từ dữ liệu)
    if ten_chuan in dynamic_map:
        return dynamic_map[ten_chuan]
    if ten_lower in dynamic_map:
        return dynamic_map[ten_lower]
    
    # Level 4: Khớp MỀM - tên ngành chứa keyword của map
    # Ưu tiên COMMON_MAJOR_MAP trước (chính xác hơn), sau đó dynamic_map
    best_match = None
    best_len = 0
    
    for keyword, code in COMMON_MAJOR_MAP.items():
        if keyword in ten_lower and len(keyword) > best_len:
            best_match = code
            best_len = len(keyword)
    
    if best_match and best_len >= 4:  # Keyword phải đủ dài (>= 4 ký tự) để tránh khớp nhầm
        return best_match
    
    # Level 4b: Thử dynamic_map (keyword matching) 
    for keyword, code in dynamic_map.items():
        keyword_lower = str(keyword).lower()
        if keyword_lower in ten_lower and len(keyword_lower) > best_len:
            best_match = code
            best_len = len(keyword_lower)
    
    if best_match and best_len >= 4:
        return best_match
    
    # Level 5: Giữ nguyên mã gốc - đây là ngành đặc thù chỉ có ở 1 trường
    return ma

def create_rag_pipeline(df):
    print("--- Dang xu ly nhanh RAG ---")
    rag_documents = []
    df_clean = df.dropna(subset=['diem_chuan']).copy()
    
    # Gom nhóm theo ngành chuẩn để mỗi năm chỉ trả ra 1 mức điểm duy nhất cho 1 ngành, tránh RAG bị nhiễu do các chương trình phụ (như CLC, VJU,...)
    # KHÔNG gom theo ten_nganh_chuan vì cùng 1 ma_nganh_chuan có thể có nhiều tên khác nhau (VD: "Khoa học máy tính" vs "Khoa học Máy tính")
    df_grouped = df_clean.groupby(['nam', 'ma_truong', 'ten_truong', 'ma_nganh_chuan', 'ma_to_hop']).agg({
        'diem_chuan': 'mean', 
        'chi_tieu': 'first',
        'ten_nganh_chuan': 'first',
        'phuong_thuc': 'first'
    }).reset_index()
    
    for _, row in df_grouped.iterrows():
        nam = int(row['nam'])
        ten_truong = str(row['ten_truong']).strip()
        ma_truong = str(row['ma_truong']).strip()
        # Quan trong: Dung ten va ma chuan hoa
        ten_nganh = str(row['ten_nganh_chuan']).strip()
        ma_nganh = str(row['ma_nganh_chuan']).strip()
        
        ma_to_hop = str(row['ma_to_hop']).strip()
        diem = round(float(row['diem_chuan']), 2)
        chi_tieu = str(row['chi_tieu']).strip()
        phuong_thuc = str(row['phuong_thuc']).strip()
        
        content = f"Vào năm {nam}, trường {ten_truong} (mã trường: {ma_truong}) lấy mức điểm chuẩn là {diem} điểm cho ngành {ten_nganh} (mã ngành: {ma_nganh})"
        if ma_to_hop and ma_to_hop != 'nan':
            content += f", xét tuyển theo tổ hợp môn {ma_to_hop}."
        else:
            content += "."
            
        if chi_tieu and chi_tieu != 'nan':
            content += f" Chỉ tiêu tuyển sinh của ngành là {chi_tieu}."
            
        if phuong_thuc and phuong_thuc != 'nan':
            content += f" Phương thức xét tuyển: {phuong_thuc}."

        doc = {
            "page_content": content,
            "metadata": {
                "nam": nam,
                "ma_truong": ma_truong,
                "ten_truong": ten_truong,
                "ma_nganh": ma_nganh, # Luu ma chuan hoa vao metadata
                "ten_nganh": ten_nganh,
                "ma_to_hop": ma_to_hop
            }
        }
        rag_documents.append(doc)
        
    output_path = os.path.join('d:\\', 'Project_AI', 'data', 'rag_processed_data.json')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rag_documents, f, ensure_ascii=False, indent=4)
    print(f"Da tao {len(rag_documents)} tai lieu JSON cho RAG tai {output_path}")

def create_ml_pipeline(df):
    print("--- Dang xu ly nhanh ML ---")
    df['diem_chuan'] = pd.to_numeric(df['diem_chuan'], errors='coerce')
    df_clean = df.dropna(subset=['diem_chuan']).copy()
    df_clean['nam'] = df_clean['nam'].astype(int)
    
    keys = ['ma_truong', 'ten_truong', 'ma_nganh', 'ten_nganh', 'ma_nganh_chuan', 'ten_nganh_chuan', 'ma_to_hop', 'phuong_thuc']
    pivot_df = df_clean.pivot_table(index=keys, columns='nam', values='diem_chuan', aggfunc='mean').reset_index()
    
    year_cols_original = [c for c in pivot_df.columns if isinstance(c, int)]
    year_cols_new = [f"diem_chuan_{c}" for c in year_cols_original]
    rename_mapping = {col: f"diem_chuan_{col}" for col in year_cols_original}
    pivot_df.rename(columns=rename_mapping, inplace=True)
    
    pivot_df['so_nam_tuyen_sinh'] = pivot_df[year_cols_new].notna().sum(axis=1)
    
    # 1. Fill các giá trị NaN bằng Nội suy và Back/Forward Fill (Nội suy nếu thiếu ở giữa, Forward fill nếu thiếu năm cuối)
    sorted_year_cols = sorted(year_cols_new)
    temp_scores = pivot_df[sorted_year_cols].T
    temp_scores = temp_scores.interpolate(method='linear').bfill().ffill()
    pivot_df[sorted_year_cols] = temp_scores.T
    
    # 2. Xây dựng các features (Đảm bảo có đủ 3 năm gần nhất, giả sử bộ data là 2023, 2024, 2025)
    # Lấy tự động 3 năm gần nhất có trong dữ liệu
    latest_years = sorted_year_cols[-3:]
    if len(latest_years) == 3:
        y1, y2, y3 = latest_years[0], latest_years[1], latest_years[2]
        pivot_df['trung_binh_3nam'] = pivot_df[[y1, y2, y3]].mean(axis=1).round(2)
        pivot_df['xu_huong_gannhat'] = pivot_df[y3] - pivot_df[y2]  # Ví dụ: 2025 - 2024
        # Phù hợp với feature gốc đòi hỏi xu_huong_24_25
        if y3 == 'diem_chuan_2025' and y2 == 'diem_chuan_2024':
            pivot_df['xu_huong_24_25'] = pivot_df['xu_huong_gannhat']
    
    output_path = os.path.join('d:\\', 'Project_AI', 'data', 'ml_processed_data.csv')
    pivot_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Da xu ly ML: {len(pivot_df)} mau tai {output_path}")
    return output_path

def create_training_data(ml_csv_path):
    """
    Sinh dữ liệu huấn luyện (synthetic) từ dữ liệu thật đã tiền xử lý.
    Logic: Với mỗi ngành/trường/tổ hợp, tạo N thí sinh giả lập có điểm phân bố
    quanh điểm chuẩn, dùng sigmoid để gán nhãn đỗ/rớt.
    """
    print("--- Dang sinh du lieu huan luyen (training data) ---")
    df = pd.read_csv(ml_csv_path)
    
    # Đảm bảo các cột điểm chuẩn tồn tại
    year_cols = [c for c in df.columns if c.startswith('diem_chuan_')]
    if len(year_cols) < 3:
        print(f"CANH BAO: Chi co {len(year_cols)} cot diem chuan, can it nhat 3.")
        return
    
    sorted_years = sorted(year_cols)
    y1, y2, y3 = sorted_years[-3], sorted_years[-2], sorted_years[-1]  # 2023, 2024, 2025
    
    records = []
    KHU_VUC = {"KV1": 0.75, "KV2": 0.5, "KV2NT": 0.25, "KV3": 0.0}
    
    for _, row in df.iterrows():
        dc_latest = row[y3]  # diem_chuan_2025
        dc_prev = row[y2]    # diem_chuan_2024
        dc_old = row[y1]     # diem_chuan_2023
        
        # Bỏ qua nếu thiếu điểm chuẩn
        if pd.isna(dc_latest) or pd.isna(dc_prev) or pd.isna(dc_old):
            continue
        
        tb = round(float((dc_old + dc_prev + dc_latest) / 3), 2)
        xu_huong = round(float(dc_latest - dc_prev), 2)
        
        # Sinh 30 thí sinh giả lập mỗi ngành/trường/tổ hợp
        for _ in range(30):
            kv = np.random.choice(list(KHU_VUC.keys()), p=[0.15, 0.25, 0.35, 0.25])
            cong = KHU_VUC[kv]
            
            diem_raw = float(round(float(np.random.normal(loc=dc_latest, scale=2.5)), 2))
            diem_raw = max(0.0, min(30.0, diem_raw))
            diem_co_uu_tien = float(round(float(diem_raw + cong), 2))
            chenh_lech = float(round(float(diem_co_uu_tien - dc_latest), 2))
            
            # Xác suất đỗ dùng sigmoid
            prob = 1 / (1 + np.exp(-2.5 * chenh_lech))
            ket_qua = int(np.random.random() < prob)
            
            records.append({
                "diem_thi_sinh": diem_raw,
                "diem_cong_kv": cong,
                y1: dc_old,
                y2: dc_prev,
                y3: dc_latest,
                "trung_binh_3nam": tb,
                "xu_huong_24_25": xu_huong,
                "chenh_lech": chenh_lech,
                "ket_qua": ket_qua,
            })
    
    train_df = pd.DataFrame(records)
    output_path = os.path.join('d:\\', 'Project_AI', 'data', 'processed', 'training_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    do_rate = train_df["ket_qua"].mean()
    print(f"Da sinh {len(train_df)} mau huan luyen | Ti le do: {do_rate:.1%} tai {output_path}")

def main():
    print("BAT DAU XU LY DU LIEU DA NHANH (RAG & ML)")
    # Duong dan data goc - Nam trong thu muc notebooks/.../data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(script_dir, 'data')
    csv_files = glob.glob(os.path.join(base_data_dir, 'Diem_Chuan_*.csv'))
    
    if not csv_files:
        print(f"Khong tim thay file Du lieu goc tai {base_data_dir}.")
        return
        
    print(f"Tim thay {len(csv_files)} file CSV.")
    df_list = []
    for f in csv_files:
        temp_df = pd.read_csv(f, dtype={'nam': str, 'ma_truong': str})
        df_list.append(temp_df)
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # --- CHUAN HOA DU LIEU TOAN CUC ---
    print("--- Dang chuan hoa ma nganh va ten nganh toan he thong ---")
    combined_df['ten_nganh_chuan'] = combined_df['ten_nganh'].apply(normalize_ten_nganh)
    dynamic_map = get_dynamic_major_map(combined_df)
    combined_df['ma_nganh_chuan'] = combined_df.apply(
        lambda row: normalize_ma_nganh(row['ma_nganh'], row['ten_nganh_chuan'], dynamic_map), 
        axis=1
    )
    
    # Khoi chay cac nhanh voi du lieu da chuan hoa
    create_rag_pipeline(combined_df.copy())
    ml_csv_path = create_ml_pipeline(combined_df.copy())
    create_training_data(ml_csv_path)
    print("HOAN TAT!")

if __name__ == "__main__":
    main()
