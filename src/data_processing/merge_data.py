import pandas as pd
import os
import re

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
    ten = re.sub(r'(?i)(\s*\(.*\)\s*|-.*|chất lượng cao|chuyên ngành.*|cttt.*|chương trình.*|tiên tiến.*|liên kết.*|định hướng.*|tăng cường.*|học bằng.*|song bằng.*|lớp chọn.*)', '', ten).strip()
    ten = re.sub(r'[\*\s:;,]+$', '', ten).strip()
    return ten.capitalize()


def get_dynamic_major_map(df):
    d_map = {}
    for _, row in df.iterrows():
        ma = str(row['ma_nganh']).strip()
        match = re.search(r'(\d{7})', ma)
        if match:
            code = match.group(1)
            ten_chuan = normalize_ten_nganh(row['ten_nganh'])
            ten_lower = ten_chuan.lower().strip()
            if ten_lower and ten_lower not in d_map:
                d_map[ten_lower] = code

            ten_goc = str(row['ten_nganh']).strip().lower()
            ten_goc_clean = re.sub(r'\s*\(.*\)\s*', '', ten_goc).strip()
            if ten_goc_clean and ten_goc_clean not in d_map:
                d_map[ten_goc_clean] = code
    return d_map


def normalize_ma_nganh(ma, ten_nganh, dynamic_map):
    ma = str(ma).strip()
    ten_chuan = normalize_ten_nganh(ten_nganh)
    ten_lower = str(ten_chuan).lower().strip()
    ten_lower = re.sub(r'[\*\s:;,]+$', '', ten_lower).strip()

    # 1. Ưu tiên Mapping cứng (Quy chuẩn của chúng ta)
    if ten_lower in COMMON_MAJOR_MAP:
        return COMMON_MAJOR_MAP[ten_lower]
        
    # 2. Ưu tiên Mapping động (Đã tìm thấy mã 7 chữ số chuẩn cùng tên này ở dòng khác)
    if ten_chuan in dynamic_map:
        return dynamic_map[ten_chuan]
    if ten_lower in dynamic_map:
        return dynamic_map[ten_lower]

    # 3. Nếu không có mapping, mới dùng mã 7 chữ số gốc (nếu có)
    match = re.search(r'(\d{7})', ma)
    if match:
        return match.group(1)

    # 4. Cuối cùng, thử tìm kiếm keyword một lần nữa
    best_match = None
    best_len = 0
    for keyword, code in COMMON_MAJOR_MAP.items():
        if keyword in ten_lower and len(keyword) > best_len:
            best_match = code
            best_len = len(keyword)
    for keyword, code in dynamic_map.items():
        keyword_lower = str(keyword).lower()
        if keyword_lower in ten_lower and len(keyword_lower) > best_len:
            best_match = code
            best_len = len(keyword_lower)

    if best_match and best_len >= 4:
        return best_match

    return ma


def normalize_ten_truong(ten):
    ten = str(ten).strip()
    ten = re.sub(r'\s+', ' ', ten)
    return ten.title()


def merge_data():
    """Gộp dữ liệu từ các năm 2023, 2024, 2025 thành file ml_processed_data.csv"""

    print("🔄 Đang gộp dữ liệu từ các năm...")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")

    # Đọc dữ liệu từ các năm
    files = {
        2023: os.path.join(data_dir, "diem_chuan_2023.csv"),
        2024: os.path.join(data_dir, "diem_chuan_2024.csv"),
        2025: os.path.join(data_dir, "diem_chuan_2025.csv")
    }

    dfs = []
    for year, file_path in files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['nam'] = year  # Thêm cột năm nếu chưa có
            dfs.append(df)
            print(f"✓ Đọc {len(df)} hàng từ {year}")
        else:
            print(f"⚠ Không tìm thấy file {file_path}")

    if not dfs:
        print("❌ Không có dữ liệu nào để gộp!")
        return

    # Gộp tất cả dữ liệu
    merged_df = pd.concat(dfs, ignore_index=True)

    # Lưu giữ tên và mã raw nếu cần đối chiếu
    merged_df['ten_truong_raw'] = merged_df['ten_truong']
    merged_df['ma_nganh_raw'] = merged_df['ma_nganh']
    merged_df['ten_nganh_raw'] = merged_df['ten_nganh']

    merged_df['ten_nganh_chuan'] = merged_df['ten_nganh'].apply(normalize_ten_nganh)
    dynamic_map = get_dynamic_major_map(merged_df)
    merged_df['ma_nganh_chuan'] = merged_df.apply(
        lambda row: normalize_ma_nganh(row['ma_nganh'], row['ten_nganh_chuan'], dynamic_map), axis=1
    )
    merged_df['ten_truong_chuan'] = merged_df['ten_truong'].apply(normalize_ten_truong)

    # Chuẩn hóa tên trường theo mã trường
    canonical_names = merged_df.groupby('ma_truong')['ten_truong_chuan'].agg(
        lambda names: names.mode().iloc[0] if not names.mode().empty else names.iloc[0]
    )
    merged_df['ten_truong_chuan'] = merged_df['ma_truong'].map(canonical_names)

    # Ghi lại cột chính bằng phiên bản chuẩn để tránh nhìn thấy nhiều biến thể
    merged_df['ten_truong'] = merged_df['ten_truong_chuan']
    merged_df['ten_nganh'] = merged_df['ten_nganh_chuan']
    merged_df['ma_nganh'] = merged_df['ma_nganh_chuan']

    before = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['ma_truong', 'ma_nganh_chuan', 'ma_to_hop', 'nam'])
    after = len(merged_df)
    if after < before:
        print(f"✓ Loại trùng lặp: {before-after} bản ghi trùng")

    # Lưu file gộp
    output_path = os.path.join(base_dir, "data", "ml_processed_data.csv")
    merged_df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"✓ Đã gộp thành công: {len(merged_df)} hàng")
    print(f"✓ Lưu tại: {output_path}")

    # Thống kê
    print(f"\n📊 Thống kê:")
    print(f"  - Số trường: {merged_df['ma_truong'].nunique()}")
    print(f"  - Số ngành: {merged_df['ma_nganh'].nunique()}")
    print(f"  - Số tổ hợp: {merged_df['ma_to_hop'].nunique()}")

if __name__ == "__main__":
    merge_data()
