import pandas as pd
import numpy as np
import os
import sys
# Thêm đường dẫn gốc của project vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_priority_scores

# Tải điểm ưu tiên
PRIORITY_MAP = load_priority_scores()

def generate_synthetic_data():
    """Tạo dữ liệu synthetic cho huấn luyện mô hình ML"""

    print("🔄 Đang tạo dữ liệu synthetic...")
    
    # Lấy danh sách các khu vực có sẵn từ CSV
    available_kvs = list(PRIORITY_MAP.keys())
    if not available_kvs:
        available_kvs = ['KV1', 'KV2', 'KV2NT', 'KV3']

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_path = os.path.join(base_dir, "data", "ml_processed_data.csv")
    output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_data.csv")

    # Đọc dữ liệu gốc
    df = pd.read_csv(input_path)
    print(f"✓ Đọc {len(df)} hàng dữ liệu gốc")

    # Tạo dữ liệu lịch sử điểm chuẩn theo ngành và tổ hợp
    historical_data = {}

    # Gom nhóm theo mã ngành và tổ hợp để tạo dữ liệu lịch sử
    grouped = df.groupby(['ma_nganh', 'ma_to_hop'])

    print(f"✓ Tìm thấy {len(grouped)} nhóm ngành/tổ hợp")

    for (ma_nganh, ma_to_hop), group in grouped:
        key = f"{ma_nganh}|{ma_to_hop}"

        # Lấy điểm chuẩn theo năm
        diem_2023 = group[group['nam'] == 2023]['diem_chuan'].mean() if 2023 in group['nam'].values else np.nan
        diem_2024 = group[group['nam'] == 2024]['diem_chuan'].mean() if 2024 in group['nam'].values else np.nan
        diem_2025 = group[group['nam'] == 2025]['diem_chuan'].mean() if 2025 in group['nam'].values else np.nan

        # Nếu thiếu dữ liệu năm trước, tạo dựa trên năm hiện tại với biến động ngẫu nhiên
        if pd.isna(diem_2023) and not pd.isna(diem_2024):
            diem_2023 = diem_2024 + np.random.normal(0, 0.5)  # Biến động ±0.5
        if pd.isna(diem_2024) and not pd.isna(diem_2025):
            diem_2024 = diem_2025 + np.random.normal(0, 0.3)  # Biến động ±0.3
        if pd.isna(diem_2023) and pd.isna(diem_2024) and not pd.isna(diem_2025):
            diem_2024 = diem_2025 + np.random.normal(0, 0.3)
            diem_2023 = diem_2024 + np.random.normal(0, 0.5)

        # Lấy thông tin ngành
        ten_nganh = group['ten_nganh'].iloc[0]

        historical_data[key] = {
            'ma_nganh': ma_nganh,
            'ten_nganh': ten_nganh,
            'ma_to_hop': ma_to_hop,
            'diem_chuan_2023': round(diem_2023, 2) if not pd.isna(diem_2023) else np.nan,
            'diem_chuan_2024': round(diem_2024, 2) if not pd.isna(diem_2024) else np.nan,
            'diem_chuan_2025': round(diem_2025, 2) if not pd.isna(diem_2025) else np.nan
        }

    print(f"✓ Tạo dữ liệu lịch sử cho {len(historical_data)} ngành/tổ hợp")
    valid_historical = {k: v for k, v in historical_data.items() if not (pd.isna(v['diem_chuan_2023']) or pd.isna(v['diem_chuan_2024']) or pd.isna(v['diem_chuan_2025']))}
    print(f"✓ Có {len(valid_historical)} ngành/tổ hợp có đủ dữ liệu lịch sử")

    # Tạo dữ liệu synthetic
    synthetic_data = []

    print(f"✓ Bắt đầu tạo synthetic data từ {len(valid_historical)} ngành/tổ hợp hợp lệ")

    for key, info in valid_historical.items():
        # Debug: In thông tin ngành đầu tiên
        if len(synthetic_data) == 0:
            print(f"  Debug - Ngành đầu tiên: {key}")
            print(f"    2023: {info['diem_chuan_2023']}, 2024: {info['diem_chuan_2024']}, 2025: {info['diem_chuan_2025']}")

        # Tạo nhiều thí sinh cho mỗi ngành/tổ hợp
        for i in range(10):  # 10 thí sinh cho mỗi ngành/tổ hợp
            # Chọn ngẫu nhiên một năm để tạo điểm thi
            nam = np.random.choice([2023, 2024, 2025])
            diem_chuan = info[f'diem_chuan_{nam}']

            # Tạo điểm thi ngẫu nhiên quanh điểm chuẩn
            diem_thi = np.random.normal(diem_chuan, 2.0)
            diem_thi = max(0, min(30, diem_thi))  # Giới hạn 0-30

            # Xác định kết quả (đỗ/rớt)
            ket_qua = 1 if diem_thi >= diem_chuan else 0

            # Thêm ưu tiên khu vực (ngẫu nhiên từ danh sách có sẵn)
            khu_vuc = np.random.choice(available_kvs)
            diem_cong_kv = float(PRIORITY_MAP.get(khu_vuc, 0.0))

            synthetic_data.append({
                'ma_nganh': info['ma_nganh'],
                'ten_nganh': info['ten_nganh'],
                'ma_to_hop': info['ma_to_hop'],
                'nam': nam,
                'diem_thi_sinh': round(diem_thi, 2),
                'diem_cong_kv': diem_cong_kv,
                'khu_vuc': khu_vuc, # Thêm cột khu_vuc để sau này dễ update
                'diem_chuan_2023': info['diem_chuan_2023'],
                'diem_chuan_2024': info['diem_chuan_2024'],
                'diem_chuan_2025': info['diem_chuan_2025'],
                'ket_qua': ket_qua
            })

    print(f"✓ Tạo được {len(synthetic_data)} mẫu synthetic")

    # Tạo DataFrame
    synthetic_df = pd.DataFrame(synthetic_data)

    # Tính các đặc trưng bổ sung
    synthetic_df['trung_binh_3nam'] = synthetic_df[['diem_chuan_2023', 'diem_chuan_2024', 'diem_chuan_2025']].mean(axis=1)
    synthetic_df['xu_huong_24_25'] = synthetic_df['diem_chuan_2025'] - synthetic_df['diem_chuan_2024']
    synthetic_df['chenh_lech'] = synthetic_df['diem_thi_sinh'] - synthetic_df['trung_binh_3nam']

    # Lưu file
    synthetic_df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"✓ Tạo thành công: {len(synthetic_df)} mẫu huấn luyện")
    print(f"✓ Lưu tại: {output_path}")

    # Thống kê
    print(f"\n📊 Thống kê:")
    print(f"  - Đỗ: {int(synthetic_df['ket_qua'].sum())}")
    print(f"  - Rớt: {len(synthetic_df) - int(synthetic_df['ket_qua'].sum())}")
    print(".1f")

if __name__ == "__main__":
    generate_synthetic_data()