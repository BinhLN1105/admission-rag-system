import sys
import os
import pandas as pd

# Thêm đường dẫn gốc của project vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_priority_scores

# Tải điểm ưu tiên lần đầu
PRIORITY_MAP = load_priority_scores()

def calculate_priority_score(khu_vuc):
    """Tính điểm ưu tiên theo khu vực"""
    if not khu_vuc:
        return 0.0
        
    # Chuẩn hóa format: KV2-NT -> KV2NT
    kv_norm = str(khu_vuc).upper().replace("-", "")
    
    return float(PRIORITY_MAP.get(kv_norm, 0.0))

def process_priority_scores():
    """Xử lý điểm ưu tiên trong dữ liệu"""

    print("🔄 Đang xử lý điểm ưu tiên...")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_path = os.path.join(base_dir, "data", "processed", "training_data.csv")

    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Giả sử cột khu_vuc có format như 'KV1', 'KV2', etc.
    # Nếu chưa có, có thể thêm sau
    if 'khu_vuc' not in df.columns:
        print("⚠ Cột 'khu_vuc' chưa có, bỏ qua xử lý ưu tiên")
        return

    # Tính điểm cộng ưu tiên
    df['diem_cong_kv'] = df['khu_vuc'].apply(calculate_priority_score)

    # Lưu lại
    df.to_csv(input_path, index=False, encoding='utf-8')

    print(f"✓ Đã cập nhật điểm ưu tiên cho {len(df)} mẫu")

if __name__ == "__main__":
    process_priority_scores()
