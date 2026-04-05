import csv
import os
import re

def extract_combinations():
    input_file = r"d:\Dev\Code\admission-rag-system\thuthapdata2025\diem_chuan_2025.csv"
    output_file = r"d:\Dev\Code\admission-rag-system\data\raw\to_hop_mon.csv"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    unique_blocks = set()
    
    try:
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                blocks_raw = row.get("Tổ hợp môn", "")
                if not blocks_raw:
                    continue
                
                # Làm sạch cơ bản: in hoa, xóa \r \n, \t thành dấu cách
                blocks_raw = blocks_raw.upper().replace('\r', ' ').replace('\n', ' ')
                
                # Cắt chuỗi bằng các ký tự phân cách thường gặp
                parts = re.split(r'[;|,|\-|\/|\\|\(|\)]+', blocks_raw)
                
                for p in parts:
                    words = p.split() # tách bằng space
                    for w in words:
                        w = w.strip()
                        # Lọc bằng Regex:
                        # - 1..2 chữ cái + 1..3 số (vd: A00, D145, AH1, DD2, M01)
                        # - Hoặc chữ + số + chữ (vd: A0C, A0T)
                        if re.match(r'^[A-Z]{1,2}\d{1,3}$', w) or re.match(r'^[A-Z]\d[A-Z]$', w):
                            unique_blocks.add(w)
                        
        sorted_blocks = sorted(list(unique_blocks))
        
        predefined = {
            "A00": ["Khối A", "Toán", "Vật lý", "Hóa học"],
            "A01": ["Khối A1", "Toán", "Vật lý", "Tiếng Anh"],
            "B00": ["Khối B", "Toán", "Hóa học", "Sinh học"],
            "C00": ["Khối C", "Ngữ văn", "Lịch sử", "Địa lý"],
            "D01": ["Khối D1", "Toán", "Ngữ văn", "Tiếng Anh"],
            "D07": ["Khối D7", "Toán", "Hóa học", "Tiếng Anh"],
            "D08": ["Khối D8", "Toán", "Sinh học", "Tiếng Anh"],
            "D09": ["Khối D9", "Toán", "Lịch sử", "Tiếng Anh"],
            "D10": ["Khối D10", "Toán", "Địa lý", "Tiếng Anh"]
        }
        
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ma_to_hop", "ten_to_hop", "mon_1", "mon_2", "mon_3"])
            for block in sorted_blocks:
                if block in predefined:
                    writer.writerow([block] + predefined[block])
                else:
                    writer.writerow([block, "0", "0", "0", "0"])
                
        print(f"=>{'='*40}")
        print(f"Đã LÀM SẠCH và trích xuất thành công {len(sorted_blocks)} mã tổ hợp môn chuẩn.")
        print(f"Những rác văn bản (Hát, Toán, \r\n...) đã bị loại bỏ hoàn toàn.")
        print(f"File output tại: {output_file}")
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {input_file}.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    extract_combinations()
