import requests
import json
import csv
import re
import os

def get_standard_name(code):
    """
    Hàm chuẩn hóa tên khối thi (A00 -> Khối A, A01 -> Khối A1, D01 -> Khối D1)
    như bạn mô tả, giúp dữ liệu đẹp hơn.
    """
    code = code.upper()
    predefined = {
        "A00": "Khối A",
        "B00": "Khối B",
        "C00": "Khối C",
        "D00": "Khối D", 
        "V00": "Khối V",
        "H00": "Khối H",
        "M00": "Khối M",
        "N00": "Khối N",
        "T00": "Khối T",
        "K00": "Khối K",
        "R00": "Khối R",
        "S00": "Khối S",
    }
    
    if code in predefined:
        return predefined[code]
        
    # Logic cắt bớt số 0 ở giữa: VD: A01 -> A1, D07 -> D7
    match = re.match(r'^([A-Z])0([1-9])$', code)
    if match:
        return f"Khối {match.group(1)}{match.group(2)}"
        
    return f"Khối {code}"

def main():
    url = "https://huongnghiep.hocmai.vn/wp-admin/admin-ajax.php?action=get_tohop_list_detail"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    print("Đang gọi API tới Hocmai để lấy dữ liệu tổ hợp môn...")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return
        
    output_dir = r"d:\Dev\Code\admission-rag-system\thuthapdata2025"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "to_hop_mon.csv")
    
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        # Ghi Header
        writer.writerow(["ma_to_hop", "ten_to_hop", "mon_1", "mon_2", "mon_3"])
        
        count = 0
        for item in data:
            code = str(item.get("value", "")).upper().strip()
            if not code:
                continue
                
            label = str(item.get("label", "")).strip()
            
            # Khởi tạo giá trị mặc định là 0
            ten = get_standard_name(code)
            mon1 = "0"
            mon2 = "0"
            mon3 = "0"
            
            # Trích xuất dữ liệu bên trong dấu ngoặc đơn: (Toán, Vật lí, Hóa học)
            match = re.search(r'\((.*?)\)', label)
            if match:
                subjects_str = match.group(1)
                # Tách bằng dấu phẩy
                subjects = [s.strip() for s in subjects_str.split(',')]
                
                if len(subjects) > 0: mon1 = subjects[0]
                if len(subjects) > 1: mon2 = subjects[1]
                if len(subjects) > 2: mon3 = subjects[2]
                
            writer.writerow([code, ten, mon1, mon2, mon3])
            count += 1
            
    print(f"=>{'='*40}")
    print(f"Đã LÀM SẠCH và bóc tách thành công {count} tổ hợp môn từ Hocmai.")
    print(f"File CSV gọn gàng đã được lưu tại:\n{output_file}")

if __name__ == "__main__":
    main()
