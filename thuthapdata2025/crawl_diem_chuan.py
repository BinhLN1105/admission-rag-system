import requests
import json
import csv
import os
import concurrent.futures

def get_schools():
    """
    Gọi API để lấy danh sách tất cả các trường đại học/cao đẳng
    """
    url = "https://diemthi.tuyensinh247.com/api/school/search?q="
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return data.get("data", [])
    except Exception as e:
        print(f"Lỗi khi lấy danh sách trường: {e}")
    return []

def extract_jsons(text):
    """
    Hàm phân tích và lấy toàn bộ các JSON Object điểm chuẩn có trong mã HTML trả về.
    Vì dữ liệu được Render bằng Next.js (App Router) nên bảng HTML thường không đẩy 
    ra thẻ <table> thuần ở lần tải đầu, thay vào đó dữ liệu nằm nguyên dưới dạng JSON payload.
    """
    results = []
    
    # Chuẩn hóa nếu như chuỗi bị serialize dạng escape javascript
    text_unescaped = text.replace('\\"', '"')
    parts = text_unescaped.split('{"id":')
    if len(parts) <= 1:
        return results
        
    for part in parts[1:]:
        s = '{"id":' + part
        depth = 0
        in_string = False
        escape = False
        end_idx = -1
        
        # Stack-based JSON boundary detect
        for i, char in enumerate(s):
            if char == '"' and not escape:
                in_string = not in_string
            elif char == '\\' and not escape:
                escape = True
                continue
            elif char == '{' and not in_string:
                depth += 1
            elif char == '}' and not in_string:
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
            escape = False
            
        if end_idx != -1:
            json_str = s[:end_idx+1]
            try:
                obj = json.loads(json_str)
                # Chỉ lọc những object có chứa thông tin điểm thi
                if "school_id" in obj and "mark" in obj and "name" in obj and "code" in obj:
                    results.append(obj)
            except:
                pass
    return results

def get_cutoff_scores_worker(school):
    """
    Worker xử lý trong luồng (Thread), tải và parse dữ liệu của 1 trường cụ thể đa luồng
    """
    alias = school.get("alias")
    code = school.get("code")
    if not alias or not code:
        return school, []
    
    url = f"https://diemthi.tuyensinh247.com/diem-chuan/{alias}-{code}.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return school, []
        return school, extract_jsons(response.text)
    except Exception as e:
        # Timeout hoặc exception mạng
        return school, []

def main():
    print("Bắt đầu gọi API lấy danh sách các trường...")
    schools = get_schools()
    print(f"Tìm thấy {len(schools)} trường.")
    
    # Dictionary chứa tất cả dữ liệu ngành học, key là (mã trường_mã ngành_tên ngành)
    records = {} 
    
    # Tập hợp các loại phương thức xét tuyển chung 
    all_methods = set()
    
    print("\n[CHẾ ĐỘ ĐA LUỒNG] Script đang kích hoạt xử lý 15 Request song song để tối đa tốc độ.")
    print("Quá trình này sẽ diễn ra Rất Nhanh (ước tính 10-20 giây cho >500 trường). Vui lòng chờ...\n")

    count = 0
    # Mở 15 luồng tải về đồng thời để rút ngắn thời gian F1 1 trường từ 5 phút xuống 15 giây
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_school = {executor.submit(get_cutoff_scores_worker, school): school for school in schools}
        
        # `as_completed` sẽ nhận kết quả ngay khi bất kỳ một luồng (trường) nào quét xong trước
        for future in concurrent.futures.as_completed(future_to_school):
            count += 1
            school = future_to_school[future]
            school_code = school.get("code", "")
            school_name = school.get("name", "")
            
            try:
                _, objects = future.result()
                collected_for_this_school = 0
                
                for obj in objects:
                    major_code = (obj.get("code") or "").strip()
                    major_name = (obj.get("name") or "").strip()
                    method = (obj.get("admission_name") or "").strip()
                    mark = obj.get("mark", 0)
                    block = obj.get("block", "")
                    note = obj.get("introtext", "")
                    
                    if not major_code or not method:
                        continue
                        
                    # Gộp lại dựa trên "Mã trường" + "Mã Ngành" + "Tên Ngành"
                    key = f"{school_code}_{major_code}_{major_name}" 
                    if key not in records:
                        records[key] = {
                            "Mã trường": school_code,
                            "Tên trường": school_name,
                            "Mã ngành": major_code,
                            "Tên ngành": major_name,
                            "Tổ hợp môn": set(),
                            "Ghi chú": set()
                        }
                    
                    all_methods.add(method)
                    # Lưu điểm cho tương ứng tên phương thức
                    records[key][method] = mark
                    collected_for_this_school += 1
                    
                    if block:
                        records[key]["Tổ hợp môn"].add(block)
                    if note:
                        records[key]["Ghi chú"].add(note)
                        
                if collected_for_this_school > 0:
                    print(f"[{count:03}/{len(schools)}] [Thành công] {school_name} - {collected_for_this_school} ngành/điểm.")
                else:
                    print(f"[{count:03}/{len(schools)}] [Trống rỗng] {school_name} chưa có hoặc sai mã.")
                    
            except Exception as exc:
                print(f"[{count:03}/{len(schools)}] [Lỗi mạng] {school_name}: {exc}")

    print(f"\nĐã gom dữ liệu xong 100%. Đang tiến hành tạo CSV với '{len(all_methods)}' CỘT phương thức xét tuyển...")
    
    # Sắp xếp tên cột phương thức ngang
    sorted_methods = sorted(list(all_methods))
    
    output_dir = r"d:\Dev\Code\admission-rag-system\thuthapdata2025"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "diem_chuan_2025.csv")
    
    # Fieldnames chuẩn
    fieldnames = ["Mã trường", "Tên trường", "Mã ngành", "Tên ngành", "Tổ hợp môn"] + sorted_methods + ["Ghi chú"]
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for key, rec in records.items():
            row = {
                "Mã trường": rec["Mã trường"],
                "Tên trường": rec["Tên trường"],
                "Mã ngành": rec["Mã ngành"],
                "Tên ngành": rec["Tên ngành"],
                "Tổ hợp môn": " | ".join(filter(None, rec["Tổ hợp môn"])),
                "Ghi chú": " | ".join(filter(None, rec["Ghi chú"]))
            }
            # Nếu ngành chưa có phương thức này thì để 0
            for m in sorted_methods:
                val = rec.get(m)
                row[m] = val if val is not None else 0
                
            writer.writerow(row)
            
    print(f"\n=> TỐC ĐỘ SIÊU CAO HOÀN TẤT. Xem file CSV tại:\n{output_file}")

if __name__ == "__main__":
    main()
