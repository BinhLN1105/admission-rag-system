import requests
import json
import csv
import os
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# Đường dẫn file 2025 đã làm trước đó
CSV_2025_PATH = r"d:\Dev\Code\admission-rag-system\thuthapdata2025\diem_chuan_2025.csv"

def get_unique_school_codes_from_2025():
    """
    Lấy danh sách mã trường duy nhất từ diem_chuan_2025.csv để lấp đầy dữ liệu 2024
    """
    codes = set()
    try:
        with open(CSV_2025_PATH, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("Mã trường")
                if code:
                    codes.add(code.strip())
    except Exception as e:
        print(f"Lỗi đọc file 2025: {e}")
    return list(codes)

def find_school_info(school_code):
    """
    Sử dụng input_college để search ID trường trên VnExpress kết hợp Retry
    """
    url = "https://diemthi.vnexpress.net/tra-cuu-dai-hoc/loadcollege"
    params = {
        "location_id": -1,
        "input_college": school_code,
        "offset": 0,
        "limit": 20,
        "college_type": 2
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://diemthi.vnexpress.net/"
    }
    
    for _ in range(3): # Thử lại tối đa 3 lần nếu mạng lỗi
        try:
            res = requests.get(url, params=params, headers=headers, timeout=15)
            if res.status_code == 200:
                data = res.json()
                html = data.get("html", "")
                if not html:
                    break
                    
                soup = BeautifulSoup(html, "html.parser")
                items = soup.find_all("li", class_="lookup__result")
                for li in items:
                    code_div = li.find("div", class_="lookup__result-code")
                    s_code = code_div.text.strip() if code_div else ""
                    
                    # Kiểm tra độ khớp của mã trường
                    if s_code.upper() == school_code.upper() or school_code.upper() in s_code.upper():
                        name_div = li.find("div", class_="lookup__result-name")
                        s_name = name_div.find("strong").text.strip() if name_div and name_div.find("strong") else ""
                        
                        btn = li.find("button", class_="btn-favourite")
                        s_id = btn.get("data-id") if btn else ""
                        if s_id:
                            return {"id": s_id, "code": s_code, "name": s_name}
            time.sleep(1)
        except Exception:
            time.sleep(1)
            pass
    return None

def fetch_school_and_scrape(school_code):
    """
    Tìm ID -> Lấy danh sách điểm chuẩn -> Trả về mảng records kết hợp Retry
    """
    school_info = find_school_info(school_code)
    if not school_info:
        # Trường không có dữ liệu trên VnExpress
        return []
    
    s_id = school_info["id"]
    s_code = school_info["code"]
    s_name = school_info["name"]
    
    url = f"https://diemthi.vnexpress.net/tra-cuu-dai-hoc/loadbenchmark/id/{s_id}/year/2024/sortby/1/block_name/all"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    results = []
    
    for _ in range(3): # Thử lại 3 lần nếu có lỗi ConnectionPool
        try:
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code == 200:
                data = res.json()
                html = data.get("html", "")
                if not html:
                    break
                    
                soup = BeautifulSoup(html, "html.parser")
                table = soup.find("table", class_="university__table")
                if not table:
                    break
                    
                rows = table.find("tbody").find_all("tr", class_="university__benchmark")
                for row in rows:
                    if "university__benchmark--chart" in row.get("class", []):
                        continue
                        
                    tds = row.find_all("td")
                    if len(tds) < 6:
                        continue
                        
                    # Lấy Tên, Mã ngành
                    name_tag = tds[1].find("strong")
                    m_name = name_tag.text.strip() if name_tag else tds[1].text.strip()
                    spans = tds[1].find_all("span")
                    m_code = spans[-1].text.strip() if len(spans) > 0 else ""
                    
                    # Lấy Điểm
                    score_span = tds[2].find("span")
                    score = score_span.text.strip() if score_span else "0"
                    
                    # Lấy Tổ hợp môn
                    a_tags = tds[3].find_all("a")
                    blocks = " | ".join([a.text.strip() for a in a_tags]) if a_tags else tds[3].text.strip()
                    
                    # Ghi chú
                    note = tds[5].text.strip()
                    
                    results.append({
                        "Mã trường": s_code,
                        "Tên trường": s_name,
                        "Mã ngành": m_code,
                        "Tên ngành": m_name,
                        "Tổ hợp môn": blocks,
                        "Điểm thi THPT": score,
                        "Ghi chú": note
                    })
                return results
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
            
    print(f"Lỗi: Không thể cào điểm trường {s_code} sau 3 lần thử.")
    return []

def main():
    print("Đang quét đọc toàn bộ Mã trường từ diem_chuan_2025.csv...")
    codes = get_unique_school_codes_from_2025()
    
    if not codes:
        print("Không tìm thấy mã trường nào (Hoặc file 2025 trống).")
        return
        
    print(f"Tìm thấy kho {len(codes)} Mã trường. Bắt đầu tìm kiếm đối chiếu vào VnExpress...\n")
    
    all_records = []
    
    # Sử dụng 10 workers để giảm rủi ro HTTPS Connection Pool Limit của requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        completed = 0
        for result in executor.map(fetch_school_and_scrape, codes):
            if result:
                all_records.extend(result)
            completed += 1
            if completed % 30 == 0:
                print(f"Tiến độ: Đã tra cứu {completed}/{len(codes)} trường...")
                
    output_dir = r"d:\Dev\Code\admission-rag-system\thuthapdata2025"
    output_file = os.path.join(output_dir, "diem_chuan_2024.csv")
    
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ["Mã trường", "Tên trường", "Mã ngành", "Tên ngành", "Tổ hợp môn", "Điểm thi THPT", "Ghi chú"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_records:
            writer.writerow(row)
            
    print(f"\n=> Đã thu hoạch thành công {len(all_records)} dòng điểm chuẩn 2024 (bao phủ rộng hơn rất nhiều).")
    print(f"=> Dữ liệu đã lưu tại: {output_file}")

if __name__ == "__main__":
    main()
