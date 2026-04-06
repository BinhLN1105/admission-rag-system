import requests
from bs4 import BeautifulSoup
import json
import time
import re
import colorama
from colorama import Fore, Style

# Bật tự động reset màu
colorama.init(autoreset=True)

def quet_danh_sach_truong_api():
    print(f"{Fore.YELLOW}{Style.BRIGHT}--- BẮT ĐẦU CÀO DANH SÁCH LINK TỪ API ---")
    
    # URL gốc của API
    api_url = "https://diemthi.vnexpress.net/tra-cuu-dai-hoc/loadcollege"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest" # Định dạng bắt buộc để báo server đây là request AJAX
    }
    
    danh_sach_url = set()
    limit = 20
    
    # Quét theo từ điển từ khóa để lấy tất cả các trường đại học (Vét cả những trường bị ẩn khỏi list mặc định)
    keywords = ['Đại học', 'Học viện', 'Khoa ', 'Phân hiệu', 'Nhạc viện', 'Viện ', 'Sĩ quan', 'Trường']
    for kw in keywords:
        offset = 0
        while True:
            # Xây dựng các tham số (parameters) truyền vào API
            params = {
                "location_id": "-1",   # -1 nghĩa là lấy Toàn quốc
                "input_college": kw,   # Tìm kiếm theo từ khóa
                "offset": offset,      # Vị trí bắt đầu lấy
                "limit": limit,        # Số lượng trường lấy mỗi lần
                "college_type": ""     # Truyền chuỗi rỗng
            }
            
            print(f"{Fore.CYAN}Đang quét từ khóa '{kw}' từ offset {offset} đến {offset + limit}...")
            
            try:
                response = requests.get(api_url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                
                # Phân tích chuỗi JSON trả về
                data = response.json()
                html_content = data.get('html', '').strip()
                
                # ĐIỀU KIỆN DỪNG: Nếu html rỗng nghĩa là đã lấy hết toàn bộ trường đại học
                if not html_content:
                    print(f"{Fore.GREEN}{Style.BRIGHT}Đã lấy hết từ khóa '{kw}'! Chuyển sang từ khóa tiếp theo.")
                    break
                    
                # Đưa HTML vào bóc tách
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Tìm tất cả các thẻ a
                cac_the_a = soup.find_all('a', href=True)
                
                for a in cac_the_a:
                    href = a.get('href')
                    name = a.text.lower() # Lấy tên trường để lọc
                    
                    # Bỏ qua các trường Cao đẳng bằng cách check cả tên lẫn link
                    if "cao đẳng" in name or "cao dang" in name or "/cao-dang-" in href or "-cao-dang-" in href:
                        continue
                    
                    # Lọc đúng các link trỏ đến chi tiết trường (Có dạng /tra-cuu-dai-hoc/ten-truong-123 hoặc tương tự)
                    if re.match(r'^/tra-cuu-dai-hoc/[a-z0-9\-]+-\d+$', href):
                        full_url = "https://diemthi.vnexpress.net" + href
                        danh_sach_url.add(full_url)
                        
                # Tăng offset lên để lấy trang tiếp theo
                offset += limit
                
                # Ngủ 1 giây để thân thiện với server
                time.sleep(1)
                
            except Exception as e:
                print(f"{Fore.RED}Lỗi khi gọi API ở offset {offset}: {e}")
                break # Dừng vòng lặp nếu lỗi mạng nặng
            
    return list(danh_sach_url)

def run_scraper():
    cac_link_truong = quet_danh_sach_truong_api()
    
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}--- TỔNG KẾT ---")
    print(f"{Fore.GREEN}Đã tìm thấy TỔNG CỘNG {len(cac_link_truong)} trường đại học.")
    
    import os
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', 'danh_sach_url_truong.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cac_link_truong, f, indent=4)
        
    print(f"{Fore.CYAN}Đã lưu sạch sẽ vào file: {file_path}")
    return file_path

if __name__ == "__main__":
    run_scraper()
