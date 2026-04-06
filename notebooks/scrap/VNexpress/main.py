import json
import os
import time
import csv
import re
import requests
from bs4 import BeautifulSoup
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)

from scraper_utils import scrape_diem_chuan_api
from scraper import run_scraper
import preprocess_data

def lay_thong_tin_truong(url):
    """Truy cập trang chính của trường để bóc tách Tên và Mã trường"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Lấy Tên trường
        h2_tag = soup.find('h2', class_='university__header-title')
        ten_truong = h2_tag.text.strip() if h2_tag else "N/A"
        
        # 2. Lấy Mã trường (Bóc tách từ chuỗi "Mã trường: BKA")
        ma_truong = "N/A"
        span_code = soup.find('span', class_='university__header-code')
        if span_code:
            # Dùng Regex để tìm các chữ cái/số nằm sau chữ "Mã trường:"
            match = re.search(r'Mã trường:\s*([A-Za-z0-9]+)', span_code.get_text('\n'))
            if match:
                ma_truong = match.group(1)
                
        return ten_truong, ma_truong
        
    except Exception as e:
        print(f"Lỗi khi lấy thông tin trường từ {url}: {e}")
        return "N/A", "N/A"

def chay_thu_thap_va_xuat_csv():
    # Chạy lệnh thu thập link trường trước
    file_json_path = run_scraper()

    print(f"{Fore.YELLOW}{Style.BRIGHT}--- BẮT ĐẦU CÀO VÀ XUẤT DỮ LIỆU CSV ---")
    
    try:
        with open(file_json_path, 'r', encoding='utf-8') as f:
            danh_sach_url = json.load(f)
    except FileNotFoundError:
        print(f"{Fore.RED}Lỗi: Không tìm thấy file {file_json_path}.")
        return

    du_lieu_theo_nam = {}
    tong_so_truong = len(danh_sach_url)

    for vi_tri, url_truong in enumerate(danh_sach_url):
        # BƯỚC 1: Cào thông tin cơ bản của trường
        ten_truong, ma_truong = lay_thong_tin_truong(url_truong)
        
        print(f"\n{Fore.CYAN}[{vi_tri + 1}/{tong_so_truong}] Đang xử lý: {Style.BRIGHT}{ten_truong} {Style.NORMAL}({ma_truong})")
        
        # BƯỚC 2: Gọi API cào điểm của 3 năm (từ file test.py)
        du_lieu_1_truong = scrape_diem_chuan_api(url_truong, danh_sach_nam=["2023", "2024", "2025"])
        
        # BƯỚC 3: Xử lý và tách tổ hợp môn
        for nganh in du_lieu_1_truong:
            nam = nganh.get('nam')
            to_hop_mon_goc = nganh.get('to_hop_mon', '')
            
            # Tách chuỗi "A00, A01" thành list ["A00", "A01"]
            cac_to_hop = [th.strip() for th in to_hop_mon_goc.split(',')] if to_hop_mon_goc else [""]
            
            if nam not in du_lieu_theo_nam:
                du_lieu_theo_nam[nam] = []
                
            for to_hop in cac_to_hop:
                row = {
                    "ma_truong": ma_truong,
                    "ten_truong": ten_truong,
                    "ma_nganh": nganh.get('ma_nganh', ''),
                    "ten_nganh": nganh.get('ten_nganh', ''),
                    "ma_to_hop": to_hop,
                    "nam": nam,
                    "diem_chuan": nganh.get('diem_chuan', ''),
                    "chi_tieu": "", # Dữ liệu này VnExpress thực sự không có, để trống
                    "phuong_thuc": "THPT" 
                }
                du_lieu_theo_nam[nam].append(row)
                
        time.sleep(1) 
        
    # Xuất ra file CSV theo từng năm
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}--- ĐANG XUẤT FILE CSV ---")
    cac_cot = ["ma_truong", "ten_truong", "ma_nganh", "ten_nganh", "ma_to_hop", "nam", "diem_chuan", "chi_tieu", "phuong_thuc"]
    
    # Tạo thư mục data nếu chưa có
    os.makedirs('data', exist_ok=True)
    
    for nam, danh_sach_dong in du_lieu_theo_nam.items():
        ten_file = f"Diem_Chuan_{nam}.csv"
        # Sinh file trong data
        duong_dan_file = os.path.join('data', ten_file)
        
        with open(duong_dan_file, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cac_cot)
            writer.writeheader()
            writer.writerows(danh_sach_dong)
            
        print(f"{Fore.GREEN}[+] Đã lưu thành công: {duong_dan_file} ({len(danh_sach_dong)} dòng)")

    print(f"\n{Fore.YELLOW}{Style.BRIGHT}--- ĐANG XỬ LÝ DỮ LIỆU RAG VÀ ML ---")
    preprocess_data.main()

if __name__ == "__main__":
    chay_thu_thap_va_xuat_csv()