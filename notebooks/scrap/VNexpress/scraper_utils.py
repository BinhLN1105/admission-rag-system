import requests
from bs4 import BeautifulSoup
import re
import time

def quet_danh_sach_truong():
    """Lấy toàn bộ link các trường từ API"""
    api_url = "https://diemthi.vnexpress.net/tra-cuu-dai-hoc/loadcollege"
    headers = {"X-Requested-With": "XMLHttpRequest"}
    danh_sach_url = set()
    offset = 0
    while True:
        params = {"location_id": "-1", "offset": offset, "limit": 20, "college_type": "2"}
        data = requests.get(api_url, headers=headers, params=params).json()
        html = data.get('html', '').strip()
        if not html: break
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=re.compile(r'/tra-cuu-dai-hoc/[a-z0-9\-]+-\d+$')):
            danh_sach_url.add("https://diemthi.vnexpress.net" + a['href'])
        offset += 20
        time.sleep(0.5)
    return list(danh_sach_url)

def lay_thong_tin_truong(url):
    """Lấy tên và mã trường"""
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        ten = soup.find('h2', class_='university__header-title').text.strip()
        span = soup.find('span', class_='university__header-code')
        ma = re.search(r'Mã trường:\s*([A-Za-z0-9]+)', span.get_text('\n')).group(1) if span else "N/A"
        return ten, ma
    except: return "N/A", "N/A"

def cào_diem_api(url_truong, truong_id, nam):
    """Cào điểm theo năm"""
    api_url = f"https://diemthi.vnexpress.net/tra-cuu-dai-hoc/loadbenchmark/id/{truong_id}/year/{nam}/sortby/1/block_name/all"
    try:
        data = requests.get(api_url, headers={"X-Requested-With": "XMLHttpRequest"}).json()
        soup = BeautifulSoup(data.get('html', ''), 'html.parser')
        rows = soup.find_all('tr', class_=lambda c: c and 'university__benchmark' in c)
        results = []
        for r in rows:
            tds = r.find_all('td')
            if len(tds) >= 5:
                results.append({
                    "ten_nganh": tds[1].find('a').text.strip(),
                    "ma_nganh": tds[1].find_all('span')[-1].text.strip(),
                    "diem": tds[2].find('span').text.strip(),
                    "to_hop": tds[3].text.strip()
                })
        return results
    except: return []

def scrape_diem_chuan_api(url_truong, danh_sach_nam):
    """Cào điểm của một trường qua nhiều năm"""
    match = re.search(r'-(\d+)$', url_truong)
    if not match:
        return []
    truong_id = match.group(1)
    
    results = []
    for nam in danh_sach_nam:
        du_lieu_nam = cào_diem_api(url_truong, truong_id, nam)
        for nganh in du_lieu_nam:
            results.append({
                "nam": nam,
                "ma_nganh": nganh.get("ma_nganh"),
                "ten_nganh": nganh.get("ten_nganh"),
                "to_hop_mon": nganh.get("to_hop"),
                "diem_chuan": nganh.get("diem")
            })
    return results