# Thu Thập Dữ Liệu Điểm Chuẩn

Web app tự động hóa thu thập dữ liệu điểm chuẩn từ ảnh → file Excel Master.

## Cài đặt

```bash
pip install -r thuthapdata/webapp/requirements.txt
```

## Chạy Web App

```bash
python thuthapdata/webapp/app.py
```

Truy cập **http://localhost:8001**

## Quy trình sử dụng

1. **Tải file mẫu** → nhấn "Tải file mẫu" trên web app
2. **Dùng AI tách dữ liệu từ ảnh** → dùng prompt có sẵn trên giao diện, gửi kèm ảnh bảng điểm cho Gemini / ChatGPT → copy kết quả vào file mẫu
3. **Upload & Xử lý** → upload file lên web app → nhấn "Xử lý"
4. **Tải Master Excel** → file `diem_chuan_master.xlsx` đã có đầy đủ 2 sheet:
   - `diem_chuan`: điểm chuẩn chi tiết
   - `truong`: thông tin trường

## Cấu trúc file nhập liệu (`input_raw.xlsx`)

| Cột | Ý nghĩa | Ví dụ |
|---|---|---|
| `ma_tuyen_sinh` | Mã tuyển sinh trường | BKA, UIT, NEU |
| `ten_nganh` | Tên ngành học | Công nghệ Thông tin |
| `to_hop` | Tổ hợp môn | A00, A01, D01 |
| `nam` | Năm xét tuyển | 2025 |
| `diem_chuan` | Điểm chuẩn | 27.50 |
| `chi_tieu` | Chỉ tiêu (có thể để trống) | 100 |

## Danh sách mã tuyển sinh hỗ trợ

| Mã | Trường |
|---|---|
| BKA | ĐH Bách Khoa Hà Nội |
| UET | ĐH Công nghệ - ĐHQGHN |
| NEU | ĐH Kinh tế Quốc dân |
| PTIT | ĐH Bưu chính Viễn thông |
| HUST | ĐH Khoa học Tự nhiên HN |
| UIT | ĐH Công nghệ Thông tin TP.HCM |
| FTU | ĐH Ngoại thương |
| RMIT | ĐH RMIT Việt Nam |
| DUT | ĐH Bách Khoa Đà Nẵng |
| CTU | ĐH Cần Thơ |
| FPT | ĐH FPT |
| HCMUT | ĐH Bách Khoa TP.HCM |
| ... | Xem thêm trong file template (sheet Ma_Tuyen_Sinh) |

> Để thêm trường mới, chỉnh sửa `TRUONG_DB` trong `scripts/process.py`
