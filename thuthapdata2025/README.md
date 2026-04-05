# Data Mining 2025 - Admission RAG System

Thư mục `thuthapdata2025` chứa toàn bộ công cụ và mã nguồn (scripts) tự động cào và tinh chế dữ liệu (Data Scraping & Mining) từ các nền tảng giáo dục lớn. Các tập dữ liệu sau khi chạy sẽ được dọn dẹp sạch sẽ (Cleaned) để bơm làm Knowledge Base (Cơ sở tri thức) cho hệ thống Admission RAG System.

## Yêu cầu môi trường (Prerequisites)

Dự án yêu cầu cài đặt Python 3.x. Trước khi chạy các script, hãy kích hoạt môi trường ảo (nếu có) và thiết lập thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```
*(Các thư viện chính bao gồm: `requests`, `beautifulsoup4`)*

---

## Danh sách các công cụ (Scripts)

### 1. `crawl_diem_chuan.py` (Cào dữ liệu điểm chuẩn)
Công cụ chủ lực để bóc tách điểm chuẩn Đại Học/Cao Đẳng từ API và hệ thống (Server-rendered Next.js) của trang Tuyensinh247.

- **Đặc điểm:** Tích hợp đa luồng CPU (Multi-threading với 15 workers) giúp vượt rào tốc độ, gánh ~590 trường chỉ trong vài chục giây. Script có khả năng bóc tách sâu các JSON Payload ẩn do Next.js sinh ra.
- **Tính năng Pivot:** Auto-detect tất cả các phương thức xét tuyển (Điểm thi THPT, Đánh giá tư duy, Xét học bạ...) và dàn thành các cột ngang duy nhất trên 1 dòng của 1 trường & ngành. Ngành nào không có phương thức tương ứng tự động set giá trị `0`.
- **Dữ liệu đầu ra:** Cấu trúc ma trận tại file `diem_chuan_2025.csv`.

**Cách chạy:**
```bash
python crawl_diem_chuan.py
```

### 2. `crawl_to_hop_hocmai.py` (Chuẩn hóa dải băng quy chiếu Tổ hợp môn)
Sử dụng API ẩn của Hocmai.vn để tải thư viện từ điển chứa danh sách các tổ hợp môn xét tuyển.

- **Đặc điểm:** Bóc tách RegEx dữ liệu Label (Vd: `A00 (Toán, Vật lí, Hóa học)`) thành 3 môn riêng biệt.
- **Tính năng:** Tự động quy chuẩn tên khối cho đẹp mắt (`A00` -> Khối A, `D01` -> Khối D1) và điền số `0` cho các ô trống / môn bị khuyết đối với tổ hợp năng khiếu lạ.
- **Dữ liệu đầu ra:** Trả file tại `to_hop_mon.csv` với format 5 cột: `ma_to_hop, ten_to_hop, mon_1, mon_2, mon_3`.

**Cách chạy:**
```bash
python crawl_to_hop_hocmai.py
```

### 3. `extract_to_hop_mon.py` (Tool nội bộ)
Tool đọc lược file `diem_chuan_2025.csv` (phát sinh từ Script số 1), lấy cột Tổ hợp môn hiện tại để cắt gọn, nhổ rác văn bản (\r, \n, "Hát", "Năng khiếu"...) và thống kê bộ codes duy nhất (A00, C19...) đẩy về file tổng hợp ở source base.

---

## Kiến trúc thư mục quy chiếu (Quy trình tiêu chuẩn)
Để Refresh (cập nhật mới) toàn bộ dữ liệu hệ thống, hãy chạy theo thứ tự sau:

1. Chạy `crawl_to_hop_hocmai.py` để làm Data Dictionary các khối môn và 3 môn thi.
2. Chạy `crawl_diem_chuan.py` để lật lấy file hàng chục ngàn dòng điểm chuẩn Đại học năm mới nhất.
3. Chạy `extract_to_hop_mon.py` để cross-check (nếu thấy cần thiết cho nhu cầu nội bộ khác).

## Các file Note / References
- `yeucau.md`: Tài liệu đặc tả ban đầu quá trình Inspection Frontend để thiết kế luồng cào dữ liệu này.

---

## Cấu trúc dữ liệu (CSV Schemas)

Để dữ liệu có thể dễ dàng được nhúng (Embedding) vào Vector Database của quá trình làm RAG, các file CSV được thiết kế với cấu trúc cực kỳ tối ưu:

### 1. Bảng Trọng Tâm `diem_chuan_2025.csv`
Chứa thông tin điểm chuẩn của tất cả các trường/ngành thu thập được. Mỗi ngành chiếm **duy nhất 1 dòng** (thể hiện thiết kế data pivot trải dài).
- **Mã trường**: Mã quy chuẩn của trường (VD: BKA, QSB, KHA,...)
- **Tên trường**: Tên đầy đủ được làm sạch (VD: Đại Học Bách Khoa Hà Nội)
- **Mã ngành**: Mã số ngành học do từng trường cung cấp (VD: 7480201)
- **Tên ngành**: Tên của khối ngành hoặc phân ngành cụ thể (VD: Khoa học máy tính)
- **Tổ hợp môn**: Danh sách các khối được phép đăng ký xét tuyển, phân cách bởi `|` (VD: A00 | A01 | D01)
- **Các cột phương thức ngang ([Điểm thi THPT, Xét học bạ, ĐGNL...])**: Điểm chuẩn trúng tuyển tương ứng với đúng phương thức xét tuyển trải trên ngang. Nếu ngành đó không có phương thức này, tự động điền cứng giá trị **`0`**.
- **Ghi chú**: Các quy định quy đổi phụ thêm (nhân hệ số, chứng chỉ IELTS) do trường ghi nhận (nếu có).

### 2. Từ Điển Tham Chiếu `to_hop_mon.csv`
Bảng bộ đệm Dictionary (Metadata) phục vụ cho RAG System định nghĩa các "Mã Khối" thành "Thông tin ngôn ngữ tự nhiên".
- **ma_to_hop**: Tổ hợp mã ký hiệu cứng (VD: A00, B00, AH5, D145,...)
- **ten_to_hop**: Tên khối được phiên dịch dạng text thân thiện để Chatbot đọc mượt (VD: Khối A, Khối D1, Khối D145)
- **mon_1**: Môn thi bắt buộc số 1 (VD: Toán). Giá trị `0` nếu phương thức không công bố chi tiết.
- **mon_2**: Môn thi bắt buộc số 2 (VD: Vật lí). Giá trị `0` nếu bị khuyết.
- **mon_3**: Môn thi bắt buộc số 3 (VD: Hóa học). Giá trị `0` nếu bị khuyết.
