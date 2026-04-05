"""
app.py — FastAPI Web App cho thuthapdata
Chạy: python thuthapdata/webapp/app.py
Truy cập: http://localhost:8001
"""

import os
import sys
import json
import asyncio
import traceback
import io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import shutil
import pandas as pd

# Thêm thư mục cha vào sys.path để import scripts
WEBAPP_DIR     = os.path.dirname(os.path.abspath(__file__))
THUTHAPDATADIR = os.path.dirname(WEBAPP_DIR)
BASE_DIR       = os.path.dirname(THUTHAPDATADIR)
sys.path.insert(0, THUTHAPDATADIR)

from scripts.process         import process, MASTER_FILE
from scripts.create_template import create_template, TEMPLATE_FILE

app = FastAPI(title="Thu Thập Dữ Liệu Điểm Chuẩn")

# Static files
static_dir = os.path.join(WEBAPP_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

INPUT_DIR = os.path.join(THUTHAPDATADIR, "input")
os.makedirs(INPUT_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    html_path = os.path.join(static_dir, "index.html")
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(
        content=content, 
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/download-template")
async def download_template():
    """Tạo (nếu chưa có) và trả về file template."""
    if not os.path.exists(TEMPLATE_FILE):
        try:
            create_template()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return FileResponse(
        TEMPLATE_FILE,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="input_raw_template.xlsx"
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload file input_raw.xlsx vào thư mục input/."""
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file .xlsx")

    save_path = os.path.join(INPUT_DIR, "input_raw.xlsx")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Preview nhanh
    try:
        df = pd.read_excel(save_path)
        rows = len(df)
        cols = df.columns.tolist()
        preview = df.head(5).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file: {e}")

    return JSONResponse({
        "message": f"Upload thành công! ({rows} dòng)",
        "rows": rows,
        "columns": cols,
        "preview": preview,
    })


@app.post("/process")
async def run_process():
    """Chạy script process.py và trả về kết quả (từ file upload)."""
    input_path = os.path.join(INPUT_DIR, "input_raw.xlsx")
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail="Chưa có file input_raw.xlsx. Hãy upload trước!")

    try:
        results, preview_data = process(input_path)
        return JSONResponse({
            "success": True,
            "total":   results["total"],
            "added":   results["added"],
            "skipped": results["skipped"],
            "logs":    results["success"] + results["warnings"] + results["errors"],
            "preview": preview_data,
        })
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{e}\n\n{tb}")


class TextData(BaseModel):
    text: str

@app.post("/process-text")
async def process_text(data: TextData):
    """Xử lý trực tiếp dữ liệu text (TSV) paste từ AI."""
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Không có dữ liệu text")
    
    try:
        # Parse TSV
        df = pd.read_csv(io.StringIO(data.text), sep="\t")
        df.columns = [str(c).strip().lower().replace(" ", "_").replace('"', '') for c in df.columns]
        
        required_cols = {"ma_tuyen_sinh", "ten_nganh", "to_hop", "nam", "diem_chuan"}
        missing = required_cols - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Text thiếu cột: {missing}. Phải có đúng các header chuẩn.")
            
        save_path = os.path.join(INPUT_DIR, "input_raw.xlsx")
        df.to_excel(save_path, index=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc text: {e}")

    try:
        results, preview_data = process(save_path)
        return JSONResponse({
            "success": True,
            "total":   results["total"],
            "added":   results["added"],
            "skipped": results["skipped"],
            "logs":    results["success"] + results["warnings"] + results["errors"],
            "preview": preview_data,
        })
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{e}\n\n{tb}")


@app.get("/preview-master")
async def preview_master():
    """Trả về 100 dòng đầu của file master."""
    if not os.path.exists(MASTER_FILE):
        return JSONResponse({"diem_chuan": [], "truong": []})
    try:
        df_dc = pd.read_excel(MASTER_FILE, sheet_name="diem_chuan")
        df_tr = pd.read_excel(MASTER_FILE, sheet_name="truong")
        return JSONResponse({
            "diem_chuan": df_dc.to_dict(orient="records"),
            "truong":     df_tr.to_dict(orient="records"),
            "total_dc":   len(df_dc),
            "total_tr":   len(df_tr),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-master")
async def download_master():
    """Tải file diem_chuan_master.xlsx."""
    if not os.path.exists(MASTER_FILE):
        raise HTTPException(status_code=404, detail="Chưa có file master. Hãy xử lý dữ liệu trước!")
    return FileResponse(
        MASTER_FILE,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="diem_chuan_master.xlsx"
    )


if __name__ == "__main__":
    print("🚀 Khởi động Web App Thu Thập Dữ Liệu...")
    print("   → Truy cập: http://localhost:8001")
    # Tạo template nếu chưa có
    if not os.path.exists(TEMPLATE_FILE):
        try:
            create_template()
            print("   ✅ Đã tạo file template mẫu")
        except Exception as e:
            print(f"   ⚠️  Không tạo được template: {e}")
    uvicorn.run(app, host="0.0.0.0", port=8001)
