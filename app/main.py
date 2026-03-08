import os
import sys
import torch


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes import router

app = FastAPI(title="Hệ Thống Tư Vấn Tuyển Sinh RAG + ML")

app.include_router(router, prefix="/api")

# Phục vụ file tĩnh
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app", "static")), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "app", "static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
