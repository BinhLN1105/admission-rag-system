from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routes import router, get_pipeline

# ==================== CONFIG ====================
BASE_DIR = Path(__file__).resolve().parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý khởi động và tắt server"""
    print("=" * 80)
    print("🚀 HỆ THỐNG TƯ VẤN TUYỂN SINH AI 2026 ĐANG KHỞI ĐỘNG...")
    print(f"📍 BASE_DIR: {BASE_DIR}")
    
    # Preload model AI
    try:
        get_pipeline()
        print("✅ AI Pipeline đã preload thành công!")
    except Exception as e:
        print(f"⚠️ Không thể preload pipeline: {e}")
    
    yield
    print("🛑 Server đã tắt.")
    print("=" * 80)


# ==================== KHỞI TẠO FASTAPI ====================
app = FastAPI(
    title="Hệ Thống Tư Vấn Tuyển Sinh AI 2026",
    description="AI tư vấn chọn ngành - chọn trường Đại học năm 2026",
    version="1.0.0",
    lifespan=lifespan,
    debug=True
)

# ==================== THƯ MỤC ====================
templates_dir = BASE_DIR / "app" / "templates"
static_dir = BASE_DIR / "app" / "static"

print("🔍 KIỂM TRA CẤU TRÚC DỰ ÁN...")
print(f"✅ Templates folder: {templates_dir.exists()} → {templates_dir}")
print(f"✅ Static folder   : {static_dir.exists()} → {static_dir}")

if templates_dir.exists():
    print("\n📄 Các file template tìm thấy:")
    for f in sorted(templates_dir.glob("**/*.html")):
        print(f"   ✓ {f.relative_to(BASE_DIR)}")

# ==================== STATIC FILES ====================
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    print("✅ Static files đã mount thành công")
else:
    print("⚠️ Cảnh báo: Không tìm thấy thư mục static!")

# ==================== TEMPLATES ====================
templates = Jinja2Templates(directory=str(templates_dir))

# ==================== MIDDLEWARE ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ROUTER (API) ====================
app.include_router(router, prefix="/api")

# ==================== HTML ROUTES ====================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("pages/index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("pages/about.html", {"request": request})

@app.get("/features", response_class=HTMLResponse)
async def features_page(request: Request):
    return templates.TemplateResponse("pages/features.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    return templates.TemplateResponse("pages/contact.html", {"request": request})

@app.get("/info", response_class=HTMLResponse)
async def info_page(request: Request):
    return templates.TemplateResponse("pages/info.html", {"request": request})

# ==================== CÁC ROUTE SAU NÀY CÓ THỂ THÊM ====================
# @app.get("/search", response_class=HTMLResponse)
# async def search_page(request: Request):
#     return templates.TemplateResponse("pages/search.html", {"request": request})


# ==================== CHẠY SERVER ====================
if __name__ == "__main__":
    import uvicorn
    print("\n🌐 Server đang chạy tại: http://127.0.0.1:8000")
    print("   Nhấn Ctrl + C để dừng server\n")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"]   # Chỉ reload thư mục app cho nhẹ
    )