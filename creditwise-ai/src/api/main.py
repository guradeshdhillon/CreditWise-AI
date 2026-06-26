"""
main.py
FastAPI application factory for CreditWise AI.

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.api.routes import router, load_artifacts

_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = _ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts before accepting any requests."""
    load_artifacts()
    yield
    # Graceful shutdown (no cleanup needed)


app = FastAPI(
    title="CreditWise AI",
    description=(
        "## 🏦 CreditWise AI — Loan Approval Intelligence\n\n"
        "Production-grade loan approval prediction API with:\n"
        "- **XGBoost** model tuned with Optuna (ROC-AUC optimised)\n"
        "- **SHAP** feature importance for every prediction\n"
        "- **Claude AI** plain-English decision explanations\n\n"
        "Built by Guradesh Dhillon — Computer Engineering, AI & Data Science specialization."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Tighten to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


# Serve frontend HTML pages
@app.get("/", tags=["Frontend"])
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/what-if", tags=["Frontend"])
async def serve_what_if():
    return FileResponse(FRONTEND_DIR / "what-if.html")


@app.get("/fairness", tags=["Frontend"])
async def serve_fairness():
    return FileResponse(FRONTEND_DIR / "fairness.html")


@app.get("/favicon.svg", tags=["Frontend"])
async def serve_favicon_svg():
    return FileResponse(FRONTEND_DIR / "favicon.svg")


@app.get("/favicon.ico", tags=["Frontend"])
async def serve_favicon_ico():
    return FileResponse(FRONTEND_DIR / "favicon.svg")


# Mount the frontend directory for static assets (images, CSS, JS etc.)
# If the directory doesn't exist yet at startup, we create it dynamically.
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

