"""FastAPI application — Acadza Question Uploading App.

Forked from acadza-pdf-generator/app/main.py patterns.
Start with: python -m uvicorn app.main:app --reload --port 8000
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.db import init_db
from app.utils import _safe_print

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# ── Jinja2 ──
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ── Concurrency ──
_task_semaphore: Optional[asyncio.Semaphore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, create dirs. Shutdown: cleanup."""
    global _task_semaphore

    _safe_print("=" * 60)
    _safe_print("  ACADZA Question Uploading App")
    _safe_print("=" * 60)

    # Init database
    init_db()
    _safe_print("[OK] Database initialized")

    # Ensure session directories exist
    Path(settings.sessions_dir).mkdir(parents=True, exist_ok=True)
    _safe_print(f"[OK] Sessions dir: {settings.sessions_dir}")

    # Concurrency limiter
    _task_semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)
    _safe_print(f"[OK] Max concurrent tasks: {settings.max_concurrent_tasks}")

    _safe_print("-" * 60)
    _safe_print(f"  Ready at http://localhost:{settings.port}")
    _safe_print("-" * 60)

    yield

    _safe_print("[SHUTDOWN] Cleaning up...")


# ── App ──
app = FastAPI(
    title="Acadza Question Uploading App",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Import & include routers ──
from app.routes.sessions import router as sessions_router
from app.routes.stage1_mathpix import router as stage1_router
from app.routes.stage2_mapper import router as stage2_router
from app.routes.stage3_cleaner import router as stage3_router
from app.routes.stage4_metadata import router as stage4_router
from app.routes.stage5_watermark import router as stage5_router
from app.routes.stage6_images import router as stage6_router
from app.routes.stage7_latex_tag import router as stage7_router
from app.routes.stage8_duplicates import router as stage8_router
from app.routes.stage9_quality import router as stage9_router
from app.routes.stage10_upload import router as stage10_router

app.include_router(sessions_router)
app.include_router(stage1_router, prefix="/api/sessions/{session_id}/stage1", tags=["Stage 1"])
app.include_router(stage2_router, prefix="/api/sessions/{session_id}/stage2", tags=["Stage 2"])
app.include_router(stage3_router, prefix="/api/sessions/{session_id}/stage3", tags=["Stage 3"])
app.include_router(stage4_router, prefix="/api/sessions/{session_id}/stage4", tags=["Stage 4"])
app.include_router(stage5_router, prefix="/api/sessions/{session_id}/stage5", tags=["Stage 5"])
app.include_router(stage6_router, prefix="/api/sessions/{session_id}/stage6", tags=["Stage 6"])
app.include_router(stage7_router, prefix="/api/sessions/{session_id}/stage7", tags=["Stage 7"])
app.include_router(stage8_router, prefix="/api/sessions/{session_id}/stage8", tags=["Stage 8"])
app.include_router(stage9_router, prefix="/api/sessions/{session_id}/stage9", tags=["Stage 9"])
app.include_router(stage10_router, prefix="/api/sessions/{session_id}/stage10", tags=["Stage 10"])


# ── Dashboard page ──
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard — list sessions, create new."""
    from app.db import list_sessions
    sessions = list_sessions()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "sessions": sessions,
    })


# ── Session detail page (renders the correct stage template) ──
@app.get("/session/{session_id}", response_class=HTMLResponse)
async def session_page(request: Request, session_id: str, stage: Optional[str] = None):
    """Session detail page — shows the current or requested stage."""
    from app.db import get_session, get_all_checkpoints, get_session_stats
    from app.state_machine import STAGES, STAGE_LABELS, STAGE_SHORT, get_stage_status, is_stage_accessible

    session = get_session(session_id)
    if session is None:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "sessions": [],
            "error": f"Session {session_id} not found",
        })

    current_stage = session["current_stage"]

    # Treat legacy "created" stage as "stage1_upload" for sidebar + template
    if current_stage == "created":
        current_stage = "stage1_upload"

    # Default to current stage if none requested
    if stage is None:
        stage = current_stage

    # Build sidebar data
    sidebar_stages = []
    for s in STAGES:
        if s in ("created", "completed"):
            continue
        sidebar_stages.append({
            "key": s,
            "label": STAGE_LABELS.get(s, s),
            "short": STAGE_SHORT.get(s, s),
            "status": get_stage_status(current_stage, s),
            "accessible": is_stage_accessible(current_stage, s),
        })

    # Add "Complete" item when pipeline is done
    if current_stage == "completed":
        sidebar_stages.append({
            "key": "completed",
            "label": "10. Complete",
            "short": "Done",
            "status": "completed",
            "accessible": True,
        })

    checkpoints = get_all_checkpoints(session_id)
    stats = get_session_stats(session_id)

    # Determine which template to render
    template_map = {
        "created": "stage1.html",
        "stage1_upload": "stage1.html",
        "stage2_mapping": "stage2.html",
        "stage3_cleaning": "stage3.html",
        "stage4_metadata": "stage4.html",
        "stage5_watermark": "stage5.html",
        "stage6_images": "stage6.html",
        "stage7_batch": "stage7.html",
        "stage8_duplicates": "stage8.html",
        "stage9_quality": "stage9.html",
        "stage10_upload": "completed.html",
        "completed": "completed.html",
    }
    template_name = template_map.get(stage, "stage1.html")

    return templates.TemplateResponse(template_name, {
        "request": request,
        "session": session,
        "current_stage": current_stage,
        "view_stage": stage,
        "sidebar_stages": sidebar_stages,
        "checkpoints": checkpoints,
        "stats": stats,
    })


# ── Health check ──
@app.get("/health")
async def health():
    return {"status": "ok", "app": "Acadza Question Uploading App", "version": "1.0.0"}
