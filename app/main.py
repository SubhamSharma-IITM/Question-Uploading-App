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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.db import init_db
from app.firebase_auth import verify_session, create_session, delete_session as delete_auth_session, verify_firebase_token, get_user_info, SESSION_COOKIE_NAME
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

# ── Auth Middleware ──
# Paths that don't require authentication
_PUBLIC_PATHS = {"/login", "/auth/verify", "/auth/logout", "/health"}
_PUBLIC_PREFIXES = ("/static/",)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow public paths
        if path in _PUBLIC_PATHS or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        # Check session cookie
        session_token = request.cookies.get(SESSION_COOKIE_NAME)
        session_data = verify_session(session_token)

        if not session_data:
            # Fallback: if display cookie exists, restore session (survives --reload)
            display = request.cookies.get("acadza_display", "")
            if display and "|" in display:
                parts = display.split("|", 1)
                session_data = {"name": parts[0], "role": parts[1], "phone": "", "uid": "restored"}
                # Re-create server session so verify works on next request
                new_token = create_session("", "restored")
                from app.firebase_auth import _sessions
                if new_token in _sessions:
                    _sessions[new_token]["name"] = parts[0]
                    _sessions[new_token]["role"] = parts[1]
                # Set user state before call_next so templates can access it
                request.state.user = session_data
                response = await call_next(request)
                response.set_cookie(SESSION_COOKIE_NAME, new_token, max_age=7*24*3600, httponly=True, samesite="lax")
                return response

            # No valid session at all
            if path.startswith("/api/"):
                return JSONResponse(status_code=401, content={"success": False, "error": "Not authenticated"})
            return RedirectResponse(url="/login", status_code=302)

        # Attach user info to request state
        request.state.user = session_data
        return await call_next(request)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth gate (added after CORS so CORS headers are still set on 401s)
app.add_middleware(AuthMiddleware)

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


# ── Auth pages ──
def _firebase_ctx():
    """Firebase config dict for login template."""
    return {
        "firebase_api_key": settings.firebase_api_key,
        "firebase_auth_domain": settings.firebase_auth_domain,
        "firebase_project_id": settings.firebase_project_id,
        "firebase_storage_bucket": settings.firebase_storage_bucket,
        "firebase_messaging_sender_id": settings.firebase_messaging_sender_id,
        "firebase_app_id": settings.firebase_app_id,
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page with Firebase Phone Auth."""
    # If already logged in, redirect to dashboard
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if verify_session(session_token):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, **_firebase_ctx()})


@app.post("/auth/verify")
async def auth_verify(request: Request):
    """Verify Firebase ID token and create session."""
    body = await request.json()
    id_token = body.get("id_token", "")

    if not id_token:
        _safe_print("[AUTH] Missing id_token in request")
        return JSONResponse(content={"success": False, "error": "Missing token"})

    _safe_print(f"[AUTH] Verifying Firebase token (length={len(id_token)})...")
    user_info = await verify_firebase_token(id_token)
    if not user_info:
        _safe_print("[AUTH] Token verification FAILED")
        return JSONResponse(content={"success": False, "error": "Invalid token"})

    user_detail = get_user_info(user_info["phone"])
    _safe_print(f"[AUTH] Verified: {user_detail['name']} ({user_detail['role']}), phone={user_info['phone']}")
    session_token = create_session(user_info["phone"], user_info["uid"])

    response = JSONResponse(content={
        "success": True,
        "redirect": "/",
        "user_name": user_detail["name"],
        "user_role": user_detail["role"],
    })
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=7 * 24 * 3600,
        httponly=True,
    )
    # Display cookie (readable by JS/templates for personalization)
    response.set_cookie(
        key="acadza_display",
        value=f"{user_detail['name']}|{user_detail['role']}",
        max_age=7 * 24 * 3600,
        httponly=False,
        samesite="lax",
    )
    return response


@app.get("/auth/logout")
async def auth_logout(request: Request):
    """Clear session and redirect to login."""
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if session_token:
        delete_auth_session(session_token)
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(SESSION_COOKIE_NAME)
    response.delete_cookie("acadza_display")
    return response


# ── Dashboard page ──
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard -- list sessions, create new."""
    from app.db import list_sessions, get_session_stats, get_checkpoint
    sessions = list_sessions()
    for s in sessions:
        s["stats"] = get_session_stats(s["id"])
        cp = get_checkpoint(s["id"], "stage10_upload")
        if cp and isinstance(cp.get("data"), str):
            import json as _json
            try:
                cp["data"] = _json.loads(cp["data"])
            except Exception:
                cp["data"] = {}
        s["uploaded_count"] = (cp.get("data") or {}).get("posted", 0) if cp else 0
    # Get user info from auth session
    user = getattr(request.state, "user", {})
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "sessions": sessions,
        "user_name": user.get("name", ""),
        "user_role": user.get("role", ""),
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

    # When pipeline is completed, stage10_upload is already in the list
    # with status "completed" -- no need to add a duplicate entry

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

    user = getattr(request.state, "user", {})
    return templates.TemplateResponse(template_name, {
        "request": request,
        "session": session,
        "current_stage": current_stage,
        "view_stage": stage,
        "sidebar_stages": sidebar_stages,
        "checkpoints": checkpoints,
        "stats": stats,
        "user_name": user.get("name", ""),
        "user_role": user.get("role", ""),
    })


# ── Health check ──
@app.get("/health")
async def health():
    return {"status": "ok", "app": "Acadza Question Uploading App", "version": "1.0.0"}
