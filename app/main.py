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
from app.firebase_auth import (
    verify_session, create_session, delete_session as delete_auth_session,
    verify_pin, get_user_info, check_rate_limit, record_failed_attempt,
    clear_rate_limit, SESSION_COOKIE_NAME,
)
from app.utils import _safe_print

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# ── Jinja2 ──
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ── Concurrency ──
_task_semaphore: Optional[asyncio.Semaphore] = None


def _reattribute_ghost_sessions(sessions_dir: str) -> None:
    """Fix sessions with empty created_by by reading session_config.json.

    Display cookie fallback used to create sessions with phone="" after
    server restarts. This scans ALL sessions and re-attributes from disk
    config where possible, or logs unattributed sessions for manual fix.
    """
    from app.db import list_sessions, _get_conn, _lock
    import json as _json

    sdir = Path(sessions_dir)
    fixed = 0
    unattributed = 0

    sessions = list_sessions()
    for s in sessions:
        sid = s.get("id", s.get("session_id", ""))
        if not sid or s.get("created_by", ""):
            continue  # Already attributed

        # Try session_config.json for phone
        phone = ""
        cfg_path = sdir / sid / "session_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = _json.load(f)
                phone = cfg.get("created_by", "")
            except Exception:
                pass

        if phone:
            with _lock:
                conn = _get_conn()
                conn.execute("UPDATE sessions SET created_by = ? WHERE id = ?",
                             (phone, sid))
                conn.commit()
            fixed += 1
            _safe_print(f"[REATTR] Fixed ghost session {sid[:8]} -> {phone[-4:]}")
        else:
            unattributed += 1
            _safe_print(f"[REATTR] Unattributed ghost: {sid[:8]} ({s.get('name', '?')}) stage={s.get('current_stage', '?')}")

    if fixed:
        _safe_print(f"[REATTR] Re-attributed {fixed} ghost session(s)")
    if unattributed:
        _safe_print(f"[REATTR] {unattributed} session(s) still unattributed (need manual fix)")


def _recover_orphan_sessions(sessions_dir: str) -> None:
    """Scan session directories and re-register any that are missing from SQLite.

    OOM crashes can destroy SQLite WAL records while session files survive on
    persistent disk. This recovers those orphaned sessions on startup.
    """
    from app.db import get_session, create_session
    import json as _json

    sdir = Path(sessions_dir)
    if not sdir.exists():
        return

    recovered = 0
    for d in sorted(sdir.iterdir()):
        if not d.is_dir():
            continue
        session_id = d.name
        if get_session(session_id) is not None:
            continue  # already in DB

        # Orphan found -- reconstruct from disk files
        config = None
        name = session_id[:8]
        created_by = ""
        config_path = d / "session_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = _json.load(f)
                name = config.get("pdf_filename", name)
                if name.endswith(".pdf"):
                    name = name[:-4]
                created_by = config.get("created_by", "")
            except Exception:
                pass

        # Detect current stage from checkpoint files / directory structure
        stage = "stage1_upload"
        upload_logs_dir = d / "upload_logs"
        if (upload_logs_dir / "combined_upload_log.csv").exists():
            stage = "completed"
        elif (d / "quality_check" / "vote_summary.json").exists():
            stage = "stage10_upload"
        elif (d / "quality_check" / "qa_config.json").exists():
            stage = "stage9_quality"
        elif (d / "duplicate_reports" / "cross_duplicates.json").exists():
            stage = "stage8_duplicates"
        elif any((d / "batch_files").glob("*.jsonl")) if (d / "batch_files").exists() else False:
            stage = "stage7_latex_tag"
        elif (d / "metadata_v2").exists() and any((d / "metadata_v2").glob("*.json")):
            stage = "stage5_watermark"
        elif any((d / "cleaned_output").glob("*.json")) if (d / "cleaned_output").exists() else False:
            stage = "stage3_cleaning"
        elif any((d / "mapped_output").glob("*_mapped.json")) if (d / "mapped_output").exists() else False:
            stage = "stage2_mapping"

        # Use directory mtime as creation time
        import datetime
        created_at = datetime.datetime.fromtimestamp(
            d.stat().st_mtime, tz=datetime.timezone.utc
        ).isoformat()

        try:
            create_session(session_id, name, created_at, config=config,
                           initial_stage=stage, created_by=created_by)
            recovered += 1
            _safe_print(f"[RECOVERY] Restored orphan session: {session_id} ({name}) at {stage}")
        except Exception as e:
            _safe_print(f"[RECOVERY] Failed to restore {session_id}: {e}")

    if recovered:
        _safe_print(f"[RECOVERY] Restored {recovered} orphan session(s)")


def _backfill_upload_log(sessions_dir: str) -> None:
    """Backfill upload_log for completed sessions that have no stats rows.

    Reads combined_upload_log.csv from each completed session's upload_logs/
    directory. Only runs for sessions with zero upload_log entries in SQLite.
    """
    import csv
    from app.db import list_sessions, log_upload, clear_upload_log

    sdir = Path(sessions_dir)
    sessions = list_sessions()
    backfilled = 0

    for s in sessions:
        if s["current_stage"] != "completed":
            continue

        sid = s["id"] if "id" in s else s.get("session_id", "")
        if not sid:
            continue

        # Check if upload_log already has entries for this session
        from app.db import _get_conn, _lock
        with _lock:
            conn = _get_conn()
            count = conn.execute(
                "SELECT COUNT(*) FROM upload_log WHERE session_id = ?", (sid,)
            ).fetchone()[0]

        if count > 0:
            continue  # Already has stats

        # Find combined_upload_log.csv on disk
        csv_path = sdir / sid / "upload_logs" / "combined_upload_log.csv"
        if not csv_path.exists():
            continue

        # Get user phone from session
        user_phone = s.get("created_by", "")
        if not user_phone:
            # Try session_config.json
            cfg_path = sdir / sid / "session_config.json"
            if cfg_path.exists():
                try:
                    import json as _json
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = _json.load(f)
                    user_phone = cfg.get("created_by", "")
                except Exception:
                    pass
            # Also fix the DB record so future lookups work
            if user_phone:
                with _lock:
                    conn = _get_conn()
                    conn.execute("UPDATE sessions SET created_by = ? WHERE id = ?",
                                 (user_phone, sid))
                    conn.commit()
                _safe_print(f"[BACKFILL] Fixed created_by for {sid[:8]} -> {user_phone[-4:]}")
        if not user_phone:
            continue  # Cannot attribute

        # Parse CSV and aggregate per subject + error level
        try:
            subject_counts = {}  # {subject: {1: count, 2: count}}
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = row.get("UploadedQuestionID", "")
                    if status in ("QUARANTINED", "FAILED", "DRY_RUN") or status.startswith("SKIPPED"):
                        continue
                    if not status or len(status) < 10:
                        continue  # Not a valid OID
                    el = int(row.get("Error Level", 2))
                    subj = row.get("Subject", "Unknown")
                    if subj not in subject_counts:
                        subject_counts[subj] = {1: 0, 2: 0}
                    subject_counts[subj][el] += 1

            if not subject_counts:
                continue

            # Clear any partial rows and re-insert
            clear_upload_log(sid)
            total = 0
            for subj, counts in subject_counts.items():
                if counts[1] > 0:
                    log_upload(sid, user_phone, 1, subj, counts[1])
                    total += counts[1]
                if counts[2] > 0:
                    log_upload(sid, user_phone, 2, subj, counts[2])
                    total += counts[2]

            backfilled += 1
            _safe_print(f"[BACKFILL] {sid[:8]} ({s.get('name', '?')}): {total} entries logged for {user_phone[-4:]}")

        except Exception as e:
            _safe_print(f"[BACKFILL] Failed for {sid[:8]}: {e}")

    if backfilled:
        _safe_print(f"[BACKFILL] Filled upload_log for {backfilled} session(s)")


def _cleanup_completed_sessions(sessions_dir: str) -> None:
    """Free disk by removing heavy directories from completed sessions.

    Keeps upload_logs/ and session_config.json (needed for stats backfill
    and orphan recovery). Deletes images, metadata, batches, etc.
    Only touches sessions that have a combined_upload_log.csv (confirmed complete).
    """
    import shutil
    from app.db import list_sessions

    sdir = Path(sessions_dir)
    if not sdir.exists():
        return

    # Identify active sessions (not completed) — never touch these
    sessions = list_sessions()
    active_ids = {s["id"] for s in sessions if s["current_stage"] != "completed"}

    HEAVY_DIRS = [
        "images_original", "images_cleaned", "metadata_v2", "mapped_output",
        "cleaned_output", "batch_jsonl", "batch_results", "batch_files",
        "duplicate_reports", "duplicates", "quality_check", "quality",
        "quarantine", "backups", "input", "mathpix_output",
        "session_embeddings", "smiles_output",
    ]

    cleaned = 0
    freed_mb = 0
    for d in sorted(sdir.iterdir()):
        if not d.is_dir():
            continue
        sid = d.name
        if sid in active_ids:
            continue

        # Only clean sessions with a completed CSV (confirmed uploaded)
        csv_path = d / "upload_logs" / "combined_upload_log.csv"
        if not csv_path.exists():
            continue

        session_freed = 0
        for hd in HEAVY_DIRS:
            target = d / hd
            if target.is_dir():
                try:
                    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                    shutil.rmtree(target)
                    session_freed += size
                except Exception:
                    pass

        if session_freed > 0:
            cleaned += 1
            freed_mb += session_freed / (1024 * 1024)

    if cleaned:
        _safe_print(f"[CLEANUP] Freed {freed_mb:.0f}MB from {cleaned} completed session(s)")


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

    # Seed users table from registry (ensures all users exist before first login)
    from app.db import seed_users_from_registry
    from app.firebase_auth import USER_REGISTRY
    seed_users_from_registry(USER_REGISTRY)
    _safe_print(f"[OK] Seeded {len(USER_REGISTRY)} users")

    # Ensure session directories exist
    Path(settings.sessions_dir).mkdir(parents=True, exist_ok=True)
    _safe_print(f"[OK] Sessions dir: {settings.sessions_dir}")

    # Recover orphan sessions (directories on disk but not in SQLite)
    # Prevents data loss from OOM crashes that kill SQLite WAL
    _recover_orphan_sessions(settings.sessions_dir)

    # Fix ghost sessions created with empty created_by (display cookie fallback bug)
    _reattribute_ghost_sessions(settings.sessions_dir)

    # Backfill upload_log for completed sessions missing stats
    _backfill_upload_log(settings.sessions_dir)

    # Selective cleanup: free disk for completed sessions (keep CSVs + config)
    _cleanup_completed_sessions(settings.sessions_dir)

    # Video cleanup: delete expired videos (>7 days)
    from app.routes.video import cleanup_expired_videos
    cleaned = cleanup_expired_videos()
    if cleaned:
        _safe_print(f"[CLEANUP] Deleted {cleaned} expired video(s)")

    # Concurrency limiter -- prevents OOM from concurrent heavy operations
    from app.utils import init_semaphore
    init_semaphore(settings.max_concurrent_tasks)
    _safe_print(f"[OK] Max concurrent tasks: {settings.max_concurrent_tasks}")

    # Capture event loop for thread-safe SSE emit
    from app.sse import sse_manager
    sse_manager.set_loop(asyncio.get_running_loop())
    _safe_print("[OK] SSE event loop registered")

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
_PUBLIC_PREFIXES = ("/static/", "/video/api/status/", "/video/api/stream/", "/video/api/proxy/", "/video/api/check/", "/video/api/log", "/video/player")


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
                # Resolve phone from USER_REGISTRY by matching name
                from app.firebase_auth import USER_REGISTRY
                resolved_phone = ""
                for ph, info in USER_REGISTRY.items():
                    if info.get("name") == parts[0]:
                        resolved_phone = ph
                        break
                session_data = {"name": parts[0], "role": parts[1], "phone": resolved_phone, "uid": "restored"}
                # Track login for restored sessions
                if resolved_phone:
                    from app.db import upsert_user, update_user_login
                    upsert_user(resolved_phone, parts[0], parts[1])
                    update_user_login(resolved_phone)
                # Re-create server session so verify works on next request
                new_token = create_session(resolved_phone, parts[0], parts[1])
                from app.firebase_auth import _sessions
                if new_token in _sessions:
                    _sessions[new_token]["name"] = parts[0]
                    _sessions[new_token]["role"] = parts[1]
                    _sessions[new_token]["phone"] = resolved_phone
                # Set user state before call_next so templates can access it
                request.state.user = session_data
                response = await call_next(request)
                response.set_cookie(SESSION_COOKIE_NAME, new_token, max_age=7*24*3600, httponly=True, samesite="lax")
                return response

            # No valid session at all
            if path.startswith("/api/"):
                return JSONResponse(status_code=401, content={"success": False, "error": "Not authenticated"})
            # Redirect to login with ?next= so we can bounce back after auth
            from urllib.parse import quote
            redirect_url = f"/login?next={quote(str(request.url.path), safe='')}"
            return RedirectResponse(url=redirect_url, status_code=302)

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

# ── Heartbeat (client pings every 5 min for activity tracking) ──
@app.post("/api/heartbeat")
async def heartbeat(request: Request):
    """Record a heartbeat from the client."""
    user = getattr(request.state, "user", {})
    phone = user.get("phone", "")
    if not phone:
        return JSONResponse(status_code=401, content={"success": False})
    body = await request.json()
    from app.db import update_heartbeat, update_user_last_active
    update_heartbeat(phone, body.get("session_id"), body.get("stage"))
    update_user_last_active(phone)
    return {"success": True}


# ── Dashboard stats API ──
@app.get("/api/dashboard/stats")
async def dashboard_stats(request: Request, days: int = 7):
    """Get dashboard stats for the logged-in user.

    Architecture: personal stats = upload_log WHERE user_phone.
    Team stats = server-side aggregation of each user's upload_log.
    No global data storage. Global = derived from personal.
    """
    user = getattr(request.state, "user", {})
    phone = user.get("phone", "")
    role = user.get("role", "Team")

    from app.db import (
        get_upload_stats, get_heartbeats, list_users,
        list_sessions_by_user,
    )

    # Personal stats from upload_log -- filtered by user_phone only
    my_today = get_upload_stats(user_phone=phone, days=1)
    my_yesterday = get_upload_stats(user_phone=phone, days=2)
    my_yesterday_only = {
        "err1": max(my_yesterday["err1"] - my_today["err1"], 0),
        "err2": max(my_yesterday["err2"] - my_today["err2"], 0),
        "by_subject": {},
    }
    # Compute per-subject yesterday-only
    for subj, vals in my_yesterday.get("by_subject", {}).items():
        today_subj = my_today.get("by_subject", {}).get(subj, {"err1": 0, "err2": 0})
        my_yesterday_only["by_subject"][subj] = {
            "err1": max(vals["err1"] - today_subj["err1"], 0),
            "err2": max(vals["err2"] - today_subj["err2"], 0),
        }
    my_range = get_upload_stats(user_phone=phone, days=days)
    my_all_time = get_upload_stats(user_phone=phone, days=9999)

    # Session counts -- PERSONAL only
    my_sessions_all = list_sessions_by_user(phone) if phone else []
    my_active = [s for s in my_sessions_all if s["current_stage"] != "completed"]
    my_completed = [s for s in my_sessions_all if s["current_stage"] == "completed"]

    # In-process question count (from metadata files of active sessions)
    from app.pipeline.metadata_stats import compute_live_stats
    my_in_process = 0
    for s in my_active:
        session_dir = Path(settings.sessions_dir) / s["id"]
        if session_dir.exists():
            st = compute_live_stats(session_dir)
            my_in_process += st.get("total", 0)

    # Last session for this user
    last_session = None
    if my_sessions_all:
        ls = my_sessions_all[0]
        last_session = {"name": ls["name"], "id": ls["id"], "stage": ls["current_stage"]}

    result = {
        "today": my_today,
        "yesterday": my_yesterday_only,
        "range": my_range,
        "range_days": days,
        "role": role,
        "active_sessions": len(my_active),
        "completed_sessions": len(my_completed),
        "in_process": my_in_process,
        "last_session": last_session,
        "all_time": {
            "err1": my_all_time["err1"],
            "err2": my_all_time["err2"],
            "by_subject": my_all_time.get("by_subject", {}),
            "sessions_uploaded": len(my_completed),
        },
    }

    # Admin: team stats -- per-user aggregation from upload_log, NO global data fetch
    if role in ("Admin", "Architect"):
        all_users = list_users()
        team_merged = []
        for u in all_users:
            p = u["phone"]
            u_today = get_upload_stats(user_phone=p, days=1)
            u_yesterday_full = get_upload_stats(user_phone=p, days=2)
            u_yesterday_only = {
                "err1": max(u_yesterday_full["err1"] - u_today["err1"], 0),
                "err2": max(u_yesterday_full["err2"] - u_today["err2"], 0),
            }
            u_range = get_upload_stats(user_phone=p, days=days)
            u_all_time = get_upload_stats(user_phone=p, days=9999)

            # Session counts per user -- each user's own sessions only
            u_sessions = list_sessions_by_user(p)
            u_active = [s for s in u_sessions if s["current_stage"] != "completed"]
            u_completed = [s for s in u_sessions if s["current_stage"] == "completed"]

            team_merged.append({
                "phone": p,
                "name": u.get("name", ""),
                "role": u.get("role", "Team"),
                "last_login": u.get("last_login") or "",
                "today_err1": u_today["err1"],
                "today_err2": u_today["err2"],
                "today_by_subject": u_today.get("by_subject", {}),
                "yesterday_err1": u_yesterday_only["err1"],
                "yesterday_err2": u_yesterday_only["err2"],
                "range_err1": u_range["err1"],
                "range_err2": u_range["err2"],
                "range_by_subject": u_range.get("by_subject", {}),
                "all_time_err1": u_all_time["err1"],
                "all_time_err2": u_all_time["err2"],
                "all_time_by_subject": u_all_time.get("by_subject", {}),
                "active_sessions": len(u_active),
                "completed_sessions": len(u_completed),
                "total_sessions": len(u_sessions),
            })
        result["team"] = team_merged
        result["heartbeats"] = get_heartbeats()

    return {"success": True, "data": result}


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
from app.routes.video import router as video_router

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
app.include_router(video_router, prefix="/video", tags=["Video"])


# ── Auth pages ──
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/"):
    """Login page with phone + PIN."""
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if verify_session(session_token):
        return RedirectResponse(url=next or "/", status_code=302)
    return templates.TemplateResponse(request, "login.html", context={
        "next_url": next or "/",
    })


@app.post("/auth/verify")
async def auth_verify(request: Request):
    """Verify phone + PIN and create session."""
    body = await request.json()
    phone = body.get("phone", "").strip()
    pin = body.get("pin", "").strip()

    if not phone or not pin:
        return JSONResponse(content={"success": False, "error": "Phone and PIN required"})

    # Normalize phone
    full_phone = "+91" + phone if not phone.startswith("+") else phone

    # Check rate limit
    locked_seconds = check_rate_limit(full_phone)
    if locked_seconds:
        mins = (locked_seconds // 60) + 1
        return JSONResponse(content={
            "success": False,
            "error": f"Too many attempts. Locked for {mins} min.",
            "locked": True,
            "locked_seconds": locked_seconds,
        })

    # Verify PIN
    user_info = verify_pin(phone, pin)
    if not user_info:
        remaining = record_failed_attempt(full_phone)
        _safe_print(f"[AUTH] Failed login for {full_phone[-4:]}, {remaining} attempts left")
        if remaining > 0:
            return JSONResponse(content={
                "success": False,
                "error": f"Invalid PIN. {remaining} attempt{'s' if remaining != 1 else ''} left.",
                "attempts_left": remaining,
            })
        else:
            return JSONResponse(content={
                "success": False,
                "error": f"Account locked for 15 minutes.",
                "locked": True,
                "locked_seconds": 15 * 60,
            })

    # Success — clear rate limit and create session
    clear_rate_limit(full_phone)
    _safe_print(f"[AUTH] Login: {user_info['name']} ({user_info['role']})")

    # Record login in user DB (upsert + update last_login)
    from app.db import upsert_user, update_user_login
    upsert_user(user_info["phone"], user_info["name"], user_info["role"])
    update_user_login(user_info["phone"])

    session_token = create_session(user_info["phone"], user_info["name"], user_info["role"])

    # Check for redirect target from ?next= param
    redirect_to = body.get("next", "/") or "/"

    response = JSONResponse(content={
        "success": True,
        "redirect": redirect_to,
        "user_name": user_info["name"],
        "user_role": user_info["role"],
    })
    # Session cookie (no max_age = dies on browser close)
    # This ensures browser close forces re-login
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        samesite="lax",
    )
    # Display cookie: session-scoped (no max_age) for --reload recovery
    # Dies on browser close, same as auth cookie
    response.set_cookie(
        key="acadza_display",
        value=f"{user_info['name']}|{user_info['role']}",
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
    from app.db import list_sessions_by_user, get_session_stats, get_checkpoint
    import json as _json

    user = getattr(request.state, "user", {})
    phone = user.get("phone", "")
    sessions = list_sessions_by_user(phone) if phone else []
    active_count = 0
    completed_count = 0
    in_process = 0

    for s in sessions:
        s["stats"] = get_session_stats(s["id"])

        # Track active vs completed
        if s["current_stage"] == "completed":
            completed_count += 1
        else:
            active_count += 1
            in_process += s["stats"].get("total", 0)

        # Get upload counts (err1/err2) from Stage 10 checkpoint
        cp = get_checkpoint(s["id"], "stage10_upload")
        if cp and isinstance(cp.get("data"), str):
            try:
                cp["data"] = _json.loads(cp["data"])
            except Exception:
                cp["data"] = {}
        cp_data = (cp.get("data") or {}) if cp else {}
        s["upload_err1"] = cp_data.get("posted_err1", 0)
        s["upload_err2"] = cp_data.get("posted_err2", 0)
        # Fallback: if no split, use total posted count as err1+err2 combined
        if not s["upload_err1"] and not s["upload_err2"] and cp_data.get("posted", 0):
            s["upload_err1"] = cp_data.get("posted", 0)

    return templates.TemplateResponse(request, "dashboard.html", context={
        "sessions": sessions,
        "user_name": user.get("name", ""),
        "user_role": user.get("role", ""),
        "active_count": active_count,
        "completed_count": completed_count,
        "in_process": in_process,
    })


# ── Session detail page (renders the correct stage template) ──
@app.get("/session/{session_id}", response_class=HTMLResponse)
async def session_page(request: Request, session_id: str, stage: Optional[str] = None):
    """Session detail page — shows the current or requested stage."""
    from app.db import get_session, get_all_checkpoints, get_session_stats
    from app.state_machine import STAGES, STAGE_LABELS, STAGE_SHORT, get_stage_status, is_stage_accessible

    session = get_session(session_id)
    if session is None:
        return templates.TemplateResponse(request, "dashboard.html", context={
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
    return templates.TemplateResponse(request, template_name, context={
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
