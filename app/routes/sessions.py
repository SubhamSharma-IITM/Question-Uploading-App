"""Session management routes: create, list, resume, delete, SSE logs."""

import shutil
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.db import (
    create_session,
    delete_session,
    get_session,
    get_session_stats,
    list_sessions,
    list_sessions_by_user,
    update_session,
    get_all_checkpoints,
)
from app.models import APIResponse, CreateSessionRequest, ErrorDetail
from app.sse import sse_manager
from app.state_machine import next_stage
from app.utils import now_iso, short_uuid

router = APIRouter(tags=["Sessions"])


@router.post("/api/sessions")
async def create_new_session(req: CreateSessionRequest, request: Request):
    """Create a new upload session."""
    session_id = short_uuid()
    created_at = now_iso()

    # Get logged-in user phone for ownership tracking
    user = getattr(request.state, "user", {})
    created_by = user.get("phone", "")

    # Create session directory
    session_dir = Path(settings.sessions_dir) / session_id
    for subdir in [
        "input", "mathpix_output", "mapped_output", "cleaned_output",
        "metadata_v2", "images_original", "images_cleaned",
        "batch_jsonl", "batch_results", "duplicates", "quality",
        "quarantine", "backups",
    ]:
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Persist created_by to disk so orphan recovery can attribute the session
    import json as _json
    config_path = session_dir / "session_config.json"
    disk_config = {"created_by": created_by, "session_name": req.name}
    config_path.write_text(_json.dumps(disk_config, ensure_ascii=False), encoding="utf-8")

    # Insert into DB — start at stage1_upload so sidebar highlights Step 1
    # and first "Advance" goes to stage2 (not back to stage1)
    create_session(session_id, req.name, created_at, req.config,
                   initial_stage="stage1_upload", created_by=created_by)

    return APIResponse(data={
        "session_id": session_id,
        "name": req.name,
        "created_at": created_at,
    })


@router.get("/api/sessions")
async def list_all_sessions(request: Request):
    """List sessions — each user sees only their own sessions."""
    user = getattr(request.state, "user", {})
    phone = user.get("phone", "")

    sessions = list_sessions_by_user(phone) if phone else []

    result = []
    for s in sessions:
        stats = get_session_stats(s["id"])
        result.append({**s, "stats": stats})
    return APIResponse(data=result)


@router.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Get session detail with checkpoints and stats."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message=f"Session {session_id} not found"),
            ).model_dump(),
        )

    checkpoints = get_all_checkpoints(session_id)
    stats = get_session_stats(session_id)

    return APIResponse(data={
        **session,
        "checkpoints": checkpoints,
        "stats": stats,
    })


@router.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Delete a session and all its data."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message=f"Session {session_id} not found"),
            ).model_dump(),
        )

    # Remove session directory
    session_dir = Path(settings.sessions_dir) / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)

    # Remove from DB (cascades to all related tables)
    delete_session(session_id)

    return APIResponse(data={"deleted": session_id})


@router.post("/api/sessions/{session_id}/advance")
async def advance_session(session_id: str):
    """Advance session to the next stage — requires current stage checkpoint completed."""
    from app.db import get_checkpoint

    session = get_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message=f"Session {session_id} not found"),
            ).model_dump(),
        )

    # Normalize legacy "created" stage — it's equivalent to "stage1_upload"
    current = session["current_stage"]
    if current == "created":
        current = "stage1_upload"

    # Gate: check that current stage is actually completed
    checkpoint = get_checkpoint(session_id, current)
    # Allow advancement if stage is completed OR running in background
    # "smiles_rendered" and "skipped" are legacy/safety — stages should save "completed"
    ADVANCEABLE_STATUSES = {"completed", "background", "smiles_rendered", "skipped"}

    # Stage 5 safety net: auto-complete if all sub-steps are done but status
    # never transitioned (e.g. SMILES scan didn't run or errored)
    if (current == "stage5_watermark"
            and checkpoint is not None
            and checkpoint.get("status") not in ADVANCEABLE_STATUSES):
        cp_data = checkpoint.get("data") or {}
        if (cp_data.get("watermark_step") == "completed"
                and cp_data.get("quarantine_step") == "completed"):
            from app.db import upsert_checkpoint
            upsert_checkpoint(session_id, current, "completed", cp_data)
            checkpoint = get_checkpoint(session_id, current)

    if checkpoint is None or checkpoint.get("status") not in ADVANCEABLE_STATUSES:
        stage_label = current.replace("_", " ").replace("stage", "Stage ")
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                error=ErrorDetail(
                    code="STAGE_NOT_COMPLETE",
                    message=f"Cannot advance: {stage_label} is not complete yet. Finish the current stage first.",
                ),
            ).model_dump(),
        )

    nxt = next_stage(current)
    if nxt is None:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="INVALID_TRANSITION", message="Session is already completed"),
            ).model_dump(),
        )

    update_session(session_id, current_stage=nxt)

    # ── FIX-A2: write in-flight FAISS shadow on Stage 8 -> Stage 9 advance ──
    # Closes the Stage 8 -> Stage 10 race window. Concurrent sessions' Stage 8
    # cross-check reads this shadow, so they see our surviving vectors as
    # duplicates even though we haven't uploaded yet.
    # Failure here is NON-fatal: the advance still succeeds. Race window
    # stays open for this one session only; nightly back-fill/post-day sweep
    # is the residual safety net.
    if current == "stage8_duplicates" and nxt == "stage9_quality":
        try:
            from app.pipeline.stage8_duplicates import DuplicateCheckStage
            stage = DuplicateCheckStage(session_id)
            session_name = session.get("name", "") or ""
            operator = session.get("created_by", "") or ""
            result = stage.write_inflight_shadow(
                session_name=session_name, operator=operator
            )
            return APIResponse(data={
                "session_id": session_id, "stage": nxt,
                "shadow_written": int(result.get("written", 0)),
            })
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning(
                f"FIX-A2 shadow write failed for {session_id} on stage8->9 "
                f"advance: {e}"
            )
            return APIResponse(data={
                "session_id": session_id, "stage": nxt,
                "shadow_written": 0,
                "shadow_error": str(e),
            })

    return APIResponse(data={"session_id": session_id, "stage": nxt})


@router.get("/api/admin/ghost-sessions")
async def list_ghost_sessions(request: Request):
    """Admin-only: list sessions with empty created_by (ghost sessions)."""
    user = getattr(request.state, "user", {})
    if user.get("role") not in ("Admin", "Architect"):
        return JSONResponse(status_code=403, content={"success": False, "error": "Admin only"})

    all_sessions = list_sessions()
    ghosts = []
    for s in all_sessions:
        if not s.get("created_by", ""):
            ghosts.append({
                "id": s["id"],
                "name": s.get("name", "?"),
                "current_stage": s.get("current_stage", "?"),
                "created_at": s.get("created_at", ""),
            })
    return APIResponse(data={"ghost_count": len(ghosts), "ghosts": ghosts})


@router.post("/api/admin/reattribute")
async def reattribute_sessions(request: Request):
    """Admin-only: assign ghost sessions to a user phone.

    Body: {"session_ids": [...], "phone": "+91..."}
    """
    user = getattr(request.state, "user", {})
    if user.get("role") not in ("Admin", "Architect"):
        return JSONResponse(status_code=403, content={"success": False, "error": "Admin only"})

    body = await request.json()
    session_ids = body.get("session_ids", [])
    phone = body.get("phone", "")
    if not phone or not session_ids:
        return JSONResponse(status_code=400, content={"success": False, "error": "session_ids and phone required"})

    from app.db import _get_conn, _lock
    import json as _json
    fixed = 0
    for sid in session_ids:
        with _lock:
            conn = _get_conn()
            conn.execute("UPDATE sessions SET created_by = ? WHERE id = ?", (phone, sid))
            conn.commit()
        # Also fix session_config.json on disk
        cfg_path = Path(settings.sessions_dir) / sid / "session_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = _json.load(f)
                cfg["created_by"] = phone
                with open(cfg_path, "w", encoding="utf-8") as f:
                    _json.dump(cfg, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        fixed += 1

    return APIResponse(data={"fixed": fixed, "phone": phone})


@router.get("/api/sessions/{session_id}/quarantine-count")
async def get_quarantine_count(session_id: str):
    """Get the current quarantine count for sidebar display."""
    from app.db import get_quarantine
    session = get_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
            ).model_dump(),
        )
    records = get_quarantine(session_id)
    return APIResponse(data={"count": len(records)})


@router.get("/api/sessions/{session_id}/live-stats")
async def get_live_stats(session_id: str):
    """Get live stats from metadata files -- single source of truth.

    Returns per-subject counts of triplets, images, SMILES, error levels,
    quarantine status -- all computed from the current metadata_v2 files.
    """
    session = get_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
            ).model_dump(),
        )

    from app.db import get_quarantined_qids
    from app.pipeline.metadata_stats import compute_live_stats

    session_dir = Path(settings.sessions_dir) / session_id
    quarantined = get_quarantined_qids(session_id)
    stats = compute_live_stats(session_dir, quarantined)
    return APIResponse(data=stats)


@router.get("/api/sessions/{session_id}/logs/history")
async def get_log_history(session_id: str, stage: str = None):
    """Get historical log entries from DB for a session (persisted across navigation)."""
    from app.db import get_logs
    session = get_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message=f"Session {session_id} not found"),
            ).model_dump(),
        )

    logs = get_logs(session_id, limit=500)
    # Optionally filter by stage
    if stage:
        logs = [l for l in logs if l.get("stage") == stage]
    return APIResponse(data=logs)


@router.get("/api/sessions/{session_id}/logs")
async def session_log_stream(request: Request, session_id: str):
    """SSE stream for live session logs."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message=f"Session {session_id} not found"),
            ).model_dump(),
        )

    return EventSourceResponse(sse_manager.stream(session_id), ping=15)
