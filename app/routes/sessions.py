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
async def list_all_sessions():
    """List all sessions with stats."""
    sessions = list_sessions()
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
    return APIResponse(data={"session_id": session_id, "stage": nxt})


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

    return EventSourceResponse(sse_manager.stream(session_id))
