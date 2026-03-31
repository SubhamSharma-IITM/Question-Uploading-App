"""Stage 6: Image CDN Upload — routes for scanning, uploading, and skipping.

Upload runs as background task to avoid Render proxy timeout.
"""

import asyncio
from pathlib import Path
from typing import Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import get_session, update_session
from app.models import APIResponse, ErrorDetail

router = APIRouter()
_bg_tasks: Dict[str, asyncio.Task] = {}


def _get_session_or_404(session_id: str):
    session = get_session(session_id)
    if session is None:
        return None, JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
            ).model_dump(),
        )
    return session, None


@router.get("/scan")
async def scan_images(session_id: str):
    """Scan metadata for image references and check what needs uploading."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage6_images import ImageCDNStage
    stage = ImageCDNStage(session_id, session_dir)
    result = stage.scan()
    return APIResponse(data=result)


@router.get("/credentials")
async def check_credentials(session_id: str):
    """Check if CDN API endpoint and dependencies are ready."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage6_images import ImageCDNStage
    stage = ImageCDNStage(session_id, session_dir)
    result = stage.check_credentials()
    return APIResponse(data=result)


@router.post("/upload")
async def upload_images(session_id: str):
    """Upload images to Acadza CDN — runs as background task."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    if session_id in _bg_tasks and not _bg_tasks[session_id].done():
        return APIResponse(data={"error": "Upload already running for this session"})

    session_dir = Path(settings.sessions_dir) / session_id

    from app.db import upsert_checkpoint
    upsert_checkpoint(session_id, "stage6_images", "running", {})

    async def _run_upload():
        try:
            from app.pipeline.stage6_images import ImageCDNStage
            stage = ImageCDNStage(session_id, session_dir)
            result = await asyncio.to_thread(stage.upload_images)
        except Exception as e:
            from app.db import upsert_checkpoint as _ucp, insert_log
            _ucp(session_id, "stage6_images", "failed", {"error": str(e)})
            insert_log(session_id, "stage6_images", "error", f"Upload failed: {e}")
            return
        finally:
            _bg_tasks.pop(session_id, None)

        if "error" not in result:
            update_session(session_id, current_stage="stage6_images")
            from app.db import upsert_checkpoint as _ucp
            _ucp(session_id, "stage6_images", "completed", {
                "uploaded": result.get("uploaded", 0),
                "cached": result.get("cached", 0),
                "replaced": result.get("replaced", 0),
                "files_modified": result.get("files_modified", 0),
                "failed": result.get("failed", 0),
            })
        else:
            from app.db import upsert_checkpoint as _ucp
            _ucp(session_id, "stage6_images", "failed", result)

    task = asyncio.create_task(_run_upload())
    _bg_tasks[session_id] = task
    return APIResponse(data={"status": "started", "message": "Image upload running in background."})


@router.get("/progress")
async def upload_progress(session_id: str):
    """Get upload progress from checkpoint + live task state."""
    from app.db import get_checkpoint
    import json as _json
    cp = get_checkpoint(session_id, "stage6_images")
    if cp and isinstance(cp.get("data"), str):
        try:
            cp["data"] = _json.loads(cp["data"])
        except Exception:
            cp["data"] = {}
    result = cp or {"status": "not_started"}
    is_running = session_id in _bg_tasks and not _bg_tasks[session_id].done()
    if is_running:
        result["bg_running"] = True
    return APIResponse(data=result)


@router.post("/skip")
async def skip_images(session_id: str):
    """Skip image CDN upload (no images or deferred)."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage6_images import ImageCDNStage
    stage = ImageCDNStage(session_id, session_dir)
    result = stage.skip()

    from app.db import insert_log, upsert_checkpoint
    from app.sse import sse_manager
    insert_log(session_id, "stage6_images", "info", "Image CDN upload skipped")
    sse_manager.emit(session_id, {"type": "log", "level": "info", "message": "Image CDN upload skipped"})
    upsert_checkpoint(session_id, "stage6_images", "completed", {"skipped": True})

    return APIResponse(data=result)
