"""Stage 6: Image CDN Upload — routes for scanning, uploading, and skipping."""

import asyncio
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import get_session, update_session
from app.models import APIResponse, ErrorDetail

router = APIRouter()


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
    """Upload images to Acadza CDN and replace local references in metadata."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage6_images import ImageCDNStage
    stage = ImageCDNStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.upload_images)

    if "error" not in result:
        update_session(session_id, current_stage="stage6_images")
        from app.db import upsert_checkpoint
        upsert_checkpoint(session_id, "stage6_images", "completed", {
            "uploaded": result.get("uploaded", 0),
            "cached": result.get("cached", 0),
            "replaced": result.get("replaced", 0),
            "files_modified": result.get("files_modified", 0),
            "failed": result.get("failed", 0),
        })

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
