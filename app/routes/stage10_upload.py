"""Stage 10: Upload to DB — routes for scan, dry-run, upload, CSV download."""

import asyncio
import io
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

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
async def scan_upload(session_id: str):
    """Scan metadata to see what's ready for upload."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage10_upload import UploadStage
    stage = UploadStage(session_id, session_dir)
    result = stage.scan()
    return APIResponse(data=result)


@router.post("/dry-run")
async def dry_run_upload(session_id: str):
    """Dry run — generate CSV without actually uploading."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage10_upload import UploadStage
    stage = UploadStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.upload, dry_run=True)
    return APIResponse(data=result)


@router.post("/upload")
async def upload_to_db(session_id: str):
    """Upload all ready questions to Acadza DB API."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage10_upload import UploadStage
    stage = UploadStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.upload, dry_run=False)

    if "error" not in result:
        from app.db import upsert_checkpoint, insert_log
        from app.sse import sse_manager
        upsert_checkpoint(session_id, "stage10_upload", "completed", {
            "posted": result.get("posted", 0),
            "failed": result.get("failed", 0),
            "skipped": result.get("skipped", 0),
        })
        msg = f"Upload complete: {result.get('posted', 0)} posted, {result.get('failed', 0)} failed"
        insert_log(session_id, "stage10_upload", "success", msg)
        sse_manager.emit(session_id, {"type": "log", "level": "success", "message": msg})

    return APIResponse(data=result)


@router.get("/download-csv")
async def download_csv(session_id: str):
    """Download the combined upload log CSV."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage10_upload import UploadStage
    stage = UploadStage(session_id, session_dir)
    csv_content = stage.export_csv()

    if not csv_content:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message="No upload CSV found — run dry-run or upload first"),
            ).model_dump(),
        )

    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=upload_log.csv"},
    )
