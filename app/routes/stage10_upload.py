"""Stage 10: Upload to DB -- routes for scan, upload, CSV/ZIP download, absorb.

Upload and absorb run as background tasks to avoid Render proxy timeout.
"""

import asyncio
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Dict


def _safe_csv_name(session: dict) -> str:
    """Sanitize session name for use in CSV filenames."""
    name = session.get("name", "session") if session else "session"
    return re.sub(r'[^\w\-]', '_', name)

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

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
    """Upload all ready questions to Acadza DB API — background task."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    if session_id in _bg_tasks and not _bg_tasks[session_id].done():
        return APIResponse(data={"error": "Upload already running for this session"})

    session_dir = Path(settings.sessions_dir) / session_id

    from app.db import upsert_checkpoint
    upsert_checkpoint(session_id, "stage10_upload", "running", {})

    async def _run_upload():
        try:
            from app.pipeline.stage10_upload import UploadStage
            stage = UploadStage(session_id, session_dir)
            from app.utils import guarded_thread
            result = await guarded_thread(stage.upload, dry_run=False)
        except Exception as e:
            from app.db import upsert_checkpoint as _ucp, insert_log
            _ucp(session_id, "stage10_upload", "failed", {"error": str(e)})
            insert_log(session_id, "stage10_upload", "error", f"Upload failed: {e}")
            return

        if "error" not in result:
            from app.db import upsert_checkpoint as _ucp, insert_log
            from app.sse import sse_manager
            _ucp(session_id, "stage10_upload", "completed", {
                "posted": result.get("posted", 0),
                "failed": result.get("failed", 0),
                "skipped": result.get("skipped", 0),
            })
            msg = f"Upload complete: {result.get('posted', 0)} posted, {result.get('failed', 0)} failed"
            insert_log(session_id, "stage10_upload", "success", msg)
            sse_manager.emit(session_id, {"type": "log", "level": "success", "message": msg})
        else:
            from app.db import upsert_checkpoint as _ucp
            _ucp(session_id, "stage10_upload", "failed", result)

    def _task_done(t: asyncio.Task):
        _bg_tasks.pop(session_id, None)

    task = asyncio.create_task(_run_upload())
    task.add_done_callback(_task_done)
    _bg_tasks[session_id] = task
    return APIResponse(data={"status": "started", "message": "Upload running in background."})


@router.get("/upload-progress")
async def upload_progress(session_id: str):
    """Get upload progress from checkpoint + task state."""
    from app.db import get_checkpoint
    cp = get_checkpoint(session_id, "stage10_upload")
    if cp and isinstance(cp.get("data"), str):
        try:
            cp["data"] = json.loads(cp["data"])
        except Exception:
            cp["data"] = {}
    result = cp or {"status": "not_started"}
    is_running = session_id in _bg_tasks and not _bg_tasks[session_id].done()
    if is_running:
        result["bg_running"] = True
    return APIResponse(data=result)


@router.post("/absorb")
async def absorb_to_local_db(session_id: str):
    """Absorb uploaded questions into local JSONL DB + update FAISS index.

    After upload, this:
    1. Reads upload_id_map.json for meta_qid → MongoDB _id mappings
    2. Builds proper DB entries with real _id
    3. Appends to the local JSONL database
    4. Generates embeddings and adds to global FAISS cache
    Only processes error=1/2 entries (not quarantined).
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage10_upload import UploadStage
    stage = UploadStage(session_id, session_dir)
    from app.utils import guarded_thread
    result = await guarded_thread(stage.absorb_to_local_db)

    if "error" not in result:
        from app.db import insert_log
        from app.sse import sse_manager
        msg = f"Absorbed {result.get('absorbed', 0)} entries to local DB, {result.get('embedded', 0)} vectors added to FAISS"
        insert_log(session_id, "stage10_upload", "success", msg)
        sse_manager.emit(session_id, {"type": "log", "level": "success", "message": msg})

    return APIResponse(data=result)


@router.get("/download-stats")
async def download_pipeline_stats(session_id: str):
    """Download comprehensive pipeline stats CSV with per-stage metrics,
    model-wise accuracy, image counts, error type splits, and final accuracy.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage10_upload import UploadStage
    stage = UploadStage(session_id, session_dir)
    csv_content = await asyncio.to_thread(stage.generate_pipeline_stats_csv)

    if not csv_content:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message="Could not generate pipeline stats"),
            ).model_dump(),
        )

    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={_safe_csv_name(session)}_pipeline_stats.csv"},
    )


@router.get("/download-csv")
async def download_csv(session_id: str, error_level: int = 0):
    """Download the combined upload log CSV, optionally filtered by error level.

    error_level=0: all entries (default)
    error_level=1: only error=1 (clean) entries
    error_level=2: only error=2 (review) entries
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage10_upload import UploadStage
    stage = UploadStage(session_id, session_dir)
    csv_content = stage.export_csv(error_level=error_level if error_level > 0 else None)

    if not csv_content:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message="No upload CSV found — run dry-run or upload first"),
            ).model_dump(),
        )

    suffix = f"_error{error_level}" if error_level > 0 else ""
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={_safe_csv_name(session)}_upload_log{suffix}.csv"},
    )


@router.get("/download-stats-zip")
async def download_stats_zip(session_id: str):
    """Download all Stage 10 stats as a single ZIP (upload logs + pipeline stats)."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id
    safe = _safe_csv_name(session)

    from app.pipeline.stage10_upload import UploadStage
    stage = UploadStage(session_id, session_dir)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Upload log (all)
        csv_all = stage.export_csv(error_level=None)
        if csv_all:
            zf.writestr(f"{safe}_upload_log.csv", csv_all)
        # Error=1 CSV
        csv_e1 = stage.export_csv(error_level=1)
        if csv_e1:
            zf.writestr(f"{safe}_upload_log_error1.csv", csv_e1)
        # Error=2 CSV
        csv_e2 = stage.export_csv(error_level=2)
        if csv_e2:
            zf.writestr(f"{safe}_upload_log_error2.csv", csv_e2)
        # Pipeline stats
        stats_csv = stage.generate_pipeline_stats_csv()
        if stats_csv:
            zf.writestr(f"{safe}_pipeline_stats.csv", stats_csv)

        # Cost report (actual token usage from batch results)
        try:
            from app.pipeline.cost_tracker import generate_cost_report
            cost_report = generate_cost_report(str(session_dir))
            zf.writestr(f"{safe}_cost_report.json", json.dumps(cost_report, indent=2))
        except Exception:
            pass  # Cost report is optional

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={safe}_all_stats.zip"},
    )


@router.post("/complete")
async def mark_session_complete(session_id: str):
    """Mark session as globally complete (current_stage='completed')."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    update_session(session_id, current_stage="completed")
    return APIResponse(data={"status": "completed"})
