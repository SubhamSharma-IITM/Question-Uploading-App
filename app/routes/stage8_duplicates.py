"""Stage 8: Duplicate Checking — routes for self-check, cross-check, reports, quarantine."""

import asyncio
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body
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
async def scan_duplicates(session_id: str):
    """Scan metadata for duplicate check readiness."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage8_duplicates import DuplicateCheckStage
    stage = DuplicateCheckStage(session_id, session_dir)
    result = stage.scan()
    return APIResponse(data=result)


@router.post("/self-check")
async def self_duplicate_check(session_id: str, threshold: float = 0.85):
    """Run self-duplicate check within the session."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage8_duplicates import DuplicateCheckStage
    stage = DuplicateCheckStage(session_id, session_dir)

    try:
        result = await asyncio.to_thread(stage.self_check, threshold=threshold)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="SELF_CHECK_ERROR", message=str(e)),
            ).model_dump(),
        )

    # Mark stage as completed so advancement works after individual checks
    if "error" not in result:
        from app.db import upsert_checkpoint
        upsert_checkpoint(session_id, "stage8_duplicates", "completed", {
            "self_duplicates": result.get("duplicates_found", 0),
        })

    return APIResponse(data=result)


@router.post("/cross-check")
async def cross_duplicate_check(session_id: str, threshold: float = 0.85):
    """Run cross-DB duplicate check against existing question database."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage8_duplicates import DuplicateCheckStage
    stage = DuplicateCheckStage(session_id, session_dir)

    try:
        result = await asyncio.to_thread(stage.cross_check, threshold=threshold)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="CROSS_CHECK_ERROR", message=str(e)),
            ).model_dump(),
        )

    # Mark stage as completed so advancement works after individual checks
    if "error" not in result:
        from app.db import upsert_checkpoint
        upsert_checkpoint(session_id, "stage8_duplicates", "completed", {
            "cross_duplicates": result.get("duplicates_found", 0),
            "db_count": result.get("db_questions", 0),
        })

    return APIResponse(data=result)


@router.get("/report")
async def get_report(session_id: str, report_type: str = "self"):
    """Get a previously generated duplicate report."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage8_duplicates import DuplicateCheckStage
    stage = DuplicateCheckStage(session_id, session_dir)
    result = stage.get_report(report_type=report_type)
    return APIResponse(data=result)


@router.get("/download-csv")
async def download_csv(session_id: str, report_type: str = "self"):
    """Download duplicate report as CSV."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage8_duplicates import DuplicateCheckStage
    stage = DuplicateCheckStage(session_id, session_dir)
    csv_content = stage.export_csv(report_type=report_type)

    if not csv_content:
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message=f"No {report_type} duplicate report found"),
            ).model_dump(),
        )

    import io
    filename = f"{report_type}_duplicates.csv"
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/quarantine")
async def quarantine_duplicates(session_id: str, qids: Optional[List[str]] = Body(None)):
    """Mark duplicate QIDs for quarantine. Requires at least 1 unique question remaining."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    if not qids:
        return APIResponse(data={"quarantined": 0, "message": "No QIDs provided"})

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage8_duplicates import DuplicateCheckStage
    stage = DuplicateCheckStage(session_id, session_dir)

    # Check that at least 1 question will remain after quarantine
    all_entries = stage._load_all_metadata()
    all_qids = {e.get("qid") for e in all_entries}
    remaining = all_qids - set(qids)
    if not remaining:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                error=ErrorDetail(
                    code="ALL_QUARANTINED",
                    message=f"Cannot quarantine all {len(all_qids)} questions. "
                            f"At least 1 unique question must remain for the pipeline to continue. "
                            f"Uncheck some duplicates to keep them.",
                ),
            ).model_dump(),
        )

    result = stage.quarantine_duplicates(qids)
    result["remaining_questions"] = len(remaining)
    return APIResponse(data=result)


@router.post("/skip")
async def skip_duplicates(session_id: str, reason: str = "manual"):
    """Skip duplicate checking (PYQ mode, testing, or small datasets)."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    from app.db import insert_log, upsert_checkpoint
    from app.sse import sse_manager
    msg = f"Duplicate checking skipped (reason: {reason})"
    insert_log(session_id, "stage8_duplicates", "info", msg)
    sse_manager.emit(session_id, {"type": "log", "level": "info", "message": msg})
    upsert_checkpoint(session_id, "stage8_duplicates", "completed", {"skipped": True, "reason": reason})

    return APIResponse(data={"skipped": True, "reason": reason})


@router.post("/run-all")
async def run_all_checks(session_id: str, threshold: float = 0.85):
    """Run both self-check and cross-check in sequence."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage8_duplicates import DuplicateCheckStage
    stage = DuplicateCheckStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.execute, threshold=threshold)

    if "error" not in result.get("self_check", {}) and "error" not in result.get("cross_check", {}):
        update_session(session_id, current_stage="stage8_duplicates")

    return APIResponse(data=result)
