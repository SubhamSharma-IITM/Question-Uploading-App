"""Stage 7: LaTeX + Tagging Batch — routes for building, submitting, polling, and merging."""

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
async def scan_batch(session_id: str):
    """Scan metadata and report what batches would be built."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage7_latex_tag import LatexTagStage
    stage = LatexTagStage(session_id, session_dir)
    result = stage.scan()
    return APIResponse(data=result)


@router.get("/credentials")
async def check_credentials(session_id: str):
    """Check if OpenAI API key is configured."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage7_latex_tag import LatexTagStage
    stage = LatexTagStage(session_id, session_dir)
    result = stage.check_credentials()
    return APIResponse(data=result)


@router.get("/preview")
async def preview_mathml(session_id: str, offset: int = 0, count: int = 5, subject: str = ""):
    """Return rendered MathML preview samples from metadata."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage7_latex_tag import LatexTagStage
    stage = LatexTagStage(session_id, session_dir)
    entries = stage._load_all_metadata()

    # Get unique subjects
    subjects = sorted(set(e.get("subject", "Unknown") for e in entries))

    # Filter by subject if specified
    if subject:
        entries = [e for e in entries if e.get("subject", "") == subject]

    total_available = len(entries)

    # Slice
    samples = []
    for e in entries[offset:offset + count]:
        qid = e.get("qid", "?")
        qtype = e.get("questionType", "")
        qdata = e.get(qtype, {})
        q_html = qdata.get("question", "") or qdata.get("ques", "")
        has_mathml = "<math" in q_html if q_html else False
        samples.append({
            "qid": qid,
            "subject": e.get("subject", ""),
            "question_html": q_html[:3000],  # cap for transport
            "has_mathml": has_mathml,
        })

    return APIResponse(data={
        "samples": samples,
        "subjects": subjects,
        "total_available": total_available,
        "offset": offset,
    })


@router.post("/convert-mathml")
async def convert_all_mathml(session_id: str):
    """Convert ALL LaTeX to MathML locally. Only failures need the LaTeX batch."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage7_latex_tag import LatexTagStage
    stage = LatexTagStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.convert_all_mathml)
    return APIResponse(data=result)


@router.post("/build-batch")
async def build_batch(session_id: str):
    """Build JSONL batch files for tagging and LaTeX conversion."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage7_latex_tag import LatexTagStage
    stage = LatexTagStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.build_batch)
    return APIResponse(data=result)


@router.post("/submit")
async def submit_batch(session_id: str):
    """Submit batch files to OpenAI Batch API."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage7_latex_tag import LatexTagStage
    stage = LatexTagStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.submit_batch)
    return APIResponse(data=result)


@router.get("/poll")
async def poll_batch(session_id: str):
    """Poll batch processing status."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage7_latex_tag import LatexTagStage
    stage = LatexTagStage(session_id, session_dir)
    result = stage.poll_status()
    return APIResponse(data=result)


@router.post("/stop")
async def stop_batch(session_id: str):
    """Request stop of batch processing. Cancels OpenAI batches and preserves partial results."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    from app.db import insert_log
    insert_log(session_id, "stage7_batch", "warning", "Batch stop requested by user")

    session_dir = Path(settings.sessions_dir) / session_id
    from app.pipeline.stage7_latex_tag import LatexTagStage
    stage = LatexTagStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.cancel_batches)
    return APIResponse(data=result)


@router.post("/background")
async def run_in_background(session_id: str):
    """Mark batch as running in background — user can advance to Stage 8."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    from app.db import insert_log, upsert_checkpoint
    insert_log(session_id, "stage7_batch", "info", "Batch running in background — user advanced")
    upsert_checkpoint(session_id, "stage7_batch", "background", {
        "background": True,
        "convert_done": True,
        "build_done": True,
        "submit_done": True,
    })

    return APIResponse(data={"status": "background"})


@router.post("/download")
async def download_results(session_id: str):
    """Download and merge batch results into metadata."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage7_latex_tag import LatexTagStage
    stage = LatexTagStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.download_results)

    if "error" not in result:
        from app.db import upsert_checkpoint
        upsert_checkpoint(session_id, "stage7_batch", "completed", {
            "tags_merged": result.get("tags_merged", 0),
            "latex_merged": result.get("latex_merged", 0),
            "mathml_converted": result.get("mathml_converted", 0),
        })

    return APIResponse(data=result)
