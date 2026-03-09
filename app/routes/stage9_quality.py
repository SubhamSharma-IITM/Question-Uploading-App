"""Stage 9: Quality Verifier — routes for QA batch build, submit, poll, vote merge.

Submit runs as a background asyncio task so the HTTP response returns immediately.
The user polls for progress and can stop processing via the /stop endpoint.
"""

import asyncio
import io
import json
import zipfile
from pathlib import Path
from typing import Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.db import get_session, update_session
from app.models import APIResponse, ErrorDetail

router = APIRouter()

# Track running background tasks per session+pass_key
_running_tasks: Dict[str, asyncio.Task] = {}
# Track stage instances for stop support
_running_stages: Dict[str, object] = {}


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
async def scan_qa(session_id: str):
    """Scan metadata and determine QA requirements."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage9_quality import QualityVerifierStage
    stage = QualityVerifierStage(session_id, session_dir)
    result = stage.scan()
    return APIResponse(data=result)


@router.post("/build")
async def build_qa_batch(
    session_id: str,
    extra_context: str = "",
    strict_mode: bool = False,
    check_latex: bool = True,
):
    """Build QA batch files for all 3 passes."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage9_quality import QualityVerifierStage
    stage = QualityVerifierStage(session_id, session_dir)
    result = await asyncio.to_thread(
        stage.build_qa_batch,
        extra_context=extra_context,
        strict_mode=strict_mode,
        check_latex=check_latex,
    )
    return APIResponse(data=result)


@router.post("/submit")
async def submit_qa_pass(session_id: str, pass_key: str = "pass1"):
    """Submit a specific QA pass to Batch API (all passes).

    All 3 passes now use Batch API (OpenAI or Gemini). Returns immediately
    with batch_id. Use /poll to track progress.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage9_quality import QualityVerifierStage
    stage = QualityVerifierStage(session_id, session_dir)

    try:
        result = await asyncio.to_thread(stage.submit_pass, pass_key=pass_key)
        return APIResponse(data=result)
    except Exception as e:
        return APIResponse(data={"error": str(e), "pass": pass_key})


@router.post("/submit-all")
async def submit_all_passes(session_id: str):
    """Submit all 3 passes to Batch API (OpenAI + Gemini).

    All passes return immediately with batch_id. Use /poll to track progress.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id
    from app.pipeline.stage9_quality import QualityVerifierStage, PASS_MODELS

    started = []
    for pass_key in PASS_MODELS:
        stage = QualityVerifierStage(session_id, session_dir)
        try:
            result = await asyncio.to_thread(stage.submit_pass, pass_key=pass_key)
            started.append({"pass": pass_key, **result})
        except Exception as e:
            started.append({"pass": pass_key, "status": "error", "error": str(e)})

    return APIResponse(data={"passes": started, "message": "All passes submitted to Batch API. Poll for progress."})


@router.post("/stop")
async def stop_qa(session_id: str, pass_key: str = ""):
    """Stop one or all running QA passes for this session."""
    from app.pipeline.stage9_quality import QualityVerifierStage, PASS_MODELS
    stopped = []

    keys_to_stop = [f"{session_id}:{pass_key}"] if pass_key else [
        f"{session_id}:{pk}" for pk in PASS_MODELS
    ]

    for tk in keys_to_stop:
        stage = _running_stages.get(tk)
        if stage and isinstance(stage, QualityVerifierStage):
            stage.request_stop()
            stopped.append(tk.split(":")[1])

    if stopped:
        return APIResponse(data={"status": "stopping", "passes": stopped,
                                 "message": f"Stop requested for: {', '.join(stopped)}"})
    return APIResponse(data={"status": "not_running",
                             "message": "No active passes to stop"})


@router.get("/poll")
async def poll_qa_passes(session_id: str):
    """Poll status of all QA passes with live progress."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage9_quality import QualityVerifierStage, PASS_MODELS
    stage = QualityVerifierStage(session_id, session_dir)
    result = stage.poll_passes()

    # Also check if background tasks are still alive (override status if needed)
    for pass_key in PASS_MODELS:
        task_key = f"{session_id}:{pass_key}"
        task = _running_tasks.get(task_key)
        if task and not task.done():
            if result["passes"].get(pass_key, {}).get("status") != "running":
                result["passes"][pass_key]["status"] = "running"

    # Recalculate flags after potential overrides
    any_running = any(s["status"] == "running" for s in result["passes"].values())
    all_done = all(s["status"] in ("completed", "stopped") for s in result["passes"].values())
    any_started = any(s["status"] != "not_started" for s in result["passes"].values())
    result["any_running"] = any_running
    result["all_complete"] = all_done and any_started
    result["can_vote"] = all_done and any_started

    return APIResponse(data=result)


@router.post("/download-batch")
async def download_batch_results(session_id: str, pass_key: str = ""):
    """Download completed OpenAI batch results and build ledger files.

    If pass_key is specified, downloads that pass only. Otherwise downloads all
    OpenAI passes that have completed batches.

    Note: poll_passes() auto-downloads when batch_complete, so this endpoint
    is mainly for manual retry if auto-download failed.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage9_quality import QualityVerifierStage
    stage = QualityVerifierStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.download_pass_results, pass_key=pass_key)
    return APIResponse(data=result)


@router.post("/vote")
async def merge_votes(session_id: str):
    """Merge votes from all 3 passes and determine error levels."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage9_quality import QualityVerifierStage
    stage = QualityVerifierStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.merge_votes)

    if result.get("status") != "stub" and "error" not in result:
        update_session(session_id, current_stage="stage9_quality")

    return APIResponse(data=result)


@router.get("/csv")
async def download_qa_csv(session_id: str, category: str = "3of3"):
    """Download QA CSV for a category (3of3 or 2of3)."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage9_quality import QualityVerifierStage
    stage = QualityVerifierStage(session_id, session_dir)
    csv_content = stage.export_csv(category=category)

    if not csv_content:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message=f"No {category} CSV found -- run vote merge first"),
        ).model_dump())

    filename = f"qa_{category}_{session_id[:8]}.csv"
    return StreamingResponse(
        io.BytesIO(csv_content.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/summary")
async def qa_summary(session_id: str):
    """Get full pipeline summary with per-stage stats, breakdowns, and QA results."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id
    stats = _compute_pipeline_summary(session_id, session_dir)
    return APIResponse(data=stats)


@router.get("/download-ready")
async def download_ready_files(session_id: str):
    """Download all upload-ready metadata_v2 files as a ZIP archive.

    Filters out quarantined questions (error >= 3) and returns only clean entries.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id
    meta_dir = session_dir / "metadata_v2_output"
    if not meta_dir.exists():
        meta_dir = session_dir / "metadata_v2"
    if not meta_dir.exists():
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="No metadata_v2 files found"),
        ).model_dump())

    # Collect all metadata files (flat or subdirectory)
    meta_files = []
    for sub in sorted(meta_dir.iterdir()):
        if sub.is_dir():
            for f in sub.glob("*_metadata_v2.json"):
                if "backup" not in f.name:
                    meta_files.append((f"{sub.name}/{f.name}", f))
        elif sub.suffix == ".json" and "metadata_v2" in sub.name and "backup" not in sub.name:
            meta_files.append((sub.name, sub))

    if not meta_files:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="No metadata_v2 files found"),
        ).model_dump())

    # Build ZIP in memory — filter out error >= 3 questions
    buf = io.BytesIO()
    total_ready = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_name, fpath in meta_files:
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    ready = [q for q in data if not (isinstance(q.get("error"), int) and q["error"] >= 3)]
                    total_ready += len(ready)
                    zf.writestr(rel_name, json.dumps(ready, indent=2, ensure_ascii=False))
            except Exception:
                continue

    buf.seek(0)
    session_name = session.get("name", session_id).replace(" ", "_")
    filename = f"{session_name}_ready_{total_ready}q.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/skip")
async def skip_qa(session_id: str):
    """Skip QA verification (taggedBy is AIT/NIT, not AI-verified)."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage9_quality import QualityVerifierStage
    stage = QualityVerifierStage(session_id, session_dir)
    result = stage.skip()

    from app.db import insert_log, upsert_checkpoint
    from app.sse import sse_manager
    insert_log(session_id, "stage9_quality", "info", "Quality verification skipped")
    sse_manager.emit(session_id, {"type": "log", "level": "info", "message": "Quality verification skipped"})
    upsert_checkpoint(session_id, "stage9_quality", "completed", {"skipped": True})

    return APIResponse(data=result)


@router.post("/complete")
async def mark_complete(session_id: str):
    """Mark the pipeline as complete after QA passes. Returns comprehensive stats."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    update_session(session_id, current_stage="completed")

    from app.db import insert_log, upsert_checkpoint, get_all_checkpoints
    from app.sse import sse_manager
    insert_log(session_id, "stage9_quality", "success", "Pipeline complete! Questions ready for upload.")
    sse_manager.emit(session_id, {"type": "log", "level": "success", "message": "Pipeline complete!"})
    upsert_checkpoint(session_id, "stage9_quality", "completed", {"pipeline_complete": True})

    # Compute comprehensive pipeline stats from checkpoints and metadata
    session_dir = Path(settings.sessions_dir) / session_id
    stats = _compute_pipeline_summary(session_id, session_dir)

    return APIResponse(data={
        "status": "completed",
        "message": "Pipeline complete! Questions ready for upload.",
        "summary": stats,
    })


def _compute_pipeline_summary(session_id: str, session_dir: Path) -> dict:
    """Compute comprehensive pipeline stats from checkpoints and metadata files."""
    import json
    from app.db import get_all_checkpoints

    summary = {
        "total_processed": 0,
        "total_quarantined": 0,
        "total_ready": 0,
        "by_subject": {},
        "by_question_type": {},
        "stage_stats": {},
    }

    # Load metadata_v2 files for question-level stats
    meta_dir = session_dir / "metadata_v2_output"
    if not meta_dir.exists():
        meta_dir = session_dir / "metadata_v2"

    all_questions = []
    if meta_dir.exists():
        for sub in sorted(meta_dir.iterdir()):
            if sub.is_dir():
                for f in sub.glob("*_metadata_v2.json"):
                    if "backup" not in f.name:
                        try:
                            with open(f, "r", encoding="utf-8") as fh:
                                data = json.load(fh)
                            if isinstance(data, list):
                                all_questions.extend(data)
                        except Exception:
                            continue
            elif sub.suffix == ".json" and "metadata_v2" in sub.name and "backup" not in sub.name:
                try:
                    with open(sub, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        all_questions.extend(data)
                except Exception:
                    continue

    summary["total_processed"] = len(all_questions)

    # Count by subject
    for q in all_questions:
        subj = q.get("subject", "Unknown")
        summary["by_subject"][subj] = summary["by_subject"].get(subj, 0) + 1
        qtype = q.get("questionType", "unknown")
        summary["by_question_type"][qtype] = summary["by_question_type"].get(qtype, 0) + 1

    # Count quarantined: error >= 3 in metadata + questions in quarantine DB table
    quarantined_by_error = 0
    for q in all_questions:
        error_val = q.get("error", 2)
        if isinstance(error_val, int) and error_val >= 3:
            quarantined_by_error += 1

    # Also count questions removed via quarantine DB (duplicates, etc.)
    from app.db import get_quarantine
    quarantine_records = get_quarantine(session_id)
    quarantined_by_db = len(quarantine_records)

    summary["total_quarantined"] = quarantined_by_error + quarantined_by_db
    summary["quarantined_by_error"] = quarantined_by_error
    summary["quarantined_by_duplicate"] = quarantined_by_db
    summary["total_ready"] = len(all_questions) - quarantined_by_error

    # Per-stage checkpoint stats
    checkpoints = get_all_checkpoints(session_id)
    for cp in checkpoints:
        stage = cp.get("stage", "")
        cp_data = cp.get("data")
        if isinstance(cp_data, str):
            try:
                cp_data = json.loads(cp_data)
            except Exception:
                cp_data = {}
        if not isinstance(cp_data, dict):
            cp_data = {}
        summary["stage_stats"][stage] = {
            "status": cp.get("status", "unknown"),
            **{k: v for k, v in cp_data.items() if k != "pipeline_complete"},
        }

    # QA vote results (from vote_summary.json if exists)
    vote_summary_path = session_dir / "quality_check" / "vote_summary.json"
    if vote_summary_path.exists():
        try:
            with open(vote_summary_path, "r") as f:
                vote_data = json.load(f)
            summary["qa_votes"] = vote_data.get("error_levels", {})
        except Exception:
            pass

    # CDN image stats (from images_cdn_map.json)
    cdn_map_path = session_dir / "images_cdn_map.json"
    if cdn_map_path.exists():
        try:
            with open(cdn_map_path, "r") as f:
                cdn_data = json.load(f)
            summary["images_uploaded"] = len(cdn_data)
        except Exception:
            pass

    # Duplicate stats (from checkpoint data)
    dup_checkpoint = summary["stage_stats"].get("stage8_duplicates", {})
    if dup_checkpoint:
        summary["duplicates"] = {
            k: v for k, v in dup_checkpoint.items()
            if k in ("self_duplicates", "cross_duplicates", "total_quarantined")
        }

    return summary
