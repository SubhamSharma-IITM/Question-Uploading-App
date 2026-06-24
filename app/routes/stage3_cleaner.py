"""Stage 3: Cleaner — routes with actual pipeline adapter."""

import asyncio
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import get_session, update_session
from app.models import APIResponse, ErrorDetail

router = APIRouter()


@router.post("/dry-run")
async def cleaner_dry_run(session_id: str):
    """Preview cleaning changes without applying."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage3_cleaner import CleanerStage
    stage = CleanerStage(session_id, session_dir)

    result = await asyncio.to_thread(stage.dry_run)
    return APIResponse(data=result)


@router.get("/preview")
async def cleaner_preview(session_id: str, count: int = 7):
    """Get before/after cleaning comparison for random samples."""
    import json, random

    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id
    mapped_dir = session_dir / "mapped_output"
    cleaned_dir = session_dir / "cleaned_output"

    # Load both original and cleaned
    originals = []
    for f in sorted(mapped_dir.glob("*_mapped.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                originals.extend(json.load(fh))
        except Exception:
            continue

    cleaned = []
    for f in sorted(cleaned_dir.glob("*_cleaned.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                cleaned.extend(json.load(fh))
        except Exception:
            continue

    # Build before/after pairs by qid or question_number
    orig_map = {e.get("qid", e.get("question_number")): e for e in originals}
    clean_map = {e.get("qid", e.get("question_number")): e for e in cleaned}

    pairs = []
    for key in orig_map:
        if key in clean_map:
            pairs.append({"before": orig_map[key], "after": clean_map[key]})

    samples = random.sample(pairs, min(count, len(pairs))) if pairs else []
    return APIResponse(data={"samples": samples, "total": len(pairs)})


@router.post("/apply")
async def cleaner_apply(session_id: str):
    """Apply cleaning changes."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage3_cleaner import CleanerStage
    stage = CleanerStage(session_id, session_dir)

    result = await asyncio.to_thread(stage.execute)

    if "error" not in result:
        update_session(session_id, current_stage="stage3_cleaning")
        from app.db import upsert_checkpoint
        upsert_checkpoint(session_id, "stage3_cleaning", "completed", {
            "questions_cleaned": result.get("questions_cleaned", 0),
            "fields_modified": result.get("fields_modified", 0),
        })

    return APIResponse(data=result)
