"""Stage 4: Metadata V2 creation — routes with actual pipeline adapter."""

import asyncio
import json
import random
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import get_session, update_session
from app.models import APIResponse, ErrorDetail

router = APIRouter()


@router.post("/convert")
async def create_metadata(
    session_id: str,
    source: str = "Unknown Source",
    test_code: str = "TEST01",
    tagged_by: str = "AITAI",
    difficulty: str = "MEDIUM",
    is_pyq: bool = False,
    prev_exam: Optional[str] = None,
    pyq_year_from: Optional[int] = None,
    pyq_year_to: Optional[int] = None,
):
    """Create metadata_v2 files from cleaned output."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage4_metadata import MetadataMapperStage
    stage = MetadataMapperStage(session_id, session_dir)

    result = await asyncio.to_thread(
        stage.execute,
        source=source,
        test_code=test_code,
        tagged_by=tagged_by,
        difficulty=difficulty,
        is_pyq=is_pyq,
        prev_exam=prev_exam,
        pyq_year_from=pyq_year_from,
        pyq_year_to=pyq_year_to,
    )

    if "error" not in result:
        update_session(session_id, current_stage="stage4_metadata")
        from app.db import upsert_checkpoint
        upsert_checkpoint(session_id, "stage4_metadata", "completed", {
            "total": result.get("total", 0),
            "triplets": result.get("triplets", 0),
            "non_triplets_quarantined": result.get("non_triplets_quarantined", 0),
        })

    return APIResponse(data=result)


@router.get("/preview")
async def preview_metadata(session_id: str, count: int = 7):
    """Preview random metadata_v2 JSON entries."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id
    meta_dir = session_dir / "metadata_v2"

    all_entries = []
    for f in sorted(meta_dir.glob("*_metadata_v2.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    all_entries.extend(data)
        except Exception:
            continue

    if not all_entries:
        return APIResponse(data={"samples": [], "total": 0})

    samples = random.sample(all_entries, min(count, len(all_entries)))
    return APIResponse(data={"samples": samples, "total": len(all_entries)})
