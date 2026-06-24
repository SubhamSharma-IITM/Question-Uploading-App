"""Stage 5: Watermark Removal — routes with chroma detection + preview pane.

V3 flow:
1. /scan — discover images
2. /apply — chroma detection + Fix1 pipeline (batch)
3. /preview_images — paginated preview with chroma labels (GET, offset/limit)
4. /quarantine_images — bulk quarantine from user review (POST)
5. /smiles/scan + /smiles/render — SMILES rendering (only after quarantine done)
"""

import asyncio
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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


def _get_stage(session_id: str):
    from app.pipeline.stage5_watermark import WatermarkStage
    session_dir = Path(settings.sessions_dir) / session_id
    return WatermarkStage(session_id, session_dir)


@router.get("/scan")
async def scan_images(session_id: str):
    """Scan session for images and return count/stats.

    If no images found in images_original/, attempt auto-discovery from
    alternative directories (mathpix_output/images/, test data, etc.).
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    stage = _get_stage(session_id)
    result = stage.scan_images()

    # If scan found 0 images, try auto-discovery from alternative locations
    if result.get("total_images", 0) == 0:
        import shutil
        from app.db import insert_log
        from app.sse import sse_manager

        session_dir = Path(settings.sessions_dir) / session_id
        exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        img_dir = session_dir / "images_original"
        img_dir.mkdir(parents=True, exist_ok=True)
        discovered = 0

        # Try 1: mathpix_output/images/ (MathPIX OCR output)
        mathpix_images = session_dir / "mathpix_output" / "images"
        if mathpix_images.exists():
            for img in mathpix_images.iterdir():
                if img.suffix.lower() in exts:
                    shutil.copy2(img, img_dir / img.name)
                    discovered += 1

        # Try 2: mathpix_output/ root (some OCR configs put images here)
        if discovered == 0:
            mathpix_root = session_dir / "mathpix_output"
            if mathpix_root.exists():
                for img in mathpix_root.iterdir():
                    if img.is_file() and img.suffix.lower() in exts:
                        shutil.copy2(img, img_dir / img.name)
                        discovered += 1

        if discovered > 0:
            insert_log(session_id, "stage5_watermark", "info",
                       f"Auto-discovered {discovered} images from alternative directories")
            sse_manager.emit(session_id, {
                "type": "log", "level": "info",
                "message": f"Auto-discovered {discovered} images from alternative directories"
            })
            result = stage.scan_images()

    return APIResponse(data=result)


@router.get("/preview")
async def preview_samples(session_id: str, count: int = 7, cleaned: bool = False):
    """Legacy: random image samples as base64 for preview."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    stage = _get_stage(session_id)
    result = await asyncio.to_thread(stage.preview_samples, count=count, cleaned=cleaned)
    return APIResponse(data=result)


@router.get("/preview_images")
async def preview_images(session_id: str, offset: int = 0, limit: int = 20):
    """Paginated image preview with chroma detection labels.

    Returns per-image: original base64, cleaned base64 (if WM detected),
    chroma label, filename. Used by the Stage 5 preview pane.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    stage = _get_stage(session_id)
    result = await asyncio.to_thread(stage.get_preview_images, offset=offset, limit=limit)
    return APIResponse(data=result)


@router.get("/image_question_map")
async def image_question_map(session_id: str):
    """Map image filenames to question QIDs from metadata_v2.

    Returns {filename: qid} for all images referenced in metadata.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    stage = _get_stage(session_id)
    mapping = await asyncio.to_thread(stage.get_image_question_map)
    return APIResponse(data=mapping)


@router.post("/process-single")
async def process_single(session_id: str, filename: str):
    """Process a single image and return before/after."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    stage = _get_stage(session_id)
    result = await asyncio.to_thread(stage.process_single, filename)
    return APIResponse(data=result)


@router.post("/apply")
async def apply_watermark_removal(session_id: str):
    """Batch process all images with chroma detection gate.

    Does NOT mark stage as completed — user must review preview pane first.
    Checkpoint status set to 'in_progress' with watermark_step='completed'.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    stage = _get_stage(session_id)
    result = await asyncio.to_thread(stage.execute)

    if "error" not in result:
        update_session(session_id, current_stage="stage5_watermark")

    return APIResponse(data=result)


class QuarantineRequest(BaseModel):
    qids: List[str]


@router.post("/quarantine_images")
async def quarantine_images(session_id: str, body: QuarantineRequest):
    """Bulk quarantine questions after user image review.

    Receives list of QIDs to quarantine. Removes them from metadata_v2
    and records in quarantine DB. Called by "Quarantine Unselected Images" button.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    stage = _get_stage(session_id)
    result = await asyncio.to_thread(stage.quarantine_images, body.qids)

    # Update sidebar quarantine count
    from app.sse import sse_manager
    sse_manager.emit(session_id, {
        "type": "log", "level": "info",
        "message": f"Quarantined {result['quarantined']} questions from image review"
    })

    return APIResponse(data=result)


@router.post("/confirm_review")
async def confirm_review(session_id: str):
    """Confirm image review is complete (even if 0 quarantined).

    Called when user finishes reviewing images but doesn't quarantine any.
    Marks quarantine_step as completed. Also runs inline SMILES check --
    if no SMILES tags exist, immediately marks checkpoint as completed
    so the advance gate passes without needing a separate scan call.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    stage = _get_stage(session_id)
    cp_data = stage._load_checkpoint_data()
    if cp_data.get("quarantine_step") != "completed":
        cp_data["quarantine_step"] = "completed"
        cp_data["quarantined_qids"] = []
        cp_data["quarantined_count"] = 0
        # Save FIRST so scan_smiles sees quarantine_step=completed
        stage.save_checkpoint("in_progress", cp_data)
        stage.log("info", "Image review confirmed -- no quarantines")

    # Inline SMILES check: scan_smiles auto-completes if no SMILES
    smiles_result = stage.scan_smiles()
    if smiles_result.get("has_smiles", False):
        stage.log("info", "SMILES tags found -- render needed before advance")

    return APIResponse(data={"confirmed": True})


@router.post("/skip")
async def skip_watermark(session_id: str):
    """Skip watermark removal entirely (no watermark issues in images)."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id
    stage = _get_stage(session_id)
    img_dir = stage._get_image_dir()
    out_dir = session_dir / "images_cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    count = 0
    if img_dir.exists():
        for f in img_dir.iterdir():
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                shutil.copy2(f, out_dir / f.name)
                count += 1

    from app.db import insert_log, upsert_checkpoint
    from app.sse import sse_manager
    insert_log(session_id, "stage5_watermark", "info", f"Skipped watermark removal - {count} images copied as-is")
    sse_manager.emit(session_id, {"type": "log", "level": "info", "message": f"Skipped - {count} images copied"})
    upsert_checkpoint(session_id, "stage5_watermark", "completed", {
        "skipped": True, "count": count,
        "watermark_step": "completed",
        "quarantine_step": "completed",
    })

    return APIResponse(data={"skipped": True, "count": count})


# ── SMILES detection + rendering ──────────────────────────────────────

@router.get("/smiles/scan")
async def scan_smiles(session_id: str):
    """Scan session metadata for <smiles> tags."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    stage = _get_stage(session_id)
    result = stage.scan_smiles()
    return APIResponse(data=result)


@router.post("/smiles/render")
async def render_smiles(session_id: str):
    """Render all <smiles> tags to SVG <img> tags in metadata."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    stage = _get_stage(session_id)
    result = await asyncio.to_thread(stage.render_smiles)
    return APIResponse(data=result)
