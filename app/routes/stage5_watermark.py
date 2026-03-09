"""Stage 5: Watermark Removal — routes with image preview."""

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
    """Scan session for images and return count/stats.

    If no images found in images_original/, attempt auto-discovery from
    alternative directories (mathpix_output/images/, test data, etc.).
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage5_watermark import WatermarkStage
    stage = WatermarkStage(session_id, session_dir)
    result = stage.scan_images()

    # If scan found 0 images, try auto-discovery from alternative locations
    if result.get("total_images", 0) == 0:
        import shutil
        from app.db import insert_log
        from app.sse import sse_manager

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
            # Re-scan after discovery
            result = stage.scan_images()

    return APIResponse(data=result)


@router.get("/preview")
async def preview_samples(session_id: str, count: int = 7, cleaned: bool = False):
    """Get random image samples as base64 for preview.

    cleaned=false: show originals only
    cleaned=true: show before/after pairs
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage5_watermark import WatermarkStage
    stage = WatermarkStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.preview_samples, count=count, cleaned=cleaned)
    return APIResponse(data=result)


@router.post("/process-single")
async def process_single(session_id: str, filename: str):
    """Process a single image and return before/after."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage5_watermark import WatermarkStage
    stage = WatermarkStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.process_single, filename)
    return APIResponse(data=result)


@router.post("/apply")
async def apply_watermark_removal(session_id: str):
    """Batch process all images for watermark removal."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage5_watermark import WatermarkStage
    stage = WatermarkStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.execute)

    if "error" not in result:
        update_session(session_id, current_stage="stage5_watermark")
        from app.db import upsert_checkpoint
        upsert_checkpoint(session_id, "stage5_watermark", "completed", {
            "total": result.get("total", 0),
            "processed": result.get("processed", 0),
            "failed": result.get("failed", 0),
        })

    return APIResponse(data=result)


@router.post("/skip")
async def skip_watermark(session_id: str):
    """Skip watermark removal (no watermark issues in images)."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    # Use stage's _get_image_dir() to benefit from fallback logic
    from app.pipeline.stage5_watermark import WatermarkStage
    stage = WatermarkStage(session_id, session_dir)
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
    upsert_checkpoint(session_id, "stage5_watermark", "completed", {"skipped": True, "count": count})

    return APIResponse(data={"skipped": True, "count": count})


# ── SMILES detection + rendering ──────────────────────────────────────

@router.get("/smiles/scan")
async def scan_smiles(session_id: str):
    """Scan session metadata for <smiles> tags.

    Called on page load to auto-detect SMILES presence and activate/deactivate
    the SMILES rendering section in the UI.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage5_watermark import WatermarkStage
    stage = WatermarkStage(session_id, session_dir)
    result = stage.scan_smiles()
    return APIResponse(data=result)


@router.post("/smiles/render")
async def render_smiles(session_id: str):
    """Render all <smiles> tags to SVG <img> tags in metadata.

    Replaces <smiles>CCO</smiles> with <img src="data:image/svg+xml;base64,...">.
    These base64 images will be uploaded to CDN in Stage 6 alongside regular images.
    """
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage5_watermark import WatermarkStage
    stage = WatermarkStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.render_smiles)
    return APIResponse(data=result)
