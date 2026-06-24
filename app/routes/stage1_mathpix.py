"""Stage 1: PDF Upload + MathPIX OCR routes."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import get_session, update_session, upsert_checkpoint
from app.models import APIResponse, ErrorDetail
from app.utils import now_iso

router = APIRouter()


@router.post("/upload")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    """Upload PDF to session input directory."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id
    input_dir = session_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    file_path = input_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    update_session(session_id, pdf_filename=file.filename)

    from app.db import insert_log
    from app.sse import sse_manager
    insert_log(session_id, "stage1_upload", "success", f"Uploaded {file.filename} ({len(content):,} bytes)")
    sse_manager.emit(session_id, {"type": "log", "level": "success", "message": f"Uploaded {file.filename}"})

    return APIResponse(data={"filename": file.filename, "size": len(content)})


@router.post("/start-ocr")
async def start_ocr(session_id: str):
    """Submit uploaded PDF to MathPIX for OCR processing."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id

    # Find the uploaded PDF
    input_dir = session_dir / "input"
    pdf_files = list(input_dir.glob("*.pdf")) if input_dir.exists() else []
    if not pdf_files:
        return APIResponse(data={"error": "No PDF found. Upload a PDF first."})

    pdf_path = pdf_files[0]

    from app.db import insert_log
    from app.sse import sse_manager

    try:
        from app.pipeline.stage1_mathpix import MathPIXStage
        stage = MathPIXStage(session_id, session_dir)
        result = await asyncio.to_thread(stage.upload_pdf, str(pdf_path))
    except Exception as e:
        err_msg = f"MathPIX OCR error: {e}"
        insert_log(session_id, "stage1_upload", "error", err_msg)
        sse_manager.emit(session_id, {"type": "log", "level": "error", "message": err_msg})
        return APIResponse(data={"error": err_msg})

    if "error" in result:
        insert_log(session_id, "stage1_upload", "error", result["error"])
        sse_manager.emit(session_id, {"type": "log", "level": "error", "message": result["error"]})
    else:
        msg = f"MathPIX OCR started — pdf_id: {result.get('pdf_id', '?')}"
        insert_log(session_id, "stage1_upload", "success", msg)
        sse_manager.emit(session_id, {"type": "log", "level": "success", "message": msg})
        upsert_checkpoint(session_id, "stage1_upload", "running", {"pdf_id": result.get("pdf_id")})

    return APIResponse(data=result)


@router.get("/poll-ocr")
async def poll_ocr(session_id: str):
    """Poll MathPIX for OCR processing status."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage1_mathpix import MathPIXStage
    stage = MathPIXStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.poll_status)

    from app.sse import sse_manager
    if result.get("status") == "completed":
        sse_manager.emit(session_id, {
            "type": "log", "level": "success",
            "message": f"MathPIX OCR complete — {result.get('num_pages', '?')} pages",
        })

    return APIResponse(data=result)


@router.post("/download-ocr")
async def download_ocr(session_id: str):
    """Download and extract MMD results from MathPIX."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage1_mathpix import MathPIXStage
    stage = MathPIXStage(session_id, session_dir)
    result = await asyncio.to_thread(stage.download_results)

    from app.db import insert_log, upsert_checkpoint
    from app.sse import sse_manager
    if "error" in result:
        insert_log(session_id, "stage1_upload", "error", result["error"])
        sse_manager.emit(session_id, {"type": "log", "level": "error", "message": result["error"]})
    else:
        msg = f"Downloaded: {result.get('mmd_filename', '?')} ({result.get('mmd_lines', 0)} lines), {result.get('images_count', 0)} images"
        insert_log(session_id, "stage1_upload", "success", msg)
        sse_manager.emit(session_id, {"type": "log", "level": "success", "message": msg})

        # Mark Stage 1 as completed so user can advance
        upsert_checkpoint(session_id, "stage1_upload", "completed", {
            "mmd_filename": result.get("mmd_filename"),
            "mmd_lines": result.get("mmd_lines", 0),
            "images_count": result.get("images_count", 0),
        })

        # Save pdf_filename to session_config for downstream test_code derivation
        import json
        config_path = session_dir / "session_config.json"
        try:
            config = {}
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
            # Read pdf_filename from mathpix_status
            status_path = session_dir / "mathpix_output" / "mathpix_status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                config["pdf_filename"] = status.get("pdf_filename", "")
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except Exception:
            pass

    return APIResponse(data=result)


@router.get("/credentials")
async def check_credentials(session_id: str):
    """Check MathPIX credential status."""
    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id

    from app.pipeline.stage1_mathpix import MathPIXStage
    stage = MathPIXStage(session_id, session_dir)
    result = stage.check_credentials()
    return APIResponse(data=result)


@router.get("/status")
async def mathpix_status(session_id: str):
    """Get Stage 1 status."""
    from app.db import get_checkpoint
    cp = get_checkpoint(session_id, "stage1_upload")
    return APIResponse(data=cp or {"status": "not_started"})


@router.get("/token-check")
async def token_safety_check(session_id: str):
    """Check MMD file token count against each LLM model's context window.

    Called after Stage 1 completes to determine which models are safe to use.
    Returns per-model safety info so Stage 2 can disable unsafe models in the dropdown.
    """
    from app.pipeline.stage2_llm_mapper import estimate_model_compatibility

    session_dir = Path(settings.sessions_dir) / session_id
    mmd_dir = session_dir / "mathpix_output"

    if not mmd_dir.exists():
        return APIResponse(data={"error": "No mathpix_output directory found"})

    # Find MMD file(s) — exclude part files for the main check
    mmd_files = sorted([
        f for f in mmd_dir.iterdir()
        if f.suffix == ".mmd" and f.is_file() and not f.name.startswith("part")
    ])
    if not mmd_files:
        return APIResponse(data={"error": "No MMD files found"})

    # Read all MMD content (may be multiple files that get concatenated in Stage 2)
    total_text = ""
    for f in mmd_files:
        total_text += f.read_text(encoding="utf-8")

    result = estimate_model_compatibility(total_text)
    result["mmd_files"] = len(mmd_files)
    result["total_chars"] = len(total_text)
    result["total_lines"] = total_text.count("\n") + 1

    # Check if split parts already exist
    part_files = sorted([f for f in mmd_dir.iterdir() if f.name.startswith("part") and f.suffix == ".mmd"])
    if part_files:
        result["has_parts"] = True
        result["part_count"] = len(part_files)
        result["part_info"] = []
        for pf in part_files:
            ptext = pf.read_text(encoding="utf-8")
            pcompat = estimate_model_compatibility(ptext)
            result["part_info"].append({
                "name": pf.name,
                "lines": ptext.count("\n") + 1,
                "gemini_safe": pcompat.get("gemini", {}).get("safe", False),
                "anthropic_safe": pcompat.get("anthropic", {}).get("safe", False),
            })

    return APIResponse(data=result)


@router.post("/auto-split")
async def auto_split_mmd(session_id: str, num_parts: int = 2):
    """Auto-split the MMD file into N parts at paragraph boundaries.

    Used when the document is too large for all models.
    """
    session = get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content=APIResponse(
            success=False, error=ErrorDetail(code="NOT_FOUND", message="Session not found"),
        ).model_dump())

    session_dir = Path(settings.sessions_dir) / session_id
    mmd_dir = session_dir / "mathpix_output"

    if not mmd_dir.exists():
        return APIResponse(data={"error": "No mathpix_output directory found"})

    # Find main MMD file (exclude existing parts)
    mmd_files = sorted([
        f for f in mmd_dir.iterdir()
        if f.suffix == ".mmd" and f.is_file() and not f.name.startswith("part")
    ])
    if not mmd_files:
        return APIResponse(data={"error": "No MMD files found"})

    full_text = ""
    for f in mmd_files:
        full_text += f.read_text(encoding="utf-8")

    lines = full_text.splitlines(keepends=True)
    total_lines = len(lines)

    if total_lines < 100:
        return APIResponse(data={"error": "Document too small to split"})

    # Split at paragraph boundaries (blank lines) near target positions
    target_size = total_lines // num_parts
    parts = []
    current_start = 0

    for part_idx in range(num_parts - 1):
        target_end = target_size * (part_idx + 1)
        # Search for a blank line within ±50 lines of target
        best_split = target_end
        for offset in range(0, 80):
            for candidate in [target_end + offset, target_end - offset]:
                if 0 < candidate < total_lines and lines[candidate].strip() == "":
                    best_split = candidate + 1  # split AFTER the blank line
                    break
            else:
                continue
            break

        parts.append(lines[current_start:best_split])
        current_start = best_split

    # Last part gets the remainder
    parts.append(lines[current_start:])

    # Write part files
    from app.pipeline.stage2_llm_mapper import estimate_model_compatibility
    from app.db import insert_log

    part_info = []
    for idx, part_lines in enumerate(parts, start=1):
        part_text = "".join(part_lines)
        part_path = mmd_dir / f"part{idx}_combined.mmd"
        part_path.write_text(part_text, encoding="utf-8")

        compat = estimate_model_compatibility(part_text)
        info = {
            "name": part_path.name,
            "lines": len(part_lines),
            "gemini_safe": compat.get("gemini", {}).get("safe", False),
            "anthropic_safe": compat.get("anthropic", {}).get("safe", False),
        }
        part_info.append(info)
        insert_log(session_id, "stage1_upload", "info",
                    f"Part {idx}: {len(part_lines)} lines — Gemini: {'OK' if info['gemini_safe'] else 'EXCEEDS'}, Claude: {'OK' if info['anthropic_safe'] else 'EXCEEDS'}")

    insert_log(session_id, "stage1_upload", "success",
               f"Split document into {num_parts} parts ({total_lines} total lines)")

    return APIResponse(data={
        "parts": num_parts,
        "total_lines": total_lines,
        "part_info": part_info,
    })
