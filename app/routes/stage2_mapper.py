"""Stage 2: LLM Mapper — routes with actual pipeline adapter.

Extraction runs as a background asyncio task so the HTTP response returns
immediately (prevents Render/proxy timeout on long Gemini calls).
Frontend polls /progress for status updates.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import get_session, update_session
from app.models import APIResponse, ErrorDetail

router = APIRouter()

# Track running stage instances for stop support (keyed by session_id)
_running_stages: Dict[str, "LLMMapperStage"] = {}
# Track background extraction tasks
_bg_tasks: Dict[str, asyncio.Task] = {}


def _get_session_or_404(session_id: str):
    session = get_session(session_id)
    if session is None:
        return None, JSONResponse(
            status_code=404,
            content=APIResponse(
                success=False,
                error=ErrorDetail(code="NOT_FOUND", message=f"Session {session_id} not found"),
            ).model_dump(),
        )
    return session, None


@router.post("/test-api")
async def test_api(session_id: str, provider: str = "gemini"):
    """Test Gemini/Anthropic API key connectivity."""
    try:
        if provider == "anthropic":
            if not settings.anthropic_api_key:
                return APIResponse(data={"status": "error", "message": "Anthropic API key not set in .env"})
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=50,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            )
            return APIResponse(data={"status": "ok", "message": f"Anthropic connected ({response.model})"})
        else:
            if not settings.gemini_api_key:
                return APIResponse(data={"status": "error", "message": "Gemini API key not set in .env"})
            from google import genai
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents="Reply with exactly: OK",
            )
            return APIResponse(data={"status": "ok", "message": "Gemini connected"})
    except Exception as e:
        return APIResponse(data={"status": "error", "message": f"Connection failed: {str(e)[:200]}"})


@router.post("/start")
async def start_mapping(
    session_id: str,
    max_questions: int = 30,
    llm_provider: str = "gemini",
    is_pyq: bool = False,
    prev_exam: Optional[str] = None,
    pyq_year_from: Optional[int] = None,
    pyq_year_to: Optional[int] = None,
    extra_context: Optional[str] = None,
    resume: bool = False,
    part: Optional[int] = None,
):
    """Start LLM-based question extraction (Gemini or Anthropic)."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id

    # Save config to session_config.json so downstream stages can read it
    config_path = session_dir / "session_config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    config["is_pyq"] = is_pyq
    config["prev_exam"] = prev_exam or ""
    config["pyq_year_from"] = pyq_year_from
    config["pyq_year_to"] = pyq_year_to
    config["extra_context"] = extra_context or ""
    config["llm_provider"] = llm_provider
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # If already running for this session, reject
    if session_id in _bg_tasks and not _bg_tasks[session_id].done():
        return APIResponse(data={"error": "Extraction already running for this session"})

    from app.pipeline.stage2_llm_mapper import LLMMapperStage
    stage = LLMMapperStage(session_id, session_dir)
    _running_stages[session_id] = stage

    # Mark as running in checkpoint so frontend knows
    from app.db import upsert_checkpoint
    upsert_checkpoint(session_id, "stage2_mapping", "running", {
        "provider": llm_provider, "max_questions": max_questions,
    })

    # Background runner — runs extraction, saves checkpoint on completion
    async def _run_extraction():
        try:
            result = await asyncio.to_thread(
                stage.execute,
                max_questions=max_questions,
                llm_provider=llm_provider,
                is_pyq=is_pyq,
                prev_exam=prev_exam or "",
                pyq_year_from=pyq_year_from,
                pyq_year_to=pyq_year_to,
                extra_context=extra_context or "",
                resume=resume,
                part=part,
            )
        except Exception as e:
            from app.db import upsert_checkpoint as _ucp, insert_log
            _ucp(session_id, "stage2_mapping", "failed", {"error": str(e)})
            insert_log(session_id, "stage2_mapping", "error", f"Extraction failed: {e}")
            return

        # Process result BEFORE cleanup (no finally block that races)
        if "error" not in result:
            update_session(session_id, current_stage="stage2_mapping")
            from app.db import upsert_checkpoint as _ucp, get_checkpoint

            cp_data = {
                "questions": result.get("questions", 0),
                "triplets": result.get("triplets", 0),
                "provider": result.get("provider", llm_provider),
                "by_subject": result.get("by_subject", {}),
            }

            if part:
                existing_cp = get_checkpoint(session_id, "stage2_mapping") or {}
                existing_data = existing_cp.get("data", {}) if isinstance(existing_cp, dict) else {}
                parts_done = existing_data.get("parts_done", {})
                parts_done[str(part)] = True
                cp_data["part"] = part
                cp_data["parts_done"] = parts_done
                if existing_data.get("questions"):
                    cp_data["questions"] += existing_data["questions"]
                    cp_data["triplets"] += existing_data.get("triplets", 0)
                    for subj, count in existing_data.get("by_subject", {}).items():
                        cp_data["by_subject"][subj] = cp_data["by_subject"].get(subj, 0) + count
                all_done = len(parts_done) >= 2
                status = "completed" if all_done else "partial"
            else:
                status = "completed"

            _ucp(session_id, "stage2_mapping", status, cp_data)
        else:
            from app.db import upsert_checkpoint as _ucp
            _ucp(session_id, "stage2_mapping", "failed", result)

    def _task_done_callback(task: asyncio.Task):
        """Cleanup tracking dicts AFTER task is truly finished."""
        _running_stages.pop(session_id, None)
        _bg_tasks.pop(session_id, None)
        # Log any unexpected exception
        try:
            exc = task.exception()
            if exc:
                from app.db import insert_log
                insert_log(session_id, "stage2_mapping", "error",
                           f"Background task crashed: {exc}")
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            pass

    # Spawn background task — returns immediately
    task = asyncio.create_task(_run_extraction())
    task.add_done_callback(_task_done_callback)
    _bg_tasks[session_id] = task

    return APIResponse(data={"status": "started", "message": "Extraction started in background. Poll /progress for updates."})


@router.get("/progress")
async def mapping_progress(session_id: str):
    """Get mapping progress from checkpoint + live task state."""
    from app.db import get_checkpoint
    import json as _json
    cp = get_checkpoint(session_id, "stage2_mapping")
    if cp and isinstance(cp.get("data"), str):
        try:
            cp["data"] = _json.loads(cp["data"])
        except Exception:
            cp["data"] = {}
    result = cp or {"status": "not_started"}

    # Overlay live task state
    is_running = session_id in _bg_tasks and not _bg_tasks[session_id].done()
    if is_running:
        result["bg_running"] = True
    elif session_id in _bg_tasks:
        # Task finished but not cleaned up yet
        result["bg_running"] = False
        _bg_tasks.pop(session_id, None)

    return APIResponse(data=result)


@router.get("/preview")
async def preview_samples(session_id: str, count: int = 7):
    """Get random sample questions from mapped output."""
    import json, random

    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id
    mapped_dir = session_dir / "mapped_output"

    all_questions = []

    # First try partial_output.json (available during/after extraction)
    partial_path = mapped_dir / "partial_output.json"
    if partial_path.exists():
        try:
            with open(partial_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    all_questions.extend(data)
        except Exception:
            pass

    # Then try per-subject files
    if not all_questions:
        for f in sorted(mapped_dir.glob("*.json")):
            if f.name == "partial_output.json":
                continue
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        all_questions.extend(data)
                    elif isinstance(data, dict) and "questions" in data:
                        all_questions.extend(data["questions"])
            except Exception:
                continue

    if not all_questions:
        return APIResponse(data={"samples": [], "total": 0})

    samples = random.sample(all_questions, min(count, len(all_questions)))
    return APIResponse(data={"samples": samples, "total": len(all_questions)})


@router.get("/config")
async def get_session_config(session_id: str):
    """Get saved session configuration (PYQ settings, extra context, etc.)."""
    session, err = _get_session_or_404(session_id)
    if err:
        return err

    session_dir = Path(settings.sessions_dir) / session_id
    config_path = session_dir / "session_config.json"

    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return APIResponse(data=config)


@router.post("/stop")
async def stop_mapping(session_id: str):
    """Graceful stop of mapping — sets stop flag so current batch finishes then loop exits."""
    stage = _running_stages.get(session_id)
    if stage:
        stage.request_stop()
        return APIResponse(data={"status": "stopping", "message": "Stop requested — current batch will complete then save."})
    return APIResponse(data={"status": "not_running", "message": "No extraction running for this session."})
