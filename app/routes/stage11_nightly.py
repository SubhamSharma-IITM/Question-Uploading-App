"""Stage 11 nightly + on-spot routes.

Endpoints:
  GET    /admin/stage11/status                       -- is a Stage 11 run currently active?
  GET    /admin/stage11/runs/{date}                  -- read run report (YYYY-MM-DD IST)
  GET    /admin/stage11/history                      -- read history.json (full ledger)
  POST   /admin/stage11/trigger?date=YYYY-MM-DD&force=true  -- manual nightly trigger
  POST   /admin/stage11/kill                         -- create killswitch flag
  DELETE /admin/stage11/kill                         -- clear killswitch flag
  POST   /api/sessions/{sid}/stage11/on-spot         -- operator one-off run
  GET    /api/sessions/{sid}/stage11/on-spot/status  -- poll session run state

All endpoints under /admin/stage11/* are auth-protected per the global
middleware. The on-spot endpoints follow the same session-auth pattern as
other stage routes.
"""
from __future__ import annotations
import json
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="", tags=["stage11"])


def _scheduler():
    from app.pipeline.stage11_nightly import get_scheduler
    return get_scheduler()


# ── Admin: status + reports + history ─────────────────────────────────
@router.get("/admin/stage11/status")
async def admin_status():
    s = _scheduler()
    return {
        "running": s.is_running(),
        "nightly_enabled": bool(getattr(s.settings, "stage11_nightly_enabled", False)),
        "dryrun": bool(getattr(s.settings, "stage11_dryrun", False)),
        "floor_date": getattr(s.settings, "stage11_floor_date", "2026-05-18"),
        "killswitch_active": s.killswitch_active(),
        "next_trigger_utc": s.compute_next_trigger_utc().isoformat(),
    }


@router.get("/admin/stage11/runs/{ist_date}")
async def admin_get_run(ist_date: str):
    s = _scheduler()
    try:
        d = _date.fromisoformat(ist_date)
    except ValueError:
        raise HTTPException(400, f"date must be YYYY-MM-DD; got {ist_date!r}")
    path = s.runs_dir() / f"{d.isoformat()}.json"
    if not path.exists():
        return {"status": "not_found", "date": d.isoformat()}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"failed to read run report: {e}")


@router.get("/admin/stage11/history")
async def admin_history(limit: int = Query(default=200, ge=1, le=10000)):
    """Return last N entries from history.json (most recent first by last_attempt_ts)."""
    s = _scheduler()
    hist = s.load_history()
    # Sort by last_attempt_ts desc
    items = sorted(
        hist.items(),
        key=lambda kv: kv[1].get("last_attempt_ts", ""),
        reverse=True,
    )[:limit]
    return {
        "total_entries": len(hist),
        "returned": len(items),
        "entries": {qid: info for qid, info in items},
    }


# ── Admin: manual trigger ─────────────────────────────────────────────
@router.post("/admin/stage11/trigger")
async def admin_trigger(
    target_date: Optional[str] = Query(default=None, alias="date"),
    force: bool = Query(default=False),
):
    """Manually trigger a Stage 11 nightly run for `date` (IST).
    If date is omitted, uses previous IST day (same as nightly default).
    force=true re-runs even if {date}.json already exists.
    """
    s = _scheduler()
    if s.is_running():
        raise HTTPException(409, "Stage 11 run already in progress")
    if target_date:
        try:
            d = _date.fromisoformat(target_date)
        except ValueError:
            raise HTTPException(400, f"date must be YYYY-MM-DD; got {target_date!r}")
    else:
        d = None
    import asyncio
    result = await asyncio.to_thread(s.run_one_night, d, force)
    return result


@router.post("/admin/stage11/kill")
async def admin_killswitch_set():
    s = _scheduler()
    p = s.killswitch_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"killswitch set at {datetime.now(timezone.utc).isoformat()}\n",
                  encoding="utf-8")
    return {"status": "killswitch_set", "path": str(p)}


@router.delete("/admin/stage11/kill")
async def admin_killswitch_clear():
    s = _scheduler()
    p = s.killswitch_path()
    if p.exists():
        p.unlink()
        return {"status": "killswitch_cleared", "path": str(p)}
    return {"status": "no_killswitch", "path": str(p)}


# ── Operator: on-spot trigger ─────────────────────────────────────────
@router.post("/api/sessions/{session_id}/stage11/on-spot")
async def operator_on_spot(session_id: str):
    """Operator clicks "Run Stage 11" button on Stage 10 UI.
    Runs Stage 11 for ONLY this session's err=2 OIDs. Refuses if any other
    Stage 11 run (nightly or on-spot) is active. Marks promoted OIDs with
    on_spot=True in history so the nightly skips them.
    """
    s = _scheduler()

    # Look up operator from session record (DB)
    try:
        from app.db import get_session
        sess = get_session(session_id)
        if not sess:
            raise HTTPException(404, f"session {session_id} not found")
        operator = sess.get("created_by", "") or ""
    except HTTPException:
        raise
    except Exception:
        operator = ""

    if s.is_running():
        raise HTTPException(409, "Another Stage 11 run is in progress; try again shortly")

    import asyncio
    result = await asyncio.to_thread(s.run_on_spot, session_id, operator)
    if result.get("status") == "already_running":
        raise HTTPException(409, result.get("reason", "already_running"))
    return result


@router.get("/api/sessions/{session_id}/stage11/on-spot/status")
async def operator_on_spot_status(session_id: str):
    """Returns whether a Stage 11 run is currently active (any session) and
    this session's history summary (counts of promoted/flagged/failed_retry).
    """
    s = _scheduler()
    hist = s.load_history()
    promoted = flagged = failed_retry = 0
    on_spot = 0
    for qid, info in hist.items():
        if info.get("on_spot_session") != session_id:
            continue
        on_spot += 1
        status = info.get("status", "")
        if status == "promoted": promoted += 1
        elif status == "flagged": flagged += 1
        else: failed_retry += 1
    return {
        "any_stage11_running": s.is_running(),
        "session_id": session_id,
        "on_spot_count": on_spot,
        "promoted": promoted,
        "flagged": flagged,
        "failed_retry": failed_retry,
    }
