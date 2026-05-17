"""Stage 11 nightly + on-spot routes.

Endpoints:
  GET    /admin/stage11/status                          -- is a Stage 11 run currently active?
  GET    /admin/stage11/runs/{date}                     -- read run report (YYYY-MM-DD IST)
  GET    /admin/stage11/history                         -- read history.json (full ledger)
  POST   /admin/stage11/trigger?date=YYYY-MM-DD&force=true  -- manual nightly trigger
  POST   /admin/stage11/kill                            -- create killswitch flag
  DELETE /admin/stage11/kill                            -- clear killswitch flag
  GET    /api/dashboard/stage11/last-night              -- last-night summary card
  POST   /api/sessions/{sid}/stage11/on-spot            -- operator one-off run (Admin only)
  POST   /api/sessions/{sid}/stage11/preflight          -- popup pre-flight (cohort + memory)
  GET    /api/sessions/{sid}/stage11/on-spot/status     -- poll session run state

All endpoints under /admin/stage11/* are auth-protected per the global
middleware. The on-spot endpoints additionally enforce role=Admin|Architect
per the 2026-05-17 user spec ("admins should only have the option").
"""
from __future__ import annotations
import csv
import json
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="", tags=["stage11"])

ADMIN_ROLES = ("Admin", "Architect")


def _require_admin(request: Request) -> dict:
    """Returns the user dict on success; raises 403 HTTPException otherwise."""
    user = getattr(request.state, "user", {}) or {}
    if user.get("role") not in ADMIN_ROLES:
        raise HTTPException(403, "Admin only")
    return user


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


# ── Dashboard: last-night summary card ────────────────────────────────
@router.get("/api/dashboard/stage11/last-night")
async def dashboard_last_night(request: Request,
                                 days_back: int = Query(default=3, ge=1, le=14)):
    """Return the most recent Stage 11 run report (admin sees full per-op
    breakdown; team members see their own slice only).

    Searches `stage11_runs/{date}.json` for the last `days_back` IST dates
    in reverse; returns the first one found. Returns 200 with `found=False`
    if no run report exists yet.
    """
    s = _scheduler()
    user = getattr(request.state, "user", {}) or {}
    role = user.get("role", "Team")
    phone = user.get("phone", "")

    IST = timezone(timedelta(hours=5, minutes=30))
    today_ist = datetime.now(IST).date()

    runs_dir = s.runs_dir()
    report = None
    report_date = None
    for offset in range(0, days_back + 1):
        d = today_ist - timedelta(days=offset)
        p = runs_dir / f"{d.isoformat()}.json"
        if not p.exists():
            continue
        try:
            report = json.loads(p.read_text(encoding="utf-8"))
            report_date = d.isoformat()
            break
        except Exception:
            continue

    if report is None:
        return {
            "found": False,
            "reason": (f"No Stage 11 run report in last {days_back} day(s); "
                        f"nightly fires at 00:30 IST starting "
                        f"{getattr(s.settings, 'stage11_floor_date', '2026-05-18')}"),
        }

    promoted = int(report.get("promoted", 0) or 0)
    flagged = int(report.get("flagged", 0) or 0)
    failed_retry = int(report.get("failed_retry", 0) or 0)
    cohort_verified = int(report.get("cohort_verified", 0) or 0)
    per_op = report.get("per_operator", {}) or {}

    # Personal slice: only show this user's per-op row (admin sees full map)
    if role in ADMIN_ROLES:
        per_operator_visible = per_op
        my_slice = per_op.get(phone, None)
    else:
        per_operator_visible = None
        my_slice = per_op.get(phone, None)

    return {
        "found": True,
        "report_date_ist": report_date,
        "status": report.get("status", "unknown"),
        "dryrun": bool(report.get("dryrun", False)),
        "cohort_verified": cohort_verified,
        "promoted": promoted,
        "flagged": flagged,
        "failed_retry": failed_retry,
        "started_at_utc": report.get("started_at_utc"),
        "ended_at_utc": report.get("ended_at_utc"),
        "is_admin": role in ADMIN_ROLES,
        "per_operator": per_operator_visible,
        "my_slice": my_slice,
    }


# ── Operator: pre-flight check (popup display) ────────────────────────
@router.post("/api/sessions/{session_id}/stage11/preflight")
async def operator_preflight(request: Request, session_id: str):
    """Pre-flight for the popup: cohort size + memory check.
    Admin-only (matches the on-spot route). Does NOT start any work.
    Returns: {cohort_size, memory_ok, memory_free_mb, memory_required_mb,
              any_stage11_running, can_proceed, blockers: [...]}.
    """
    user = _require_admin(request)
    s = _scheduler()

    blockers = []

    # Memory check
    mem_check = s.check_memory_available()
    if mem_check is not None:
        blockers.append(mem_check)
        memory_ok = False
        memory_free_mb = int(mem_check.get("free_mb", 0))
        memory_required_mb = int(mem_check.get("required_mb",
                                                  s.STAGE11_MIN_ANON_MB))
    else:
        memory_ok = True
        memory_free_mb = None  # cgroup unavailable (Windows dev) -- not blocking
        memory_required_mb = int(s.STAGE11_MIN_ANON_MB)

    # Concurrency check
    any_running = s.is_running()
    if any_running:
        blockers.append({"reason": "Another Stage 11 run is in progress"})

    # Killswitch check
    if s.killswitch_active():
        blockers.append({"reason": "Killswitch flag is active "
                                     "(admin must clear before any run)"})

    # Cohort size: count err=2 in this session's combined_upload_log.csv
    # minus any OIDs already promoted/flagged in history.
    from app.config import settings as _cfg
    session_dir = Path(_cfg.sessions_dir) / session_id
    cohort_size = 0
    csv_missing = True
    log_path = session_dir / "upload_logs" / "combined_upload_log.csv"
    if not log_path.exists():
        log_path = session_dir / "combined_upload_log.csv"
    if log_path.exists():
        csv_missing = False
        history = s.load_history()
        try:
            with open(log_path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    qid = (r.get("UploadedQuestionID") or "").strip()
                    err = (r.get("Error Level") or "").strip()
                    if not qid or len(qid) != 24 or err != "2":
                        continue
                    h = history.get(qid)
                    if h and h.get("status") in ("promoted", "flagged"):
                        continue
                    cohort_size += 1
        except Exception as e:
            blockers.append({"reason": f"CSV read failed: {e}"})
    else:
        blockers.append({"reason": "combined_upload_log.csv not found "
                                     "(has this session uploaded yet?)"})

    if cohort_size == 0 and not csv_missing:
        blockers.append({"reason": "No err=2 OIDs left to promote in this "
                                     "session"})

    return {
        "session_id": session_id,
        "operator": user.get("phone", ""),
        "cohort_size": cohort_size,
        "memory_ok": memory_ok,
        "memory_free_mb": memory_free_mb,
        "memory_required_mb": memory_required_mb,
        "any_stage11_running": any_running,
        "killswitch_active": s.killswitch_active(),
        "can_proceed": len(blockers) == 0,
        "blockers": blockers,
    }


# ── Operator: on-spot trigger ─────────────────────────────────────────
@router.post("/api/sessions/{session_id}/stage11/on-spot")
async def operator_on_spot(request: Request, session_id: str):
    """Operator clicks "Run Stage 11" button on Stage 10 UI. Admin-only.
    Runs Stage 11 for ONLY this session's err=2 OIDs. Refuses if any other
    Stage 11 run (nightly or on-spot) is active. Marks promoted OIDs with
    on_spot=True in history so the nightly skips them.

    Per 2026-05-17 spec, this is an Admin-only feature ("admins should
    only have the option in the dashboard to open the stage eleven gate").
    """
    user = _require_admin(request)
    s = _scheduler()

    # Operator identity: prefer the LOGGED-IN admin's phone (so the
    # on_spot_by attribution reflects who actually clicked the button).
    # Falls back to session.created_by if the user dict is empty.
    operator = user.get("phone", "")
    if not operator:
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
        raise HTTPException(409, "Another Stage 11 run is in progress; "
                                  "try again shortly")

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
