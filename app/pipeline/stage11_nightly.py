"""Stage 11 nightly scheduler.

Fires at 00:30 IST each day. Cohort selection:
  - Previous IST day's err=2 uploads (from data/sessions/*/combined_upload_log.csv)
  - Plus any prior-day OIDs in history with status `failed_retry`
  - Floor: STAGE11_FLOOR_DATE (default 2026-05-18). Older err=2 NOT swept.
  - Excluded: OIDs whose history.status is `promoted` or `flagged`

Live-multiget verification before processing: drops any OID whose live
error != 2 (handles operator manual fixes between upload and tonight).

Per-OID outcomes -> history.json status:
  promoted     : err=2 -> err=1 PUT succeeded (excluded from future sweeps)
  flagged      : LLM judged "cannot fix" (excluded; faculty review)
  failed_retry : OpenAI/network/auth error or killswitch hit; retried tomorrow

Trust Stage 8 (post-2026-05-17 fix bundle); no promote-time dedup gate.

Resilience:
  - Strict 00:30 IST schedule; missed nights roll into next as failed_retry
  - Per-OID backups (chmod 444) by stage11_correction.process_batch
  - Per-chunk killswitch check (file: data/stage11_killswitch.flag)
  - Idempotent: skip if data/stage11_runs/{date}.json already exists
    (unless force=True via manual trigger)
"""
from __future__ import annotations
import asyncio
import csv
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


class Stage11NightlyScheduler:
    """Owns the asyncio sleeper loop + the run executor."""

    def __init__(self, settings_obj):
        self.settings = settings_obj
        self._stop = asyncio.Event()
        self._run_lock = threading.Lock()  # guards concurrent run_one_night calls

    # ── Paths ────────────────────────────────────────────────────────
    def history_path(self) -> Path:
        return Path(self.settings.data_dir) / "stage11_history.json"

    def runs_dir(self) -> Path:
        return Path(self.settings.data_dir) / "stage11_runs"

    def backups_dir(self) -> Path:
        return Path(self.settings.data_dir) / "stage11_backups"

    def killswitch_path(self) -> Path:
        return Path(self.settings.data_dir) / "stage11_killswitch.flag"

    # ── History I/O (atomic) ─────────────────────────────────────────
    def load_history(self) -> Dict[str, Any]:
        p = self.history_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"history.json corrupt: {e}; starting fresh")
            return {}

    def save_history(self, history: dict):
        p = self.history_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    def _atomic_write_json(self, data: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, path)

    # ── Scheduling ───────────────────────────────────────────────────
    def compute_next_trigger_utc(self) -> datetime:
        h = int(getattr(self.settings, "stage11_trigger_hour_ist", 0))
        m = int(getattr(self.settings, "stage11_trigger_min_ist", 30))
        now_ist = datetime.now(IST)
        target_ist = now_ist.replace(hour=h, minute=m, second=0, microsecond=0)
        if target_ist <= now_ist:
            target_ist += timedelta(days=1)
        return target_ist.astimezone(timezone.utc)

    # ── Killswitch ───────────────────────────────────────────────────
    def killswitch_active(self) -> bool:
        return self.killswitch_path().exists()

    # ── OOM pre-flight (per user spec 2026-05-17) ────────────────────
    # Stage 11 LLM calls + correction loop fit easily in ~300-500 MB anon.
    # Floor of 500 MB allocatable keeps headroom for concurrent dedup or
    # absorption jobs that might fire from Stage 8/10 on the same box.
    STAGE11_MIN_ANON_MB = 500.0

    def check_memory_available(self) -> Optional[Dict[str, Any]]:
        """Return None if memory is sufficient; otherwise a dict explaining
        the refusal. Caller surfaces to UI.
        """
        try:
            from app.pipeline.stage8_duplicates import _read_cgroup_anon_allocatable_mb
            free_mb = _read_cgroup_anon_allocatable_mb()
        except Exception:
            free_mb = None
        if free_mb is None:
            # Non-cgroup system (Windows local dev) -- skip the guard
            return None
        if free_mb < self.STAGE11_MIN_ANON_MB:
            return {
                "status": "insufficient_memory",
                "free_mb": int(free_mb),
                "required_mb": int(self.STAGE11_MIN_ANON_MB),
                "reason": (f"Only {free_mb:.0f} MB anon allocatable; "
                            f"Stage 11 needs >= {self.STAGE11_MIN_ANON_MB:.0f} MB. "
                            f"Wait for other sessions to finish, then retry."),
            }
        return None

    # ── Cohort selection ─────────────────────────────────────────────
    def _session_date_ist(self, session_id: str, session_dir: Path) -> Optional[date]:
        """Determine the IST date when this session's uploads landed.
        Source: Stage 10 checkpoint completed timestamp; fallback to CSV mtime.
        """
        try:
            from app.db import get_checkpoint
            cp = get_checkpoint(session_id, "stage10_upload")
            if cp:
                ts_str = cp.get("updated_at") or cp.get("created_at")
                if ts_str:
                    try:
                        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt.astimezone(IST).date()
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            logger.debug(f"get_checkpoint failed for {session_id}: {e}")

        # Fallback: CSV file mtime
        for rel in ("upload_logs/combined_upload_log.csv", "combined_upload_log.csv"):
            p = session_dir / rel
            if p.exists():
                try:
                    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                    return mtime.astimezone(IST).date()
                except Exception:
                    continue
        return None

    def _session_operator(self, session_id: str, session_dir: Path) -> str:
        """Operator (uploaded_by) for this session.
        Source: get_session().created_by; fallback to session_config.json.
        """
        try:
            from app.db import get_session
            sess = get_session(session_id)
            if sess and sess.get("created_by"):
                return str(sess["created_by"])
        except Exception as e:
            logger.debug(f"get_session failed for {session_id}: {e}")
        cfg_path = session_dir / "session_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                return str(cfg.get("created_by", ""))
            except Exception:
                pass
        return ""

    def _build_error_description(self, csv_row: Dict[str, str]) -> str:
        """Build error_description from Pass1/2/3 Errors columns.
        Format matches the historical Stage 11 cohort CSVs.
        """
        parts = []
        for pass_num in (1, 2, 3):
            errs = (csv_row.get(f"Pass{pass_num} Errors") or "").strip()
            if errs:
                parts.append(f"[Pass{pass_num}] {errs}")
        return " || ".join(parts)

    def select_cohort(self, target_date_ist: date, history: Dict[str, Any],
                       floor_date: date) -> List[Dict[str, Any]]:
        """Returns rows ready for stage11_correction.process_batch.
        Each row: {qid, error_description, q_type, view_link, meta_qid,
                   session_id, uploaded_by}.
        """
        rows: List[Dict[str, Any]] = []
        seen = set()
        skipped_promoted = 0
        skipped_flagged = 0
        sessions_scanned = 0
        sessions_in_window = 0

        sessions_dir = Path(self.settings.sessions_dir)
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                sessions_scanned += 1
                session_id = session_dir.name

                # Date attribution: session-level (combined_upload_log.csv has
                # no per-row timestamp).
                d_ist = self._session_date_ist(session_id, session_dir)
                if d_ist is None:
                    continue
                if d_ist < floor_date:
                    continue
                if d_ist != target_date_ist:
                    continue
                sessions_in_window += 1

                # Locate the CSV (Stage 10 writes to upload_logs/ subfolder)
                log_path = session_dir / "upload_logs" / "combined_upload_log.csv"
                if not log_path.exists():
                    log_path = session_dir / "combined_upload_log.csv"
                    if not log_path.exists():
                        continue

                # Pre-compute operator (one DB call per session)
                uploaded_by = self._session_operator(session_id, session_dir)

                try:
                    with open(log_path, encoding="utf-8") as f:
                        for r in csv.DictReader(f):
                            qid = (r.get("UploadedQuestionID") or "").strip()
                            if not qid or len(qid) != 24:
                                continue
                            err = (r.get("Error Level") or "").strip()
                            if err != "2":
                                continue
                            if qid in seen:
                                continue
                            h = history.get(qid)
                            if h:
                                if h.get("status") == "promoted":
                                    skipped_promoted += 1
                                    continue
                                if h.get("status") == "flagged":
                                    skipped_flagged += 1
                                    continue
                            seen.add(qid)
                            rows.append({
                                "qid": qid,
                                "error_description": self._build_error_description(r),
                                "q_type": (r.get("Question Type") or "scq").strip(),
                                "view_link": (r.get("View Link") or "").strip(),
                                "meta_qid": (r.get("Question META ID") or "").strip(),
                                "session_id": session_id,
                                "uploaded_by": uploaded_by,
                            })
                except Exception as e:
                    logger.warning(f"Failed to read {log_path}: {e}")

        # Source B: failed_retry from history
        for qid, info in history.items():
            if qid in seen:
                continue
            if info.get("status") != "failed_retry":
                continue
            row = dict(info.get("input_row", {}))
            row["qid"] = qid
            seen.add(qid)
            rows.append(row)

        logger.info(
            f"Cohort for {target_date_ist}: {len(rows)} OIDs "
            f"(scanned {sessions_scanned} sessions, {sessions_in_window} in window; "
            f"skipped {skipped_promoted} promoted, {skipped_flagged} flagged)"
        )
        return rows

    # ── Live-multiget verification ───────────────────────────────────
    def filter_to_live_err2(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drops OIDs whose live error != 2 (handles concurrent operator fixes)."""
        if not rows:
            return rows
        try:
            from app.pipeline.auth import authenticated_request
        except ImportError:
            logger.warning("auth.authenticated_request unavailable; skipping live verify")
            return rows

        qids = [r["qid"] for r in rows]
        keep_set = set()
        for i in range(0, len(qids), 100):
            chunk = qids[i:i + 100]
            try:
                resp = authenticated_request(
                    "post", "https://api.acadza.in/question/multiget",
                    json={"ids": chunk}, timeout=60,
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"multiget chunk {i//100} status={resp.status_code}; keeping all"
                    )
                    keep_set.update(chunk)
                    continue
                items = resp.json().get("questions") or []
                live_by_id = {}
                for it in items:
                    _id = it.get("_id")
                    if isinstance(_id, dict):
                        _id = _id.get("$oid")
                    live_by_id[_id] = it.get("error")
                for qid in chunk:
                    if live_by_id.get(qid) == 2:
                        keep_set.add(qid)
            except Exception as e:
                logger.warning(f"multiget exception: {e}; keeping chunk")
                keep_set.update(chunk)

        filtered = [r for r in rows if r["qid"] in keep_set]
        logger.info(f"Live verify: {len(filtered)}/{len(rows)} still at err=2")
        return filtered

    # ── Run executor (sync; called via asyncio.to_thread) ────────────
    def run_one_night(self, target_date_ist: Optional[date] = None,
                      force: bool = False) -> Dict[str, Any]:
        """Synchronous entry point. Returns the report dict."""
        # Guard against concurrent invocation (e.g., scheduler + manual trigger)
        if not self._run_lock.acquire(blocking=False):
            logger.warning("Stage 11 nightly run already in progress; refusing concurrent start")
            return {"status": "already_running"}
        try:
            return self._run_one_night_inner(target_date_ist, force)
        finally:
            self._run_lock.release()

    def _run_one_night_inner(self, target_date_ist: Optional[date],
                              force: bool) -> Dict[str, Any]:
        if target_date_ist is None:
            target_date_ist = (datetime.now(IST) - timedelta(days=1)).date()

        floor_date = date.fromisoformat(
            getattr(self.settings, "stage11_floor_date", "2026-05-18")
        )
        if target_date_ist < floor_date:
            logger.info(f"Target {target_date_ist} below floor {floor_date}; skipping")
            return {"status": "below_floor", "target_date_ist": target_date_ist.isoformat()}

        report_path = self.runs_dir() / f"{target_date_ist.isoformat()}.json"
        if report_path.exists() and not force:
            logger.info(f"Run for {target_date_ist} already done; not re-running")
            existing = {}
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                pass
            existing["status"] = existing.get("status") or "already_done"
            return existing

        if self.killswitch_active():
            logger.warning("Killswitch active; aborting at start")
            return {"status": "killswitch", "target_date_ist": target_date_ist.isoformat()}

        # OOM pre-flight with retry (per user spec 2026-05-17). Nightly fires
        # at 00:30 IST low-load window; if memory tight, wait + retry. Up to
        # 3 retries x 30 min = 1.5 hr total wait. If still OOM after that,
        # give up; tomorrow's run picks the OIDs up as failed_retry.
        # ON-SPOT path does NOT retry -- operator gets immediate error.
        import time as _time
        MAX_OOM_RETRIES = 3
        OOM_RETRY_WAIT_S = 1800  # 30 minutes
        for attempt in range(MAX_OOM_RETRIES + 1):
            mem_check = self.check_memory_available()
            if mem_check is None:
                break  # memory OK
            if attempt >= MAX_OOM_RETRIES:
                logger.error(
                    f"Stage 11 nightly OOM after {MAX_OOM_RETRIES} retries: {mem_check}"
                )
                mem_check["target_date_ist"] = target_date_ist.isoformat()
                mem_check["retries_exhausted"] = MAX_OOM_RETRIES
                return mem_check
            logger.warning(
                f"Stage 11 nightly OOM (attempt {attempt+1}/{MAX_OOM_RETRIES+1}): "
                f"free={mem_check.get('free_mb')}MB need={mem_check.get('required_mb')}MB; "
                f"sleeping {OOM_RETRY_WAIT_S}s then retrying..."
            )
            if self.killswitch_active():
                return {"status": "killswitch_during_oom_retry",
                        "target_date_ist": target_date_ist.isoformat()}
            _time.sleep(OOM_RETRY_WAIT_S)

        history = self.load_history()
        raw_cohort = self.select_cohort(target_date_ist, history, floor_date)
        verified = self.filter_to_live_err2(raw_cohort)

        dryrun = bool(getattr(self.settings, "stage11_dryrun", False))
        report = {
            "target_date_ist": target_date_ist.isoformat(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "cohort_raw": len(raw_cohort),
            "cohort_verified": len(verified),
            "dryrun": dryrun,
            "status": "running",
            "promoted": 0,
            "flagged": 0,
            "failed_retry": 0,
            "per_operator": {},
        }
        self._atomic_write_json(report, report_path)

        if not verified:
            report["status"] = "done_empty"
            report["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            self._atomic_write_json(report, report_path)
            return report

        if dryrun:
            report["status"] = "dryrun_done"
            report["dryrun_sample"] = [r["qid"] for r in verified[:20]]
            report["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            self._atomic_write_json(report, report_path)
            logger.info(f"DRY-RUN done for {target_date_ist}: cohort={len(verified)}")
            return report

        # Live execute in chunks with killswitch check
        try:
            from app.pipeline.stage11_correction import process_batch
        except ImportError as e:
            report["status"] = "error_import"
            report["error"] = str(e)
            report["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            self._atomic_write_json(report, report_path)
            logger.error(f"process_batch import failed: {e}")
            return report

        backup_dir = self.backups_dir() / target_date_ist.isoformat()
        backup_dir.mkdir(parents=True, exist_ok=True)

        CHUNK = 50
        all_results: List[Any] = []
        for i in range(0, len(verified), CHUNK):
            if self.killswitch_active():
                logger.warning(
                    f"Killswitch active after {i}/{len(verified)}; halting and saving state"
                )
                break
            chunk = verified[i:i + CHUNK]
            try:
                chunk_results = process_batch(
                    chunk,
                    set_error_to_1=True,
                    do_update_api=True,
                    backup_dir=backup_dir,
                    max_workers=5,
                    logger=lambda lvl, msg: logger.log(
                        logging.INFO if lvl == "info" else logging.WARNING,
                        f"[stage11-nightly] {msg}",
                    ),
                )
                all_results.extend(chunk_results)
            except Exception as e:
                logger.error(f"chunk {i//CHUNK} failed: {e}", exc_info=True)
                # All OIDs in chunk -> failed_retry (handled below)
            # Incremental report + history write per chunk so an OOM crash
            # mid-run doesn't lose progress. The cron picks up partial state
            # next night via failed_retry semantics.
            self._tally_into_report(report, all_results)
            self._atomic_write_json(report, report_path)
            self._update_history_from_results(history, all_results, verified[:i + CHUNK])
            self.save_history(history)

        # Final history update from results
        self._update_history_from_results(history, all_results, verified)
        self.save_history(history)

        report["status"] = "done" if not self.killswitch_active() else "halted_killswitch"
        report["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._tally_into_report(report, all_results)  # final tally
        self._atomic_write_json(report, report_path)

        # Write companion CSVs per user spec (2026-05-17):
        #   - {date}_faculty.csv      : OIDs flagged by this run (for faculty review)
        #   - {date}_operator_log.csv : per-OID outcome with timestamp + operator
        self._write_faculty_csv(target_date_ist, all_results, verified)
        self._write_operator_log_csv(target_date_ist, all_results, verified)
        logger.info(
            f"Stage 11 nightly DONE for {target_date_ist}: "
            f"promoted={report['promoted']} flagged={report['flagged']} "
            f"failed_retry={report['failed_retry']}"
        )
        return report

    # ── Tally + history helpers ──────────────────────────────────────
    def _qresult_attr(self, r, key, default=None):
        """QuestionResult may be dataclass or dict; handle both."""
        if hasattr(r, key):
            return getattr(r, key)
        if isinstance(r, dict):
            return r.get(key, default)
        return default

    def _classify_outcome(self, r) -> str:
        """Map QuestionResult.status to our 3-bucket status.
        Ground-truth status values per stage11_correction.py:1492-1929:
          - 'updated'                          -> promoted
          - 'flagged' / 'flagged_*'            -> flagged (LLM rejected)
          - 'skipped_*'                        -> flagged (malformed input, not retryable)
          - 'approved_pending_update'          -> failed_retry (should be 'updated' if do_update_api=True)
          - 'api_fetch_error: *' / 'exception: *' / 'update_api_error: *'
            / 'backup_save_error: *' / 'update_blocked_no_backup_dir' -> failed_retry
        """
        status = str(self._qresult_attr(r, "status", ""))
        if status == "updated":
            return "promoted"
        if status.startswith("flagged"):
            return "flagged"
        if status.startswith("skipped_"):
            return "flagged"
        if status == "approved_pending_update":
            return "failed_retry"
        # Anything starting with these prefixes is a transient/retryable failure
        retryable_prefixes = (
            "api_fetch_error", "exception:", "update_api_error",
            "backup_save_error",
        )
        if any(status.startswith(p) for p in retryable_prefixes):
            return "failed_retry"
        if status == "update_blocked_no_backup_dir":
            return "failed_retry"
        # Unknown -- default to failed_retry for safety
        return "failed_retry"

    def _tally_into_report(self, report: dict, results: list):
        promoted = flagged = failed_retry = 0
        per_op = {}
        for r in results:
            bucket = self._classify_outcome(r)
            if bucket == "promoted":
                promoted += 1
            elif bucket == "flagged":
                flagged += 1
            else:
                failed_retry += 1
            op = self._qresult_attr(r, "uploaded_by") or ""
            if op:
                per_op.setdefault(op, {"promoted": 0, "flagged": 0, "failed_retry": 0})
                per_op[op][bucket] += 1
        report["promoted"] = promoted
        report["flagged"] = flagged
        report["failed_retry"] = failed_retry
        report["per_operator"] = per_op

    def _update_history_from_results(self, history: dict, results: list,
                                       rows: list):
        results_by_qid = {}
        for r in results:
            qid = self._qresult_attr(r, "qid")
            if qid:
                results_by_qid[qid] = r
        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            qid = row["qid"]
            r = results_by_qid.get(qid)
            entry = history.setdefault(qid, {"first_seen": now, "attempts": []})
            entry["last_attempt_ts"] = now
            if r is None:
                entry["status"] = "failed_retry"
                entry["input_row"] = row
                entry["attempts"].append({"ts": now, "outcome": "skipped_or_chunk_failed"})
            else:
                bucket = self._classify_outcome(r)
                entry["status"] = bucket
                if bucket == "promoted":
                    entry.pop("input_row", None)
                else:
                    entry["input_row"] = row
                entry["attempts"].append({
                    "ts": now,
                    "outcome": bucket,
                    "raw_status": self._qresult_attr(r, "status", ""),
                })
            # Trim attempts to last 5
            entry["attempts"] = entry["attempts"][-5:]

    # ── On-spot manual trigger (per user spec 2026-05-17) ────────────
    def run_on_spot(self, session_id: str, operator: str) -> Dict[str, Any]:
        """Operator clicks "Run Stage 11" button on Stage 10 UI.
        Processes ONLY this session's err=2 OIDs (not whole-night cohort).
        Marks history entries with on_spot=True so the nightly skips them.
        Refuses if any other Stage 11 run (nightly or on-spot) is active.
        """
        if not self._run_lock.acquire(blocking=False):
            return {"status": "already_running",
                    "reason": "Another Stage 11 run is in progress"}
        try:
            return self._run_on_spot_inner(session_id, operator)
        finally:
            self._run_lock.release()

    def _run_on_spot_inner(self, session_id: str, operator: str) -> Dict[str, Any]:
        if self.killswitch_active():
            return {"status": "killswitch", "reason": "killswitch_flag_present"}

        # OOM pre-flight (per user spec 2026-05-17). UNLIKE the nightly,
        # on-spot does NOT retry on OOM -- operator gets an immediate
        # error and the frontend mutes the button. Operator can retry
        # manually after memory frees up.
        mem_check = self.check_memory_available()
        if mem_check is not None:
            logger.error(
                f"Stage 11 on-spot OOM pre-check failed for session {session_id}: "
                f"{mem_check}"
            )
            mem_check["session_id"] = session_id
            mem_check["retry_policy"] = "manual_only_no_auto_retry"
            return mem_check

        sessions_dir = Path(self.settings.sessions_dir)
        session_dir = sessions_dir / session_id
        if not session_dir.is_dir():
            return {"status": "error", "reason": "session_not_found"}

        log_path = session_dir / "upload_logs" / "combined_upload_log.csv"
        if not log_path.exists():
            log_path = session_dir / "combined_upload_log.csv"
            if not log_path.exists():
                return {"status": "error", "reason": "csv_missing",
                        "session_id": session_id}

        history = self.load_history()
        rows: List[Dict[str, Any]] = []
        try:
            with open(log_path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    qid = (r.get("UploadedQuestionID") or "").strip()
                    if not qid or len(qid) != 24:
                        continue
                    err = (r.get("Error Level") or "").strip()
                    if err != "2":
                        continue
                    h = history.get(qid)
                    if h and h.get("status") in ("promoted", "flagged"):
                        continue
                    rows.append({
                        "qid": qid,
                        "error_description": self._build_error_description(r),
                        "q_type": (r.get("Question Type") or "scq").strip(),
                        "view_link": (r.get("View Link") or "").strip(),
                        "meta_qid": (r.get("Question META ID") or "").strip(),
                        "session_id": session_id,
                        "uploaded_by": operator,
                    })
        except Exception as e:
            return {"status": "error", "reason": f"csv_read_failed: {e}",
                    "session_id": session_id}

        if not rows:
            return {"status": "done_empty", "session_id": session_id,
                    "cohort_size": 0}

        verified = self.filter_to_live_err2(rows)
        if not verified:
            return {"status": "done_empty_after_verify", "session_id": session_id,
                    "cohort_raw": len(rows), "cohort_verified": 0}

        try:
            from app.pipeline.stage11_correction import process_batch
        except ImportError as e:
            return {"status": "error_import", "error": str(e)}

        backup_dir = (self.backups_dir() /
                      f"on_spot_{session_id}_"
                      f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Process in chunks of 50 with per-chunk history save so an OOM
        # crash or browser-close mid-run preserves work-so-far (per user
        # spec 2026-05-17: "things are saved in real time and written").
        CHUNK = 50
        results = []
        for i in range(0, len(verified), CHUNK):
            if self.killswitch_active():
                logger.warning(
                    f"on-spot killswitch tripped after {i}/{len(verified)}; halting"
                )
                break
            chunk = verified[i:i + CHUNK]
            try:
                chunk_results = process_batch(
                    chunk,
                    set_error_to_1=True,
                    do_update_api=True,
                    backup_dir=backup_dir,
                    max_workers=5,
                    logger=lambda lvl, msg: logger.log(
                        logging.INFO if lvl == "info" else logging.WARNING,
                        f"[stage11-on-spot {session_id}] {msg}",
                    ),
                )
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"on-spot chunk {i // CHUNK} crashed: {e}", exc_info=True)
                # leave chunk OIDs without results -> get tagged failed_retry below
            # Incremental save: history persisted after each chunk
            self._update_history_from_results(history, results, verified[:i + CHUNK])
            for row in chunk:
                qid = row["qid"]
                if qid in history:
                    history[qid]["on_spot"] = True
                    history[qid]["on_spot_by"] = operator
                    history[qid]["on_spot_session"] = session_id
                    history[qid]["on_spot_ts_utc"] = datetime.now(timezone.utc).isoformat()
            self.save_history(history)

        # Per-OID per-operator summary
        promoted = flagged = failed_retry = 0
        for r in results:
            b = self._classify_outcome(r)
            if b == "promoted": promoted += 1
            elif b == "flagged": flagged += 1
            else: failed_retry += 1

        return {
            "status": "done",
            "session_id": session_id,
            "operator": operator,
            "cohort_raw": len(rows),
            "cohort_verified": len(verified),
            "promoted": promoted,
            "flagged": flagged,
            "failed_retry": failed_retry,
            "backup_dir": str(backup_dir),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def is_running(self) -> bool:
        """Whether ANY Stage 11 run (nightly or on-spot) holds the lock."""
        # Try to acquire (non-blocking); if we can, nothing is running.
        if self._run_lock.acquire(blocking=False):
            self._run_lock.release()
            return False
        return True

    # ── Companion CSV outputs (per user spec 2026-05-17) ─────────────
    def _write_faculty_csv(self, target_date_ist: date, results: list,
                            rows: list):
        """Per-night faculty-review CSV: OIDs that ended up `flagged`
        (LLM judged "cannot fix" or input malformed). Faculty review these.
        Excludes API-failure (failed_retry) — those auto-retry tomorrow.
        """
        path = self.runs_dir() / f"{target_date_ist.isoformat()}_faculty.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows_by_qid = {r["qid"]: r for r in rows}

        flagged_count = 0
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "oid", "view_link", "q_type", "uploaded_by", "session_id",
                "error_description", "flag_reason", "raw_status", "flagged_at_utc",
            ])
            writer.writeheader()
            now_utc = datetime.now(timezone.utc).isoformat()
            for r in results:
                if self._classify_outcome(r) != "flagged":
                    continue
                qid = self._qresult_attr(r, "qid", "")
                src = rows_by_qid.get(qid, {})
                writer.writerow({
                    "oid": qid,
                    "view_link": src.get("view_link", "")
                                  or self._qresult_attr(r, "view_link", ""),
                    "q_type": src.get("q_type", "")
                              or self._qresult_attr(r, "q_type", ""),
                    "uploaded_by": src.get("uploaded_by", ""),
                    "session_id": src.get("session_id", ""),
                    "error_description": src.get("error_description", "")
                                          or self._qresult_attr(r, "error_description", ""),
                    "flag_reason": self._qresult_attr(r, "flag_reason", "")
                                   or self._qresult_attr(r, "sideline_reason", ""),
                    "raw_status": self._qresult_attr(r, "status", ""),
                    "flagged_at_utc": now_utc,
                })
                flagged_count += 1
        os.replace(tmp, path)
        logger.info(f"Faculty CSV: {flagged_count} OIDs -> {path.name}")

    def _write_operator_log_csv(self, target_date_ist: date, results: list,
                                 rows: list):
        """Per-night, per-OID detailed log: timestamp + outcome + operator.
        One row per OID in the cohort. Used by the dashboard widget to
        compute per-operator nightly totals.
        """
        path = self.runs_dir() / f"{target_date_ist.isoformat()}_operator_log.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows_by_qid = {r["qid"]: r for r in rows}
        results_by_qid = {self._qresult_attr(r, "qid", ""): r for r in results}

        tmp = path.with_suffix(path.suffix + ".tmp")
        now_utc = datetime.now(timezone.utc).isoformat()
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "oid", "uploaded_by", "session_id", "outcome", "raw_status",
                "attempted_at_utc", "view_link", "q_type",
            ])
            writer.writeheader()
            for row in rows:
                qid = row["qid"]
                r = results_by_qid.get(qid)
                outcome = self._classify_outcome(r) if r is not None else "failed_retry"
                writer.writerow({
                    "oid": qid,
                    "uploaded_by": row.get("uploaded_by", ""),
                    "session_id": row.get("session_id", ""),
                    "outcome": outcome,
                    "raw_status": self._qresult_attr(r, "status", "") if r is not None else "",
                    "attempted_at_utc": now_utc,
                    "view_link": row.get("view_link", ""),
                    "q_type": row.get("q_type", ""),
                })
        os.replace(tmp, path)
        logger.info(f"Operator-log CSV: {len(rows)} rows -> {path.name}")

    # ── Async scheduler entry point ──────────────────────────────────
    async def start(self):
        if not bool(getattr(self.settings, "stage11_nightly_enabled", False)):
            logger.info("Stage 11 nightly DISABLED via STAGE11_NIGHTLY_ENABLED")
            return
        logger.info("Stage 11 nightly: scheduler starting")
        while not self._stop.is_set():
            next_t = self.compute_next_trigger_utc()
            wait_s = max(1.0, (next_t - datetime.now(timezone.utc)).total_seconds())
            logger.info(
                f"Stage 11 nightly: next fire at {next_t.isoformat()} "
                f"({wait_s:.0f}s = {wait_s/3600:.1f}h)"
            )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=wait_s)
                break  # stop event set during sleep
            except asyncio.TimeoutError:
                pass  # timer fired
            try:
                await asyncio.to_thread(self.run_one_night)
            except Exception as e:
                logger.error(f"Stage 11 nightly run crashed: {e}", exc_info=True)
        logger.info("Stage 11 nightly: scheduler stopped")

    def stop(self):
        self._stop.set()


# ── Singleton accessor ────────────────────────────────────────────────
_singleton: Optional[Stage11NightlyScheduler] = None


def get_scheduler() -> Stage11NightlyScheduler:
    global _singleton
    if _singleton is None:
        from app.config import settings
        _singleton = Stage11NightlyScheduler(settings)
    return _singleton
