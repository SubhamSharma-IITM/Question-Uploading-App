"""SQLite database singleton — thread-safe, WAL mode.

Forked from acadza-pdf-generator/app/analytics.py pattern.
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings

_DB_DIR = Path(settings.data_dir)
_DB_PATH = _DB_DIR / "app.db"
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    """Get or create singleton DB connection."""
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                current_stage TEXT DEFAULT 'stage1_upload',
                pdf_filename TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config JSON
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                session_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                started_at TEXT,
                completed_at TEXT,
                data JSON,
                PRIMARY KEY (session_id, stage),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                task_type TEXT NOT NULL,
                status TEXT DEFAULT 'queued',
                progress_pct REAL DEFAULT 0,
                progress_msg TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                result JSON,
                error TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                stage TEXT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS quarantine (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                qid TEXT NOT NULL,
                stage TEXT NOT NULL,
                reason TEXT NOT NULL,
                data JSON,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS questions (
                session_id TEXT NOT NULL,
                qid TEXT NOT NULL,
                subject TEXT,
                question_type TEXT,
                current_stage TEXT,
                has_answer BOOLEAN DEFAULT FALSE,
                has_solution BOOLEAN DEFAULT FALSE,
                quarantined BOOLEAN DEFAULT FALSE,
                PRIMARY KEY (session_id, qid),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_logs_session ON logs(session_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
            CREATE INDEX IF NOT EXISTS idx_quarantine_session ON quarantine(session_id);
        """)
        conn.commit()


# ── Session CRUD ──


def create_session(session_id: str, name: str, created_at: str,
                   config: Optional[Dict] = None,
                   initial_stage: str = "stage1_upload") -> None:
    """Insert a new session. Defaults to stage1_upload so sidebar highlights Step 1."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO sessions (id, name, current_stage, created_at, updated_at, config) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, name, initial_stage, created_at, created_at, json.dumps(config) if config else None),
        )
        conn.commit()


def get_session(session_id: str) -> Optional[Dict]:
    """Get a session by ID."""
    with _lock:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("config"):
            d["config"] = json.loads(d["config"])
        return d


def list_sessions() -> List[Dict]:
    """List all sessions, newest first."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("config"):
                d["config"] = json.loads(d["config"])
            results.append(d)
        return results


def update_session(session_id: str, **kwargs) -> None:
    """Update session fields. Pass field=value pairs."""
    if not kwargs:
        return
    from app.utils import now_iso
    kwargs["updated_at"] = now_iso()
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [session_id]
    with _lock:
        conn = _get_conn()
        conn.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?", values)
        conn.commit()


def delete_session(session_id: str) -> None:
    """Delete a session and all related data (cascades)."""
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()


# ── Checkpoint CRUD ──


def upsert_checkpoint(session_id: str, stage: str, status: str,
                      data: Optional[Dict] = None, **kwargs) -> None:
    """Insert or update a stage checkpoint."""
    from app.utils import now_iso
    now = now_iso()
    started = now if status == "running" else kwargs.get("started_at")
    completed = now if status == "completed" else kwargs.get("completed_at")
    data_json = json.dumps(data) if data else None

    with _lock:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO checkpoints (session_id, stage, status, started_at, completed_at, data)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, stage) DO UPDATE SET
                status = excluded.status,
                started_at = COALESCE(excluded.started_at, checkpoints.started_at),
                completed_at = excluded.completed_at,
                data = COALESCE(excluded.data, checkpoints.data)
        """, (session_id, stage, status, started, completed, data_json))
        conn.commit()


def get_checkpoint(session_id: str, stage: str) -> Optional[Dict]:
    """Get checkpoint for a specific stage."""
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM checkpoints WHERE session_id = ? AND stage = ?",
            (session_id, stage),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("data"):
            d["data"] = json.loads(d["data"])
        return d


def get_all_checkpoints(session_id: str) -> List[Dict]:
    """Get all checkpoints for a session."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY stage",
            (session_id,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("data"):
                d["data"] = json.loads(d["data"])
            results.append(d)
        return results


# ── Log entries ──


def insert_log(session_id: str, stage: Optional[str], level: str,
               message: str) -> int:
    """Insert a log entry. Returns the log ID."""
    from app.utils import now_iso
    with _lock:
        conn = _get_conn()
        cursor = conn.execute(
            "INSERT INTO logs (session_id, stage, level, message, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, stage, level, message, now_iso()),
        )
        conn.commit()
        return cursor.lastrowid


def get_logs(session_id: str, after_id: int = 0, limit: int = 200) -> List[Dict]:
    """Get logs for a session, optionally after a specific ID."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM logs WHERE session_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
            (session_id, after_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


# ── Task queue ──


def create_task(task_id: str, session_id: str, stage: str,
                task_type: str) -> None:
    """Insert a new background task."""
    from app.utils import now_iso
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO tasks (id, session_id, stage, task_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, session_id, stage, task_type, now_iso()),
        )
        conn.commit()


def update_task(task_id: str, **kwargs) -> None:
    """Update task fields."""
    if not kwargs:
        return
    # Serialize result/error if dicts
    if "result" in kwargs and isinstance(kwargs["result"], dict):
        kwargs["result"] = json.dumps(kwargs["result"])
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    with _lock:
        conn = _get_conn()
        conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", values)
        conn.commit()


def get_task(task_id: str) -> Optional[Dict]:
    """Get a task by ID."""
    with _lock:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("result"):
            try:
                d["result"] = json.loads(d["result"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d


# ── Quarantine ──


def quarantine_question(session_id: str, qid: str, stage: str,
                        reason: str, data: Optional[Dict] = None) -> None:
    """Add a question to quarantine."""
    from app.utils import now_iso
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO quarantine (session_id, qid, stage, reason, data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, qid, stage, reason, json.dumps(data) if data else None, now_iso()),
        )
        conn.commit()


def get_quarantine(session_id: str) -> List[Dict]:
    """Get all quarantined questions for a session."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM quarantine WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_quarantined_qids(session_id: str) -> set:
    """Get the set of quarantined QIDs for a session.

    Convenience wrapper used by all downstream stages to filter their input.
    """
    records = get_quarantine(session_id)
    return {r["qid"] for r in records if r.get("qid")}


# ── Question tracking ──


def upsert_question(session_id: str, qid: str, **kwargs) -> None:
    """Insert or update a question tracking record."""
    fields = {k: v for k, v in kwargs.items() if v is not None}
    with _lock:
        conn = _get_conn()
        # Check if exists
        existing = conn.execute(
            "SELECT 1 FROM questions WHERE session_id = ? AND qid = ?",
            (session_id, qid),
        ).fetchone()
        if existing:
            if fields:
                set_clause = ", ".join(f"{k} = ?" for k in fields)
                values = list(fields.values()) + [session_id, qid]
                conn.execute(
                    f"UPDATE questions SET {set_clause} WHERE session_id = ? AND qid = ?",
                    values,
                )
        else:
            all_fields = {"session_id": session_id, "qid": qid, **fields}
            cols = ", ".join(all_fields.keys())
            placeholders = ", ".join("?" for _ in all_fields)
            conn.execute(
                f"INSERT INTO questions ({cols}) VALUES ({placeholders})",
                list(all_fields.values()),
            )
        conn.commit()


def get_session_stats(session_id: str) -> Dict:
    """Get question stats for a session."""
    with _lock:
        conn = _get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        with_answer = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE session_id = ? AND has_answer = 1",
            (session_id,),
        ).fetchone()[0]
        with_solution = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE session_id = ? AND has_solution = 1",
            (session_id,),
        ).fetchone()[0]
        quarantined = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE session_id = ? AND quarantined = 1",
            (session_id,),
        ).fetchone()[0]
        by_subject = {}
        rows = conn.execute(
            "SELECT subject, COUNT(*) as cnt FROM questions WHERE session_id = ? GROUP BY subject",
            (session_id,),
        ).fetchall()
        for row in rows:
            by_subject[row["subject"] or "Unknown"] = row["cnt"]
        return {
            "total": total,
            "with_answer": with_answer,
            "with_solution": with_solution,
            "quarantined": quarantined,
            "by_subject": by_subject,
        }
