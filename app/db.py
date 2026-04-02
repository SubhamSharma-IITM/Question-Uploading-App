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
        _conn.execute("PRAGMA busy_timeout=5000")   # wait 5s on lock instead of failing
        _conn.execute("PRAGMA synchronous=NORMAL")   # flush on checkpoint, not every commit (faster + WAL-safe)
        _conn.execute("PRAGMA wal_autocheckpoint=50") # checkpoint every 50 pages (~200KB) to keep WAL small
    return _conn


def force_wal_checkpoint() -> None:
    """Force WAL checkpoint — flushes all WAL data to main DB file.

    Call after critical writes (session create, stage advance, upload complete)
    to ensure data survives OOM kills. On OOM, uncommitted WAL pages are lost
    but checkpointed data is safe in the main .db file.
    """
    try:
        with _lock:
            conn = _get_conn()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass  # non-fatal, WAL will checkpoint eventually


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
                created_by TEXT,
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

            CREATE TABLE IF NOT EXISTS users (
                phone TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'Team',
                created_at TEXT NOT NULL,
                last_login TEXT,
                last_active TEXT,
                properties JSON
            );

            CREATE TABLE IF NOT EXISTS upload_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_phone TEXT NOT NULL,
                error_level INTEGER NOT NULL,
                subject TEXT,
                count INTEGER DEFAULT 0,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS heartbeat (
                user_phone TEXT PRIMARY KEY,
                last_ping TEXT NOT NULL,
                current_session TEXT,
                current_stage TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_logs_session ON logs(session_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
            CREATE INDEX IF NOT EXISTS idx_quarantine_session ON quarantine(session_id);
            CREATE INDEX IF NOT EXISTS idx_upload_log_user ON upload_log(user_phone);
            CREATE INDEX IF NOT EXISTS idx_upload_log_date ON upload_log(uploaded_at);
        """)

        # ── Auto-migration: add created_by to sessions if missing ──
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
        if "created_by" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN created_by TEXT")

        # Index on created_by (safe to run after migration)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_by ON sessions(created_by)")
        conn.commit()


# ── Session CRUD ──


def create_session(session_id: str, name: str, created_at: str,
                   config: Optional[Dict] = None,
                   initial_stage: str = "stage1_upload",
                   created_by: str = "") -> None:
    """Insert a new session. Defaults to stage1_upload so sidebar highlights Step 1."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO sessions (id, name, current_stage, created_by, created_at, updated_at, config) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, name, initial_stage, created_by, created_at, created_at, json.dumps(config) if config else None),
        )
        conn.commit()
    force_wal_checkpoint()  # flush to survive OOM


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
    # Checkpoint on stage advances (critical state change)
    if "current_stage" in kwargs:
        force_wal_checkpoint()


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


# ── User management ──


def upsert_user(phone: str, name: str, role: str,
                properties: Optional[Dict] = None) -> None:
    """Insert or update a user record."""
    from app.utils import now_iso
    now = now_iso()
    with _lock:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO users (phone, name, role, created_at, properties)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(phone) DO UPDATE SET
                name = excluded.name,
                role = excluded.role,
                properties = COALESCE(excluded.properties, users.properties)
        """, (phone, name, role, now, json.dumps(properties) if properties else None))
        conn.commit()


def update_user_login(phone: str) -> None:
    """Update last_login timestamp for a user."""
    from app.utils import now_iso
    with _lock:
        conn = _get_conn()
        conn.execute("UPDATE users SET last_login = ? WHERE phone = ?",
                      (now_iso(), phone))
        conn.commit()


def update_user_last_active(phone: str) -> None:
    """Update last_active timestamp for a user (called on every heartbeat)."""
    from app.utils import now_iso
    with _lock:
        conn = _get_conn()
        conn.execute("UPDATE users SET last_active = ? WHERE phone = ?",
                      (now_iso(), phone))
        conn.commit()


def get_user(phone: str) -> Optional[Dict]:
    """Get a user by phone."""
    with _lock:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("properties"):
            d["properties"] = json.loads(d["properties"])
        return d


def list_users(role: Optional[str] = None) -> List[Dict]:
    """List all users, optionally filtered by role."""
    with _lock:
        conn = _get_conn()
        if role:
            rows = conn.execute(
                "SELECT * FROM users WHERE role = ? ORDER BY name", (role,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM users ORDER BY name").fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("properties"):
                d["properties"] = json.loads(d["properties"])
            results.append(d)
        return results


# ── Upload log (daily stats tracking) ──


def log_upload(session_id: str, user_phone: str, error_level: int,
               subject: str, count: int) -> None:
    """Record an upload event for daily stats tracking."""
    from app.utils import now_iso
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO upload_log (session_id, user_phone, error_level, subject, count, uploaded_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, user_phone, error_level, subject, count, now_iso()),
        )
        conn.commit()


def get_upload_stats(user_phone: Optional[str] = None,
                     days: int = 1) -> Dict:
    """Get upload stats for a user (or all users) over N days.

    Returns: {err1: int, err2: int, by_subject: {subj: {err1, err2}}, by_day: [{date, err1, err2}]}
    """
    with _lock:
        conn = _get_conn()
        # IST offset (+5:30) so "today" boundaries align with Indian midnight, not UTC
        # days=1 means "today only", days=2 means "yesterday+today", etc.
        _IST = "'+5 hours', '+30 minutes'"
        date_filter = f"date(uploaded_at, {_IST}) >= date('now', {_IST}, '-{days - 1} days')"
        user_filter = "AND user_phone = ?" if user_phone else ""
        params = (user_phone,) if user_phone else ()

        # Totals by error level
        rows = conn.execute(f"""
            SELECT error_level, SUM(count) as total
            FROM upload_log WHERE {date_filter} {user_filter}
            GROUP BY error_level
        """, params).fetchall()
        err1 = sum(r["total"] for r in rows if r["error_level"] == 1)
        err2 = sum(r["total"] for r in rows if r["error_level"] == 2)

        # By subject
        by_subject = {}
        rows = conn.execute(f"""
            SELECT subject, error_level, SUM(count) as total
            FROM upload_log WHERE {date_filter} {user_filter}
            GROUP BY subject, error_level
        """, params).fetchall()
        for r in rows:
            subj = r["subject"] or "Unknown"
            if subj not in by_subject:
                by_subject[subj] = {"err1": 0, "err2": 0}
            if r["error_level"] == 1:
                by_subject[subj]["err1"] += r["total"]
            else:
                by_subject[subj]["err2"] += r["total"]

        # By day
        by_day = []
        rows = conn.execute(f"""
            SELECT date(uploaded_at, {_IST}) as day, error_level, SUM(count) as total
            FROM upload_log WHERE {date_filter} {user_filter}
            GROUP BY date(uploaded_at, {_IST}), error_level
            ORDER BY date(uploaded_at, {_IST})
        """, params).fetchall()
        day_map = {}
        for r in rows:
            d = r["day"]
            if d not in day_map:
                day_map[d] = {"date": d, "err1": 0, "err2": 0}
            if r["error_level"] == 1:
                day_map[d]["err1"] += r["total"]
            else:
                day_map[d]["err2"] += r["total"]
        by_day = list(day_map.values())

        return {"err1": err1, "err2": err2, "by_subject": by_subject, "by_day": by_day}


def get_user_daily_stats(user_phone: str) -> Dict:
    """Get today's stats for a specific user."""
    return get_upload_stats(user_phone=user_phone, days=1)


def get_all_users_stats(days: int = 7) -> List[Dict]:
    """Get upload stats per user over N days (for admin panel)."""
    with _lock:
        conn = _get_conn()
        _IST = "'+5 hours', '+30 minutes'"
        date_filter = f"date(uploaded_at, {_IST}) >= date('now', {_IST}, '-{days} days')"
        rows = conn.execute(f"""
            SELECT user_phone, error_level, SUM(count) as total
            FROM upload_log WHERE {date_filter}
            GROUP BY user_phone, error_level
        """).fetchall()
        user_map = {}
        for r in rows:
            phone = r["user_phone"]
            if phone not in user_map:
                user_map[phone] = {"phone": phone, "err1": 0, "err2": 0}
            if r["error_level"] == 1:
                user_map[phone]["err1"] += r["total"]
            else:
                user_map[phone]["err2"] += r["total"]
        return list(user_map.values())


# ── Heartbeat (activity tracking) ──


def update_heartbeat(user_phone: str, session_id: Optional[str] = None,
                     stage: Optional[str] = None) -> None:
    """Update heartbeat for a user."""
    from app.utils import now_iso
    with _lock:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO heartbeat (user_phone, last_ping, current_session, current_stage)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_phone) DO UPDATE SET
                last_ping = excluded.last_ping,
                current_session = COALESCE(excluded.current_session, heartbeat.current_session),
                current_stage = COALESCE(excluded.current_stage, heartbeat.current_stage)
        """, (user_phone, now_iso(), session_id, stage))
        conn.commit()


def get_heartbeats() -> List[Dict]:
    """Get all heartbeat records (for admin panel)."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM heartbeat ORDER BY last_ping DESC").fetchall()
        return [dict(r) for r in rows]


def seed_users_from_registry(registry: Dict[str, dict]) -> None:
    """Seed users table from the in-memory USER_REGISTRY.

    Called at startup so the users table always has all registered users,
    even if they haven't logged in yet.
    """
    from app.utils import now_iso
    now = now_iso()
    with _lock:
        conn = _get_conn()
        for phone, info in registry.items():
            conn.execute("""
                INSERT INTO users (phone, name, role, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(phone) DO UPDATE SET
                    name = excluded.name,
                    role = excluded.role
            """, (phone, info["name"], info["role"], now))
        conn.commit()


def get_checkpoint_upload_stats() -> Dict:
    """Get aggregate upload stats from stage10 checkpoint data.

    This is the fallback/primary source for upload stats since
    upload_log may not have been populated for historical uploads.
    Returns: {total_err1, total_err2, by_session: [{id, name, err1, err2, created_by, completed_at}]}
    """
    with _lock:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT s.id, s.name, s.created_by, s.created_at,
                   c.data, c.completed_at
            FROM sessions s
            JOIN checkpoints c ON s.id = c.session_id
            WHERE c.stage = 'stage10_upload'
              AND c.status = 'completed'
        """).fetchall()

        total_err1 = 0
        total_err2 = 0
        by_session = []
        by_user = {}

        for row in rows:
            data_raw = row["data"]
            if not data_raw:
                continue
            try:
                data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
            except Exception:
                continue

            err1 = data.get("posted_err1", 0)
            err2 = data.get("posted_err2", 0)
            # Fallback: if no split counts, use total posted
            if not err1 and not err2 and data.get("posted", 0):
                err1 = data.get("posted", 0)

            total_err1 += err1
            total_err2 += err2

            created_by = row["created_by"] or ""
            by_session.append({
                "id": row["id"],
                "name": row["name"],
                "err1": err1,
                "err2": err2,
                "created_by": created_by,
                "completed_at": row["completed_at"] or row["created_at"],
            })

            # Aggregate per user
            if created_by:
                if created_by not in by_user:
                    by_user[created_by] = {"err1": 0, "err2": 0}
                by_user[created_by]["err1"] += err1
                by_user[created_by]["err2"] += err2

        return {
            "total_err1": total_err1,
            "total_err2": total_err2,
            "by_session": by_session,
            "by_user": by_user,
        }


def list_sessions_by_user(user_phone: str) -> List[Dict]:
    """List sessions created by a specific user."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM sessions WHERE created_by = ? ORDER BY created_at DESC",
            (user_phone,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("config"):
                d["config"] = json.loads(d["config"])
            results.append(d)
        return results
