"""Base pipeline stage adapter — ABC that all stage adapters implement.

Each stage adapter forks the corresponding CLI script, replaces print()
with self.log(), replaces argparse with method params, replaces file I/O
with session directory paths.

Includes ledger system for durable state persistence:
- ledger_get(key) → read a sub-step state from checkpoint data
- ledger_set(key, value) → atomically update a sub-step state
- ledger_verify_file(path) → confirm a file exists on disk
- ledger_get_all() → read entire checkpoint data dict

Ledger data is stored in the SQLite checkpoint `data` JSON column,
inside a `sub_steps` dict. This provides durable state persistence
across server restarts without additional files.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.backup import create_backup
from app.db import insert_log, upsert_checkpoint, get_checkpoint
from app.sse import sse_manager


class BasePipelineStage(ABC):
    """Interface all stage adapters implement.

    Provides logging, progress, backup, and ledger-based state persistence.
    """

    def __init__(self, session_id: str, session_dir: Path):
        self.session_id = session_id
        self.session_dir = session_dir
        self._stop_requested = False

    # ── Logging & Progress ────────────────────────────────────────────

    def log(self, level: str, message: str) -> None:
        """Log to DB + emit via SSE."""
        insert_log(self.session_id, self.stage_name, level, message)
        sse_manager.emit(self.session_id, {
            "type": "log",
            "stage": self.stage_name,
            "level": level,
            "message": message,
        })

    def progress(self, pct: float, message: str = "") -> None:
        """Emit a progress update via SSE."""
        sse_manager.emit(self.session_id, {
            "type": "progress",
            "stage": self.stage_name,
            "pct": pct,
            "message": message,
        })

    def backup(self, file_path: Path) -> Path:
        """Create timestamped backup before mutation."""
        backup_dir = self.session_dir / "backups"
        return create_backup(file_path, backup_dir, self.stage_name)

    # ── Stop / Cancel ─────────────────────────────────────────────────

    def request_stop(self) -> None:
        """Request graceful stop. Checked by long-running loops."""
        self._stop_requested = True

    @property
    def should_stop(self) -> bool:
        return self._stop_requested

    # ── Checkpoint (stage-level) ──────────────────────────────────────

    def save_checkpoint(self, status: str, data: Optional[Dict] = None) -> None:
        """Save stage checkpoint for resume. Merges with existing sub_steps."""
        existing = self._load_checkpoint_data()
        if data:
            existing.update(data)
        upsert_checkpoint(self.session_id, self.stage_name, status, existing)

    def get_checkpoint_status(self) -> Optional[str]:
        """Get current checkpoint status for this stage."""
        cp = get_checkpoint(self.session_id, self.stage_name)
        return cp.get("status") if cp else None

    # ── Ledger (sub-step level) ───────────────────────────────────────
    #
    # The ledger stores sub-step states inside the checkpoint `data` dict
    # under the key `sub_steps`. Example:
    #   {"sub_steps": {"batch_submit": {"status": "submitted", "batch_id": "..."}}}
    #
    # Ledger updates are atomic — read existing, merge, write back.

    def _load_checkpoint_data(self) -> Dict:
        """Load the full checkpoint data dict (or empty dict if none)."""
        cp = get_checkpoint(self.session_id, self.stage_name)
        if not cp:
            return {}
        data = cp.get("data", {})
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                data = {}
        return data if isinstance(data, dict) else {}

    def ledger_get_all(self) -> Dict:
        """Read the entire sub_steps dict from checkpoint."""
        data = self._load_checkpoint_data()
        return data.get("sub_steps", {})

    def ledger_get(self, key: str) -> Optional[Dict]:
        """Read a single sub-step state from the ledger.

        Args:
            key: Sub-step name (e.g. 'batch_submit', 'pass1', 'merge')

        Returns:
            Sub-step dict or None if not found.
        """
        steps = self.ledger_get_all()
        return steps.get(key)

    def ledger_set(self, key: str, value: Dict, status: Optional[str] = None) -> None:
        """Atomically update a sub-step in the ledger.

        Reads existing checkpoint data, merges the sub-step update,
        writes back. If status is provided, also updates the checkpoint status.

        Args:
            key: Sub-step name (e.g. 'batch_submit')
            value: Sub-step state dict (e.g. {"status": "submitted", "batch_id": "..."})
            status: Optional checkpoint status to set (e.g. 'in_progress', 'completed')
        """
        data = self._load_checkpoint_data()
        if "sub_steps" not in data:
            data["sub_steps"] = {}
        data["sub_steps"][key] = value
        cp_status = status or self.get_checkpoint_status() or "in_progress"
        upsert_checkpoint(self.session_id, self.stage_name, cp_status, data)

    def ledger_verify_file(self, relative_path: str) -> bool:
        """Verify a file exists on disk relative to session directory.

        Args:
            relative_path: Path relative to session_dir (e.g. 'quality_check/pass1_batch_results.jsonl')

        Returns:
            True if file exists and is non-empty.
        """
        full_path = self.session_dir / relative_path
        return full_path.exists() and full_path.stat().st_size > 0

    def ledger_verify_files(self, paths: List[str]) -> Dict[str, bool]:
        """Verify multiple files exist. Returns {path: exists_bool}."""
        return {p: self.ledger_verify_file(p) for p in paths}

    # ── OpenAI Batch Splitting ──────────────────────────────────────────
    #
    # OpenAI Batch API has a 200MB file size limit (209715200 bytes).
    # Use 190MB safety margin. Split by complete JSONL lines.

    OPENAI_BATCH_MAX_BYTES = 190 * 1024 * 1024  # 190 MB

    def split_jsonl_by_size(self, jsonl_path: "Path", max_bytes: int = 0) -> List["Path"]:
        """Split a JSONL file into chunks that each fit under max_bytes.

        Splits by complete lines (never breaks a JSON object mid-line).
        Returns list of chunk file paths. If no split needed, returns [original_path].
        """
        if max_bytes <= 0:
            max_bytes = self.OPENAI_BATCH_MAX_BYTES

        file_size = jsonl_path.stat().st_size
        if file_size <= max_bytes:
            return [jsonl_path]

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]

        avg_line_size = file_size / max(len(lines), 1)
        lines_per_chunk = max(1, int(max_bytes / avg_line_size))

        chunks: List["Path"] = []
        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_path = jsonl_path.with_suffix(f".chunk{len(chunks)}.jsonl")
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.writelines(chunk_lines)
            # Safety: if a single chunk still exceeds limit, split in half
            if chunk_path.stat().st_size > max_bytes and len(chunk_lines) > 1:
                chunk_path.unlink()
                mid = len(chunk_lines) // 2
                for sub_lines, suffix in [(chunk_lines[:mid], f".chunk{len(chunks)}a"),
                                          (chunk_lines[mid:], f".chunk{len(chunks)}b")]:
                    sp = jsonl_path.with_suffix(f"{suffix}.jsonl")
                    with open(sp, "w", encoding="utf-8") as f:
                        f.writelines(sub_lines)
                    chunks.append(sp)
            else:
                chunks.append(chunk_path)

        self.log("info", f"Split {jsonl_path.name} ({file_size // (1024*1024)}MB) into {len(chunks)} chunks")
        return chunks

    # ── Abstract interface ────────────────────────────────────────────

    @abstractmethod
    def dry_run(self, **kwargs) -> Dict[str, Any]:
        """Preview what the stage will do, without making changes."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the stage. Returns result dict."""
        ...

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """The stage identifier (e.g. 'stage1_upload')."""
        ...
