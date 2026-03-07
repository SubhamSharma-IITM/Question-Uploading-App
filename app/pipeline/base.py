"""Base pipeline stage adapter — ABC that all stage adapters implement.

Each stage adapter forks the corresponding CLI script, replaces print()
with self.log(), replaces argparse with method params, replaces file I/O
with session directory paths.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from app.backup import create_backup
from app.db import insert_log, upsert_checkpoint
from app.sse import sse_manager


class BasePipelineStage(ABC):
    """Interface all stage adapters implement."""

    def __init__(self, session_id: str, session_dir: Path):
        self.session_id = session_id
        self.session_dir = session_dir

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

    def save_checkpoint(self, status: str, data: Optional[Dict] = None) -> None:
        """Save stage checkpoint for resume."""
        upsert_checkpoint(self.session_id, self.stage_name, status, data)

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
