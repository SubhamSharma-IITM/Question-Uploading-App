"""Timestamped backup utility.

Mandatory backup policy: BEFORE any file transformation, create a
timestamped backup copy for revert capability.
"""

import shutil
from pathlib import Path

from app.utils import now_stamp, _safe_print


def create_backup(file_path: Path, backup_dir: Path, stage: str) -> Path:
    """Create a timestamped backup of a file.

    Returns the path to the backup file.
    Pattern: backups/{stage}_{filename}_{YYYYMMDD_HHMMSS}.json
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    stem = file_path.stem
    suffix = file_path.suffix
    stamp = now_stamp()
    backup_name = f"{stage}_{stem}_{stamp}{suffix}"
    backup_path = backup_dir / backup_name
    shutil.copy2(file_path, backup_path)
    _safe_print(f"  Backup: {backup_path.name}")
    return backup_path


def list_backups(backup_dir: Path, stage: str = None) -> list:
    """List all backups, optionally filtered by stage prefix."""
    if not backup_dir.exists():
        return []
    backups = sorted(backup_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if stage:
        backups = [b for b in backups if b.name.startswith(f"{stage}_")]
    return backups
