"""Shared utilities: safe print, JSON helpers, logging formatters, concurrency."""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Concurrency guard for memory-intensive operations ──
# Prevents OOM from concurrent FAISS/embedding operations on 2GB Render.
# Initialized by lifespan in main.py, used by routes.
_task_semaphore: asyncio.Semaphore = None  # type: ignore


def init_semaphore(max_tasks: int = 3) -> None:
    """Called at startup to initialize the global semaphore."""
    global _task_semaphore
    _task_semaphore = asyncio.Semaphore(max_tasks)


async def guarded_thread(func, *args, **kwargs):
    """Run a blocking function in a thread, guarded by the global semaphore.

    Limits concurrent memory-intensive operations (FAISS, embeddings, large
    metadata processing). Prevents OOM on 2GB instances when multiple users
    trigger heavy operations simultaneously.
    """
    if _task_semaphore is None:
        return await asyncio.to_thread(func, *args, **kwargs)
    async with _task_semaphore:
        return await asyncio.to_thread(func, *args, **kwargs)


def _safe_print(text: str) -> None:
    """Print with ASCII fallback for Windows cp1252 console."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def now_stamp() -> str:
    """Current local time as YYYYMMDD_HHMMSS for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> Any:
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """Save data to JSON file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def short_uuid() -> str:
    """Generate a short 8-char UUID."""
    import uuid
    return uuid.uuid4().hex[:8]


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
