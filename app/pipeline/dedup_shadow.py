"""Per-session in-flight FAISS shadow files.

Purpose
-------
Closes the Stage 8 -> Stage 10 race window without polluting the global
`db_reference` FAISS index. The original FIX-A migrated absorption *code*
into `stage8_duplicates.py` but left the *trigger* at Stage 10 post-upload.
That left a 30+ minute race window where Operator B's Stage 8 cross-check
could not see Operator A's already-survived-Stage-8 vectors because A was
still in Stage 9 QA.

How it works
------------
At Stage 8 -> Stage 9 advance, a per-session "in-flight" FAISS index is
written to ``data/sessions/{sid}/dedup_shadow/inflight.{faiss,json}``
containing only the surviving (non-quarantined) session vectors.

Stage 8 cross-check (in any session) reads the global ``db_reference``
index PLUS every OTHER session's inflight shadow and unions matches.
Operator B now sees Operator A's in-flight vectors as duplicates even
though A is still in Stage 9.

At Stage 10 post-upload absorption, A's vectors are written to global
``db_reference`` with real ``mongo_id`` keys (existing flow), and A's
shadow is then deleted. The global index never sees qid-tagged entries.

Cleanup paths
-------------
1. Stage 10 absorb success     -> shadow deleted (the canonical path)
2. App startup stale sweep     -> shadows > 24h removed (catches abandons)
3. Session deletion            -> entire session dir removed (shadow with it)

There is NO Stage 9 quarantine hook. If a question is auto-quarantined
during Stage 9, its shadow entry is harmless until either Stage 10 absorb
removes the whole shadow or the 24h stale sweep does. False-positive
duplicate matches against an eventually-quarantined qid are operationally
benign: the matching operator sees "inflight match against session X",
sees no /question/details preview, and judges accordingly.

Architectural rationale
-----------------------
- NO qid-tagged entries enter global FAISS (clean long-term hygiene).
- NO per-failure cleanup hooks (Stage 9 quarantine, Stage 10 upload fail,
  session abandonment) -- a single stale-sweep handles all failure paths.
- Shadow read at cross-check time is O(per-shadow), and shadows are tiny
  (~30-100 vectors per session) -- negligible search latency vs the 200K
  global index.
- Atomic tmp+replace writes; LOCK_EX not required since each shadow is
  written by exactly one writer (the owning session).
"""
from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SHADOW_DIR_NAME = "dedup_shadow"
SHADOW_INDEX_FILE = "inflight.faiss"
SHADOW_META_FILE = "inflight.json"
DEFAULT_STALE_HOURS = 24

# Build provenance written into every shadow metadata (matches main FAISS
# index's FIX-H stamps). Bump when the extractor or model changes.
SHADOW_TEXT_EXTRACTOR_VERSION = "v2.mathml-to-latex+img-keys"
SHADOW_EMBEDDING_MODEL = "text-embedding-3-small"


def get_shadow_dir(sessions_dir: Path, session_id: str) -> Path:
    return Path(sessions_dir) / session_id / SHADOW_DIR_NAME


def shadow_exists(sessions_dir: Path, session_id: str) -> bool:
    d = get_shadow_dir(sessions_dir, session_id)
    return (d / SHADOW_INDEX_FILE).exists() and (d / SHADOW_META_FILE).exists()


def _default_log(level: str, message: str) -> None:
    py_level = logging.INFO if level == "info" else (
        logging.WARNING if level == "warning" else (
            logging.ERROR if level == "error" else logging.INFO
        )
    )
    logger.log(py_level, message)


def write_shadow(
    sessions_dir: Path,
    session_id: str,
    session_name: str,
    operator: str,
    qids: List[str],
    texts: List[str],
    embeddings,  # numpy ndarray (N, D) or list of vectors
    log: Optional[Callable[[str, str], None]] = None,
) -> Dict:
    """Atomic write of the session's in-flight FAISS shadow.

    If the qid list is empty, any existing shadow is removed so consumers
    don't see stale data from a previous attempt.

    Returns
    -------
    {"written": int, "path": str}
    """
    if log is None:
        log = _default_log

    if len(qids) != len(texts):
        raise ValueError(f"qid count {len(qids)} != texts count {len(texts)}")

    d = get_shadow_dir(sessions_dir, session_id)
    if not qids:
        if shadow_exists(sessions_dir, session_id):
            delete_shadow(sessions_dir, session_id, log=log)
        log("info", f"No surviving vectors for session {session_id}; "
                      "shadow cleared (post-Stage-8 cohort is empty)")
        return {"written": 0, "path": str(d)}

    if len(qids) != len(embeddings):
        raise ValueError(
            f"qid count {len(qids)} != embedding count {len(embeddings)}"
        )

    import numpy as np
    import faiss

    d.mkdir(parents=True, exist_ok=True)

    vectors = np.asarray(embeddings, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (got shape {vectors.shape})")
    faiss.normalize_L2(vectors)
    dim = int(vectors.shape[1])

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    metadata = {
        "schema_version": 1,
        "session_id": session_id,
        "session_name": session_name or "",
        "operator": operator or "",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "id_map": list(qids),
        "text_map": {qid: (texts[i] or "")[:200] for i, qid in enumerate(qids)},
        "build_rule": {
            "text_extractor_version": SHADOW_TEXT_EXTRACTOR_VERSION,
            "embedding_model": SHADOW_EMBEDDING_MODEL,
            "embedding_dim": dim,
        },
    }

    idx_path = d / SHADOW_INDEX_FILE
    meta_path = d / SHADOW_META_FILE
    idx_tmp = idx_path.with_suffix(idx_path.suffix + ".tmp")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")

    faiss.write_index(index, str(idx_tmp))
    meta_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(idx_tmp, idx_path)
    os.replace(meta_tmp, meta_path)

    log("info", f"Shadow written for session {session_id}: "
                  f"{len(qids)} vectors -> {d.name}/")
    return {"written": len(qids), "path": str(d)}


def read_shadow(
    sessions_dir: Path,
    session_id: str,
    log: Optional[Callable[[str, str], None]] = None,
) -> Optional[Tuple]:
    """Read a single session's shadow.

    Returns
    -------
    (faiss_index, metadata_dict) or None if missing / corrupt / build-drift.
    """
    if log is None:
        log = _default_log

    d = get_shadow_dir(sessions_dir, session_id)
    idx_path = d / SHADOW_INDEX_FILE
    meta_path = d / SHADOW_META_FILE
    if not idx_path.exists() or not meta_path.exists():
        return None

    try:
        import faiss
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Build-rule drift check: skip shadows from older code versions.
        # (Equivalent to FIX-H drift detection on global db_reference.)
        br = meta.get("build_rule", {})
        if br.get("text_extractor_version") != SHADOW_TEXT_EXTRACTOR_VERSION:
            log("warning",
                f"Shadow {session_id} build drift "
                f"(disk={br.get('text_extractor_version')}, "
                f"code={SHADOW_TEXT_EXTRACTOR_VERSION}); skipping")
            return None
        index = faiss.read_index(str(idx_path))
        return (index, meta)
    except Exception as e:
        log("warning", f"Failed to read shadow for {session_id}: {e}")
        return None


def read_all_shadows(
    sessions_dir: Path,
    exclude_session_id: Optional[str] = None,
    log: Optional[Callable[[str, str], None]] = None,
) -> List[Tuple]:
    """Read every session's shadow except the caller's.

    Returns list of (faiss_index, metadata_dict) tuples.
    """
    if log is None:
        log = _default_log

    sessions_dir = Path(sessions_dir)
    if not sessions_dir.exists():
        return []

    shadows: List[Tuple] = []
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        sid = session_dir.name
        if exclude_session_id and sid == exclude_session_id:
            continue
        if not shadow_exists(sessions_dir, sid):
            continue
        result = read_shadow(sessions_dir, sid, log=log)
        if result is not None:
            shadows.append(result)
    return shadows


def delete_shadow(
    sessions_dir: Path,
    session_id: str,
    log: Optional[Callable[[str, str], None]] = None,
) -> bool:
    """Remove the session's shadow files. Idempotent."""
    if log is None:
        log = _default_log

    d = get_shadow_dir(sessions_dir, session_id)
    if not d.exists():
        return False
    removed = 0
    for fname in (SHADOW_INDEX_FILE, SHADOW_META_FILE):
        p = d / fname
        try:
            if p.exists():
                p.unlink()
                removed += 1
        except Exception as e:
            log("warning", f"Failed to remove {p.name}: {e}")
    if removed:
        log("info", f"Shadow removed for session {session_id} "
                      f"({removed} file(s))")
    return removed > 0


def cleanup_stale_shadows(
    sessions_dir: Path,
    max_age_hours: float = DEFAULT_STALE_HOURS,
    log: Optional[Callable[[str, str], None]] = None,
) -> Dict:
    """Delete shadow files older than `max_age_hours` (default 24h).

    Called at app startup. Catches:
      - Sessions abandoned mid-flow (browser closed, server killed, etc.)
      - Sessions where Stage 10 absorb crashed before delete_shadow fired
      - Sessions whose owner deleted the session record but disk lingered
    """
    if log is None:
        log = _default_log

    sessions_dir = Path(sessions_dir)
    if not sessions_dir.exists():
        return {"scanned": 0, "deleted": 0, "kept": 0}

    now = time.time()
    cutoff = now - max_age_hours * 3600
    scanned = deleted = kept = 0

    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        scanned += 1
        sid = session_dir.name
        meta_path = get_shadow_dir(sessions_dir, sid) / SHADOW_META_FILE
        if not meta_path.exists():
            continue
        try:
            mtime = meta_path.stat().st_mtime
            if mtime < cutoff:
                delete_shadow(sessions_dir, sid, log=log)
                deleted += 1
            else:
                kept += 1
        except Exception as e:
            log("warning", f"Failed to inspect shadow {sid}: {e}")

    log("info",
        f"Stale shadow cleanup: scanned={scanned} deleted={deleted} "
        f"kept={kept} cutoff_h={max_age_hours}")
    return {"scanned": scanned, "deleted": deleted, "kept": kept}
