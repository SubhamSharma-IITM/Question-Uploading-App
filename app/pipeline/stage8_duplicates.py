"""Stage 8: Duplicate Checking — pipeline adapter.

Two-phase duplicate detection:
  1. Self-duplicate check: find duplicates within the session
  2. Cross-DB check: compare session questions against the existing 143K question database

Engine selection (auto):
  - PRIMARY: OpenAI text-embedding-3-small + FAISS — O(n log n), semantic similarity
  - FALLBACK: Trigram Jaccard — O(n²) within subject, no API key needed

Forked from: CodeBase/PCMB_Duplicate_Checker.py
CHANGES:
  - Adapted for web app session-based architecture
  - Auto engine selection (embeddings if API key + faiss available, else trigram)
  - Detailed stage-by-stage logging for UX
  - Quarantine filtering from shared db helper
"""

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.pipeline.base import BasePipelineStage

# ── Constants ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 100
MAX_TEXT_LENGTH = 8000

# FIX-B/H: provenance + locking
TEXT_EXTRACTOR_VERSION = "v2.mathml-to-latex+img-keys"  # bump when extractor changes
FAISS_LOCK_TIMEOUT_S = 60.0
FAISS_LOCK_POLL_S = 0.005

HTML_TAG_RE = re.compile(r'<[^>]+>')
MATHML_RE = re.compile(r'<math[\s\S]*?</math>')
WHITESPACE_RE = re.compile(r'\s+')
IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)


# ── Exceptions (typed; surfaced to operator UI) ──────────────────────────
class EmbeddingBatchFailure(RuntimeError):
    """FIX-G: raised when an OpenAI embedding batch fails.
    Halts Stage 8 instead of silently zero-filling (which produced
    orthogonal vectors that matched nothing -- a silent dedup blindness).
    Operator should reload to retry."""


class FAISSLockTimeout(RuntimeError):
    """FIX-B: raised when LOCK_SH or LOCK_EX cannot be acquired within
    FAISS_LOCK_TIMEOUT_S. Operator should wait for active writer to
    finish (typically 5-10s for a 1 GB index) then reload."""


class FAISSAbsorptionInsufficientRam(RuntimeError):
    """FIX-E: raised when the OOM guard refuses to absorb due to low
    anon memory. Operator should close other sessions and reload to retry.
    Carries free_mb, needed_mb, index_mb for diagnostic UI."""
    def __init__(self, free_mb: float, needed_mb: float, index_mb: float):
        self.free_mb = free_mb
        self.needed_mb = needed_mb
        self.index_mb = index_mb
        super().__init__(
            f"FAISS absorption refused: only {free_mb:.0f} MB allocatable anon, "
            f"need {needed_mb:.0f} MB (1.2x of {index_mb:.0f} MB index). "
            f"Close other sessions and reload to retry."
        )


class FAISSMetadataDrift(RuntimeError):
    """FIX-H: raised when metadata.json's build_rule / embedding_model /
    text_extractor_version doesn't match the code's current expected
    values. Prevents silent dedup drift when extractor or model is
    upgraded but the on-disk index is stale."""


def _acquire_faiss_lock(lock_fd, mode, timeout_s: float = FAISS_LOCK_TIMEOUT_S):
    """FIX-B: cross-platform polling lock acquisition with timeout.
    mode: fcntl.LOCK_SH or fcntl.LOCK_EX (LOCK_NB is OR'd in for polling).
    Raises FAISSLockTimeout on timeout. Poll interval 5 ms keeps latency
    overhead negligible vs real production FAISS write time (~5-10s)."""
    import time as _t
    import sys
    start = _t.monotonic()
    if sys.platform == "win32":
        import msvcrt
        mode_arg = msvcrt.LK_NBLCK if mode == "EX" else msvcrt.LK_NBLCK  # Windows has no shared lock
        while True:
            try:
                msvcrt.locking(lock_fd.fileno(), mode_arg, 1)
                return True
            except (OSError, IOError):
                if _t.monotonic() - start > timeout_s:
                    raise FAISSLockTimeout(
                        f"Could not acquire {mode} lock within {timeout_s:.0f}s"
                    )
                _t.sleep(FAISS_LOCK_POLL_S)
    else:
        import fcntl
        flock_mode = fcntl.LOCK_SH if mode == "SH" else fcntl.LOCK_EX
        while True:
            try:
                fcntl.flock(lock_fd, flock_mode | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if _t.monotonic() - start > timeout_s:
                    raise FAISSLockTimeout(
                        f"Could not acquire {mode} lock within {timeout_s:.0f}s"
                    )
                _t.sleep(FAISS_LOCK_POLL_S)


def _release_faiss_lock(lock_fd):
    """FIX-B: release the FAISS lock (cross-platform)."""
    import sys
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except Exception:
        pass


def _atomic_write_faiss_and_metadata(index, metadata: dict, idx_path, meta_path):
    """FIX-B: atomic tmp + fsync + os.replace + parent-dir fsync.
    Readers always see either the complete old file or the complete new
    file -- never a torn intermediate state. Protects against partial
    writes if the process crashes mid-write."""
    import faiss
    idx_tmp = idx_path.with_suffix(idx_path.suffix + ".tmp")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    faiss.write_index(index, str(idx_tmp))
    with open(idx_tmp, "rb") as f:
        os.fsync(f.fileno())
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(idx_tmp, idx_path)
    os.replace(meta_tmp, meta_path)
    import sys
    if sys.platform != "win32":
        try:
            dir_fd = os.open(idx_path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass


# ── cgroup v2 anon allocatable memory accounting ─────────────────────────
# Used by the absorber's OOM guard. Lives here because absorption is owned
# by Stage 8 (FIX-A). Stage 10 imports these for backward compat.

def _parse_cgroup_anon_allocatable(memory_max_str: str, memory_stat_str: str) -> Optional[float]:
    """Pure parser -- testable without filesystem mocking.

    Returns allocatable MB after subtracting only NON-RECLAIMABLE categories
    from the cgroup limit (anon + slab + kernel + sock + pagetables).
    File page cache is reclaimable; kernel auto-evicts it on allocation
    pressure, so it must NOT be counted as used.

    Returns None when the limit is unbounded ("max") or input is malformed.
    """
    raw = memory_max_str.strip()
    if raw in ("max", ""):
        return None
    try:
        limit = int(raw)
    except ValueError:
        return None
    stat = {}
    for line in memory_stat_str.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            try:
                stat[parts[0]] = int(parts[1])
            except ValueError:
                pass
    non_reclaimable = (stat.get("anon", 0)
                       + stat.get("slab", 0)
                       + stat.get("kernel", 0)
                       + stat.get("sock", 0)
                       + stat.get("pagetables", 0))
    return (limit - non_reclaimable) / (1024 * 1024)


def _read_cgroup_anon_allocatable_mb() -> Optional[float]:
    """Real allocatable memory: cgroup limit minus only NON-RECLAIMABLE memory.

    Returns None on non-cgroup systems (Windows local dev) -- caller skips
    the guard entirely.
    """
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            mm = f.read()
        with open("/sys/fs/cgroup/memory.stat") as f:
            ms = f.read()
        return _parse_cgroup_anon_allocatable(mm, ms)
    except (FileNotFoundError, PermissionError, OSError):
        return None


# ── Module-level absorber (FIX-A) ────────────────────────────────────────
# FIX-A: absorption code lives in this module. Stage 10 calls into it
# after API upload completes (where mongo_ids are available). Carries
# FIX-B (LOCK_EX blocking + atomic write), FIX-E (1.2x OOM raise),
# FIX-G (zero-vec raise), FIX-H (build_rule provenance).

def absorb_db_entries_into_global(
    db_entries,
    log=None,
):
    """Generate embeddings for new DB entries and absorb them into the
    global FAISS index. This is the single canonical absorber.

    Args:
        db_entries: List of (meta_qid, mongo_id, db_entry) tuples. mongo_id
            is the 24-hex OID returned by the Acadza API after upload --
            it's the key written into FAISS's id_map.
        log: callable(level, msg) for structured logging; defaults to
            standard print with [absorb|level] prefix.

    Returns:
        {"embedded": int, "status": "ok"|"unavailable", "reason": str|None}

    Raises:
        FAISSAbsorptionInsufficientRam: OOM guard tripped (FIX-E)
        EmbeddingBatchFailure: OpenAI batch failed (FIX-G)
        FAISSLockTimeout: write lock could not be acquired within 60s (FIX-B)
    """
    if log is None:
        def log(level, msg):
            print(f"[absorb|{level}] {msg}", flush=True)

    from app.config import settings

    try:
        import faiss
        import numpy as np
    except ImportError:
        log("warning", "FAISS not available -- skipping embedding update")
        return {"embedded": 0, "status": "unavailable", "reason": "faiss_not_installed"}

    openai_key = getattr(settings, "openai_api_key", None) or ""
    if not openai_key or len(openai_key) < 10:
        log("warning", "OpenAI API key not configured -- skipping embedding update")
        return {"embedded": 0, "status": "unavailable", "reason": "openai_key_missing"}

    global_dir = Path(getattr(settings, "db_index_dir", ""))
    index_path = global_dir / "db_reference" / "faiss.index"
    meta_path = global_dir / "db_reference" / "metadata.json"
    if not index_path.exists():
        log("warning", f"Global FAISS index not found at {index_path} -- skipping")
        return {"embedded": 0, "status": "unavailable", "reason": "index_missing"}

    # FIX-E: 1.2x threshold + raise on trip (no silent return).
    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    needed_mb = index_size_mb * 1.2
    allocatable_mb = _read_cgroup_anon_allocatable_mb()
    if allocatable_mb is not None:
        log("info",
            f"Memory check: {allocatable_mb:.0f}MB allocatable (anon-based), "
            f"need {needed_mb:.0f}MB (1.2x of {index_size_mb:.0f}MB index)")
        if allocatable_mb < needed_mb:
            log("error",
                f"OOM guard tripped: {allocatable_mb:.0f}MB allocatable, "
                f"need {needed_mb:.0f}MB. Raising to halt absorption.")
            raise FAISSAbsorptionInsufficientRam(allocatable_mb, needed_mb, index_size_mb)

    log("info", "Loading global FAISS index for absorb...")
    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    id_map = metadata.get("id_map", [])
    text_map = metadata.get("text_map", {})

    log("info", f"Current index: {index.ntotal} vectors, adding {len(db_entries)} new...")

    # Shared extractor -- single source of truth between Stage 8 dedup and absorb.
    texts = []
    new_ids = []
    for meta_qid, mongo_id, db_entry in db_entries:
        text = extract_text_for_embedding(db_entry)
        if text and len(text) >= 10:
            texts.append(text)
            new_ids.append(mongo_id)
        else:
            log("warning",
                f"Empty embedding text for {meta_qid} (qtype={db_entry.get('questionType')}) -- "
                "skipping FAISS vector for this entry")

    if not texts:
        log("info", "No texts extracted for embedding")
        return {"embedded": 0, "status": "ok"}

    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

    # FIX-G: hard-fail on embedding API failure (no zero-vector fallback).
    BATCH_SIZE = 100
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        valid = [t if t.strip() else "empty" for t in batch]
        try:
            response = client.embeddings.create(model="text-embedding-3-small", input=valid)
            for item in response.data:
                all_embeddings.append(np.array(item.embedding, dtype="float32"))
        except Exception as e:
            log("error",
                f"Embedding batch {i // BATCH_SIZE + 1} failed: {e}. "
                f"Halting absorption to prevent silent dedup blindness.")
            raise EmbeddingBatchFailure(
                f"OpenAI embedding batch failed during absorption: {e}"
            ) from e

    log("info", f"Generated {len(all_embeddings)} embeddings")

    vectors = np.array(all_embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    index.add(vectors)

    for i, mongo_id in enumerate(new_ids):
        id_map.append(mongo_id)
        text_map[mongo_id] = texts[i][:500]

    # FIX-B + FIX-H: blocking LOCK_EX, atomic write, provenance stamp.
    lock_path = index_path.parent / ".faiss_write.lock"
    lock_fd = open(lock_path, "w")
    try:
        try:
            _acquire_faiss_lock(lock_fd, "EX")
        except FAISSLockTimeout as e:
            log("error", f"FAISS write lock timeout: {e}")
            lock_fd.close()
            raise

        metadata["id_map"] = id_map
        metadata["text_map"] = text_map
        metadata["embedding_model"] = EMBEDDING_MODEL
        metadata["text_extractor_version"] = TEXT_EXTRACTOR_VERSION
        metadata["build_rule"] = f"{EMBEDDING_MODEL}+{TEXT_EXTRACTOR_VERSION}"
        _atomic_write_faiss_and_metadata(index, metadata, index_path, meta_path)
    finally:
        _release_faiss_lock(lock_fd)
        lock_fd.close()

    log("success", f"FAISS index updated atomically: {index.ntotal} total vectors (added {len(new_ids)})")
    return {"embedded": len(new_ids), "status": "ok"}


def _safe_truncate_html(html: str, max_len: int = 3000) -> str:
    """Truncate HTML without cutting inside tags or MathML blocks.

    Finds the last complete tag boundary before max_len.
    Prevents 'math input error' from broken <math>...</math> blocks.
    """
    if not html or len(html) <= max_len:
        return html
    # Find the last '>' before or at max_len
    cut = html.rfind('>', 0, max_len)
    if cut == -1:
        return html[:max_len]
    truncated = html[:cut + 1]
    # If we cut inside a <math> block, find the last complete </math> before cut
    open_math = truncated.rfind('<math')
    close_math = truncated.rfind('</math>')
    if open_math > close_math:
        # Unclosed <math> — cut before it
        truncated = truncated[:open_math]
    return truncated


# ── Text helpers ─────────────────────────────────────────────────────────

def _mathml_to_latex_text(html_text: str) -> str:
    """Convert MathML blocks to LaTeX for semantic preservation in embeddings.

    Uses the production MathML→LaTeX converter from PROOF-READING_v2.0 codebase.
    Falls back to text extraction if converter fails.
    """
    try:
        from app.pipeline._mathml_to_latex import convert_mathml_to_latex
        return convert_mathml_to_latex(html_text)
    except Exception:
        # Fallback: extract text content from MathML tags
        def _extract_text(m):
            inner = re.sub(r'<[^>]+>', ' ', m.group())
            import html as html_mod
            inner = html_mod.unescape(inner)
            inner = WHITESPACE_RE.sub(' ', inner).strip()
            return f' {inner} ' if inner else ' '
        return MATHML_RE.sub(_extract_text, html_text)


def _strip_html(html: str) -> str:
    """Strip HTML tags and normalize for comparison."""
    if not html:
        return ""
    text = _mathml_to_latex_text(html)
    text = HTML_TAG_RE.sub(' ', text)
    import html as html_mod
    text = html_mod.unescape(text)
    text = WHITESPACE_RE.sub(' ', text).strip()
    return text


def _strip_html_keep_images(html: str) -> str:
    """Strip HTML but preserve image URL fingerprints for matching."""
    if not html:
        return ""
    img_urls = IMG_SRC_RE.findall(html)
    text = _mathml_to_latex_text(html)
    text = HTML_TAG_RE.sub(' ', text)
    import html as html_mod
    text = html_mod.unescape(text)
    text = WHITESPACE_RE.sub(' ', text).strip()
    if img_urls:
        img_keys = [url.split('/')[-1][:20] for url in img_urls]
        text += ' [IMGS:' + ','.join(img_keys) + ']'
    return text


def extract_text_for_embedding(entry: Dict) -> str:
    """Extract plain text from a metadata entry for embedding/comparison.

    Single source of truth — used by Stage 8 (self + cross-check) AND Stage 10
    (FAISS absorption after upload). Both must produce IDENTICAL text for the
    same entry so dedup is consistent.

    Embedding strategy per question type:
      - flat (scq/mcq/integerQuestion/subjective): question HTML stripped
      - matchQuestion: table cell text + canonical rows mapping summary
        (e.g., "1->C; 2->A,E; ..."). Both cell content AND mapping are strong
        dedup signals — two match questions with same cells but different
        mappings are NOT duplicates.
      - passageQuestion: main passage stem + FIRST sub-question stem.
        Subsequent sub-questions are correlated with first; embedding only
        the first keeps the signal focused without inflating noise.
    """
    parts = []
    strip_fn = _strip_html_keep_images
    qtype = entry.get("questionType", "")
    qtypes_to_check = [qtype] if qtype else [
        "scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion", "subjective"
    ]

    for qt in qtypes_to_check:
        qdata = entry.get(qt)
        if not isinstance(qdata, dict):
            continue

        q_text = qdata.get("question", "") or qdata.get("ques", "")
        if q_text:
            parts.append(strip_fn(q_text))

        # matchQuestion: append canonical rows mapping summary so the
        # embedding distinguishes two questions with same cells but
        # different answer mappings.
        if qt == "matchQuestion":
            rows = qdata.get("rows") or []
            if isinstance(rows, list) and rows:
                mapping_summary = "; ".join(
                    f"{i+1}->{','.join(r) if isinstance(r, list) else str(r)}"
                    for i, r in enumerate(rows)
                )
                parts.append(mapping_summary)

        # passageQuestion: passage stem + FIRST sub-question only
        if qt == "passageQuestion":
            passage = qdata.get("passage", "")
            if passage:
                parts.append(strip_fn(passage))
            sub_qs = qdata.get("questions", [])
            if isinstance(sub_qs, list) and sub_qs:
                first_sq = sub_qs[0]
                if isinstance(first_sq, dict):
                    for sub_type in ("scq", "mcq", "integerQuestion"):
                        sub_data = first_sq.get(sub_type, {})
                        if isinstance(sub_data, dict):
                            st = sub_data.get("ques", "") or sub_data.get("question", "")
                            if st:
                                parts.append(strip_fn(st))
                                break

        if parts:
            break

    text = " ".join(parts)
    return text[:MAX_TEXT_LENGTH] if len(text) > MAX_TEXT_LENGTH else text


def _normalize_for_comparison(text: str) -> str:
    """Aggressively normalize text for trigram similarity."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = WHITESPACE_RE.sub(' ', t).strip()
    return t


def _simple_similarity(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity with trigram fallback (used when FAISS unavailable)."""
    if not text_a or not text_b:
        return 0.0

    na = _normalize_for_comparison(text_a)
    nb = _normalize_for_comparison(text_b)
    if not na or not nb:
        return 0.0

    tokens_a = set(na.split())
    tokens_b = set(nb.split())
    token_sim = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if tokens_a and tokens_b else 0.0

    def trigrams(s):
        return set(s[i:i+3] for i in range(max(0, len(s) - 2)))
    tri_a = trigrams(na)
    tri_b = trigrams(nb)
    tri_sim = len(tri_a & tri_b) / len(tri_a | tri_b) if tri_a and tri_b else 0.0

    return max(token_sim, tri_sim)


# ── Engine detection ─────────────────────────────────────────────────────

def _check_faiss_available() -> bool:
    """Check if faiss-cpu is importable."""
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def _check_openai_available() -> bool:
    """Check if openai API key is configured."""
    from app.config import settings
    key = getattr(settings, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY", "")
    return bool(key and len(key) > 10)


# ── Embedding helpers (production fork) ──────────────────────────────────

def _get_openai_client():
    """Get OpenAI client for embeddings."""
    from app.config import settings
    from openai import OpenAI
    key = getattr(settings, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(api_key=key)


def _get_embeddings_batch(client, texts: List[str]) -> List:
    """Call OpenAI embeddings API for a batch of texts."""
    import numpy as np
    valid_texts = [t if t.strip() else "empty" for t in texts]
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=valid_texts)
    return [np.array(item.embedding, dtype="float32") for item in response.data]


# ── Main stage class ─────────────────────────────────────────────────────

class DuplicateCheckStage(BasePipelineStage):

    @property
    def stage_name(self) -> str:
        return "stage8_duplicates"

    def _get_metadata_dir(self) -> Path:
        d1 = self.session_dir / "metadata_v2_output"
        if d1.exists():
            return d1
        return self.session_dir / "metadata_v2"

    def _get_reports_dir(self) -> Path:
        d = self.session_dir / "duplicate_reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_index_dir(self) -> Path:
        d = self.session_dir / "duplicate_reports" / "indices"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_all_metadata(self, filter_quarantine: bool = True) -> List[Dict]:
        """Load all metadata_v2 entries from either flat or subdirectory layout."""
        meta_dir = self._get_metadata_dir()
        if not meta_dir.exists():
            return []

        all_entries = []
        for sub in sorted(meta_dir.iterdir()):
            if sub.is_dir():
                for f in sub.glob("*_metadata_v2.json"):
                    if "backup" in f.name:
                        continue
                    try:
                        with open(f, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        if isinstance(data, list):
                            all_entries.extend(data)
                    except Exception:
                        continue
            elif sub.suffix == ".json" and "metadata_v2" in sub.name and "backup" not in sub.name:
                try:
                    with open(sub, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        all_entries.extend(data)
                except Exception:
                    continue

        if filter_quarantine and all_entries:
            from app.db import get_quarantined_qids
            q_qids = get_quarantined_qids(self.session_id)
            if q_qids:
                before = len(all_entries)
                all_entries = [e for e in all_entries if e.get("qid") not in q_qids]
                skipped = before - len(all_entries)
                if skipped:
                    self.log("info", f"Filtered {skipped} quarantined questions from input")

        return all_entries

    def _extract_question_text(self, entry: Dict) -> str:
        """Method wrapper for the module-level extract_text_for_embedding."""
        return extract_text_for_embedding(entry)

    def _lookup_db_flags(self, db_qids: Set[str]) -> Dict[str, Dict]:
        """Look up protected flags for matched DB question IDs.

        Protected flags (cannot unselect in UI):
          isPrevious, positivePick, negativePick, isSolvedExample
        These questions MUST be quarantined if duplicated — they are curated DB entries.

        Scans the JSONL DB for just the matched IDs — avoids loading entire DB into memory.
        """
        from app.config import settings
        FLAG_KEYS = ("isPrevious", "positivePick", "negativePick", "isSolvedExample")

        db_path = Path(settings.db_path)
        if not db_path.exists():
            return {}

        flags_map: Dict[str, Dict] = {}
        remaining = set(db_qids)

        try:
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not remaining:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Match by OID (preferred) or qid
                    raw_id = entry.get("_id", "")
                    if isinstance(raw_id, dict) and "$oid" in raw_id:
                        oid = raw_id["$oid"]
                    elif isinstance(raw_id, str) and len(raw_id) == 24:
                        oid = raw_id
                    else:
                        oid = ""
                    qid = entry.get("qid", str(raw_id))
                    # Try matching by OID first, then by qid
                    matched_key = oid if oid in remaining else (qid if qid in remaining else None)
                    if matched_key and matched_key in remaining:
                        flags = {k: bool(entry.get(k)) for k in FLAG_KEYS}
                        flags["has_protected_flag"] = any(flags.values())
                        # Also extract question HTML for preview rendering
                        q_html = ""
                        for qt in ("scq", "mcq", "integerQuestion", "passageQuestion"):
                            obj = entry.get(qt)
                            if isinstance(obj, dict):
                                q_html = obj.get("question", obj.get("ques", ""))
                                if not q_html and qt == "passageQuestion":
                                    q_html = obj.get("passage", "")
                                if q_html:
                                    break
                        flags["question_html"] = q_html if q_html else ""
                        flags_map[matched_key] = flags
                        remaining.discard(matched_key)
        except Exception:
            pass

        return flags_map

    def _detect_engine(self) -> str:
        """Auto-detect best available engine."""
        has_faiss = _check_faiss_available()
        has_openai = _check_openai_available()
        if has_faiss and has_openai:
            return "embeddings"
        return "trigram"

    # ── Embedding-based methods (primary engine) ─────────────────────────

    def _generate_embeddings(self, texts: List[str]) -> List:
        """Generate OpenAI embeddings with batch progress logging."""
        import numpy as np
        client = _get_openai_client()

        all_embeddings = []
        total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        self.log("info", f"Generating embeddings via OpenAI {EMBEDDING_MODEL} ({len(texts)} texts, {total_batches} batches)...")

        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            batch_num = i // EMBEDDING_BATCH_SIZE + 1
            try:
                embeddings = _get_embeddings_batch(client, batch)
                all_embeddings.extend(embeddings)
                if batch_num % 5 == 0 or batch_num == total_batches:
                    self.log("info", f"  Embedding batch {batch_num}/{total_batches} done ({len(all_embeddings)}/{len(texts)})")
                    self.progress(min(40, batch_num / total_batches * 40), f"Embeddings: {len(all_embeddings)}/{len(texts)}")
            except Exception as e:
                # FIX-G: hard-fail instead of silent zero-fill. The old fallback
                # filled np.zeros, producing orthogonal vectors that matched
                # nothing -- silent dedup blindness. Halt Stage 8 so operator
                # can retry from a clean state.
                self.log("error", f"  Embedding batch {batch_num}/{total_batches} failed: {e}")
                self.log("error",
                         f"Halting Stage 8 -- {len(all_embeddings)}/{len(texts)} embeddings "
                         f"generated before failure. Reload to retry.")
                raise EmbeddingBatchFailure(
                    f"OpenAI embedding batch {batch_num}/{total_batches} failed: {e}. "
                    f"Halting Stage 8 to prevent silent dedup blindness."
                ) from e

        return all_embeddings

    def _build_faiss_index(self, embeddings, ids, texts):
        """Build a FAISS index from embeddings."""
        import faiss
        import numpy as np

        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(vectors)
        text_map = {qid: text[:500] for qid, text in zip(ids, texts)}
        self.log("info", f"Built FAISS index: {index.ntotal} vectors ({EMBEDDING_DIM}-dim)")
        return index, ids, text_map

    def _save_session_embeddings(self, ids: List[str], embeddings: List) -> None:
        """Save session embeddings to disk for reuse between self/cross checks."""
        import numpy as np
        emb_dir = self._get_index_dir() / "session_embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_dir / "vectors.npy"), np.array(embeddings, dtype="float32"))
        (emb_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")

    def _load_session_embeddings(self, expected_ids: List[str]):
        """Load saved session embeddings if they match current session questions."""
        import numpy as np
        emb_dir = self._get_index_dir() / "session_embeddings"
        vec_path = emb_dir / "vectors.npy"
        ids_path = emb_dir / "ids.json"
        if not vec_path.exists() or not ids_path.exists():
            return None
        try:
            saved_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if set(saved_ids) != set(expected_ids) or len(saved_ids) != len(expected_ids):
                return None
            vectors = np.load(str(vec_path))
            # Reorder to match expected_ids order
            id_to_idx = {qid: i for i, qid in enumerate(saved_ids)}
            reordered = [vectors[id_to_idx[qid]] for qid in expected_ids]
            return reordered
        except Exception:
            return None

    def _faiss_search(self, index, id_map, query_embedding, k=10, exclude_id=None):
        """Search FAISS index for nearest neighbors."""
        import faiss
        import numpy as np

        query = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = index.search(query, k + 1)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(id_map):
                continue
            matched_id = id_map[idx]
            if exclude_id and matched_id == exclude_id:
                continue
            results.append((matched_id, float(score)))
        return results[:k]

    def _save_faiss_index(self, index, id_map, text_map, name: str):
        """Save a FAISS index to disk for reuse.

        FIX-D: refuses 'db_reference' name. The global cache is the
        single source of truth; mutations to it MUST go through the
        absorber (which holds LOCK_EX and writes atomically). This
        function is now only for session-local indexes (e.g. 'self_session').

        FIX-H: stamps build_rule + embedding_model + text_extractor_version
        into metadata.json so future drift can be detected on load.
        """
        if name == "db_reference":
            # FIX-D: never create a session-local shadow of db_reference.
            # Old code copied this to global -- removed. Use the absorber.
            raise RuntimeError(
                "_save_faiss_index('db_reference') is disallowed (FIX-D). "
                "Use absorb_session_vectors_into_global() to mutate the global cache."
            )

        import faiss
        index_dir = self._get_index_dir() / name
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / "faiss.index"))
        # FIX-H: provenance fields
        metadata = {
            "id_map": id_map,
            "text_map": text_map,
            "dimension": EMBEDDING_DIM,
            "embedding_model": EMBEDDING_MODEL,
            "text_extractor_version": TEXT_EXTRACTOR_VERSION,
            "build_rule": f"{EMBEDDING_MODEL}+{TEXT_EXTRACTOR_VERSION}",
        }
        (index_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

    def _get_global_index_dir(self) -> Optional[Path]:
        """Get global FAISS cache directory from config (shared across sessions)."""
        from app.config import settings
        gdir = getattr(settings, "db_index_dir", None)
        if gdir:
            p = Path(gdir)
            if p.exists():
                return p
        return None

    def _load_faiss_index(self, name: str):
        """Load a previously saved FAISS index.

        FIX-B: acquires fcntl.LOCK_SH on the global lock for db_reference
        reads. Multiple readers can share; writer's LOCK_EX blocks them.
        FIX-C: drops IO_FLAG_MMAP; full-RAM read each time. ~1 GB
        production cost serialized by the upstream guarded_thread cap.
        FIX-D: db_reference is GLOBAL-ONLY -- skips session-local entirely.
        FIX-H: asserts metadata.json build_rule matches code's current
        expected value; loud warning + raise on drift.

        Search order:
          - 'db_reference': global cache only (FIX-D)
          - everything else (e.g. 'self_session'): session-local only
        """
        import faiss

        # FIX-D: db_reference is GLOBAL-ONLY. Session-local is reserved for
        # per-session indexes like 'self_session'.
        if name != "db_reference":
            index_dir = self._get_index_dir() / name
            if (index_dir / "faiss.index").exists():
                # FIX-C: full-RAM read (no IO_FLAG_MMAP)
                index = faiss.read_index(str(index_dir / "faiss.index"))
                metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
                self._assert_metadata_provenance(metadata, name, "session-local")
                self.log("info", f"Loaded session-local FAISS index '{name}': {index.ntotal} vectors (full-RAM)")
                return index, metadata["id_map"], metadata["text_map"]
            return None, None, None

        # db_reference: global cache with LOCK_SH (FIX-B)
        global_dir = self._get_global_index_dir()
        if not global_dir:
            return None, None, None
        global_index_dir = global_dir / name
        index_path = global_index_dir / "faiss.index"
        meta_path = global_index_dir / "metadata.json"
        if not index_path.exists():
            return None, None, None

        lock_path = global_index_dir / ".faiss_write.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(lock_path, "w")
        try:
            _acquire_faiss_lock(lock_fd, "SH")
            # FIX-C: full-RAM read inside the lock so a writer that releases
            # before our search starts can't change the bytes we're reading.
            index = faiss.read_index(str(index_path))
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self._assert_metadata_provenance(metadata, name, "global")
            self.log("info",
                f"Loaded global FAISS index '{name}': {index.ntotal} vectors "
                f"(full-RAM, LOCK_SH held during read)")
            return index, metadata["id_map"], metadata["text_map"]
        finally:
            _release_faiss_lock(lock_fd)
            lock_fd.close()

    def _assert_metadata_provenance(self, metadata: dict, name: str, source: str):
        """FIX-H: verify metadata.json was built with the same
        embedding_model + text_extractor_version as the current code.
        Loud-warn + raise on mismatch so silent drift can't accumulate."""
        expected_model = EMBEDDING_MODEL
        expected_ext = TEXT_EXTRACTOR_VERSION
        got_model = metadata.get("embedding_model")
        got_ext = metadata.get("text_extractor_version")
        if got_model is None and got_ext is None:
            # Pre-FIX-H index (no provenance fields). Warn but allow --
            # the production global index was built before this fix.
            # Operator should rebuild at convenience to get drift detection.
            self.log("warning",
                f"FAISS '{name}' ({source}) has no build provenance metadata. "
                f"Drift detection unavailable. Consider rebuilding the index.")
            return
        if got_model != expected_model or got_ext != expected_ext:
            raise FAISSMetadataDrift(
                f"FAISS '{name}' ({source}) build mismatch: "
                f"embedding_model={got_model!r} (expected {expected_model!r}), "
                f"text_extractor_version={got_ext!r} (expected {expected_ext!r}). "
                f"Rebuild the index before continuing."
            )

    # ── Self-duplicate (embedding engine) ────────────────────────────────

    def _self_check_embeddings(self, questions: List[Dict], threshold: float) -> Dict[str, Any]:
        """Self-duplicate check using embeddings + FAISS batch search. O(n log n)."""
        import numpy as np
        import faiss

        # Stage 1: Extract text
        self.log("info", "Stage 1/4: Extracting question text...")
        texts, ids, subjects = [], [], []
        html_map = {}  # qid -> full question HTML for preview
        for q in questions:
            text = self._extract_question_text(q)
            if text and len(text) >= 10:
                texts.append(text)
                qid = q.get("qid", "?")
                ids.append(qid)
                subjects.append(q.get("subject", "?"))
                for qt in ("scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"):
                    obj = q.get(qt)
                    if isinstance(obj, dict):
                        q_html = obj.get("question", obj.get("ques", ""))
                        # passage: fall back to passage body if `question`/`ques` empty
                        # match: question field IS the table HTML, no fallback needed
                        if not q_html and qt == "passageQuestion":
                            q_html = obj.get("passage", "")
                        if q_html:
                            html_map[qid] = q_html
                            break
        self.log("info", f"Extracted {len(texts)} question texts ({len(questions) - len(texts)} skipped)")

        # Stage 2: Generate embeddings (or reuse if already cached)
        embeddings = self._load_session_embeddings(ids)
        if embeddings is not None:
            self.log("info", "Stage 2/4: Reusing cached session embeddings")
        else:
            self.log("info", "Stage 2/4: Generating embeddings via OpenAI API...")
            embeddings = self._generate_embeddings(texts)
            self._save_session_embeddings(ids, embeddings)

        # Stage 3: Build FAISS index
        self.log("info", "Stage 3/4: Building FAISS index...")
        index, id_map, text_map = self._build_faiss_index(embeddings, ids, texts)
        self._save_faiss_index(index, id_map, text_map, "self_session")

        # Stage 4: Batch search — single FAISS call for all vectors
        self.log("info", f"Stage 4/4: Batch searching {len(ids)} vectors (single FAISS call)...")
        self.progress(50, "Batch FAISS search...")

        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        k = min(10, len(ids) - 1) if len(ids) > 1 else 1
        all_scores, all_indices = index.search(vectors, k + 1)

        subject_map = {qid: subj for qid, subj in zip(ids, subjects)}
        checked_pairs = set()
        duplicates = []

        for i in range(len(ids)):
            qid = ids[i]
            for j in range(all_scores.shape[1]):
                idx = int(all_indices[i][j])
                score = float(all_scores[i][j])

                if idx < 0 or idx >= len(id_map):
                    continue
                matched_id = id_map[idx]
                if matched_id == qid:
                    continue
                if score < threshold:
                    break  # Scores sorted descending

                pair = (min(qid, matched_id), max(qid, matched_id))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                duplicates.append({
                    "qid_a": qid,
                    "qid_b": matched_id,
                    "subject": subject_map.get(qid, "?"),
                    "similarity": round(score, 3),
                    "text_a": text_map.get(qid, "")[:150],
                    "text_b": text_map.get(matched_id, "")[:150],
                    "html_a": html_map.get(qid, ""),
                    "html_b": html_map.get(matched_id, ""),
                })

            if (i + 1) % 100 == 0:
                self.progress(50 + (i / len(ids)) * 45, f"Filtering: {i+1}/{len(ids)}")

        return {
            "total_questions": len(texts),
            "pairs_checked": len(checked_pairs),
            "duplicates_found": len(duplicates),
            "unique_qids_affected": len(set(d["qid_a"] for d in duplicates) | set(d["qid_b"] for d in duplicates)),
            "duplicates": duplicates,
            "engine": "embeddings+faiss",
        }

    # ── Self-duplicate (trigram fallback) ────────────────────────────────

    def _self_check_trigram(self, questions: List[Dict], threshold: float) -> Dict[str, Any]:
        """Self-duplicate check using trigram Jaccard. O(n²) within subject."""
        self.log("info", "Stage 1/3: Extracting question text...")
        qdata = []
        for q in questions:
            text = self._extract_question_text(q)
            if text:
                qdata.append({"qid": q.get("qid", "?"), "subject": q.get("subject", "?"), "text": text})
        self.log("info", f"Extracted {len(qdata)} question texts")

        self.log("info", "Stage 2/3: Grouping by subject...")
        by_subject = defaultdict(list)
        for q in qdata:
            by_subject[q["subject"]].append(q)

        total_pairs = sum(len(qs) * (len(qs) - 1) // 2 for qs in by_subject.values())
        self.log("info", f"Total pairs to check: {total_pairs:,}")

        self.log("info", "Stage 3/3: Computing pairwise similarity...")
        duplicates = []
        checked = 0

        for subj, subj_qs in by_subject.items():
            subj_dupes = 0
            for i in range(len(subj_qs)):
                for j in range(i + 1, len(subj_qs)):
                    sim = _simple_similarity(subj_qs[i]["text"], subj_qs[j]["text"])
                    checked += 1
                    if sim >= threshold:
                        subj_dupes += 1
                        duplicates.append({
                            "qid_a": subj_qs[i]["qid"],
                            "qid_b": subj_qs[j]["qid"],
                            "subject": subj,
                            "similarity": round(sim, 3),
                            "text_a": subj_qs[i]["text"][:150],
                            "text_b": subj_qs[j]["text"][:150],
                        })
                if checked % 500 == 0:
                    self.progress(min(95, checked / max(total_pairs, 1) * 100), f"Checked {checked:,}/{total_pairs:,}")
            self.log("info", f"  {subj}: {len(subj_qs)} questions, {subj_dupes} duplicates found")

        return {
            "total_questions": len(qdata),
            "pairs_checked": checked,
            "duplicates_found": len(duplicates),
            "unique_qids_affected": len(set(d["qid_a"] for d in duplicates) | set(d["qid_b"] for d in duplicates)),
            "duplicates": duplicates,
            "engine": "trigram_jaccard",
        }

    # ── Cross-duplicate (embedding engine) ───────────────────────────────

    def _cross_check_embeddings(self, entries: List[Dict], threshold: float) -> Dict[str, Any]:
        """Cross-duplicate check using embeddings + FAISS. O(n log n) search."""
        import numpy as np
        import faiss
        from app.config import settings

        db_path = Path(settings.db_path)
        if not db_path.exists():
            return {"error": "Reference database not found", "db_path": settings.db_path,
                    "message": "Set db_path in .env to the JSONL file with existing questions"}

        # Stage 1: Load or build DB index
        self.log("info", "Stage 1/4: Loading reference DB index...")
        self.progress(5, "Loading DB index...")

        db_index, db_id_map, db_text_map = self._load_faiss_index("db_reference")

        if db_index is None:
            # Build DB index from scratch
            self.log("info", f"No cached DB index — building from {db_path.name}...")
            db_texts, db_ids = [], []
            db_count = 0
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            db_count += 1
                        except json.JSONDecodeError:
                            continue
                        text = self._extract_question_text(entry)
                        if not text or len(text) < 10:
                            continue
                        # Prefer _id.$oid (24-digit hex OID) for DB entries
                        raw_id = entry.get("_id", "")
                        if isinstance(raw_id, dict) and "$oid" in raw_id and raw_id["$oid"]:
                            db_oid = raw_id["$oid"]
                        elif isinstance(raw_id, str) and len(raw_id) == 24:
                            db_oid = raw_id
                        else:
                            db_oid = entry.get("qid", "") or str(raw_id) or f"unknown_{db_count}"
                        if not db_oid:
                            continue
                        db_ids.append(str(db_oid))
                        db_texts.append(text)
                        if db_count % 10000 == 0:
                            self.log("info", f"  Read {db_count:,} DB entries...")
            except Exception as e:
                return {"error": f"Failed to read database: {e}"}

            self.log("info", f"DB loaded: {db_count:,} entries, {len(db_ids):,} usable")

            # Generate DB embeddings (this is the expensive part — cache result)
            self.log("info", "Generating DB embeddings (one-time cost, cached for reuse)...")
            db_embeddings = self._generate_embeddings(db_texts)

            # Build and save index
            db_index, db_id_map, db_text_map = self._build_faiss_index(db_embeddings, db_ids, db_texts)
            self._save_faiss_index(db_index, db_id_map, db_text_map, "db_reference")
            self.log("success", f"DB index built and cached: {db_index.ntotal} vectors")
        else:
            db_count = db_index.ntotal

        # Stage 2: Extract + embed session questions (reuse from self-check if available)
        self.log("info", "Stage 2/4: Extracting session question text...")
        sess_texts, sess_ids, sess_subjects = [], [], []
        sess_html_map = {}  # qid -> full question HTML for preview
        for entry in entries:
            text = self._extract_question_text(entry)
            if text and len(text) >= 10:
                sess_texts.append(text)
                qid = entry.get("qid", "?")
                sess_ids.append(qid)
                sess_subjects.append(entry.get("subject", "Unknown"))
                # Extract question HTML for preview pane (incl. match table HTML)
                for qt in ("scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"):
                    obj = entry.get(qt)
                    if isinstance(obj, dict):
                        q_html = obj.get("question", obj.get("ques", ""))
                        # For passageQuestion, use passage field
                        if not q_html and qt == "passageQuestion":
                            q_html = obj.get("passage", "")
                        if q_html:
                            sess_html_map[qid] = q_html
                            break

        # Try to reuse session embeddings from self-check (avoids double API cost)
        sess_embeddings = self._load_session_embeddings(sess_ids)
        if sess_embeddings is not None:
            self.log("info", "Stage 3/4: Reusing session embeddings from self-check (saved API cost)")
        else:
            self.log("info", "Stage 3/4: Generating session embeddings...")
            sess_embeddings = self._generate_embeddings(sess_texts)
            self._save_session_embeddings(sess_ids, sess_embeddings)

        # Stage 4: Batch search all session questions against DB index
        self.log("info", f"Stage 4/4: Batch searching {len(sess_ids)} questions against {db_index.ntotal:,} DB vectors...")
        self.progress(50, "Batch FAISS search...")

        sess_vectors = np.array(sess_embeddings, dtype="float32")
        faiss.normalize_L2(sess_vectors)
        all_scores, all_indices = db_index.search(sess_vectors, 5)

        duplicates = []
        duplicate_ids = set()

        for i in range(len(sess_ids)):
            qid = sess_ids[i]
            subj = sess_subjects[i]
            for j in range(all_scores.shape[1]):
                idx = int(all_indices[i][j])
                score = float(all_scores[i][j])

                if idx < 0 or idx >= len(db_id_map):
                    continue
                if score < threshold:
                    break  # Scores sorted descending

                matched_id = db_id_map[idx]
                duplicate_ids.add(qid)
                duplicates.append({
                    "session_qid": qid,
                    "db_qid": matched_id,  # Now contains 24-digit OID
                    "db_oid": matched_id,
                    "subject": subj,
                    "similarity": round(score, 3),
                    "session_text": sess_texts[i][:120],
                    "db_text": db_text_map.get(matched_id, "")[:120],
                    "session_html": sess_html_map.get(qid, ""),
                })
                break  # One match per session question is enough

            if (i + 1) % 50 == 0:
                self.progress(50 + (i / len(sess_ids)) * 45, f"Filtering: {i+1}/{len(sess_ids)}")

        # Enrich duplicates with DB entry flags for lock-on-flag feature
        if duplicates:
            matched_db_ids = {d["db_qid"] for d in duplicates}
            db_flags = self._lookup_db_flags(matched_db_ids)
            for d in duplicates:
                flags = db_flags.get(d["db_qid"], {})
                d["db_flags"] = flags
                # Copy DB question HTML into pair for preview rendering
                d["db_html"] = flags.get("question_html", "")

        return {
            "session_questions": len(entries),
            "db_questions": db_count,
            "pairs_checked": len(sess_ids) * min(5, db_index.ntotal),
            "duplicates_found": len(duplicates),
            "unique_qids_affected": len(duplicate_ids),
            "duplicates": duplicates,
            "engine": "embeddings+faiss",
        }

    # ── Cross-duplicate (trigram fallback) ───────────────────────────────

    def _cross_check_trigram(self, entries: List[Dict], threshold: float) -> Dict[str, Any]:
        """Cross-duplicate check using trigram Jaccard. O(n*m) per subject."""
        from app.config import settings

        db_path = Path(settings.db_path)
        if not db_path.exists():
            return {"error": "Reference database not found", "db_path": settings.db_path,
                    "message": "Set db_path in .env to the JSONL file with existing questions"}

        # Stage 1: Load DB
        self.log("info", f"Stage 1/3: Loading reference database: {db_path.name}...")
        self.progress(5, "Loading reference database...")
        db_by_subject: Dict[str, List[Dict]] = {}
        db_count = 0
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        db_count += 1
                    except json.JSONDecodeError:
                        continue
                    subj = self._normalize_subject(entry.get("subject", "Unknown"))
                    text = self._extract_question_text(entry)
                    if not text or len(text) < 10:
                        continue
                    if subj not in db_by_subject:
                        db_by_subject[subj] = []
                    db_by_subject[subj].append({
                        "qid": entry.get("qid", entry.get("_id", "?")),
                        "text": text,
                    })
                    if db_count % 10000 == 0:
                        self.progress(min(20, db_count / 150000 * 20), f"Loading DB: {db_count:,}")
        except Exception as e:
            return {"error": f"Failed to read database: {e}"}

        db_loaded = sum(len(v) for v in db_by_subject.values())
        self.log("success", f"DB loaded: {db_count:,} total, {db_loaded:,} usable")

        # Stage 2: Extract session questions
        self.log("info", "Stage 2/3: Extracting session question text...")
        session_by_subject: Dict[str, List[Dict]] = {}
        for entry in entries:
            text = self._extract_question_text(entry)
            if not text or len(text) < 10:
                continue
            subj = self._normalize_subject(entry.get("subject", "Unknown"))
            if subj not in session_by_subject:
                session_by_subject[subj] = []
            session_by_subject[subj].append({
                "qid": entry.get("qid", "?"),
                "subject_original": entry.get("subject", "Unknown"),
                "text": text,
            })

        # Stage 3: Compare
        session_total = sum(len(v) for v in session_by_subject.values())
        total_comparisons = sum(len(session_by_subject.get(s, [])) * len(db_by_subject.get(s, [])) for s in session_by_subject)
        self.log("info", f"Stage 3/3: Cross-comparing {session_total} session vs {db_loaded:,} DB ({total_comparisons:,} comparisons)...")

        duplicates = []
        checked = 0
        for subj, sess_qs in session_by_subject.items():
            db_qs = db_by_subject.get(subj, [])
            if not db_qs:
                self.log("warning", f"No DB questions for subject '{subj}' — skipping {len(sess_qs)}")
                continue
            subj_dupes = 0
            for sq_idx, sq in enumerate(sess_qs):
                best_sim, best_match = 0.0, None
                for dq in db_qs:
                    sim = _simple_similarity(sq["text"], dq["text"])
                    if sim > best_sim:
                        best_sim = sim
                        best_match = dq
                    checked += 1
                if best_sim >= threshold and best_match:
                    subj_dupes += 1
                    duplicates.append({
                        "session_qid": sq["qid"],
                        "db_qid": best_match["qid"],
                        "subject": sq.get("subject_original", subj),
                        "similarity": round(best_sim, 3),
                        "session_text": sq["text"][:120],
                        "db_text": best_match["text"][:120],
                    })
                if (sq_idx + 1) % 10 == 0:
                    self.progress(min(95, 30 + checked / max(total_comparisons, 1) * 65),
                                  f"{subj}: {sq_idx+1}/{len(sess_qs)}")
            self.log("info", f"  {subj}: {len(sess_qs)} vs {len(db_qs):,} DB -> {subj_dupes} duplicates")

        return {
            "session_questions": len(entries),
            "db_questions": db_count,
            "pairs_checked": checked,
            "duplicates_found": len(duplicates),
            "unique_qids_affected": len(set(d["session_qid"] for d in duplicates)),
            "duplicates": duplicates,
            "engine": "trigram_jaccard",
        }

    # ── Public API ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_subject(subj: str) -> str:
        """Normalize subject name for cross-comparison."""
        if not subj:
            return "unknown"
        s = subj.strip().lower()
        if s.startswith("phy"):
            return "physics"
        if s.startswith("chem") or s in ("inorganic chemistry", "organic chemistry", "physical chemistry"):
            return "chemistry"
        if s.startswith("math"):
            return "mathematics"
        if s.startswith("bio") or s in ("botany", "zoology"):
            return "biology"
        return s

    def scan(self) -> Dict[str, Any]:
        """Scan metadata for duplicate check readiness."""
        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata_v2 files found — run Stage 4 first"}

        by_subject = {}
        for entry in entries:
            subj = entry.get("subject", "Unknown")
            by_subject[subj] = by_subject.get(subj, 0) + 1

        reports_dir = self._get_reports_dir()
        existing_reports = list(reports_dir.glob("*.json"))

        from app.config import settings
        db_available = Path(settings.db_path).exists() if settings.db_path else False

        engine = self._detect_engine()
        engine_label = (
            f"OpenAI {EMBEDDING_MODEL} + FAISS (semantic, O(n log n))"
            if engine == "embeddings"
            else "Trigram Jaccard (token-based, O(n^2) within subject)"
        )

        self.log("info", f"Found {len(entries)} questions | Engine: {engine}")
        return {
            "total_questions": len(entries),
            "by_subject": by_subject,
            "db_available": db_available,
            "db_path": settings.db_path if db_available else "(not found)",
            "existing_reports": len(existing_reports),
            "embedding_engine": engine_label,
            "engine": engine,
        }

    def self_check(self, threshold: float = 0.85) -> Dict[str, Any]:
        """Self-duplicate check — auto-selects best engine."""
        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata_v2 files found"}

        engine = self._detect_engine()
        self.log("info", f"Running self-duplicate check on {len(entries)} questions (engine: {engine}, threshold: {threshold})")
        start = time.time()

        if engine == "embeddings":
            result = self._self_check_embeddings(entries, threshold)
        else:
            result = self._self_check_trigram(entries, threshold)

        elapsed = time.time() - start
        result["time_seconds"] = round(elapsed, 1)

        # Save report
        reports_dir = self._get_reports_dir()
        report = {
            "type": "self_duplicate",
            "threshold": threshold,
            "engine": result.get("engine", engine),
            "time_seconds": result["time_seconds"],
            **{k: v for k, v in result.items() if k not in ("engine", "time_seconds")},
        }
        report_path = reports_dir / "self_duplicates.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        self.log("success", f"Self-check complete: {result['duplicates_found']} pairs in {elapsed:.1f}s ({engine})")
        self.save_checkpoint("self_check_complete", {
            "duplicates": result["duplicates_found"],
            "total": result["total_questions"],
            "engine": result.get("engine", engine),
        })

        # Limit response size
        result["duplicates"] = result.get("duplicates", [])[:50]
        result["report_path"] = str(report_path)
        return result

    def cross_check(self, threshold: float = 0.85) -> Dict[str, Any]:
        """Cross-duplicate check — auto-selects best engine."""
        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata_v2 files found"}

        engine = self._detect_engine()
        self.log("info", f"Running cross-DB check on {len(entries)} questions (engine: {engine}, threshold: {threshold})")
        start = time.time()

        if engine == "embeddings":
            result = self._cross_check_embeddings(entries, threshold)
        else:
            result = self._cross_check_trigram(entries, threshold)

        elapsed = time.time() - start
        result["time_seconds"] = round(elapsed, 1)

        # Save report
        reports_dir = self._get_reports_dir()
        report = {
            "type": "cross_duplicate",
            "threshold": threshold,
            "engine": result.get("engine", engine),
            "time_seconds": result["time_seconds"],
            **{k: v for k, v in result.items() if k not in ("engine", "time_seconds")},
        }
        report_path = reports_dir / "cross_duplicates.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        self.log("success", f"Cross-check complete: {result['duplicates_found']} duplicates in {elapsed:.1f}s ({engine})")
        self.save_checkpoint("cross_check_complete", {
            "duplicates": result["duplicates_found"],
            "db_count": result.get("db_questions", 0),
            "engine": result.get("engine", engine),
        })

        result["duplicates"] = result.get("duplicates", [])[:50]
        result["report_path"] = str(report_path)
        return result

    def quarantine_duplicates(self, qids: List[str]) -> Dict[str, Any]:
        """Mark duplicate QIDs for quarantine and remove from metadata_v2 files."""
        if not qids:
            return {"quarantined": 0}

        from app.db import quarantine_question

        qid_set = set(qids)
        count = 0
        for qid in qids:
            quarantine_question(
                self.session_id, qid, self.stage_name,
                reason="duplicate",
                data={"source": "stage8_duplicate_check"},
            )
            count += 1

        meta_dir = self._get_metadata_dir()
        files_modified = 0
        questions_removed = 0

        if meta_dir.exists():
            meta_files = []
            for sub in sorted(meta_dir.iterdir()):
                if sub.is_dir():
                    meta_files.extend(sub.glob("*_metadata_v2.json"))
                elif sub.suffix == ".json" and "metadata_v2" in sub.name and "backup" not in sub.name:
                    meta_files.append(sub)

            for f in meta_files:
                if "backup" in f.name:
                    continue
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if not isinstance(data, list):
                        continue

                    original_len = len(data)
                    filtered = [q for q in data if q.get("qid") not in qid_set]
                    removed = original_len - len(filtered)

                    if removed > 0:
                        self.backup(f)
                        with open(f, "w", encoding="utf-8") as fh:
                            json.dump(filtered, fh, indent=2, ensure_ascii=False)
                        files_modified += 1
                        questions_removed += removed
                        self.log("info", f"Removed {removed} quarantined Q from {f.name}")
                except Exception as e:
                    self.log("error", f"Failed to update {f.name}: {e}")

        self.log("success", f"Quarantined {count} questions — removed {questions_removed} from {files_modified} files")
        return {
            "quarantined": count,
            "qids": qids,
            "files_modified": files_modified,
            "questions_removed": questions_removed,
        }

    def export_csv(self, report_type: str = "self") -> str:
        """Export report as CSV string."""
        import csv
        import io

        report = self.get_report(report_type=report_type)
        if "error" in report:
            return ""

        output = io.StringIO()
        writer = csv.writer(output)

        if report_type == "cross":
            writer.writerow([
                "session_qid", "session_text", "db_oid", "db_text",
                "similarity", "subject", "isPrevious", "positivePick", "negativePick", "isSolvedExample",
            ])
            for dup in report.get("duplicates", []):
                flags = dup.get("db_flags", {})
                writer.writerow([
                    dup.get("session_qid", ""), dup.get("session_text", ""),
                    dup.get("db_qid", ""), dup.get("db_text", ""),
                    dup.get("similarity", 0), dup.get("subject", ""),
                    flags.get("isPrevious", False), flags.get("positivePick", False),
                    flags.get("negativePick", False), flags.get("isSolvedExample", False),
                ])
        else:
            writer.writerow(["qid_a", "question_a", "qid_b", "question_b", "similarity", "subject"])
            for dup in report.get("duplicates", []):
                writer.writerow([
                    dup.get("qid_a", ""), dup.get("text_a", ""),
                    dup.get("qid_b", ""), dup.get("text_b", ""),
                    dup.get("similarity", 0), dup.get("subject", ""),
                ])

        return output.getvalue()

    def get_report(self, report_type: str = "self") -> Dict[str, Any]:
        """Get a previously generated duplicate report."""
        reports_dir = self._get_reports_dir()
        filename = f"{'self' if report_type == 'self' else 'cross'}_duplicates.json"
        report_path = reports_dir / filename
        if not report_path.exists():
            return {"error": f"No {report_type} duplicate report found — run the check first"}
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        return self.scan()

    def execute(self, **kwargs) -> Dict[str, Any]:
        threshold = kwargs.get("threshold", 0.85)
        self_result = self.self_check(threshold=threshold)
        cross_result = self.cross_check(threshold=threshold)
        total_dupes = self_result.get("duplicates_found", 0) + cross_result.get("duplicates_found", 0)
        self.save_checkpoint("completed", {
            "self_duplicates": self_result.get("duplicates_found", 0),
            "cross_duplicates": cross_result.get("duplicates_found", 0),
        })
        return {
            "self_check": self_result,
            "cross_check": cross_result,
            "total_duplicates": total_dupes,
        }
