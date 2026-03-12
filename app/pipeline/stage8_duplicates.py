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

HTML_TAG_RE = re.compile(r'<[^>]+>')
MATHML_RE = re.compile(r'<math[\s\S]*?</math>')
WHITESPACE_RE = re.compile(r'\s+')
IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)


# ── Text helpers ─────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """Strip HTML tags and normalize for comparison."""
    if not html:
        return ""
    text = MATHML_RE.sub(' MATH ', html)
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
    text = MATHML_RE.sub(' MATH ', html)
    text = HTML_TAG_RE.sub(' ', text)
    import html as html_mod
    text = html_mod.unescape(text)
    text = WHITESPACE_RE.sub(' ', text).strip()
    if img_urls:
        img_keys = [url.split('/')[-1][:20] for url in img_urls]
        text += ' [IMGS:' + ','.join(img_keys) + ']'
    return text


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
        """Extract plain text from a metadata entry for comparison.

        Forked from production PCMB_Duplicate_Checker.py:extract_question_text()
        Uses image URL fingerprints + passage sub-question extraction.
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

            # passageQuestion: passage + sub-questions
            if qt == "passageQuestion":
                passage = qdata.get("passage", "")
                if passage:
                    parts.append(strip_fn(passage))
                sub_qs = qdata.get("questions", [])
                if isinstance(sub_qs, list):
                    for sq in sub_qs[:2]:
                        if isinstance(sq, dict):
                            for sub_type in ("scq", "mcq", "integerQuestion"):
                                sub_data = sq.get(sub_type, {})
                                if isinstance(sub_data, dict):
                                    st = sub_data.get("ques", "") or sub_data.get("question", "")
                                    if st:
                                        parts.append(strip_fn(st))
                                        break

            if parts:
                break

        text = " ".join(parts)
        return text[:MAX_TEXT_LENGTH] if len(text) > MAX_TEXT_LENGTH else text

    def _lookup_db_flags(self, db_qids: Set[str]) -> Dict[str, Dict]:
        """Look up special flags (isPrevious, isBoard, etc.) for matched DB question IDs.

        Scans the JSONL DB for just the matched IDs — avoids loading entire DB into memory.
        """
        from app.config import settings
        FLAG_KEYS = ("isSolvedExample", "isPrevious", "isBoard", "isNCERT")

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
                    qid = entry.get("qid", entry.get("_id", "?"))
                    if isinstance(qid, dict) and "$oid" in qid:
                        qid = qid["$oid"]
                    qid = str(qid)
                    if qid in remaining:
                        flags = {k: bool(entry.get(k)) for k in FLAG_KEYS}
                        flags["has_protected_flag"] = any(flags.values())
                        # Also extract question HTML for preview rendering
                        q_html = ""
                        for qt in ("scq", "mcq", "integerQuestion", "passageQuestion"):
                            obj = entry.get(qt)
                            if isinstance(obj, dict):
                                q_html = obj.get("question", obj.get("ques", ""))
                                if q_html:
                                    break
                        flags["question_html"] = q_html[:2000] if q_html else ""
                        flags_map[qid] = flags
                        remaining.discard(qid)
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
                self.log("error", f"  Embedding batch {batch_num} failed: {e}")
                for _ in batch:
                    all_embeddings.append(np.zeros(EMBEDDING_DIM, dtype="float32"))
                time.sleep(2)

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

        Saves to session-local directory. For 'db_reference' indices, also
        copies to the global cache so other sessions can reuse it.
        """
        import faiss
        import shutil

        index_dir = self._get_index_dir() / name
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / "faiss.index"))
        metadata = {"id_map": id_map, "text_map": text_map, "dimension": EMBEDDING_DIM}
        (index_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

        # Also save db_reference to global cache for cross-session reuse
        if name == "db_reference":
            from app.config import settings
            gdir = getattr(settings, "db_index_dir", None)
            if gdir:
                global_dest = Path(gdir) / name
                global_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(index_dir / "faiss.index", global_dest / "faiss.index")
                shutil.copy2(index_dir / "metadata.json", global_dest / "metadata.json")
                self.log("info", f"Copied DB index to global cache: {global_dest}")

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

        Search order:
          1. Session-local index (session_dir/duplicate_reports/indices/{name}/)
          2. Global cache (db_index_dir/{name}/) — shared across sessions
        """
        import faiss

        # 1. Try session-local
        index_dir = self._get_index_dir() / name
        if (index_dir / "faiss.index").exists():
            index = faiss.read_index(str(index_dir / "faiss.index"))
            metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
            self.log("info", f"Loaded session-local FAISS index '{name}': {index.ntotal} vectors")
            return index, metadata["id_map"], metadata["text_map"]

        # 2. Try global cache
        global_dir = self._get_global_index_dir()
        if global_dir:
            global_index_dir = global_dir / name
            if (global_index_dir / "faiss.index").exists():
                index = faiss.read_index(str(global_index_dir / "faiss.index"))
                metadata = json.loads((global_index_dir / "metadata.json").read_text(encoding="utf-8"))
                self.log("info", f"Loaded global FAISS index '{name}': {index.ntotal} vectors")
                return index, metadata["id_map"], metadata["text_map"]

        return None, None, None

    # ── Self-duplicate (embedding engine) ────────────────────────────────

    def _self_check_embeddings(self, questions: List[Dict], threshold: float) -> Dict[str, Any]:
        """Self-duplicate check using embeddings + FAISS batch search. O(n log n)."""
        import numpy as np
        import faiss

        # Stage 1: Extract text
        self.log("info", "Stage 1/4: Extracting question text...")
        texts, ids, subjects = [], [], []
        for q in questions:
            text = self._extract_question_text(q)
            if text and len(text) >= 10:
                texts.append(text)
                ids.append(q.get("qid", "?"))
                subjects.append(q.get("subject", "?"))
        self.log("info", f"Extracted {len(texts)} question texts ({len(questions) - len(texts)} skipped)")

        # Stage 2: Generate embeddings
        self.log("info", "Stage 2/4: Generating embeddings via OpenAI API...")
        embeddings = self._generate_embeddings(texts)

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
                        qid = entry.get("qid", entry.get("_id", "?"))
                        if isinstance(qid, dict) and "$oid" in qid:
                            qid = qid["$oid"]
                        db_ids.append(str(qid))
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

        # Stage 2: Extract + embed session questions
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
                # Extract question HTML for preview
                for qt in ("scq", "mcq", "integerQuestion", "passageQuestion"):
                    obj = entry.get(qt)
                    if isinstance(obj, dict):
                        q_html = obj.get("question", obj.get("ques", ""))
                        if q_html:
                            sess_html_map[qid] = q_html[:2000]
                            break

        self.log("info", "Stage 3/4: Generating session embeddings...")
        sess_embeddings = self._generate_embeddings(sess_texts)

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
                    "db_qid": matched_id,
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
                "similarity", "subject", "isPrevious", "isBoard", "isNCERT", "isSolvedExample",
            ])
            for dup in report.get("duplicates", []):
                flags = dup.get("db_flags", {})
                writer.writerow([
                    dup.get("session_qid", ""), dup.get("session_text", ""),
                    dup.get("db_qid", ""), dup.get("db_text", ""),
                    dup.get("similarity", 0), dup.get("subject", ""),
                    flags.get("isPrevious", False), flags.get("isBoard", False),
                    flags.get("isNCERT", False), flags.get("isSolvedExample", False),
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
