"""Full DB internal duplicate check using FAISS batch search.

Searches ALL ~160K DB questions against each other using the cached
FAISS index. First updates the index with fresh OCR'd embeddings
(replacing old useless [IMGS:...] vectors), then runs a single batch
FAISS search — O(n log n) via FAISS optimized search.

Steps:
  1. Load cached FAISS index (159K vectors)
  2. Extract fresh text for OCR'd entries from updated DB
  3. Generate new embeddings ONLY for OCR'd entries (~14K)
  4. Replace old vectors in index with new OCR'd embeddings
  5. Batch search ALL vectors against index in one call
  6. Filter self-matches and pairs below threshold
  7. Save duplicate pairs + updated index

Usage:
    python scripts/db_internal_dedup.py
    python scripts/db_internal_dedup.py --threshold 0.90
    python scripts/db_internal_dedup.py --dry-run
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    print("ERROR: pip install faiss-cpu")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
APP_ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = APP_ROOT / "data" / "ocr_cache"
CONVERTED_FILE = OCR_DIR / "ocr_converted.jsonl"
DB_PATH = Path(r"D:\ACADZA_WORK\acadza-pdf-generator\data\allque030326.json")

# Existing cached FAISS index from app session
CACHED_INDEX_DIR = APP_ROOT / "data" / "sessions" / "cb376e58" / "duplicate_reports" / "indices" / "db_reference"
CACHED_FAISS = CACHED_INDEX_DIR / "faiss.index"
CACHED_META = CACHED_INDEX_DIR / "metadata.json"

# Output
OUTPUT_DIR = OCR_DIR / "dedup_results"
REPORT_FILE = OUTPUT_DIR / "dedup_report.json"
DUPLICATES_FILE = OUTPUT_DIR / "duplicate_pairs.jsonl"
NEW_INDEX_FILE = OUTPUT_DIR / "updated_faiss.index"
NEW_META_FILE = OUTPUT_DIR / "updated_metadata.json"

# ── Constants (same as app Stage 8) ──────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 100
MAX_TEXT_LENGTH = 8000
DEFAULT_THRESHOLD = 0.85
TOP_K = 10

HTML_TAG_RE = re.compile(r'<[^>]+>')
MATHML_RE = re.compile(r'<math[\s\S]*?</math>')
WHITESPACE_RE = re.compile(r'\s+')
IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)

QUESTION_TYPES = ["scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"]


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs, flush=True)


# ── Text extraction (forked from app Stage 8) ───────────────────────────

def extract_question_text(entry: dict) -> str:
    """Extract comparable question text from DB entry (question field only)."""
    import html as html_mod

    for qtype in QUESTION_TYPES:
        obj = entry.get(qtype)
        if not obj or not isinstance(obj, dict):
            continue

        q = str(obj.get("question", "") or obj.get("ques", "") or "")
        if not q or len(q.strip()) < 5:
            continue

        # Extract image URLs before stripping
        img_urls = IMG_SRC_RE.findall(q)

        # Strip MathML → MATH placeholder
        text = MATHML_RE.sub(' MATH ', q)
        # Strip HTML tags
        text = HTML_TAG_RE.sub(' ', text)
        # Unescape HTML entities
        text = html_mod.unescape(text)
        # Normalize whitespace
        text = WHITESPACE_RE.sub(' ', text).strip()

        # Append image fingerprints (for entries that still have images)
        if img_urls:
            img_keys = [url.split('/')[-1][:20] for url in img_urls]
            text += ' [IMGS:' + ','.join(img_keys) + ']'

        return text[:MAX_TEXT_LENGTH]

    return ""


def get_qid(entry: dict) -> str:
    """Get short qid from DB entry."""
    return entry.get("qid", "")


def get_oid(entry: dict) -> str:
    """Get $oid from DB entry."""
    _id = entry.get("_id") or {}
    return _id.get("$oid", "") if isinstance(_id, dict) else str(_id)


# ── Embedding generation ─────────────────────────────────────────────────

def generate_embeddings(client, texts: List[str], label: str = "") -> np.ndarray:
    """Generate embeddings in batches. Returns (N, 1536) float32 array."""
    all_embeddings = []
    total = len(texts)
    batches = (total + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for i in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        # Replace empty texts
        batch = [t if t.strip() else "empty" for t in batch]

        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in response.data:
            all_embeddings.append(item.embedding)

        batch_num = i // EMBEDDING_BATCH_SIZE + 1
        if batch_num % 20 == 0 or batch_num == batches:
            _safe_print(f"    [{label}] Batch {batch_num}/{batches}")

    return np.array(all_embeddings, dtype="float32")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full DB internal dedup")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _safe_print("=" * 60)
    _safe_print("Full DB Internal Duplicate Check (FAISS Batch Search)")
    _safe_print(f"Threshold: {args.threshold}")
    _safe_print("=" * 60)

    # Check OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or len(api_key) < 10:
        _safe_print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)
    client = OpenAI(api_key=api_key)
    _safe_print("  OpenAI client ready")

    # ── Step 1: Load existing FAISS index ──
    _safe_print("\nStep 1: Loading cached FAISS index...")
    if not CACHED_FAISS.exists():
        _safe_print(f"ERROR: Cached index not found: {CACHED_FAISS}")
        sys.exit(1)

    index = faiss.read_index(str(CACHED_FAISS))
    with open(CACHED_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    id_map = meta["id_map"]
    text_map = meta.get("text_map", {})

    _safe_print(f"  Index: {index.ntotal} vectors, dim={index.d}")
    _safe_print(f"  ID map: {len(id_map)}, Text map: {len(text_map)}")

    # Build reverse lookup: qid → index position
    qid_to_idx = {qid: i for i, qid in enumerate(id_map)}

    # ── Step 2: Get OCR'd QIDs and extract new question text ──
    _safe_print("\nStep 2: Identifying OCR'd entries that need re-embedding...")
    ocr_oids = set()
    with open(CONVERTED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ocr_oids.add(entry["_id"].get("$oid", ""))
    _safe_print(f"  OCR'd entries ($oid): {len(ocr_oids)}")

    # Read DB and extract text for OCR'd entries
    ocr_entries = []  # (qid, oid, text, subject)
    with open(DB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            oid = get_oid(entry)
            if oid not in ocr_oids:
                continue
            qid = get_qid(entry)
            text = extract_question_text(entry)
            subject = entry.get("subject", "Unknown")
            if text and len(text) >= 10 and qid:
                ocr_entries.append((qid, oid, text, subject))

    _safe_print(f"  OCR'd entries with valid text + qid: {len(ocr_entries)}")

    if args.dry_run:
        _safe_print("\n[DRY RUN — would generate embeddings and batch search]")
        _safe_print(f"  Embedding API calls needed: {len(ocr_entries) // EMBEDDING_BATCH_SIZE + 1}")
        _safe_print(f"  Total index vectors for batch search: {index.ntotal}")
        _safe_print(f"  Batch search: index.search(all_{index.ntotal}_vectors, TOP_K={TOP_K})")
        _safe_print(f"  This is O(n log n) — single FAISS call, not one-by-one")
        return

    # ── Step 3: Generate embeddings ONLY for OCR'd questions ──
    _safe_print(f"\nStep 3: Generating embeddings for {len(ocr_entries)} OCR'd questions...")
    start_embed = time.time()
    texts = [e[2] for e in ocr_entries]
    new_embeddings = generate_embeddings(client, texts, label="OCR")
    embed_time = time.time() - start_embed
    _safe_print(f"  Done in {embed_time:.1f}s")

    # ── Step 4: Update index — replace old vectors with fresh OCR'd ones ──
    _safe_print("\nStep 4: Updating FAISS index with fresh OCR'd embeddings...")

    # Reconstruct existing vectors from index
    existing_vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d)
    existing_vectors = np.reshape(existing_vectors, (index.ntotal, index.d)).copy()

    updated_count = 0
    added_count = 0
    new_id_map = list(id_map)
    new_text_map = dict(text_map)

    for i, (qid, oid, text, subject) in enumerate(ocr_entries):
        vec = new_embeddings[i:i+1].copy()
        faiss.normalize_L2(vec)

        if qid in qid_to_idx:
            # Replace existing vector (old one had useless [IMGS:...] text)
            idx = qid_to_idx[qid]
            existing_vectors[idx] = vec[0]
            new_text_map[qid] = text[:500]
            updated_count += 1
        else:
            # Add new vector
            existing_vectors = np.vstack([existing_vectors, vec])
            new_id_map.append(qid)
            qid_to_idx[qid] = len(new_id_map) - 1
            new_text_map[qid] = text[:500]
            added_count += 1

    _safe_print(f"  Vectors replaced (OCR'd): {updated_count}")
    _safe_print(f"  Vectors added (new): {added_count}")

    # Rebuild index with updated vectors
    new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    faiss.normalize_L2(existing_vectors)
    new_index.add(existing_vectors)
    total_vectors = new_index.ntotal
    _safe_print(f"  Updated index: {total_vectors} vectors")

    # ── Step 5: BATCH search ALL vectors against full index ──
    _safe_print(f"\nStep 5: Batch searching ALL {total_vectors} vectors against index (TOP_K={TOP_K})...")
    _safe_print(f"  This is a single FAISS call — O(n log n), not one-by-one")
    start_search = time.time()

    # Single batch search: search ALL vectors at once
    # FAISS handles this efficiently internally
    all_scores, all_indices = new_index.search(existing_vectors, TOP_K + 1)

    search_time = time.time() - start_search
    _safe_print(f"  FAISS batch search done in {search_time:.1f}s")

    # ── Step 6: Filter results — remove self-matches, apply threshold ──
    _safe_print(f"\nStep 6: Filtering {total_vectors} x {TOP_K} results...")
    start_filter = time.time()

    duplicates = []
    checked_pairs = set()

    for i in range(total_vectors):
        qid_a = new_id_map[i]

        for j in range(all_scores.shape[1]):
            idx = int(all_indices[i][j])
            score = float(all_scores[i][j])

            if idx < 0 or idx >= len(new_id_map):
                continue

            qid_b = new_id_map[idx]

            # Skip self-match
            if qid_b == qid_a:
                continue

            if score < args.threshold:
                break  # Scores are sorted descending, no more matches

            # Deduplicate pairs (A,B) == (B,A)
            pair_key = (min(qid_a, qid_b), max(qid_a, qid_b))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            duplicates.append({
                "qid_a": qid_a,
                "qid_b": qid_b,
                "similarity": round(score, 4),
                "text_a": new_text_map.get(qid_a, "")[:200],
                "text_b": new_text_map.get(qid_b, "")[:200],
            })

        if (i + 1) % 20000 == 0:
            _safe_print(f"    [{i+1}/{total_vectors}] {len(duplicates)} duplicate pairs so far")

    filter_time = time.time() - start_filter
    _safe_print(f"  Filtering done in {filter_time:.1f}s")
    _safe_print(f"  Total duplicate pairs found: {len(duplicates)}")

    # ── Step 7: Save results ──
    _safe_print("\nStep 7: Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save duplicate pairs (sorted by similarity descending)
    with open(DUPLICATES_FILE, "w", encoding="utf-8") as f:
        for d in sorted(duplicates, key=lambda x: -x["similarity"]):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    _safe_print(f"  Duplicates: {DUPLICATES_FILE}")

    # Save updated index
    faiss.write_index(new_index, str(NEW_INDEX_FILE))
    with open(NEW_META_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "id_map": new_id_map,
            "text_map": new_text_map,
            "dimension": EMBEDDING_DIM,
        }, f, ensure_ascii=False)
    _safe_print(f"  Updated index: {NEW_INDEX_FILE}")

    # Similarity distribution
    by_range = defaultdict(int)
    if duplicates:
        for d in duplicates:
            s = d["similarity"]
            if s >= 0.95:
                by_range["0.95-1.00"] += 1
            elif s >= 0.90:
                by_range["0.90-0.95"] += 1
            elif s >= 0.85:
                by_range["0.85-0.90"] += 1

    # Save report
    report = {
        "threshold": args.threshold,
        "total_db_vectors": total_vectors,
        "ocr_embeddings_generated": len(ocr_entries),
        "vectors_replaced": updated_count,
        "vectors_added": added_count,
        "total_duplicates_found": len(duplicates),
        "unique_qids_affected": len(
            set(d["qid_a"] for d in duplicates) | set(d["qid_b"] for d in duplicates)
        ),
        "similarity_distribution": dict(by_range),
        "embed_time_seconds": round(embed_time, 1),
        "search_time_seconds": round(search_time, 1),
        "filter_time_seconds": round(filter_time, 1),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _safe_print(f"  Report: {REPORT_FILE}")

    # Summary
    _safe_print(f"\n{'=' * 60}")
    _safe_print("DONE — Full DB Self-Dedup")
    _safe_print(f"  Total vectors searched: {total_vectors}")
    _safe_print(f"  OCR'd embeddings regenerated: {len(ocr_entries)}")
    _safe_print(f"  Duplicate pairs found: {len(duplicates)}")
    _safe_print(f"  Unique QIDs affected: {report['unique_qids_affected']}")
    if duplicates:
        _safe_print(f"  Similarity distribution:")
        for k, v in sorted(by_range.items(), reverse=True):
            _safe_print(f"    {k}: {v}")
        _safe_print(f"  Top 10 highest similarity:")
        for d in sorted(duplicates, key=lambda x: -x["similarity"])[:10]:
            _safe_print(f"    {d['qid_a']} <-> {d['qid_b']}: {d['similarity']}")
            _safe_print(f"      A: {d['text_a'][:80]}")
            _safe_print(f"      B: {d['text_b'][:80]}")


if __name__ == "__main__":
    main()
