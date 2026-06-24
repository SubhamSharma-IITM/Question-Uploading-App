"""Generate comprehensive CSV of all duplicate question pairs from DB.

Reads verified_strict.jsonl (two-pass verified duplicates) and enriches
with DB metadata: error code, subject, chapter, boolean fields, etc.

Output: data/ocr_cache/dedup_results/duplicate_pairs_full.csv
"""

import csv
import json
import sys
from pathlib import Path

# Paths
DEDUP_DIR = Path(__file__).parent.parent / "data" / "ocr_cache" / "dedup_results"
VERIFIED_FILE = DEDUP_DIR / "verified_strict.jsonl"
DB_PATH = Path(r"D:\ACADZA_WORK\data\allque030326.json")
OUTPUT_CSV = DEDUP_DIR / "duplicate_pairs_full.csv"


def load_db_metadata(db_path: Path) -> dict:
    """Load DB entries into a dict keyed by qid, extracting relevant fields."""
    print(f"Loading DB from {db_path}...")
    db = {}
    count = 0
    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1

            # Extract _id
            _id = entry.get("_id", "")
            if isinstance(_id, dict) and "$oid" in _id:
                _id = _id["$oid"]
            _id = str(_id)

            qid = str(entry.get("qid", _id))

            # Extract fields
            meta = {
                "qid": qid,
                "_id": _id,
                "error": entry.get("error", ""),
                "subject": entry.get("subject", ""),
                "chapter": entry.get("chapter", ""),
                "questionType": entry.get("questionType", ""),
                "level": entry.get("level", ""),
                # Boolean fields that actually exist in DB
                "isBuffer": entry.get("isBuffer", False),
                "isNCERT": entry.get("isNCERT", False),
                "isPrevious": entry.get("isPrevious", False),
                "isBoard": entry.get("isBoard", False),
                "isNEET": entry.get("isNEET", None),
                "isMHCET": entry.get("isMHCET", False),
                "isKCET": entry.get("isKCET", False),
                "negativePick": entry.get("negativePick", False),
                "positivePick": entry.get("positivePick", False),
                "practiceToughQuestion": entry.get("practiceToughQuestion", False),
                "tagByAI": entry.get("tagByAI", False),
                "taggedByCoachingTeacher": entry.get("taggedByCoachingTeacher", False),
                # Numeric fields that act as booleans
                "isLengthy": entry.get("isLengthy", 0),
                "smartTrick": entry.get("smartTrick", 0),
                "trap": entry.get("trap", 0),
                "sillyMistake": entry.get("sillyMistake", 0),
                "difficulty": entry.get("difficulty", ""),
            }

            # Index by both qid and _id for lookup
            db[qid] = meta
            if _id != qid:
                db[_id] = meta

            if count % 50000 == 0:
                print(f"  Loaded {count:,} entries...")

    print(f"DB loaded: {count:,} entries, {len(db):,} index keys")
    return db


def main():
    if not VERIFIED_FILE.exists():
        print(f"ERROR: {VERIFIED_FILE} not found. Run db_internal_dedup.py first.")
        sys.exit(1)
    if not DB_PATH.exists():
        print(f"ERROR: DB not found at {DB_PATH}")
        sys.exit(1)

    # Load DB metadata
    db = load_db_metadata(DB_PATH)

    # Load verified pairs
    print(f"\nLoading verified pairs from {VERIFIED_FILE}...")
    pairs = []
    with open(VERIFIED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(pairs):,} verified duplicate pairs")

    # CSV columns
    columns = [
        "original_qid", "duplicate_qid",
        "embedding_similarity", "jaccard_similarity",
        # Original question metadata
        "orig_error", "orig_subject", "orig_chapter", "orig_questionType", "orig_level",
        "orig_isBuffer", "orig_isNCERT", "orig_isPrevious", "orig_isBoard", "orig_isNEET",
        "orig_negativePick", "orig_positivePick", "orig_practiceToughQuestion",
        "orig_tagByAI", "orig_taggedByCoachingTeacher",
        "orig_isLengthy", "orig_smartTrick", "orig_trap", "orig_sillyMistake", "orig_difficulty",
        # Duplicate question metadata
        "dup_error", "dup_subject", "dup_chapter", "dup_questionType", "dup_level",
        "dup_isBuffer", "dup_isNCERT", "dup_isPrevious", "dup_isBoard", "dup_isNEET",
        "dup_negativePick", "dup_positivePick", "dup_practiceToughQuestion",
        "dup_tagByAI", "dup_taggedByCoachingTeacher",
        "dup_isLengthy", "dup_smartTrick", "dup_trap", "dup_sillyMistake", "dup_difficulty",
    ]

    meta_fields = [
        "error", "subject", "chapter", "questionType", "level",
        "isBuffer", "isNCERT", "isPrevious", "isBoard", "isNEET",
        "negativePick", "positivePick", "practiceToughQuestion",
        "tagByAI", "taggedByCoachingTeacher",
        "isLengthy", "smartTrick", "trap", "sillyMistake", "difficulty",
    ]

    # Write CSV
    print(f"\nWriting CSV to {OUTPUT_CSV}...")
    found_in_db = 0
    missing_from_db = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for pair in pairs:
            qid_a = str(pair.get("qid_a", ""))
            qid_b = str(pair.get("qid_b", ""))
            emb_sim = pair.get("similarity", pair.get("embedding_sim", 0))
            jac_sim = pair.get("jaccard", pair.get("jaccard_sim", 0))

            meta_a = db.get(qid_a, {})
            meta_b = db.get(qid_b, {})

            if meta_a:
                found_in_db += 1
            else:
                missing_from_db += 1
            if meta_b:
                found_in_db += 1
            else:
                missing_from_db += 1

            row = {
                "original_qid": qid_a,
                "duplicate_qid": qid_b,
                "embedding_similarity": round(emb_sim, 4) if emb_sim else "",
                "jaccard_similarity": round(jac_sim, 4) if jac_sim else "",
            }

            for field in meta_fields:
                row[f"orig_{field}"] = meta_a.get(field, "")
                row[f"dup_{field}"] = meta_b.get(field, "")

            writer.writerow(row)

    print(f"\nCSV written: {OUTPUT_CSV}")
    print(f"  Total pairs: {len(pairs):,}")
    print(f"  QIDs found in DB: {found_in_db:,}")
    print(f"  QIDs missing from DB: {missing_from_db:,}")

    # Summary stats
    unique_qids = set()
    error_counts = {"1": 0, "2": 0, "other": 0}
    for pair in pairs:
        for qid_key in ("qid_a", "qid_b"):
            qid = str(pair.get(qid_key, ""))
            unique_qids.add(qid)
            meta = db.get(qid, {})
            err = str(meta.get("error", ""))
            if err == "1":
                error_counts["1"] += 1
            elif err == "2":
                error_counts["2"] += 1
            else:
                error_counts["other"] += 1

    print(f"\n  Unique QIDs involved: {len(unique_qids):,}")
    print(f"  Error=1 entries: {error_counts['1']:,}")
    print(f"  Error=2 entries: {error_counts['2']:,}")
    print(f"  Other/missing error: {error_counts['other']:,}")


if __name__ == "__main__":
    main()
