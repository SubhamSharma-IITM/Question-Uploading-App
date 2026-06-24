"""Replace image-only question/solution fields in DB with OCR'd HTML.

Reads ocr_converted.jsonl, finds matching QIDs in allque030326.json,
backs up original values, replaces with converted HTML.

Creates:
  - allque030326_pre_ocr_backup.jsonl  (original values only, for rollback)
  - allque030326.json                  (updated DB, in-place rewrite)
  - ocr_replacement_report.json        (what was replaced, stats)
  - diagram_flagged_qids.json          (552 QIDs needing manual diagram work)

Usage:
    python scripts/replace_ocr_in_db.py
    python scripts/replace_ocr_in_db.py --dry-run   # preview, no write
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

APP_ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = APP_ROOT / "data" / "ocr_cache"
CONVERTED_FILE = OCR_DIR / "ocr_converted.jsonl"
QUESTION_OCR_MAP = OCR_DIR / "question_ocr_map.json"
OCR_RESULTS_MAP = OCR_DIR / "ocr_results_map.json"

DB_PATH = Path(r"D:\ACADZA_WORK\acadza-pdf-generator\data\allque030326.json")
BACKUP_PATH = DB_PATH.parent / "allque030326_pre_ocr_backup.jsonl"
REPORT_PATH = OCR_DIR / "ocr_replacement_report.json"
DIAGRAM_FLAG_PATH = OCR_DIR / "diagram_flagged_qids.json"

QUESTION_TYPES = ["scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"]

DIAGRAM_KEYWORDS = [
    "figure", "diagram", "circuit", "as shown in fig",
    "shown in the figure", "shown in figure", "given below",
    "shown above", "shown below", "the graph", "as shown",
]


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs, flush=True)


def wrap_h3(html: str) -> str:
    """Wrap in <h3> if not already wrapped — matches DB format."""
    if not html:
        return html
    stripped = html.strip()
    if stripped.lower().startswith("<h3"):
        return html
    return f"<h3>{html}</h3>"


def build_replacement_map():
    """Build QID → {field: html} from converted OCR output."""
    _safe_print("Loading converted OCR entries...")
    replacements = {}

    with open(CONVERTED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["_id"].get("$oid", "")
            if not qid:
                continue

            qtype = entry.get("qtype", "scq")
            fields = {}
            for field, data in entry.get("ocr_fields", {}).items():
                html = data.get("html", "")
                if html:
                    fields[field] = html

            if fields:
                replacements[qid] = {
                    "qtype": qtype,
                    "fields": fields,
                }

    _safe_print(f"  Loaded {len(replacements)} QIDs with replacement HTML")
    return replacements


def build_diagram_flag_set():
    """Identify QIDs whose OCR text references diagrams/figures."""
    _safe_print("Building diagram flag set...")

    diagram_qids = {}

    with open(QUESTION_OCR_MAP, "r", encoding="utf-8") as f:
        qocr = json.load(f)

    url_to_ocr = {}
    with open(OCR_RESULTS_MAP, "r", encoding="utf-8") as f:
        url_to_ocr = json.load(f).get("url_to_ocr", {})

    for key, data in qocr.items():
        qid = key.split("|")[0]
        field_path = key.split("|")[-1] if "|" in key else ""

        for r in data.get("ocr_results", []):
            url = r.get("url", "")
            text = (url_to_ocr.get(url, {}).get("text", "") or r.get("text", "")).lower()
            for kw in DIAGRAM_KEYWORDS:
                if kw in text:
                    if qid not in diagram_qids:
                        diagram_qids[qid] = []
                    diagram_qids[qid].append({
                        "field": field_path,
                        "keyword": kw,
                        "text_excerpt": text[:150],
                        "image_url": url,
                    })
                    break

    _safe_print(f"  Flagged {len(diagram_qids)} QIDs with diagram references")
    return diagram_qids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no write")
    args = parser.parse_args()

    _safe_print("=" * 60)
    _safe_print("OCR HTML → DB Replacement")
    _safe_print("=" * 60)

    if args.dry_run:
        _safe_print("[DRY RUN MODE — no files will be written]\n")

    # Build replacement map
    replacements = build_replacement_map()

    # Build diagram flag set
    diagram_qids = build_diagram_flag_set()

    # Save diagram flagged list
    if not args.dry_run:
        with open(DIAGRAM_FLAG_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "total": len(diagram_qids),
                "note": "These QIDs have diagram/figure references in OCR text. "
                        "The diagrams are baked into the original images and NOT extracted by OCR. "
                        "Manual cropping of diagrams from original images needed.",
                "qids": diagram_qids,
            }, f, ensure_ascii=False, indent=2)
        _safe_print(f"  Saved diagram flags: {DIAGRAM_FLAG_PATH}")

    # Process the DB
    _safe_print(f"\nProcessing DB: {DB_PATH}")
    start = time.time()

    stats = {
        "total_db_entries": 0,
        "matched_qids": 0,
        "question_replaced": 0,
        "solution_replaced": 0,
        "diagram_flagged": 0,
        "skipped_no_match": 0,
    }
    backup_entries = []  # just the changed entries' original values

    # Read entire DB, modify in memory, write back
    all_lines = []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                all_lines.append(line)
                continue

            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                all_lines.append(line)
                continue

            stats["total_db_entries"] += 1
            _id = entry.get("_id") or {}
            qid = _id.get("$oid", "") if isinstance(_id, dict) else str(_id)

            if qid not in replacements:
                all_lines.append(line)
                continue

            stats["matched_qids"] += 1
            repl = replacements[qid]
            qtype = repl["qtype"]
            is_diagram = qid in diagram_qids

            if is_diagram:
                stats["diagram_flagged"] += 1

            # Find the question type object in DB entry
            db_qtype_obj = entry.get(qtype)
            if not db_qtype_obj or not isinstance(db_qtype_obj, dict):
                # Try to find any matching qtype
                for qt in QUESTION_TYPES:
                    if entry.get(qt) and isinstance(entry.get(qt), dict):
                        qtype = qt
                        db_qtype_obj = entry[qt]
                        break

            if not db_qtype_obj:
                all_lines.append(line)
                stats["skipped_no_match"] += 1
                continue

            # Build backup of original values
            backup = {
                "qid": qid,
                "qtype": qtype,
                "is_diagram_flagged": is_diagram,
                "original": {},
            }

            modified = False

            for field, new_html in repl["fields"].items():
                # Map field name to DB key
                if field == "question":
                    db_key = "question"
                    if db_key not in db_qtype_obj:
                        db_key = "ques"
                elif field == "solution":
                    db_key = "solution"
                    if db_key not in db_qtype_obj:
                        db_key = "sol"
                elif field.startswith("sub_q"):
                    # passageQuestion sub-questions — skip for now
                    continue
                else:
                    continue

                # Save original
                original_val = db_qtype_obj.get(db_key, "")
                backup["original"][field] = original_val

                # Replace with OCR'd HTML wrapped in <h3>
                wrapped = wrap_h3(new_html)
                db_qtype_obj[db_key] = wrapped
                modified = True

                if field == "question":
                    stats["question_replaced"] += 1
                elif field == "solution":
                    stats["solution_replaced"] += 1

            if modified:
                entry[qtype] = db_qtype_obj
                backup_entries.append(backup)

            # Write modified entry
            all_lines.append(json.dumps(entry, ensure_ascii=False) + "\n")

            if stats["matched_qids"] % 2000 == 0:
                _safe_print(f"  [{stats['matched_qids']}] Q={stats['question_replaced']} S={stats['solution_replaced']}")

    elapsed = time.time() - start

    _safe_print(f"\n  Processing done in {elapsed:.1f}s")
    _safe_print(f"  DB entries: {stats['total_db_entries']}")
    _safe_print(f"  Matched QIDs: {stats['matched_qids']}")
    _safe_print(f"  Questions replaced: {stats['question_replaced']}")
    _safe_print(f"  Solutions replaced: {stats['solution_replaced']}")
    _safe_print(f"  Diagram-flagged: {stats['diagram_flagged']}")

    if args.dry_run:
        _safe_print("\n[DRY RUN — no files written]")
        return

    # Save backup
    _safe_print(f"\nSaving backup ({len(backup_entries)} entries)...")
    with open(BACKUP_PATH, "w", encoding="utf-8") as f:
        for b in backup_entries:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    _safe_print(f"  Backup: {BACKUP_PATH}")

    # Write updated DB
    _safe_print(f"Writing updated DB...")
    tmp = DB_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line)

    # Atomic replace
    tmp.replace(DB_PATH)
    _safe_print(f"  DB updated: {DB_PATH}")

    # Save report
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["backup_path"] = str(BACKUP_PATH)
    stats["diagram_flag_path"] = str(DIAGRAM_FLAG_PATH)
    stats["backup_entries"] = len(backup_entries)
    stats["completed_at"] = datetime.now(timezone.utc).isoformat()

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    _safe_print(f"  Report: {REPORT_PATH}")

    _safe_print(f"\n{'=' * 60}")
    _safe_print("DONE")
    _safe_print(f"  To rollback: use {BACKUP_PATH}")
    _safe_print(f"  Diagram manual work: {DIAGRAM_FLAG_PATH} ({stats['diagram_flagged']} QIDs)")


if __name__ == "__main__":
    main()
