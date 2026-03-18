#!/usr/bin/env python3
"""Batch DB patch for 3 uploaded sessions.

Applies all confirmed MathML + metadata fixes to uploaded questions via Acadza Details API.

Fixes applied per question:
1. Ampersand: Remove <mi>&amp;</mi> / <mi>&</mi> (LaTeX alignment artifact)
2. Hat: <mo stretchy="false">^</mo> -> <mo>&#x0302;</mo> (circumflex fix)
3. Lim: <msub><mo>lim</mo>...</msub> -> <munder> (underscript)
4. Displaystyle: Add <mstyle displaystyle="true"> to all <math> blocks
5. Error fix: error=1 with API_ERROR -> error=2
6. PYQ year: Set previousExamYear from session config
7. Ans strip: Remove trailing Ans.(N) from question text

Usage:
    python tools/batch_db_patch.py --session 35e7d78e --dry-run
    python tools/batch_db_patch.py --session all
    python tools/batch_db_patch.py --session 09477d7d --limit 5
"""

import argparse
import csv
import json
import glob
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

API_BASE = "https://api.acadza.in"
DETAILS_PATH = "/question/details"
UPDATE_PATH = "/question/update/field"
VIEW_URL = "https://www.acadza.com/question/view-question/"

SESSIONS = {
    "35e7d78e": {"year": 2026},
    "56589a45": {"year": 2026},
    "09477d7d": {"year": 2025},
}

# ── Fix patterns ────────────────────────────────────────────────────────

AMP_MI_RE = re.compile(r'<mi>\s*(?:&amp;|&)\s*</mi>')
LIM_SUB_RE = re.compile(r'<msub>(\s*<mo>lim</mo>)(.*?)</msub>', re.DOTALL)
DS_RE = re.compile(r'(<math[^>]*>)((?:(?!</math>).)*)(</math>)', re.DOTALL)
ANS_TAIL_RE = re.compile(
    r'\s*(?:<br\s*/?>)*\s*Ans\.?\s*[\(\[]\s*\d+\s*[\)\]]\s*(?:</\w+>)*\s*$',
    re.IGNORECASE,
)


def apply_mathml_fixes(text: str) -> Tuple[str, List[str]]:
    """Apply all 4 MathML fixes to a text field. Returns (fixed_text, list_of_fixes_applied)."""
    if not text:
        return text, []
    fixes = []

    # Fix 1: Ampersand
    new = AMP_MI_RE.sub('', text)
    if new != text:
        fixes.append("ampersand")
        text = new

    # Fix 2: Hat
    if '<mo stretchy="false">^</mo>' in text:
        text = text.replace('<mo stretchy="false">^</mo>', '<mo>&#x0302;</mo>')
        fixes.append("hat")

    # Fix 3: Lim underscript
    new = LIM_SUB_RE.sub(r'<munder>\1\2</munder>', text)
    if new != text:
        fixes.append("lim")
        text = new

    # Fix 4: Displaystyle
    def _add_ds(m):
        open_t, content, close_t = m.group(1), m.group(2), m.group(3)
        if 'displaystyle' in content:
            return m.group(0)
        return f'{open_t}<mstyle displaystyle="true">{content}</mstyle>{close_t}'

    new = DS_RE.sub(_add_ds, text)
    if new != text:
        fixes.append("displaystyle")
        text = new

    return text, fixes


def apply_ans_strip(text: str) -> Tuple[str, bool]:
    """Strip trailing Ans.(N) from question text."""
    new = ANS_TAIL_RE.sub('', text)
    return new, new != text


# ── Main ────────────────────────────────────────────────────────────────

def patch_session(session_id: str, dry_run: bool = False, limit: int = 0):
    """Patch all uploaded questions for a session."""
    sess_info = SESSIONS.get(session_id)
    if not sess_info:
        print(f"[ERR] Unknown session: {session_id}")
        return

    year = sess_info["year"]
    data_dir = Path(f"data/sessions/{session_id}")

    # Load OID map from upload logs
    oid_map = {}  # qid -> oid
    for cf in glob.glob(str(data_dir / "upload_logs" / "*_upload_log.csv")):
        with open(cf, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = row.get("UploadedQuestionID", "")
                if uid and len(uid) == 24:
                    oid_map[row["Question META ID"]] = uid

    if not oid_map:
        print(f"[ERR] No upload OIDs found for {session_id}")
        return

    # Load error fix list
    error_fix_oids = set()
    fix_file = Path("data/sessions/api_error_voting_fix.json")
    if fix_file.exists():
        with open(fix_file, "r") as f:
            all_fixes = json.load(f)
        for entry in all_fixes.get(session_id, []):
            error_fix_oids.add(entry["oid"])

    # Load ans strip list
    ans_strip_qids = set()
    patch_log = data_dir / "cleaner_patch_log.json"
    if patch_log.exists():
        with open(patch_log, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                if entry.get("type") == "ans_tail_stripped":
                    ans_strip_qids.add(entry["qid"])

    http = requests.Session()
    hdrs = {"Accept": "application/json", "Content-Type": "application/json"}

    oids = list(oid_map.items())
    if limit > 0:
        oids = oids[:limit]

    total = len(oids)
    updated = 0
    skipped = 0
    failed = 0

    print(f"\n[INFO] Session {session_id}: {total} questions, year={year}")
    print(f"[INFO] Error fix OIDs: {len(error_fix_oids)}, Ans strip QIDs: {len(ans_strip_qids)}")
    print(f"[INFO] Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    for i, (qid, oid) in enumerate(oids, 1):
        # Fetch from API
        r = http.post(f"{API_BASE}{DETAILS_PATH}",
                       headers={**hdrs, "questionid": oid}, json={})
        if r.status_code != 200:
            print(f"  [{i}/{total}] {oid} FETCH FAILED")
            failed += 1
            continue

        data = r.json().get("question", r.json())
        payload = dict(data)
        for k in ("_id", "createdAt", "updatedAt", "__v"):
            payload.pop(k, None)

        changes = []

        # MathML fixes on question + solution
        qtype = payload.get("questionType", "scq")
        qdata = payload.get(qtype, {})
        if isinstance(qdata, dict):
            for field in ("question", "solution"):
                text = qdata.get(field, "")
                if text:
                    fixed, fix_names = apply_mathml_fixes(text)
                    if fix_names:
                        qdata[field] = fixed
                        changes.extend(fix_names)

            # Ans strip (question only)
            if qid in ans_strip_qids:
                q = qdata.get("question", "")
                q_fixed, did_strip = apply_ans_strip(q)
                if did_strip:
                    qdata["question"] = q_fixed
                    changes.append("ans_strip")

        # Error fix
        if oid in error_fix_oids and payload.get("error") == 1:
            payload["error"] = 2
            changes.append("error_1_to_2")

        # Year fix
        if payload.get("previousExamYear") != year:
            payload["previousExamYear"] = year
            changes.append("year")

        if not changes:
            skipped += 1
            continue

        # Update
        if dry_run:
            print(f"  [{i}/{total}] {oid} DRY-RUN: {', '.join(set(changes))}")
            updated += 1
        else:
            r2 = http.put(f"{API_BASE}{UPDATE_PATH}",
                          headers={**hdrs, "questionid": oid}, json=payload)
            if r2.status_code == 200:
                updated += 1
                if i <= 5 or i % 25 == 0:
                    print(f"  [{i}/{total}] {oid} OK: {', '.join(set(changes))}")
            else:
                print(f"  [{i}/{total}] {oid} FAILED: {r2.status_code}")
                failed += 1

        time.sleep(0.08)

    print(f"\n[DONE] {session_id}: updated={updated}, skipped={skipped}, failed={failed}")


def main():
    ap = argparse.ArgumentParser(description="Batch DB patch for uploaded sessions")
    ap.add_argument("--session", required=True, help="Session ID or 'all'")
    ap.add_argument("--dry-run", action="store_true", help="Don't actually update")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N questions")
    args = ap.parse_args()

    if args.session == "all":
        for sid in SESSIONS:
            patch_session(sid, dry_run=args.dry_run, limit=args.limit)
    else:
        patch_session(args.session, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
