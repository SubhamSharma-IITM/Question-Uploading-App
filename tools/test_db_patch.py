#!/usr/bin/env python3
"""Test script for DB patch operations.

Tests:
1. Fetch question details via API
2. Inspect fields (error, previousExamYear, images, solution)
3. Apply corrections (error=1->2, year fix, text cleanup)
4. Update via API
5. Verify update

Usage:
    python tools/test_db_patch.py --oid 69b9777d5a699930d5df5b73 --action inspect
    python tools/test_db_patch.py --oid 69b9777d5a699930d5df5b73 --action fix-error --dry-run
    python tools/test_db_patch.py --oid 69b9777d5a699930d5df5b73 --action fix-error
    python tools/test_db_patch.py --oid 69b9777d5a699930d5df5b73 --action fix-ampersand --dry-run
"""

import argparse
import json
import re
import sys
import time
from typing import Dict, Optional, Tuple

import requests

API_BASE = "https://api.acadza.in"
DETAILS_PATH = "/question/details"
UPDATE_PATH = "/question/update/field"

VIEW_URL = "https://www.acadza.com/question/view-question/"


def get_details(session: requests.Session, oid: str) -> Optional[Dict]:
    """Fetch question details from Acadza API."""
    url = f"{API_BASE}{DETAILS_PATH}"
    headers = {"Accept": "application/json", "Content-Type": "application/json", "questionid": oid}
    r = session.post(url, headers=headers, json={})
    if r.status_code != 200:
        print(f"  [ERR] Details {oid}: HTTP {r.status_code}")
        return None
    data = r.json()
    return data.get("question", data)


def clean_payload(details: Dict) -> Dict:
    """Remove MongoDB internal fields before update."""
    cleaned = dict(details)
    for k in ("_id", "createdAt", "updatedAt", "__v"):
        cleaned.pop(k, None)
    return cleaned


def update_question(session: requests.Session, oid: str, payload: Dict, dry_run: bool = False) -> bool:
    """Update question via Acadza API."""
    if dry_run:
        print(f"  [DRY-RUN] Would update {oid}")
        return True
    url = f"{API_BASE}{UPDATE_PATH}"
    headers = {"Accept": "application/json", "Content-Type": "application/json", "questionid": oid}
    r = session.put(url, headers=headers, json=payload)
    if r.status_code != 200:
        print(f"  [ERR] Update {oid}: HTTP {r.status_code}: {r.text[:200]}")
        return False
    print(f"  [OK] Updated {oid}")
    return True


# ── Inspection ──────────────────────────────────────────────────────────

def inspect_question(details: Dict, oid: str):
    """Print key fields for inspection."""
    print(f"\n=== {oid} ===")
    print(f"  View: {VIEW_URL}{oid}")
    print(f"  questionType: {details.get('questionType', '?')}")
    print(f"  error: {details.get('error', '?')}")
    print(f"  previousExam: {details.get('previousExam', '?')}")
    print(f"  previousExamYear: {details.get('previousExamYear', '?')}")
    print(f"  subject: {details.get('subject', '?')}")

    qtype = details.get("questionType", "scq")
    qdata = details.get(qtype, {})
    if isinstance(qdata, dict):
        q_text = str(qdata.get("question", ""))
        s_text = str(qdata.get("solution", ""))
        answer = str(qdata.get("answer", details.get("answer", "")))

        # Check for issues
        q_plain = re.sub(r'<[^>]+>', '', q_text)
        ans_match = re.search(r'Ans\.?\s*[\(\[]\s*\d+', q_plain, re.IGNORECASE)
        if ans_match:
            print(f"  [!] Ans in question: {ans_match.group(0)}")

        bare_amp = len(re.findall(r'&(?!amp;|lt;|gt;|nbsp;|#\d+;|#x[0-9a-fA-F]+;|[a-zA-Z]{2,8};)', s_text))
        if bare_amp:
            print(f"  [!] Bare & in solution: {bare_amp} instances")

        imgs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', q_text + s_text)
        print(f"  Images: {len(imgs)}")
        print(f"  Answer: {answer[:50]}")
        print(f"  Q tail: ...{q_plain[-60:]}")


# ── Fix actions ─────────────────────────────────────────────────────────

def fix_error_to_2(details: Dict) -> Tuple[Dict, bool]:
    """Change error=1 to error=2."""
    payload = clean_payload(details)
    if payload.get("error") == 1:
        payload["error"] = 2
        return payload, True
    return payload, False


def fix_year(details: Dict, year: int) -> Tuple[Dict, bool]:
    """Set previousExamYear."""
    payload = clean_payload(details)
    if payload.get("previousExamYear") != year:
        payload["previousExamYear"] = year
        return payload, True
    return payload, False


def fix_ampersand_in_solution(details: Dict) -> Tuple[Dict, bool]:
    """Remove LaTeX alignment artifact <mi>&</mi> or <mi>&amp;</mi> from MathML.

    These are LaTeX column separators from \\begin{aligned} blocks that leaked
    into MathML. They have no mathematical meaning -- <mi> wrapping a bare
    ampersand is never valid content. Safe to remove entirely.
    """
    # Pattern: <mi>&</mi> or <mi>&amp;</mi> (with optional whitespace)
    AMP_MI_RE = re.compile(r'<mi>\s*(?:&amp;|&)\s*</mi>')

    payload = clean_payload(details)
    qtype = payload.get("questionType", "scq")
    qdata = payload.get(qtype, {})
    if not isinstance(qdata, dict):
        return payload, False

    changed = False
    for field in ("question", "solution"):
        text = qdata.get(field, "")
        if not text:
            continue
        cleaned = AMP_MI_RE.sub('', text)
        if cleaned != text:
            qdata[field] = cleaned
            changed = True

    return payload, changed


def fix_ans_in_question(details: Dict) -> Tuple[Dict, bool]:
    """Strip trailing Ans. (N) from question text."""
    ANS_RE = re.compile(r'\s*(?:<br\s*/?>)*\s*Ans\.?\s*[\(\[]\s*\d+\s*[\)\]]\s*(?:</\w+>)*\s*$', re.IGNORECASE)

    payload = clean_payload(details)
    qtype = payload.get("questionType", "scq")
    qdata = payload.get(qtype, {})
    if not isinstance(qdata, dict):
        return payload, False

    q = qdata.get("question", "")
    cleaned = ANS_RE.sub('', q)
    if cleaned != q:
        qdata["question"] = cleaned
        return payload, True
    return payload, False


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Test DB patch operations")
    ap.add_argument("--oid", required=True, help="Question OID (24-hex)")
    ap.add_argument("--action", default="inspect",
                    choices=["inspect", "fix-error", "fix-year", "fix-ampersand", "fix-ans", "fix-all"],
                    help="Action to perform")
    ap.add_argument("--year", type=int, default=2026, help="Year for fix-year action")
    ap.add_argument("--dry-run", action="store_true", help="Don't actually update")
    args = ap.parse_args()

    session = requests.Session()
    details = get_details(session, args.oid)
    if not details:
        print("Failed to fetch details")
        sys.exit(1)

    if args.action == "inspect":
        inspect_question(details, args.oid)
        return

    actions = {
        "fix-error": lambda d: fix_error_to_2(d),
        "fix-year": lambda d: fix_year(d, args.year),
        "fix-ampersand": lambda d: fix_ampersand_in_solution(d),
        "fix-ans": lambda d: fix_ans_in_question(d),
    }

    if args.action == "fix-all":
        payload = clean_payload(details)
        any_changed = False
        for name, fn in actions.items():
            p, changed = fn(details)
            if changed:
                # Merge changes into payload
                for k, v in p.items():
                    payload[k] = v
                any_changed = True
                print(f"  Applied: {name}")
        if any_changed:
            ok = update_question(session, args.oid, payload, args.dry_run)
            print(f"  Result: {'OK' if ok else 'FAILED'}")
        else:
            print("  No changes needed")
    else:
        fn = actions[args.action]
        payload, changed = fn(details)
        if changed:
            print(f"  Change detected for {args.action}")
            ok = update_question(session, args.oid, payload, args.dry_run)
            print(f"  Result: {'OK' if ok else 'FAILED'}")
        else:
            print(f"  No change needed for {args.action}")

    print(f"\n  View: {VIEW_URL}{args.oid}")


if __name__ == "__main__":
    main()
