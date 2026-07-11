"""
Acadza Question Error Audit Pipeline (v2.1)
==========================================
Reads a CSV or JSON/JSONL file containing a `mongodb_id` column/key,
fetches each question's details from the Acadza API, cleans the HTML
(converts MathML -> LaTeX, strips UI noise, marks images explicitly),
runs deterministic structural prechecks, runs it through two independent
LLM reviewers (M1 = gpt-4o-mini, M2 = gpt-5-mini) with a strict
checklist-style prompt, resolves disagreements, classifies confirmed
errors as faculty-solvable vs code-solvable, updates confirmed-clean
questions back to error=1 via the update API, and writes everything
to CSVs.

--------------------------------------------------------------------
WHY THIS VERSION EXISTS
--------------------------------------------------------------------
The v1 pipeline sent raw HTML (with embedded MathML and base64
drag-handle <img> spacer tags) straight to the LLMs and asked a fairly
open-ended "does this have an error?" question. Two known failure
modes slipped through as a result:

  1. A question missing an entire lettered option (e.g. option (B)
     absent) got marked clean, because nothing forced the model to
     count options.
  2. A "solution" that was really just an embedded image with almost
     no surrounding text got marked clean, because the model was never
     told to treat unreadable image content as unverifiable rather
     than assumed-correct.

v2 fixed both classes of bug with CODE, not just a better prompt:
  - clean_html_for_llm() converts MathML to LaTeX and turns every
    <img> into an explicit "[IMAGE: url]" marker so the model can
    never silently skip over image-only content.
  - structural_precheck() runs deterministic regex/structural checks
    (option count, solution length after removing image placeholders,
    leftover/broken MathML, literal "null" artifacts) BEFORE and
    AFTER the LLM pass. A qid can only be auto-marked clean
    (error=1) if it passes prechecks AND both LLMs agree it's clean.
  - The prompt is a strict, numbered checklist the model must verify
    item by item, forbidding it from marking something clean just
    because it didn't spot anything wrong on a casual read.
  - Output has ONE reconciled `error_description` column instead
    of separate m1/m2 description columns, plus a `flagged_by` column
    (llm / precheck / both) so you can see what actually caught it.

--------------------------------------------------------------------
WHAT CHANGED IN v2.1 (two real bugs fixed)
--------------------------------------------------------------------
BUG 1 - malformed MathML silently converting to an empty string instead
of raising an error:
    _mathml_elem_to_latex() had fallback branches like:
        if len(kids) >= 2: ... else: return "".join(kids)
    for msup/msub/msubsup/mfrac/mroot/mover/munder. A structurally
    malformed tag - e.g. an <msup> with only ONE child instead of the
    required base+exponent pair (this happens with genuinely broken
    source HTML, see the CFSE/Co(II) question example) - would silently
    fall through to the "else" branch and return whatever partial text
    was there (often empty), rather than raising.
    Because nothing raised, mathml_to_latex() never threw, so
    clean_html_for_llm() never inserted the "[UNPARSEABLE_MATHML]"
    marker, so detect_code_artifacts() never flagged it, so the
    question sailed through prechecks AND both LLMs as "clean" and got
    auto-updated to error=1 even though the browser itself renders
    "Invalid <msup> element" for that same MathML.
    FIX: every one of those branches now does `if len(kids) != N: raise
    ValueError(...)` instead of silently falling back. A malformed tag
    now always surfaces as [UNPARSEABLE_MATHML], which correctly blocks
    the auto-clean path and routes the qid to errors_code_solvable.csv.

BUG 2 - duplicate rows in errors_code_solvable.csv (and general row
duplication risk in errors_confirmed.csv):
    The old process_one() wrote a row for the LLM/faculty-precheck
    verdict in one code block, and then - completely separately and
    unconditionally - wrote ANOTHER row for code_flags in a second code
    block. Any qid that was flagged by both paths (e.g. an
    "image-only solution" faculty flag AND an "[UNPARSEABLE_MATHML]"
    code flag) got written to errors_code_solvable.csv TWICE, with the
    qid/mongodb_id repeated.
    FIX: all findings (LLM verdict, faculty precheck flags, code
    precheck flags) are now merged into a single deduplicated list
    BEFORE any row is written. Each qid gets AT MOST ONE row in
    errors_confirmed.csv, AT MOST ONE row in errors_faculty.csv (if it
    has any faculty-category findings), and AT MOST ONE row in
    errors_code_solvable.csv (if it has any code-category findings).
    Multiple issues within the same category are joined with ", " in a
    single description/category cell instead of producing extra rows.

--------------------------------------------------------------------
SETUP
--------------------------------------------------------------------
pip install langchain langchain-openai pandas requests pydantic python-dotenv tiktoken --break-system-packages

Set these environment variables (or put them in a .env file next to this script):
    OPENAI_API_KEY   -> for both M1 (gpt-4o-mini) and M2 (gpt-5-mini)

Update ACADZA_BEARER_TOKEN below (or set env var ACADZA_BEARER_TOKEN) -
these tokens expire, you will need to refresh it periodically.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
python acadza_error_audit.py --input path/to/file.csv --column mongodb_id
python acadza_error_audit.py --input path/to/file.json --column mongodb_id --test-mode 20
python acadza_error_audit.py --input path/to/file.csv --workers 40 --no-update

--------------------------------------------------------------------
ID COLUMN RESOLUTION
--------------------------------------------------------------------
Pass the primary column with --column (e.g. --column mongodb_id).
For each row, that column is checked first. If the value there isn't a real
Mongo _id (24-char hex), the script falls back to checking the columns listed
in FALLBACK_ID_COLUMNS (near the top of this file) on that same row.
Rows where no real _id is found anywhere (only a qid, or nothing) are NOT
sent to the API - they're logged to unresolved_qid.csv instead, since the
details API needs the real _id, not the qid.
Edit FALLBACK_ID_COLUMNS to match whatever column names your exports use.

Outputs (written to ./audit_output/ by default):
    errors_confirmed.csv        -> LLM and/or precheck agree there's an error
                                    (ONE row per qid, multiple issues comma-joined)
    errors_faculty.csv          -> subset that needs a human
                                    (ONE row per qid, multiple issues comma-joined)
    errors_code_solvable.csv    -> subset that's auto-fixable (formatting/MathML/etc.)
                                    (ONE row per qid, multiple issues comma-joined)
                                    NOTE: a qid can land here purely from the
                                    deterministic precheck even if both LLMs
                                    said "clean" - see structural_precheck().
    flagged_passage_match.csv   -> passage/match type questions, untouched
    updated_to_error1.csv       -> questions confirmed clean, updated via API
                                    (only if NO faculty or code flag fired at all)
    unresolved_qid.csv          -> rows where no real mongodb _id could be found
    audit_log.jsonl             -> full raw record of every decision (for resume/debug)
    cost_summary.json           -> approximate token usage & $ cost for M1 and M2
    run_summary.json            -> counts: errors found, faculty vs code-solvable, updated, etc.

Also written:
    all_qid_backups/<mongodb_id>.json    -> raw details JSON for EVERY qid fetched
    pre_update_backups/<mongodb_id>.json -> extra snapshot taken right before an update PUT
"""

import argparse
import json
import os
import re
import sys
import time
import threading
import csv as csv_module
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path
from typing import Optional, Literal

import pandas as pd
import requests
import tiktoken
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_openai import ChatOpenAI

# ============================================================================
# CONFIG
# ============================================================================

ACADZA_BASE_URL = "https://api.acadza.in"
ACADZA_DETAILS_URL = f"{ACADZA_BASE_URL}/question/details"
ACADZA_UPDATE_URL = f"{ACADZA_BASE_URL}/question/update/field"
ACADZA_API_KEY = "postmanrulz"

# Refresh this when it expires. Env var takes priority if set.
ACADZA_BEARER_TOKEN = os.environ.get(
    "ACADZA_BEARER_TOKEN",
    "PASTE_YOUR_BEARER_TOKEN_HERE",
)

VIEW_URL_BASE = "https://www.acadza.com/question/view-question/"
EDIT_URL_BASE = {
    "scq": "https://www.acadza.com/question/edit-question/single-question/",
    "mcq": "https://www.acadza.com/question/edit-question/multi-question/",
    "passage": "https://www.acadza.com/question/edit-question/passage-question/",
    "integer": "https://www.acadza.com/question/edit-question/integer-question/",
    "match": "https://www.acadza.com/question/edit-question/match-question/",
}

# --------------------------------------------------------------------------
# ID RESOLUTION
# --------------------------------------------------------------------------
FALLBACK_ID_COLUMNS = [
    "mongodb_id", "mongo_id", "_id", "id", "objectId",
    "question_id", "qid", "select_qid",
]

MONGO_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")


def classify_id_value(value) -> str:
    if value is None:
        return "empty"
    v = str(value).strip()
    if not v or v.lower() == "nan":
        return "empty"
    if MONGO_ID_RE.match(v):
        return "mongodb_id"
    return "qid"


def resolve_row_id(row: dict, primary_column: str):
    columns_in_order = [primary_column] + [
        c for c in FALLBACK_ID_COLUMNS if c != primary_column
    ]
    first_qid_value = None
    first_qid_column = None

    for col in columns_in_order:
        if col not in row:
            continue
        raw_val = row.get(col)
        id_type = classify_id_value(raw_val)
        if id_type == "mongodb_id":
            return str(raw_val).strip(), "mongodb_id", col
        if id_type == "qid" and first_qid_value is None:
            first_qid_value = str(raw_val).strip()
            first_qid_column = col

    if first_qid_value is not None:
        return first_qid_value, "qid", first_qid_column

    return None, "empty", None


TYPE_TO_BLOCK_KEY = {
    "scq": "scq",
    "mcq": "mcq",
    "integer": "integerQuestion",
    "passage": "passageQuestion",
    "match": "matchQuestion",
}

FACULTY_CATEGORIES = {
    "Content/Concept error",
    "Wrong answer key",
    "Missing solution",
    "Missing question text",
    "Missing data",
    "Missing option(s)",
    "Missing image",
    "Wrong diagram/graph",
    "Image-dependent values",
    "Ambiguous question",
    "Multiple correct answers",
    "Wrong chapter/subconcept",
    "Question type mismatch",
    "Visual/Layout issue",
    "Unknown/low confidence",
}

CODE_SOLVABLE_CATEGORIES = {
    "Spelling",
    "Grammar",
    "LaTeX leak",
    "Broken/unconverted MathML",
    "null artifact",
    "Formatting artifacts",
}

ALL_CATEGORIES = FACULTY_CATEGORIES | CODE_SOLVABLE_CATEGORIES

M1_MODEL = "gpt-5-mini"   # M1 - cheaper/faster first-pass reviewer
M2_MODEL = "gpt-5-mini"    # M2 - stronger second-pass reviewer

MAX_RETRIES = 5
BACKOFF_BASE = 2  # seconds

# Minimum length (in chars, after stripping [IMAGE: ...] placeholders and
# whitespace) that a solution's non-image text must have before we stop
# treating it as "essentially image-only, cannot verify".
MIN_SOLUTION_TEXT_CHARS = 25

# ============================================================================
# TOKEN / COST TRACKING
# ============================================================================
PRICING_PER_MILLION_TOKENS = {
    "m1": {"input": 0.25, "output": 2.00},   # gpt-5-mini, USD / 1M tokens
    "m2": {"input": 0.25, "output": 2.00},   # gpt-5-mini, USD / 1M tokens - EDIT before trusting
}

_token_encoding = tiktoken.get_encoding("cl100k_base")
_token_lock = threading.Lock()

TOKEN_COUNTS = {
    "m1": {"input": 0, "output": 0, "calls": 0},
    "m2": {"input": 0, "output": 0, "calls": 0},
}

_summary_lock = threading.Lock()

RUN_SUMMARY = {
    "total_processed": 0,
    "errors_confirmed": 0,
    "errors_faculty": 0,
    "errors_code_solvable": 0,
    "updated_to_error1": 0,
    "flagged_passage_match": 0,
    "unresolved_qid": 0,
    "fetch_or_llm_failed": 0,
    "no_content_skipped": 0,
    "update_failed": 0,
    "precheck_blocked_clean": 0,   # would have been auto-cleaned, precheck stopped it
}


def bump_summary(key: str, amount: int = 1):
    with _summary_lock:
        RUN_SUMMARY[key] = RUN_SUMMARY.get(key, 0) + amount


def print_run_summary():
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"Total qids processed:                {RUN_SUMMARY['total_processed']}")
    print(f"Errors confirmed (LLM and/or precheck): {RUN_SUMMARY['errors_confirmed']}")
    print(f"  -> sent to errors_faculty.csv:      {RUN_SUMMARY['errors_faculty']}")
    print(f"  -> sent to errors_code_solvable.csv:{RUN_SUMMARY['errors_code_solvable']}")
    print(f"Confirmed clean, updated to error=1: {RUN_SUMMARY['updated_to_error1']}")
    print(f"  -> blocked from auto-clean by precheck: {RUN_SUMMARY['precheck_blocked_clean']}")
    print(f"Flagged passage/match (untouched):   {RUN_SUMMARY['flagged_passage_match']}")
    print(f"Unresolved id (no real _id found):   {RUN_SUMMARY['unresolved_qid']}")
    print(f"Fetch/LLM call failures (skipped):   {RUN_SUMMARY['fetch_or_llm_failed']}")
    print(f"No question/solution content:        {RUN_SUMMARY['no_content_skipped']}")
    print(f"Update API failures:                 {RUN_SUMMARY['update_failed']}")
    print("=" * 60)

    with open(OUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(RUN_SUMMARY, f, indent=2)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_token_encoding.encode(text))


def record_tokens(model_label: str, input_text: str, output_text: str):
    input_tokens = count_tokens(input_text)
    output_tokens = count_tokens(output_text)
    with _token_lock:
        TOKEN_COUNTS[model_label]["input"] += input_tokens
        TOKEN_COUNTS[model_label]["output"] += output_tokens
        TOKEN_COUNTS[model_label]["calls"] += 1


def compute_cost(model_label: str) -> float:
    counts = TOKEN_COUNTS[model_label]
    price = PRICING_PER_MILLION_TOKENS[model_label]
    return (counts["input"] / 1_000_000 * price["input"]) + (
        counts["output"] / 1_000_000 * price["output"]
    )


def print_and_save_cost_summary():
    summary = {}
    grand_total = 0.0
    print("\n" + "=" * 60)
    print("TOKEN USAGE / COST SUMMARY (approximate)")
    print("=" * 60)
    for label, name in [("m1", f"M1 ({M1_MODEL})"), ("m2", f"M2 ({M2_MODEL})")]:
        counts = TOKEN_COUNTS[label]
        cost = compute_cost(label)
        grand_total += cost
        summary[label] = {
            "model": M1_MODEL if label == "m1" else M2_MODEL,
            "calls": counts["calls"],
            "input_tokens": counts["input"],
            "output_tokens": counts["output"],
            "estimated_cost_usd": round(cost, 4),
        }
        print(f"{name}: {counts['calls']} calls | "
              f"{counts['input']:,} input tok | {counts['output']:,} output tok | "
              f"~${cost:.4f}")
    summary["total_estimated_cost_usd"] = round(grand_total, 4)
    print(f"TOTAL ESTIMATED COST: ~${grand_total:.4f}")
    print("=" * 60)

    with open(OUT_DIR / "cost_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ============================================================================
# OUTPUT FILES / THREAD LOCKS
# ============================================================================

OUT_DIR = Path("test-100,saurabhsirerros_v4")
OUT_DIR.mkdir(exist_ok=True)

FILES = {
    "errors_confirmed": OUT_DIR / "errors_confirmed.csv",
    "errors_faculty": OUT_DIR / "errors_faculty.csv",
    "errors_code_solvable": OUT_DIR / "errors_code_solvable.csv",
    "flagged_passage_match": OUT_DIR / "flagged_passage_match.csv",
    "updated_to_error1": OUT_DIR / "updated_to_error1.csv",
    "unresolved_qid": OUT_DIR / "unresolved_qid.csv",
    "audit_log": OUT_DIR / "audit_log.jsonl",
}

HEADERS = {
    "errors_confirmed": [
        "mongodb_id", "qid", "question_type", "subject",
        "error_description", "error_category", "flagged_by",
        "resolution_type", "view_link", "edit_link",
    ],
    "errors_faculty": [
        "mongodb_id", "qid", "question_type", "subject",
        "error_description", "error_category", "flagged_by",
        "view_link", "edit_link",
    ],
    "errors_code_solvable": [
        "mongodb_id", "qid", "question_type", "subject",
        "error_description", "error_category", "flagged_by",
        "view_link", "edit_link",
    ],
    "flagged_passage_match": [
        "mongodb_id", "qid", "question_type", "subject", "view_link",
    ],
    "updated_to_error1": [
        "mongodb_id", "qid", "question_type", "subject", "view_link", "edit_link",
    ],
    "unresolved_qid": [
        "value_found", "source_column", "primary_column_value", "note",
    ],
}

_locks = {name: threading.Lock() for name in FILES}


def init_csv_files():
    for name, path in FILES.items():
        if name == "audit_log":
            continue
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv_module.writer(f)
                writer.writerow(HEADERS[name])


def append_row(name: str, row: list):
    path = FILES[name]
    with _locks[name]:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv_module.writer(f)
            writer.writerow(row)


def append_log(record: dict):
    with _locks["audit_log"]:
        with open(FILES["audit_log"], "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_already_processed() -> set:
    done = set()
    if FILES["audit_log"].exists():
        with open(FILES["audit_log"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("mongodb_id"):
                        done.add(rec["mongodb_id"])
                except json.JSONDecodeError:
                    continue
    return done


# ============================================================================
# RETRY HELPER
# ============================================================================

def with_retries(fn, *args, what="operation", **kwargs):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            wait = BACKOFF_BASE ** attempt
            print(f"  [retry {attempt}/{MAX_RETRIES}] {what} failed: {e} -> waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {MAX_RETRIES} retries: {last_exc}")


# ============================================================================
# ACADZA API CALLS
# ============================================================================

def acadza_details_headers(mongodb_id: str) -> dict:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {ACADZA_BEARER_TOKEN}",
        "Content-Type": "application/json",
        "questionId": mongodb_id,
        "api-key": ACADZA_API_KEY,
        "course": "undefined",
    }


def fetch_question_details(mongodb_id: str) -> dict:
    def _call():
        resp = requests.post(
            ACADZA_DETAILS_URL,
            headers=acadza_details_headers(mongodb_id),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    return with_retries(_call, what=f"details fetch {mongodb_id}")


def update_question_error_field(mongodb_id: str, original_payload: dict) -> dict:
    payload = dict(original_payload)
    payload["error"] = 1

    def _call():
        resp = requests.put(
            ACADZA_UPDATE_URL,
            headers={
                "Authorization": f"Bearer {ACADZA_BEARER_TOKEN}",
                "Content-Type": "application/json",
                "questionId": mongodb_id,
                "api-key": ACADZA_API_KEY,
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    return with_retries(_call, what=f"update {mongodb_id}")


def backup_qid_json(mongodb_id: str, payload: dict, subdir: str):
    backup_dir = OUT_DIR / subdir
    backup_dir.mkdir(exist_ok=True)
    with open(backup_dir / f"{mongodb_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ============================================================================
# QUESTION TYPE / TEXT EXTRACTION
# ============================================================================

def normalize_question_type(raw_type: str) -> str:
    if not raw_type:
        return ""
    t = raw_type.strip().lower()
    if t.endswith("question"):
        t = t[: -len("question")]
    return t


def extract_question_solution(details: dict, qtype: str):
    block_key = TYPE_TO_BLOCK_KEY.get(qtype)
    if not block_key:
        return None, None
    block = details.get(block_key) or {}
    question = block.get("question", "") or ""
    solution = block.get("solution", "") or ""
    return question, solution


def build_view_edit_links(mongodb_id: str, qtype: str):
    view_link = VIEW_URL_BASE + mongodb_id
    edit_base = EDIT_URL_BASE.get(qtype, "")
    edit_link = (edit_base + mongodb_id) if edit_base else ""
    return view_link, edit_link


# ============================================================================
# MATHML -> LATEX CONVERSION
# ============================================================================
# Best-effort recursive converter covering the MathML constructs that show
# up in this content (mfrac, msup, msub, msubsup, msqrt, mroot, mover,
# munder, simple mtable). Anything it can't confidently handle - INCLUDING
# structurally malformed tags with the wrong number of children (e.g. an
# <msup> missing its exponent) - now RAISES rather than silently returning
# partial/empty text. That exception is caught one level up in
# clean_html_for_llm(), which turns it into an explicit
# "[UNPARSEABLE_MATHML]" marker so nothing is ever lost silently. This is
# the fix for the bug where a malformed <msup> (base with no exponent)
# converted to an empty string instead of raising, letting a genuinely
# broken question (browser renders "Invalid <msup> element") sail through
# as "clean" and get auto-updated to error=1.

_MO_TRANSLATIONS = {
    "×": r"\times", "÷": r"\div", "−": "-", "–": "-",
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx",
    "→": r"\rightarrow", "⇌": r"\rightleftharpoons", "°": r"^\circ",
    "∘": r"^\circ", "·": r"\cdot", "⋅": r"\cdot",
}


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _mo_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.strip()
    return _MO_TRANSLATIONS.get(text, text)


def _children_latex(el) -> list:
    return [_mathml_elem_to_latex(child) for child in list(el)]


def _mixed_content(el) -> str:
    """Walks an element's mixed content properly: el.text, then for each
    child its converted latex followed by that child's *tail* text (the
    text trailing the child before the next sibling/closing tag). Plain
    "".join(children) silently drops tail text - which is exactly how a
    real bug slipped through in testing: an <mtext> containing
    <img/><br/>"Some Answer Text" lost "Some Answer Text" because it lives
    in <br/>'s tail, not in mtext's own .text."""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(_mathml_elem_to_latex(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _mathml_elem_to_latex(el) -> str:
    tag = _strip_ns(el.tag)
    text = (el.text or "").strip()

    if tag in ("math", "mrow", "mstyle", "semantics", "annotation", "mtext"):
        return _mixed_content(el)

    if tag in ("mi", "mn"):
        return text

    if tag == "mo":
        return " " + _mo_text(text) + " "

    if tag == "mspace":
        return " "

    # --- The following tags all have a STRUCTURALLY REQUIRED number of
    # children. Previously these fell back to `"".join(kids)` when the
    # count didn't match, which could silently produce empty/partial
    # output for malformed MathML. They now raise ValueError instead,
    # which propagates up and gets converted into an explicit
    # "[UNPARSEABLE_MATHML]" marker by clean_html_for_llm(). ---

    if tag == "mfrac":
        kids = _children_latex(el)
        if len(kids) != 2:
            raise ValueError(f"mfrac expects 2 children, got {len(kids)}")
        return r"\frac{" + kids[0] + "}{" + kids[1] + "}"

    if tag == "msup":
        kids = _children_latex(el)
        if len(kids) != 2:
            raise ValueError(f"msup expects 2 children (base, exponent), got {len(kids)}")
        return "{" + kids[0] + "}^{" + kids[1] + "}"

    if tag == "msub":
        kids = _children_latex(el)
        if len(kids) != 2:
            raise ValueError(f"msub expects 2 children (base, subscript), got {len(kids)}")
        return "{" + kids[0] + "}_{" + kids[1] + "}"

    if tag == "msubsup":
        kids = _children_latex(el)
        if len(kids) != 3:
            raise ValueError(f"msubsup expects 3 children (base, sub, sup), got {len(kids)}")
        return "{" + kids[0] + "}_{" + kids[1] + "}^{" + kids[2] + "}"

    if tag == "msqrt":
        kids = _children_latex(el)
        if len(kids) < 1:
            raise ValueError("msqrt expects at least 1 child")
        return r"\sqrt{" + "".join(kids) + "}"

    if tag == "mroot":
        kids = _children_latex(el)
        if len(kids) != 2:
            raise ValueError(f"mroot expects 2 children (radicand, index), got {len(kids)}")
        return r"\sqrt[" + kids[1] + "]{" + kids[0] + "}"

    if tag == "mover":
        kids = _children_latex(el)
        if len(kids) != 2:
            raise ValueError(f"mover expects 2 children (base, over), got {len(kids)}")
        over = kids[1].strip()
        if over in ("¯", "-", r"\bar"):
            return r"\overline{" + kids[0] + "}"
        if over in ("^", "\u005e"):
            return r"\hat{" + kids[0] + "}"
        return "{" + kids[0] + "}^{" + kids[1] + "}"

    if tag == "munder":
        kids = _children_latex(el)
        if len(kids) != 2:
            raise ValueError(f"munder expects 2 children (base, under), got {len(kids)}")
        return "{" + kids[0] + "}_{" + kids[1] + "}"

    if tag in ("mtable", "mtr", "mtd"):
        # Rare in this dataset - fall back to space-joined cell text so we
        # at least don't lose the content, but this is not "true" LaTeX.
        return " ".join(_children_latex(el))

    # Unknown tag (including things like <img>, <br/> that end up nested
    # inside MathML in this dataset) - fall back to mixed-content walk so
    # any trailing text is preserved rather than silently dropped.
    return _mixed_content(el)


def mathml_to_latex(mathml_string: str) -> str:
    """Convert a single <math>...</math> block to a LaTeX string.
    Raises on totally malformed XML, and now also on structurally invalid
    MathML (wrong child count for msup/msub/mfrac/etc.), so the caller can
    flag it as [UNPARSEABLE_MATHML] rather than silently losing content."""
    # ElementTree needs a single root and doesn't like bare & etc. - the
    # calling code already runs this on isolated <math>...</math> chunks.
    cleaned = mathml_string.strip()
    root = ET.fromstring(cleaned)
    latex = _mathml_elem_to_latex(root).strip()
    latex = re.sub(r"\s+", " ", latex).strip()
    return latex


# ============================================================================
# HTML CLEANING FOR LLM CONSUMPTION
# ============================================================================

_BASE64_GIF_RE = re.compile(r'<img[^>]*src="data:image/gif;base64[^"]*"[^>]*/?>', re.IGNORECASE)
_MATH_BLOCK_RE = re.compile(r'<math\b[^>]*>.*?</math>', re.IGNORECASE | re.DOTALL)
_IMG_SRC_RE = re.compile(r'<img[^>]*\bsrc="([^"]+)"[^>]*/?>', re.IGNORECASE)
_TAG_STRIP_RE = re.compile(r'<[^>]+>')


def clean_html_for_llm(html: str) -> str:
    """
    - Removes zero-signal base64 drag-handle spacer images.
    - Converts every <math>...</math> MathML block to inline $...$ LaTeX.
      Unparseable or structurally invalid blocks become an explicit
      [UNPARSEABLE_MATHML] marker (never silently dropped or emptied).
    - Converts every remaining <img src="..."> into an explicit
      "[IMAGE: url]" marker so the model is forced to acknowledge content
      it cannot actually see, instead of ignoring the tag.
    - Unescapes HTML entities (&ndash; -> -, &times; -> x, etc).
    Structural tags (h3, span, sup, sub, br) are left in place on purpose -
    they still carry some signal (e.g. sup/sub around exponents) and the
    LLM is instructed to read past markup, not be confused by it.
    """
    if not html:
        return ""

    cleaned = _BASE64_GIF_RE.sub("", html)

    def _repl_math(m: "re.Match") -> str:
        try:
            latex = mathml_to_latex(m.group(0))
            if not latex:
                # Successfully parsed but produced nothing usable (e.g. an
                # <msup> with an empty <mi></mi> exponent) - treat as
                # unparseable too, don't let it disappear as blank space.
                return " [UNPARSEABLE_MATHML] "
            return f" ${latex}$ "
        except Exception:
            return " [UNPARSEABLE_MATHML] "

    cleaned = _MATH_BLOCK_RE.sub(_repl_math, cleaned)

    def _repl_img(m: "re.Match") -> str:
        return f" [IMAGE: {m.group(1)}] "

    cleaned = _IMG_SRC_RE.sub(_repl_img, cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def strip_tags_and_images(text: str) -> str:
    """Used only for the length-based 'is this solution basically empty'
    precheck - removes [IMAGE: ...] markers and any leftover HTML tags."""
    no_images = re.sub(r"\[IMAGE:[^\]]*\]", "", text)
    no_tags = _TAG_STRIP_RE.sub("", no_images)
    return re.sub(r"\s+", " ", no_tags).strip()


# ============================================================================
# DETERMINISTIC STRUCTURAL PRECHECKS
# ============================================================================
# These run in plain Python, independent of the LLMs. They exist because an
# LLM reading a wall of markup can miss a missing option or an image-only
# solution - a regex/length check cannot.
#
# Question-type-specific expectations enforced here:
#   scq      -> exactly 4 lettered options (A)-(D); answer key is exactly
#               ONE of those letters (single correct option).
#   mcq      -> exactly 4 lettered options (A)-(D); answer key is ONE OR
#               MORE of those letters (multiple options can be correct).
#   integer  -> NO lettered options should appear in the question at all;
#               answer key must be a plain integer or decimal number
#               (e.g. "4", "-3", "2.88", "0.5") - not a letter, not text.

_OPTION_LETTER_RE = {
    letter: re.compile(rf"\(\s*{letter}\s*\)", re.IGNORECASE)
    for letter in ["A", "B", "C", "D", "E"]
}

_LEFTOVER_MATHML_RE = re.compile(
    r"</?m(row|frac|sup|sub|subsup|sqrt|root|i|n|o|text|over|under|table|tr|td|style|space)\b",
    re.IGNORECASE,
)
_NULL_ARTIFACT_RE = re.compile(r"(?<![a-zA-Z])null(?![a-zA-Z])", re.IGNORECASE)
_UNPARSEABLE_RE = re.compile(r"\[UNPARSEABLE_MATHML\]")
_ANSWER_LETTER_SPLIT_RE = re.compile(r"[,\s/|]+")


def _present_option_letters(cleaned_question: str) -> list:
    return [l for l in ["A", "B", "C", "D"] if _OPTION_LETTER_RE[l].search(cleaned_question)]


def _extra_option_letters(cleaned_question: str) -> list:
    """Options beyond D (e.g. an (E)) - rare, but worth a heads-up rather
    than silently ignoring since scq/mcq are only ever expected to have 4."""
    return [l for l in ["E"] if _OPTION_LETTER_RE[l].search(cleaned_question)]


def _parse_answer_letters(answer_key: str) -> list:
    if not answer_key:
        return []
    raw = _ANSWER_LETTER_SPLIT_RE.split(str(answer_key).strip())
    letters = [r.strip().upper() for r in raw if r.strip()]
    return letters


def detect_option_count_issue(cleaned_question: str, qtype: str) -> Optional[str]:
    """scq/mcq must show exactly 4 lettered options (A)-(D). Returns a
    faculty-flag description if the count is wrong, else None. Not applied
    to integer type (handled separately - integer questions should have
    ZERO options)."""
    if qtype not in ("scq", "mcq"):
        return None
    present = _present_option_letters(cleaned_question)
    extra = _extra_option_letters(cleaned_question)
    missing = [l for l in ["A", "B", "C", "D"] if l not in present]

    if len(present) == 4 and not extra:
        return None
    if len(present) == 0:
        return f"{qtype.upper()} question has NO lettered options (A)-(D) detected in the question text at all."
    if missing:
        return (
            f"{qtype.upper()} question shows only {len(present)}/4 expected lettered options "
            f"(missing: {', '.join(missing)}) - a standard 4-option set is expected."
        )
    if extra:
        return f"{qtype.upper()} question shows extra option letter(s) beyond D: {', '.join(extra)} - unexpected for this question type."
    return None


def detect_options_present_when_none_expected(cleaned_question: str, qtype: str) -> Optional[str]:
    """Integer-type questions should have NO lettered (A)/(B)/(C)/(D) options
    at all - if any are present, the question is likely mistagged or has
    leftover MCQ-style content."""
    if qtype != "integer":
        return None
    present = _present_option_letters(cleaned_question)
    if present:
        return (
            f"Integer-type question unexpectedly contains lettered option(s) "
            f"{', '.join(present)} in the question text - integer questions should have "
            f"no multiple-choice-style options. Possible question-type mismatch or leftover content."
        )
    return None


def detect_answer_key_issue(qtype: str, answer_key: str, cleaned_question: str) -> Optional[str]:
    """Cross-checks the answer key's shape/content against what's expected
    for the question type, and (for scq/mcq) against the options actually
    present in the question."""
    answer_key = (answer_key or "").strip()

    if qtype == "scq":
        letters = _parse_answer_letters(answer_key)
        if len(letters) == 0:
            return "SCQ answer key is empty - a single correct option letter is expected."
        if len(letters) > 1:
            return f"SCQ answer key lists multiple options ({answer_key}) but SCQ questions should have exactly ONE correct answer."
        present = _present_option_letters(cleaned_question)
        if present and letters[0] not in present:
            return f"SCQ answer key references option '{letters[0]}' which is not among the options present in the question ({', '.join(present) or 'none found'})."
        return None

    if qtype == "mcq":
        letters = _parse_answer_letters(answer_key)
        if len(letters) == 0:
            return "MCQ answer key is empty - one or more correct option letters are expected."
        present = _present_option_letters(cleaned_question)
        bad_letters = [l for l in letters if present and l not in present]
        if bad_letters:
            return f"MCQ answer key references option(s) {', '.join(bad_letters)} not present in the question ({', '.join(present) or 'none found'})."
        return None

    if qtype == "integer":
        if not answer_key:
            return "Integer-type question has an empty answer key."
        # NOTE: intentionally NOT enforcing a plain-numeric format here.
        # Integer-type answer keys can legitimately be words, letters, or
        # formulas (e.g. "zero", a chemical name, "8"), not just digits.
        # The real invariant for this question type is that it must never
        # show MCQ-style lettered options (A)/(B)/(C)/(D) in the question
        # text - that's already checked separately by
        # detect_options_present_when_none_expected(). A non-numeric
        # answer key on its own is NOT an error.
        return None

    return None


def detect_thin_solution(cleaned_solution: str) -> bool:
    """True if, after stripping [IMAGE: ...] placeholders and tags, the
    solution has almost no verifiable text left - i.e. it's effectively
    just an image with nothing a reviewer (or the LLM) can check."""
    if not cleaned_solution:
        return True
    text_only = strip_tags_and_images(cleaned_solution)
    return len(text_only) < MIN_SOLUTION_TEXT_CHARS


def detect_code_artifacts(cleaned_question: str, cleaned_solution: str) -> list:
    hits = []
    for label, text in [("question", cleaned_question), ("solution", cleaned_solution)]:
        if _LEFTOVER_MATHML_RE.search(text):
            hits.append(f"{label}: leftover/unconverted MathML tag in cleaned content")
        if _NULL_ARTIFACT_RE.search(text):
            hits.append(f"{label}: literal 'null' artifact present")
        if _UNPARSEABLE_RE.search(text):
            hits.append(f"{label}: MathML block failed to convert to LaTeX")
    return hits


def structural_precheck(cleaned_question: str, cleaned_solution: str, qtype: str, answer_key: str = "") -> dict:
    """Returns {'faculty_flags': [...], 'code_flags': [...]}.
    qtype must be one of 'scq', 'mcq', 'integer' (passage/match never reach
    here - they're flagged-and-skipped earlier in process_one)."""
    faculty_flags = []
    code_flags = []

    option_issue = detect_option_count_issue(cleaned_question, qtype)
    if option_issue:
        faculty_flags.append(option_issue)

    integer_option_issue = detect_options_present_when_none_expected(cleaned_question, qtype)
    if integer_option_issue:
        faculty_flags.append(integer_option_issue)

    answer_issue = detect_answer_key_issue(qtype, answer_key, cleaned_question)
    if answer_issue:
        faculty_flags.append(answer_issue)

    if detect_thin_solution(cleaned_solution):
        faculty_flags.append(
            "Solution is effectively image-only (no substantive verifiable text after "
            "removing [IMAGE: ...] placeholders) - faculty must confirm the image content "
            "matches the answer key."
        )

    code_flags.extend(detect_code_artifacts(cleaned_question, cleaned_solution))

    return {"faculty_flags": faculty_flags, "code_flags": code_flags}


def categorize_precheck_flag(flag_text: str) -> str:
    """Maps a precheck faculty-flag string to one of the fixed error
    categories, so the CSVs always get a real category instead of the
    literal flag sentence."""
    f = flag_text.lower()
    if "answer key" in f:
        return "Wrong answer key"
    if "lettered option" in f or "option letter" in f or "option(s)" in f:
        return "Missing option(s)"
    if "image-only" in f or "image-dependent" in f:
        return "Image-dependent values"
    if "solution" in f:
        return "Missing solution"
    return "Missing data"


def categorize_code_flag(flag_text: str) -> str:
    """Maps a precheck code-artifact flag string to one of the fixed
    code-solvable categories."""
    f = flag_text.lower()
    if "null artifact" in f:
        return "null artifact"
    if "failed to convert" in f or "unparseable" in f:
        return "Broken/unconverted MathML"
    if "leftover/unconverted mathml" in f:
        return "Broken/unconverted MathML"
    return "Formatting artifacts"


# ============================================================================
# LLM SETUP
# ============================================================================

class ErrorAnalysis(BaseModel):
    has_error: bool = Field(description="True if the question/solution has any error.")
    error_description: str = Field(
        default="", description="Concise, specific description of the error found. Empty if no error."
    )
    error_category: str = Field(
        default="",
        description=(
            "Exactly one category from this fixed list, empty if no error: "
            + ", ".join(sorted(ALL_CATEGORIES))
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(default="medium")


class RecheckAnalysis(BaseModel):
    agrees_error_exists: bool = Field(
        description="True if, after independently re-checking, you confirm the claimed error is real."
    )
    error_description: str = Field(default="")
    error_category: str = Field(default="")
    confidence: Literal["low", "medium", "high"] = Field(default="medium")


SYSTEM_PROMPT = """You are an expert exam question quality auditor for a JEE/NEET coaching platform.

You will be given cleaned question and solution content. MathML has already been converted
to inline LaTeX between $...$ signs. Any embedded image has been replaced with a literal
marker in the form [IMAGE: <url>] - you cannot see the actual image, only that one exists there.

You MUST work through this checklist explicitly, item by item, before deciding has_error.
Do NOT mark has_error=False just because nothing jumped out on a casual read - actively verify
every item below against the actual text in front of you. You will be told the question type
(scq, mcq, or integer) - apply the rule for that type exactly:

1. OPTION COUNT AND TYPE-SPECIFIC SHAPE:
   You will be given an "AUTOMATED OPTION SCAN" line before the question text, produced by a
   deterministic regex scan of the raw markup. TRUST THAT SCAN'S COUNT over your own visual
   read of the HTML - the question markup in this dataset is sometimes deeply nested or has
   unclosed/malformed tags that can make an option marker like "(D)" easy to visually lose
   track of while reading, even though it is unambiguously present in the text. Do not report
   a missing or extra option count that contradicts the automated scan unless you can point to
   a specific, concrete reason the scan itself would be wrong (e.g. a stray "(D)" appears
   inside body/explanation text rather than as an actual option marker with no option content
   after it - this is rare, so treat it as the exception, not the default assumption).
   FIRST, before applying any rule below, correctly identify what actually counts as a
   "lettered option". A real lettered answer-choice option (used identically by BOTH SCQ and
   MCQ question types - this is not specific to either one) is a labelled ANSWER CHOICE
   presented in the format "(A) <choice text>", "(B) <choice text>", etc., almost always one
   per line (often after a <br/> or at the start of the options block), where each one is a
   complete, independent candidate answer to the question. This is COMPLETELY DIFFERENT from a
   capital letter used as an algebraic constant, variable, or label INSIDE a mathematical
   expression - e.g. "(1+alpha)^7 = A + B*alpha + C*alpha^2", "A, B, C >= 0",
   "5(3A - 2B - C)". Those A/B/C are just named unknowns being solved for as part of the
   problem itself, not answer choices, and seeing them is completely normal and correct for
   SCQ, MCQ, AND integer-type questions alike. Do NOT flag a question just because capital
   letters appear somewhere in an equation - only flag based on the actual presence or absence
   of a real "(A) ... (B) ... (C) ... (D) ..." style answer-choice list.
   - SCQ: must show exactly 4 real lettered answer-choice options (A)-(D) as defined above, and
     the answer key must be EXACTLY ONE of those letters (a single correct option). Fewer than
     4 such options, extra options, or an answer key with more than one letter are all errors -
     category "Missing option(s)" or "Wrong answer key" as appropriate.
   - MCQ: must show exactly 4 real lettered answer-choice options (A)-(D) as defined above, but
     the answer key may correctly list MORE THAN ONE letter (multiple correct options is normal
     for MCQ, not an error).
   - Integer type: must show NO real lettered answer-choice options (A)/(B)/(C)/(D) as defined
     above in the question text. Algebraic constants/variables named A, B, C, etc. appearing
     inside the equation itself are expected and NOT an error. The answer key for an integer
     question does NOT have to be a plain digit-only number - it can legitimately be a word,
     name, formula, or other non-numeric value (e.g. "zero", a chemical name) and that alone is
     NOT an error. Only flag as "Question type mismatch" if there is an actual answer-choice-
     style options list "(A) ... (B) ... (C) ... (D) ..." present in the question text - that is
     the only real invariant for this question type.

2. ANSWER KEY CONSISTENCY: Confirm the answer key letter (given separately, or implied by the
   solution's final stated conclusion) corresponds to one of the options actually visible in
   the question text word-for-word. If it references an option that isn't there, or the
   solution's final conclusion doesn't match the answer key, that is an error.
   REMINDER - MCQ EXCEPTION: for MCQ questions specifically, the answer key legitimately listing
   TWO OR MORE letters (e.g. "A, D") is completely normal and correct - this is NOT an error,
   NOT "Wrong answer key", and NOT "Multiple correct answers" (that category is for SCQ, where
   only one option should ever be correct but the solution shows two or more as valid - it does
   NOT apply to MCQ, where multiple correct options are the expected, intended format). For
   MCQ, only flag the answer key if it references a letter with NO corresponding option in the
   question text, or if the solution's own worked reasoning genuinely does not support one of
   the listed letters as correct.
   CRITICAL SELF-CHECK before flagging "Wrong answer key": explicitly write down (a) the given
   answer key letter, and (b) the letter/option the solution's final conclusion actually
   supports. Only flag an error if (a) and (b) are GENUINELY DIFFERENT letters. Do not flag
   "Wrong answer key" if (a) and (b) are the same letter - this includes cases where the
   solution's wording is imprecise, uses different phrasing than the option text, or where you
   are inclined to describe a data/labeling nitpick in the solution - if the letter the
   solution concludes with is the same as the given answer key, there is NO answer-key error.
   Never produce a description that says the correct answer should be the same letter as the
   given answer key while simultaneously claiming the answer key is wrong - that is a
   self-contradiction and means you must NOT flag this error.

3. SOLUTION COMPLETENESS: The solution must contain an actual derivation or reasoning that
   ends in a stated final value/conclusion, expressed in text or LaTeX you can read - not
   just a formula setup that trails off, and not just an [IMAGE: ...] marker with no
   surrounding explanation. If the solution is mostly or entirely an [IMAGE: ...] marker with
   little to no readable text around it, you CANNOT verify it is correct - treat this as an
   error, category "Image-dependent values" or "Missing solution". Never assume an image you
   cannot see is correct.

4. IMAGE-DEPENDENT CLAIMS: Wherever an [IMAGE: ...] marker appears in a position where the
   surrounding text depends on reading that image to be understood or verified (e.g. "the
   compound shown below", a graph, a diagram with labelled values used later in the solution),
   explicitly note that you cannot verify that portion. If the rest of the question/solution
   cannot be fully verified without it, flag it rather than silently passing.

5. MATHML / LATEX ARTIFACTS: Look for literal tag fragments (e.g. <mrow>, <msup>), the literal
   word "null", double-encoded entities, or a "[UNPARSEABLE_MATHML]" marker. Any of these is an
   error - category "LaTeX leak", "Broken/unconverted MathML", or "null artifact" - these are
   code_solvable, not faculty issues.

6. CONTENT/CONCEPT CORRECTNESS: Only after 1-5 are clear, evaluate whether the chemistry/
   physics/math/biology content itself is scientifically correct and the answer key is right.

7. SPELLING / GRAMMAR / LAYOUT: Note genuine spelling or grammar errors, or garbled/broken
   markup (stray nested spans, unclosed tags) as code_solvable formatting issues.

Pick exactly one category from the fixed list you're given if there is an error. Be precise:
flag real, specific, checklist-grounded errors, not vague stylistic nitpicks. But do NOT default
to "no error" out of caution when items 1-5 above give you a concrete, checkable reason to flag
something - those are objective checks, not judgment calls.
"""

RECHECK_SYSTEM_PROMPT = """You are an expert exam question quality auditor for a JEE/NEET coaching platform.

Another reviewer independently claimed there is an error in this question/solution (using the
same MathML->LaTeX + [IMAGE: url] cleaned format you're seeing) and gave a description below.
You initially found no error. Re-examine the question and solution CAREFULLY against the same
checklist you were given originally (option count, answer key consistency, solution
completeness, image-dependent claims, MathML/LaTeX artifacts, content correctness, spelling/
layout) and independently verify: is the claimed error actually real?

Do not simply defer to the other reviewer - check it yourself against the actual text. Only
confirm agrees_error_exists=True if, on careful re-inspection, you genuinely find that specific
error (or a closely related real error) yourself.

Reminder on option counting: a real lettered answer-choice option (used identically by BOTH
SCQ and MCQ, not specific to either) is a labelled answer choice in the format
"(A) <choice text>", "(B) <choice text>", etc. Capital letters used as algebraic constants or
variables inside an equation (e.g. "A + Bx + Cx^2", "A, B, C >= 0") are NOT answer options and
their presence in an integer-type (or any type) question is normal, not an error. If the other
reviewer's claimed error is "lettered options in an integer question" or similar, verify there
is an actual answer-choice-style list before agreeing - do not agree just because capital
letters appear somewhere in the math.

Reminder on answer key consistency: before agreeing an answer key is wrong, explicitly write
down (a) the given answer key letter and (b) the letter the solution's conclusion actually
supports. Only agree if these are genuinely different letters. If they are the same letter,
the claimed error is a false positive - do not agree with it, regardless of how the other
reviewer phrased the description. Also remember: for MCQ questions, an answer key listing TWO
OR MORE letters is normal and correct, not an error - do not agree with a claimed error that
is really just "the MCQ answer key has more than one letter".
"""


def get_m1_llm():
    return ChatOpenAI(model=M1_MODEL, temperature=0).with_structured_output(ErrorAnalysis)


def get_m2_llm():
    return ChatOpenAI(model=M2_MODEL, temperature=0).with_structured_output(ErrorAnalysis)


def get_m1_recheck_llm():
    return ChatOpenAI(model=M1_MODEL, temperature=0).with_structured_output(RecheckAnalysis)


def get_m2_recheck_llm():
    return ChatOpenAI(model=M2_MODEL, temperature=0).with_structured_output(RecheckAnalysis)


def build_option_scan_hint(cleaned_question: str, qtype: str) -> str:
    """Builds a short grounding line reporting the deterministic regex-based
    option count, so the LLM doesn't have to re-derive this purely by eye
    from messy/deeply-nested HTML (a real failure mode: option markers like
    "(D)" buried in garbled, unclosed <span> tags have caused the LLM to
    miscount options that a plain regex finds without any trouble). The LLM
    is instructed to trust this scan over its own visual read unless it has
    a specific, quotable reason to doubt it."""
    if qtype not in ("scq", "mcq"):
        present = _present_option_letters(cleaned_question)
        if present:
            return (
                f"AUTOMATED OPTION SCAN (ground truth, regex-based - trust this over your own "
                f"visual read of the markup): detected lettered option marker(s) {', '.join(present)} "
                f"in the question text despite this being a {qtype} question, which should have none."
            )
        return "AUTOMATED OPTION SCAN: no lettered option markers detected (expected for this question type)."
    present = _present_option_letters(cleaned_question)
    extra = _extra_option_letters(cleaned_question)
    all_found = present + extra
    return (
        f"AUTOMATED OPTION SCAN (ground truth, regex-based - trust this over your own visual "
        f"read of the markup, especially if the HTML below looks messy or has unclosed/nested "
        f"tags): detected {len(all_found)} lettered option marker(s): "
        f"{', '.join(all_found) if all_found else '(none found)'}. "
        f"Do NOT report a different option count than this scan unless you can quote the exact "
        f"specific reason (e.g. a letter marker appears but has no actual choice text after it)."
    )


def analyze_question(llm, question_clean: str, solution_clean: str, answer_key: str, qtype: str, model_label: str) -> ErrorAnalysis:
    option_hint = build_option_scan_hint(question_clean, qtype)
    user_msg = (
        f"QUESTION TYPE: {qtype}\n"
        f"ANSWER KEY: {answer_key or '(not provided)'}\n"
        f"{option_hint}\n\n"
        f"QUESTION (cleaned):\n{question_clean}\n\n"
        f"SOLUTION (cleaned):\n{solution_clean}"
    )

    def _call():
        return llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("human", user_msg),
        ])

    result = with_retries(_call, what="LLM error analysis")
    record_tokens(model_label, input_text=SYSTEM_PROMPT + "\n" + user_msg, output_text=result.model_dump_json())
    return result


def recheck_question(llm, question_clean: str, solution_clean: str, answer_key: str, qtype: str, claimed_error: str, model_label: str) -> RecheckAnalysis:
    option_hint = build_option_scan_hint(question_clean, qtype)
    user_msg = (
        f"QUESTION TYPE: {qtype}\n"
        f"ANSWER KEY: {answer_key or '(not provided)'}\n"
        f"{option_hint}\n\n"
        f"QUESTION (cleaned):\n{question_clean}\n\n"
        f"SOLUTION (cleaned):\n{solution_clean}\n\n"
        f"OTHER REVIEWER'S CLAIMED ERROR:\n{claimed_error}"
    )

    def _call():
        return llm.invoke([
            ("system", RECHECK_SYSTEM_PROMPT),
            ("human", user_msg),
        ])

    result = with_retries(_call, what="LLM recheck")
    record_tokens(model_label, input_text=RECHECK_SYSTEM_PROMPT + "\n" + user_msg, output_text=result.model_dump_json())
    return result


# ============================================================================
# CORE PER-ROW PIPELINE
# ============================================================================

def resolution_type_for_category(category: str) -> str:
    if category in CODE_SOLVABLE_CATEGORIES:
        return "code_solvable"
    return "faculty"


def process_one(mongodb_id: str, do_update: bool):
    mongodb_id = str(mongodb_id).strip()
    if not mongodb_id:
        return

    bump_summary("total_processed")

    try:
        details = fetch_question_details(mongodb_id)
    except Exception as e:
        print(f"[SKIP] {mongodb_id}: failed to fetch details -> {e}")
        append_log({"mongodb_id": mongodb_id, "status": "fetch_failed", "error": str(e)})
        bump_summary("fetch_or_llm_failed")
        return

    try:
        backup_qid_json(mongodb_id, details, subdir="all_qid_backups")
    except Exception as e:
        print(f"[WARN] {mongodb_id}: backup failed (continuing anyway) -> {e}")

    raw_type = details.get("questionType", "")
    qtype = normalize_question_type(raw_type)
    subject = details.get("subject", "") or (
        (details.get("tagSubConcept") or [{}])[0].get("subject", "")
    )
    qid = details.get("qid", "")
    view_link, edit_link = build_view_edit_links(mongodb_id, qtype)

    if qtype in ("passage", "match"):
        append_row("flagged_passage_match", [mongodb_id, qid, qtype, subject, view_link])
        append_log({"mongodb_id": mongodb_id, "status": "flagged_passage_or_match", "qtype": qtype})
        bump_summary("flagged_passage_match")
        print(f"[FLAG] {mongodb_id} is {qtype} type - flagged, skipped.")
        return

    raw_question, raw_solution = extract_question_solution(details, qtype)
    if not raw_question and not raw_solution:
        print(f"[SKIP] {mongodb_id}: no question/solution text found for type {qtype}")
        append_log({"mongodb_id": mongodb_id, "status": "no_content", "qtype": qtype})
        bump_summary("no_content_skipped")
        return

    question_clean = clean_html_for_llm(raw_question)
    solution_clean = clean_html_for_llm(raw_solution)
    answer_key = details.get(TYPE_TO_BLOCK_KEY.get(qtype, ""), {}).get("answer", "")
    if isinstance(answer_key, list):
        answer_key = ", ".join(str(a) for a in answer_key)

    precheck = structural_precheck(question_clean, solution_clean, qtype, answer_key)

    # --- M1 and M2 independent analysis ---
    try:
        m1_llm = get_m1_llm()
        m1_result: ErrorAnalysis = analyze_question(m1_llm, question_clean, solution_clean, answer_key, qtype, "m1")
    except Exception as e:
        print(f"[SKIP] {mongodb_id}: M1 analysis failed -> {e}")
        append_log({"mongodb_id": mongodb_id, "status": "m1_failed", "error": str(e)})
        bump_summary("fetch_or_llm_failed")
        return

    try:
        m2_llm = get_m2_llm()
        m2_result: ErrorAnalysis = analyze_question(m2_llm, question_clean, solution_clean, answer_key, qtype, "m2")
    except Exception as e:
        print(f"[SKIP] {mongodb_id}: M2 analysis failed -> {e}")
        append_log({"mongodb_id": mongodb_id, "status": "m2_failed", "error": str(e)})
        bump_summary("fetch_or_llm_failed")
        return

    llm_has_error = False
    llm_category = ""
    llm_description = ""

    if m1_result.has_error and m2_result.has_error:
        llm_has_error = True
        llm_category = m1_result.error_category or m2_result.error_category
        llm_description = m1_result.error_description or m2_result.error_description

    elif not m1_result.has_error and not m2_result.has_error:
        llm_has_error = False

    else:
        if m1_result.has_error and not m2_result.has_error:
            try:
                recheck: RecheckAnalysis = recheck_question(
                    get_m2_recheck_llm(), question_clean, solution_clean, answer_key, qtype,
                    m1_result.error_description, "m2",
                )
            except Exception as e:
                print(f"[SKIP] {mongodb_id}: M2 recheck failed -> {e}")
                append_log({"mongodb_id": mongodb_id, "status": "m2_recheck_failed", "error": str(e)})
                bump_summary("fetch_or_llm_failed")
                return
            if recheck.agrees_error_exists:
                llm_has_error = True
                llm_category = m1_result.error_category or recheck.error_category
                llm_description = recheck.error_description or m1_result.error_description
        else:
            try:
                recheck: RecheckAnalysis = recheck_question(
                    get_m1_recheck_llm(), question_clean, solution_clean, answer_key, qtype,
                    m2_result.error_description, "m1",
                )
            except Exception as e:
                print(f"[SKIP] {mongodb_id}: M1 recheck failed -> {e}")
                append_log({"mongodb_id": mongodb_id, "status": "m1_recheck_failed", "error": str(e)})
                bump_summary("fetch_or_llm_failed")
                return
            if recheck.agrees_error_exists:
                llm_has_error = True
                llm_category = m2_result.error_category or recheck.error_category
                llm_description = recheck.error_description or m2_result.error_description

    # ------------------------------------------------------------------
    # RECONCILE LLM verdict + faculty precheck flags + code precheck flags
    # into ONE merged set of findings, so each qid gets AT MOST ONE row
    # per output CSV, with multiple issues comma-joined instead of the
    # qid/mongodb_id being duplicated across rows.
    #
    # (This replaces the old logic that wrote a row for the LLM/faculty
    # verdict in one block and then, separately and unconditionally,
    # wrote ANOTHER row for code_flags - which caused the duplicate rows
    # you saw for the same qid in errors_code_solvable.csv.)
    # ------------------------------------------------------------------
    faculty_flags = precheck["faculty_flags"]
    code_flags = precheck["code_flags"]

    findings = []          # list of (description, category) tuples
    seen_desc = set()      # dedup guard on description text

    def add_finding(desc: str, cat: str):
        desc = (desc or "").strip()
        key = desc.lower()
        if desc and key not in seen_desc:
            seen_desc.add(key)
            findings.append((desc, cat or "Unknown/low confidence"))

    if llm_has_error:
        add_finding(llm_description, llm_category)
    for flag in faculty_flags:
        add_finding(flag, categorize_precheck_flag(flag))
    for flag in code_flags:
        add_finding(flag, categorize_code_flag(flag))

    faculty_items = [(d, c) for d, c in findings if resolution_type_for_category(c) == "faculty"]
    code_items = [(d, c) for d, c in findings if resolution_type_for_category(c) == "code_solvable"]

    final_has_error = bool(findings)

    # which source(s) actually contributed at least one finding
    sources = set()
    if llm_has_error:
        sources.add("llm")
    if faculty_flags or code_flags:
        sources.add("precheck")
    flagged_by = "both" if len(sources) > 1 else (next(iter(sources), ""))

    def combine(items):
        descs = ", ".join(d for d, _ in items)
        cats = ", ".join(sorted(set(c for _, c in items)))
        return descs, cats

    # --- act on final decision: at most one row per CSV, per qid ---
    if final_has_error:
        all_desc, all_cat = combine(findings)
        # If a qid has ANY faculty-category finding it still needs a human,
        # regardless of also having code-solvable findings.
        resolution = "faculty" if faculty_items else "code_solvable"

        append_row(
            "errors_confirmed",
            [mongodb_id, qid, qtype, subject, all_desc, all_cat,
             flagged_by, resolution, view_link, edit_link],
        )
        bump_summary("errors_confirmed")

        # If a qid has ANY faculty-category finding, it goes to
        # errors_faculty.csv ONLY - never also to errors_code_solvable.csv,
        # even if it also has code-solvable findings. A human already has to
        # open it for the faculty issue, so the code issue is just noted
        # inside that same faculty row's description instead of creating a
        # second row for the same qid in the code-solvable sheet.
        if faculty_items:
            fac_desc, fac_cat = combine(findings)  # include code findings too, in one row
            append_row(
                "errors_faculty",
                [mongodb_id, qid, qtype, subject, fac_desc, fac_cat,
                 flagged_by, view_link, edit_link],
            )
            bump_summary("errors_faculty")
        elif code_items:
            code_desc, code_cat = combine(code_items)
            append_row(
                "errors_code_solvable",
                [mongodb_id, qid, qtype, subject, code_desc, code_cat,
                 flagged_by, view_link, edit_link],
            )
            bump_summary("errors_code_solvable")

        append_log({
            "mongodb_id": mongodb_id, "status": "error_confirmed",
            "category": all_cat, "resolution": resolution, "flagged_by": flagged_by,
        })
        print(f"[ERROR] {mongodb_id} -> {all_cat} ({resolution}, via {flagged_by})")

    blocked_from_clean = final_has_error

    if not blocked_from_clean:
        if do_update:
            try:
                backup_qid_json(mongodb_id, details, subdir="pre_update_backups")
                update_question_error_field(mongodb_id, details)
                append_row(
                    "updated_to_error1",
                    [mongodb_id, qid, qtype, subject, view_link, edit_link],
                )
                bump_summary("updated_to_error1")
                append_log({"mongodb_id": mongodb_id, "status": "updated_error1"})
                print(f"[OK] {mongodb_id} confirmed clean -> updated to error=1")
            except Exception as e:
                print(f"[FAIL] {mongodb_id}: update failed -> {e}")
                append_log({"mongodb_id": mongodb_id, "status": "update_failed", "error": str(e)})
                bump_summary("update_failed")
        else:
            append_log({"mongodb_id": mongodb_id, "status": "clean_no_update_mode"})
            print(f"[OK] {mongodb_id} confirmed clean (update skipped, --no-update set)")
    else:
        bump_summary("precheck_blocked_clean")


# ============================================================================
# INPUT LOADING
# ============================================================================

def load_rows(input_path: str) -> list:
    path = Path(input_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str)
        rows = df.to_dict("records")
    elif path.suffix.lower() in (".json", ".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                rows = json.load(f)
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported input file type: {path.suffix}")
    return rows


def resolve_ids_from_rows(rows: list, primary_column: str):
    seen = set()
    resolved = []
    unresolved = []

    for row in rows:
        value, id_type, source_col = resolve_row_id(row, primary_column)
        if id_type == "mongodb_id":
            if value not in seen:
                seen.add(value)
                resolved.append(value)
        else:
            primary_val = row.get(primary_column, "")
            unresolved.append((value, id_type, source_col, primary_val))

    return resolved, unresolved


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Acadza dual-LLM question error audit pipeline (v2.1)")
    parser.add_argument("--input", required=True, help="Path to input CSV/JSON/JSONL file")
    parser.add_argument("--column", default="mongodb_id", help="Column/key holding the mongodb _id")
    parser.add_argument(
        "--workers", type=int, default=35,
        help=(
            "Parallel worker threads (default 35). Each worker makes up to ~4 sequential "
            "API calls per qid (details fetch, M1, M2, possibly a recheck, possibly an "
            "update). If you see lots of retry/backoff messages, lower this to 15-20."
        ),
    )
    parser.add_argument("--test-mode", type=int, default=0, help="Only process first N ids (0 = all)")
    parser.add_argument("--no-update", action="store_true", help="Don't call the update API, dry-run only")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing audit_log.jsonl and reprocess everything")
    args = parser.parse_args()

    if ACADZA_BEARER_TOKEN == "PASTE_YOUR_BEARER_TOKEN_HERE":
        print("WARNING: ACADZA_BEARER_TOKEN not set. Set env var ACADZA_BEARER_TOKEN or edit the script.")
        sys.exit(1)

    init_csv_files()

    rows = load_rows(args.input)
    if args.test_mode > 0:
        rows = rows[: args.test_mode]

    ids, unresolved = resolve_ids_from_rows(rows, args.column)

    for value, id_type, source_col, primary_val in unresolved:
        note = "no valid _id found in any checked column" if id_type == "empty" else \
            f"only a qid-like value was found (column: {source_col}), not a real Mongo _id"
        append_row("unresolved_qid", [value or "", source_col or "", primary_val or "", note])

    if unresolved:
        print(f"[NOTE] {len(unresolved)} row(s) could not be resolved to a real mongodb_id "
              f"-> see audit_output/unresolved_qid.csv")
    RUN_SUMMARY["unresolved_qid"] = len(unresolved)

    if not args.no_resume:
        done = load_already_processed()
        before = len(ids)
        ids = [i for i in ids if i not in done]
        print(f"Resume: {before - len(ids)} ids already processed, {len(ids)} remaining.")

    print(f"Processing {len(ids)} ids with {args.workers} workers (update={'off' if args.no_update else 'on'})")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, mid, not args.no_update): mid for mid in ids
        }
        done_count = 0
        for future in as_completed(futures):
            mid = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[FATAL] {mid}: unhandled exception -> {e}")
            done_count += 1
            if done_count % 25 == 0:
                print(f"--- progress: {done_count}/{len(ids)} ---")

    print_run_summary()
    print_and_save_cost_summary()
    print("Done. See ./audit_output/ for all CSVs.")


if __name__ == "__main__":
    main()