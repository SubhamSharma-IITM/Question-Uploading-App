"""Extract NCERT references from DB questions.

Detects all variations of NCERT page references in question text,
extracts them cleanly, and writes NCERT questions to a separate JSONL
with an added NCERTref field.

Patterns detected:
  (NCERT Page-14 / N-14)
  (NCERT Page N-20)
  (NCERT Page-330 / N-250)
  (NCERT Page-19 & 20 / N-19 & 20)
  (NCERT Page-5/N-2)
  (NCERT Page-15/ N-12)
  NCERT Page-20 / N-20        (no parens)
  ( NCERT Page-21 / N-23)     (space after paren)
  (NCERT ( Page-334 / N-252)  (extra paren)
  (NCERT Page-173)            (no N- reference)
  Just "NCERT" standalone in solution text

Usage:
    python scripts/extract_ncert_refs.py
"""

import json
import re
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
DB_PATH = Path(r"D:\ACADZA_WORK\acadza-pdf-generator\data\allque030326.json")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ncert_extraction"
OUTPUT_JSONL = OUTPUT_DIR / "ncert_questions.jsonl"
STATS_FILE = OUTPUT_DIR / "ncert_stats.json"

QUESTION_TYPES = ["scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"]


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs, flush=True)


# ── NCERT Reference Detection ────────────────────────────────────────────

# Master regex: matches all observed NCERT reference patterns
# Captures the full reference including optional parentheses
NCERT_REF_RE = re.compile(
    r'[(\[]?\s*'                          # optional opening paren/bracket with space
    r'NCERT'                              # literal NCERT
    r'\s*\(?\s*'                          # optional extra paren, spaces
    r'(?:Page[- ]?\s*'                    # "Page-" or "Page " or "Page"
    r'[\d]+(?:\s*[/&,]\s*[\d]+)*'         # page number(s): 14, 19 & 20, 15/N-12
    r'(?:\s*/\s*N[- ]?\s*[\d]+(?:\s*[/&,]\s*[\d]+)*)?'  # optional / N-14 part
    r')?'                                 # entire Page... part is optional (might just be "NCERT")
    r'\s*\)?\s*[)\]]?',                   # optional closing paren/bracket
    re.IGNORECASE
)

# Simpler standalone NCERT detector (word boundary)
NCERT_WORD_RE = re.compile(r'(?<![a-zA-Z])NCERT(?![a-zA-Z])', re.IGNORECASE)

# HTML tag stripper
HTML_TAG_RE = re.compile(r'<[^>]+>')
HTML_ENTITY_RE = re.compile(r'&[a-z]+;|&#\d+;|&#x[0-9a-fA-F]+;')


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = HTML_TAG_RE.sub(' ', text)
    text = HTML_ENTITY_RE.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_ncert_ref(html_text: str) -> str | None:
    """Extract clean NCERT reference from HTML text.

    Returns cleaned reference like "NCERT Page-14 / N-14" or None.
    """
    if not html_text or not isinstance(html_text, str):
        return None

    # First check if NCERT word exists at all
    if not NCERT_WORD_RE.search(html_text):
        return None

    # Strip HTML first for cleaner matching
    clean = strip_html(html_text)

    # Try the full reference pattern
    match = NCERT_REF_RE.search(clean)
    if match:
        ref = match.group(0).strip()
        # Clean up: remove outer parens/brackets, extra spaces
        ref = ref.strip('()[] \t')
        ref = re.sub(r'\s+', ' ', ref).strip()
        if ref and len(ref) > 4:  # More than just "NCERT"
            return ref

    # Fallback: find NCERT and grab surrounding context
    m = NCERT_WORD_RE.search(clean)
    if m:
        # Get text from NCERT to next closing paren or end of line
        start = m.start()
        rest = clean[start:]
        # Try to capture up to closing paren or 60 chars
        end_match = re.search(r'[)\]]\s*(?:\(|$|[A-Z]\))', rest)
        if end_match:
            ref = rest[:end_match.start()].strip()
        else:
            # Just take up to 60 chars or next sentence boundary
            ref = rest[:60]
            # Cut at first option marker (A), (B), etc.
            opt_match = re.search(r'\s*\(?[A-D]\)', ref)
            if opt_match:
                ref = ref[:opt_match.start()]

        ref = ref.strip('()[] \t,.')
        ref = re.sub(r'\s+', ' ', ref).strip()
        if ref:
            return ref

    return None


def extract_ncert_from_full_html(html_text: str) -> str | None:
    """Extract NCERT ref directly from raw HTML, handling bold/italic wrappers.

    Many refs are wrapped like: <strong><em>(NCERT Page-14 / N-14)</em></strong>
    """
    if not html_text or 'NCERT' not in html_text.upper():
        return None

    # Try to find the NCERT block in raw HTML — it's usually within <strong>/<em> tags
    # Pattern: captures everything from opening paren/NCERT to closing paren
    html_ncert_re = re.compile(
        r'[(<\[]*\s*(?:<[^>]*>)*\s*'        # opening paren + HTML tags
        r'[\s(]*NCERT\s*\(?\s*'              # NCERT with optional parens
        r'(?:Page[- ]?\s*'
        r'[\d]+(?:\s*(?:[/&,]|&amp;)\s*[\d]+)*'
        r'(?:\s*/\s*N[- ]?\s*[\d]+(?:\s*(?:[/&,]|&amp;)\s*[\d]+)*)?'
        r')?'
        r'\s*\)?\s*'
        r'(?:<[^>]*>)*\s*[)\]]?',
        re.IGNORECASE
    )

    match = html_ncert_re.search(html_text)
    if match:
        raw_ref = match.group(0)
        # Strip all HTML
        clean_ref = strip_html(raw_ref).strip('()[] \t')
        clean_ref = re.sub(r'\s+', ' ', clean_ref).strip()
        if clean_ref and len(clean_ref) > 4:
            return clean_ref

    # Fallback to plain text extraction
    return extract_ncert_ref(html_text)


# ── Field extraction (same nested structure as download script) ──────────

def get_question_id(entry: dict) -> str:
    _id = entry.get("_id", {})
    if isinstance(_id, dict):
        return _id.get("$oid", str(_id))
    return str(_id)


def get_question_html(entry: dict) -> tuple[str, str, str]:
    """Get (question_html, solution_html, question_type) from nested DB entry."""
    for qtype in QUESTION_TYPES:
        obj = entry.get(qtype)
        if not obj or not isinstance(obj, dict):
            continue

        q = obj.get("question") or obj.get("ques") or ""
        s = obj.get("solution") or obj.get("sol") or ""

        # For passageQuestion, also include passage + sub-questions
        if qtype == "passageQuestion":
            passage = obj.get("passage") or obj.get("stem") or ""
            parts_q = [str(passage)] if passage else []
            parts_s = [str(s)] if s else []

            sub_qs = obj.get("questions", [])
            if isinstance(sub_qs, list):
                for sq in sub_qs:
                    if not isinstance(sq, dict): continue
                    for sub_type in ["scq", "mcq", "integerQuestion", "matchQuestion"]:
                        sub_obj = sq.get(sub_type)
                        if sub_obj and isinstance(sub_obj, dict):
                            sq_q = sub_obj.get("question") or sub_obj.get("ques") or ""
                            sq_s = sub_obj.get("solution") or sub_obj.get("sol") or ""
                            if sq_q: parts_q.append(str(sq_q))
                            if sq_s: parts_s.append(str(sq_s))
                            break

            q = " ".join(parts_q)
            s = " ".join(parts_s)

        return str(q), str(s), qtype

    return "", "", ""


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _safe_print("=" * 60)
    _safe_print("NCERT Reference Extraction")
    _safe_print("=" * 60)

    start = time.time()
    total = 0
    error1_count = 0
    ncert_in_question = 0
    ncert_in_solution = 0
    ncert_in_both = 0
    ncert_total = 0
    ncert_flag_match = 0   # isNCERT=true AND we found ref
    ncert_flag_miss = 0    # isNCERT=true but no ref in text
    found_no_flag = 0      # ref in text but isNCERT=false

    ref_patterns = {}  # track unique ref patterns

    with open(DB_PATH, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            if entry.get("error") not in (1, 2):
                continue
            error1_count += 1  # now counts error=1 + error=2

            q_html, s_html, qtype = get_question_html(entry)
            is_ncert_flag = bool(entry.get("isNCERT"))

            # Extract NCERT references
            q_ref = extract_ncert_from_full_html(q_html)
            s_ref = extract_ncert_from_full_html(s_html)

            has_q = q_ref is not None
            has_s = s_ref is not None

            if not has_q and not has_s:
                if is_ncert_flag:
                    ncert_flag_miss += 1
                    # Still include — flagged as NCERT but no text ref found
                    ncert_total += 1
                    out_entry = dict(entry)
                    out_entry["NCERTref"] = "NCERT (flag only, no text reference)"
                    f_out.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                continue

            # Found NCERT reference
            ncert_total += 1
            if has_q: ncert_in_question += 1
            if has_s: ncert_in_solution += 1
            if has_q and has_s: ncert_in_both += 1

            if is_ncert_flag:
                ncert_flag_match += 1
            else:
                found_no_flag += 1

            # Use question ref preferentially, fall back to solution ref
            primary_ref = q_ref or s_ref

            # Track pattern types
            # Normalize: "NCERT Page-14 / N-14" → "NCERT Page-X / N-X"
            normalized = re.sub(r'\d+', 'X', primary_ref)
            ref_patterns[normalized] = ref_patterns.get(normalized, 0) + 1

            # Build output entry — copy original + add NCERTref
            out_entry = dict(entry)
            out_entry["NCERTref"] = primary_ref
            if q_ref and s_ref and q_ref != s_ref:
                out_entry["NCERTref_question"] = q_ref
                out_entry["NCERTref_solution"] = s_ref

            f_out.write(json.dumps(out_entry, ensure_ascii=False) + "\n")

            # Progress
            if ncert_total % 500 == 0:
                _safe_print(f"  Found {ncert_total} NCERT questions so far... (line {total})")

    elapsed = time.time() - start

    # Stats
    stats = {
        "total_entries": total,
        "error1_entries": error1_count,
        "ncert_total": ncert_total,
        "ncert_in_question": ncert_in_question,
        "ncert_in_solution_only": ncert_in_solution - ncert_in_both,
        "ncert_in_both": ncert_in_both,
        "isNCERT_flag_true": ncert_flag_match + ncert_flag_miss,
        "isNCERT_flag_matched": ncert_flag_match,
        "isNCERT_flag_no_text_ref": ncert_flag_miss,
        "text_ref_no_flag": found_no_flag,
        "unique_ref_patterns": len(ref_patterns),
        "top_patterns": dict(sorted(ref_patterns.items(), key=lambda x: -x[1])[:20]),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    _safe_print(f"\n{'=' * 60}")
    _safe_print(f"DONE in {elapsed:.1f}s")
    _safe_print(f"  Total DB entries: {total}")
    _safe_print(f"  Error=1 entries: {error1_count}")
    _safe_print(f"  NCERT questions found: {ncert_total}")
    _safe_print(f"    In question text: {ncert_in_question}")
    _safe_print(f"    In solution only: {ncert_in_solution - ncert_in_both}")
    _safe_print(f"    In both: {ncert_in_both}")
    _safe_print(f"  isNCERT flag=true: {ncert_flag_match + ncert_flag_miss}")
    _safe_print(f"    Matched by text: {ncert_flag_match}")
    _safe_print(f"    No text ref found: {ncert_flag_miss}")
    _safe_print(f"  Text ref but no flag: {found_no_flag}")
    _safe_print(f"\n  Top reference patterns:")
    for pat, cnt in sorted(ref_patterns.items(), key=lambda x: -x[1])[:10]:
        _safe_print(f"    {cnt:5d}x  {pat}")
    _safe_print(f"\n  Output: {OUTPUT_JSONL}")
    _safe_print(f"  Stats: {STATS_FILE}")


if __name__ == "__main__":
    main()
