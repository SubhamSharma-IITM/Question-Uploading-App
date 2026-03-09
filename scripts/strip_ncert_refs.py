"""Strip NCERT references from question HTML in the extracted JSONL.

Reads ncert_questions.jsonl, removes NCERT references from question text,
keeps the extracted NCERTref field, writes cleaned output.

Usage:
    python scripts/strip_ncert_refs.py
"""

import json
import re
import time
from pathlib import Path

INPUT_JSONL = Path(__file__).resolve().parent.parent / "data" / "ncert_extraction" / "ncert_questions.jsonl"
OUTPUT_JSONL = Path(__file__).resolve().parent.parent / "data" / "ncert_extraction" / "ncert_questions_cleaned.jsonl"

QUESTION_TYPES = ["scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"]


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs, flush=True)


# ── Stripping patterns ───────────────────────────────────────────────────
# These cover all observed HTML wrapping patterns:
#
# 1. <strong><em>(NCERT Page-14 / N-14)</em></strong>
# 2. <strong>(NCERT Page-15 / N-15)</strong>
# 3. <em>(NCERT Page-15/ N-12)</em>
# 4. Plain: (NCERT Page-14 / N-14)
# 5. Plain: NCERT Page-20 / N-20
# 6. With &nbsp; before: &nbsp;<strong><em>(NCERT...
# 7. With extra paren: (NCERT ( Page-334 / N-252)
# 8. With <br /> after
# 9. Wrapped in <span> tags

# Comprehensive NCERT stripping regex — handles ALL observed patterns:
# - (NCERT Page-14 / N-14), (NCERT Page N-13), (NCERT/Page-35 / N-15)
# - With/without parens, with/without HTML tags in any order
# - (<strong><em>...), (<em><strong>...), <strong>(<em>...)
# - Space variations, &nbsp;, <br/> after

# Page reference core (shared between patterns)
_PAGE_CORE = (
    r'NCERT\s*[(/]?\s*'                         # NCERT with optional ( or /
    r'Page[\s-]*'                                # Page- or Page or Page - (any combo)
    r'[\d]+(?:\s*(?:[/&,]|&amp;)\s*[\d]+)*'      # page number(s)
    r'(?:\s*/\s*N[\s-]*[\d]+(?:\s*(?:[/&,]|&amp;)\s*[\d]+)*)?'  # optional / N-X
    r'(?:\s*[|/I]?\s*\d{4}\s*,?\s*[A-Z]?)?'     # optional year/exam code: | 2021, C
)

# Pattern 1: with parentheses — parens can be inside or outside HTML tags
NCERT_STRIP_RE = re.compile(
    r'(?:&nbsp;|\s)*'                             # leading whitespace
    r'(?:<(?:strong|em|b|i|span|p|li|ol)[^>]*>\s*)*'  # opening HTML tags
    r'[(\[]\s*'                                   # opening paren
    r'(?:<(?:strong|em|b|i|span)[^>]*>\s*)*'      # more HTML tags after paren
    + _PAGE_CORE +
    r'\s*\)?\s*'                                  # optional inner close paren
    r'(?:</(?:strong|em|b|i|span|p|li|ol)>\s*)*'  # closing HTML tags
    r'[)\]]?\s*'                                  # optional closing paren
    r'(?:</(?:strong|em|b|i|span|p|li|ol)>\s*)*'  # more closing tags
    r'(?:<br\s*/?>\s*)?',                         # optional trailing <br/>
    re.IGNORECASE
)

# Pattern 2: without parentheses
NCERT_STRIP_NOPARENS_RE = re.compile(
    r'(?:&nbsp;|\s)*'
    r'(?:<(?:strong|em|b|i|span)[^>]*>\s*)*'
    + _PAGE_CORE +
    r'\s*\)?\s*'
    r'(?:</(?:strong|em|b|i|span)>\s*)*'
    r'(?:<br\s*/?>\s*)?',
    re.IGNORECASE
)

# Solution-only: standalone NCERT mentions (e.g., "as per NCERT" — don't strip these,
# they're part of the explanation text, not references)


def strip_ncert_from_html(html: str) -> tuple[str, int]:
    """Remove NCERT page references from HTML.

    Returns (cleaned_html, count_of_removals).
    Only strips structured references like (NCERT Page-14 / N-14),
    NOT inline mentions like "as per NCERT textbook".
    """
    if not html or 'NCERT' not in html.upper():
        return html, 0

    count = 0

    # First pass: with parentheses (most common)
    new_html, n = NCERT_STRIP_RE.subn('', html)
    count += n

    # Second pass: without parentheses (less common)
    if 'NCERT' in new_html.upper():
        new_html2, n2 = NCERT_STRIP_NOPARENS_RE.subn('', new_html)
        count += n2
        new_html = new_html2

    # Third pass: catch-all for any remaining structured NCERT refs
    # Matches (...NCERT...) OR <tag>(...NCERT...)</tag> in any nesting order
    # IMPORTANT: only strip formatting tags, preserve structural tags (h3, p, div, li, ol, ul)
    if 'NCERT' in new_html.upper():
        CATCHALL = re.compile(
            r'(?:&nbsp;|\s)*'
            r'(?:<(?:strong|em|b|i|span)[^>]*>\s*)*'  # formatting tags only
            r'[(\[]\s*'                  # opening paren
            r'(?:<(?:strong|em|b|i|span)[^>]*>\s*)*'  # more formatting tags after paren
            r'\s*NCERT[^)\]]{0,100}'     # NCERT + up to 100 chars of ref content
            r'[)\]]\s*'                  # closing paren
            r'(?:</(?:strong|em|b|i|span)>\s*)*'  # formatting closing tags only
            r'(?:<br\s*/?>\s*)?',
            re.IGNORECASE
        )
        new_html3, n3 = CATCHALL.subn('', new_html)
        count += n3
        new_html = new_html3

    # Fourth pass: bare NCERT Page/N- refs without parens
    # Covers: "NCERT Page-367", "NCERT/Page N-50", "NCERT Page-N-201", "NCERT Page- N-129"
    # IMPORTANT: only strip formatting tags, preserve structural tags (h3, p, div, li)
    if re.search(r'NCERT\s*[/]?\s*Page', new_html, re.IGNORECASE):
        BARE_REF = re.compile(
            r'(?:&nbsp;|\s)*'
            r'(?:<(?:strong|em|b|i|span)[^>]*>\s*)*'
            r'NCERT\s*[/]?\s*Page[\s-]*'
            r'(?:'
            r'N?[\s-]*[\d]+(?:\s*(?:[/&,]|&amp;)\s*[\d]+)*'   # N-50 or 367 or N-201
            r'(?:\s*/?\s*N[\s-]*[\d]+(?:\s*(?:[/&,]|&amp;)\s*[\d]+)*)?'  # optional / N-X
            r'|'
            r'[\d]+(?:\s*(?:[/&,]|&amp;)\s*[\d]+)*'   # just digits
            r')'
            r'(?:\s*[|/I]?\s*\d{4}\s*,?\s*[A-Z]?)?' # optional | 2021, C or /2021, A
            r'\s*'
            r'(?:</(?:strong|em|b|i|span)>\s*)*'
            r'(?:<br\s*/?>\s*)?',
            re.IGNORECASE
        )
        new_html4, n4 = BARE_REF.subn('', new_html)
        count += n4
        new_html = new_html4

    # Fifth pass: hardcoded remaining edge cases
    # These cover year/exam codes, broken refs, and deeply nested tag combos
    HARDCODED = [
        # <li>NCERT Page-XXX</li> (bare ref in list item)
        (re.compile(r'<li>\s*NCERT\s*[/]?\s*Page[^<]{0,40}</li>', re.IGNORECASE), ''),
        # (NCERT Page-XXX/N-XXX | YYYY, X) — parens with any tags inside/outside
        # Covers: (<em>NCERT Page-239/N-180 I 2023, C)</em></strong>
        #         <em>(NCERT Page-488 | 2021, C)</em></strong>
        #         (NCERT Page-511/N-400/2021, A)</strong></em>
        # IMPORTANT: only strip formatting tags (em/strong/b/i/span), NOT structural (h3/p/div/li)
        (re.compile(
            r'(?:&nbsp;|\s)*'
            r'(?:<(?:strong|em|b|i|span)[^>]*>\s*)*'  # formatting tags before paren
            r'[(\[]\s*'                  # opening paren
            r'(?:<(?:strong|em|b|i|span)[^>]*>\s*)*'  # formatting tags after paren
            r'NCERT\s*[/]?\s*Page[\s-]*' # NCERT Page with flexible spacing
            r'[^)\]]{0,120}'            # everything up to closing paren
            r'[)\]]\s*'                  # closing paren
            r'(?:</(?:strong|em|b|i|span)>\s*)*',  # only formatting closing tags
            re.IGNORECASE
        ), ''),
        # Bare NCERT Page ref with year/exam code, inside spans/tags, NO parens
        # Covers: NCERT Page-305/N-282 | 2019, C</span></span></em></strong><br />
        # IMPORTANT: only strip formatting tags, preserve structural tags
        (re.compile(
            r'(?:&nbsp;|\s)*'
            r'(?:<(?:strong|em|b|i|span)[^>]*>\s*)*'
            r'NCERT\s*[/]?\s*Page[\s-]*'
            r'[\d]+(?:\s*[/]\s*N?[\s-]*[\d]+)*'
            r'(?:\s*[|/I]?\s*\d{4}\s*,?\s*[A-Z]?)?'
            r'\s*'
            r'(?:</(?:strong|em|b|i|span)>\s*)*'
            r'(?:<br\s*/?>)?',
            re.IGNORECASE
        ), ''),
        # NCERT Page followed by </span> then <img> (broken ref, no page number)
        (re.compile(
            r'(?:&nbsp;|\s)*'
            r'(?:<(?:strong|em|b|i|span)[^>]*>\s*)*'
            r'NCERT\s+Page\s*'
            r'(?:</(?:strong|em|b|i|span)>\s*)*'
            r'(?=<img)',
            re.IGNORECASE
        ), ''),
    ]
    for pat, repl in HARDCODED:
        if 'NCERT' not in new_html.upper():
            break
        new_html5, n5 = pat.subn(repl, new_html)
        count += n5
        new_html = new_html5

    # Clean up artifacts: double <br/>, empty tags, trailing &nbsp;
    new_html = re.sub(r'(<br\s*/?>\s*){2,}', '<br />\n', new_html)
    new_html = re.sub(r'<(strong|em|b|i|span)>\s*</\1>', '', new_html)
    new_html = re.sub(r'<(strong|em|b|i|span)>\s*<(strong|em|b|i|span)>\s*</\2>\s*</\1>', '', new_html)
    new_html = re.sub(r'&nbsp;\s*<br', '<br', new_html)

    return new_html, count


def process_entry(entry: dict) -> tuple[dict, int]:
    """Strip NCERT refs from all question fields in entry."""
    total_stripped = 0

    for qtype in QUESTION_TYPES:
        obj = entry.get(qtype)
        if not obj or not isinstance(obj, dict):
            continue

        # Question field
        for field in ["question", "ques"]:
            val = obj.get(field)
            if val and isinstance(val, str) and 'NCERT' in val.upper():
                cleaned, n = strip_ncert_from_html(val)
                if n > 0:
                    obj[field] = cleaned
                    total_stripped += n

        # Solution field — only strip structured refs, not inline mentions
        for field in ["solution", "sol"]:
            val = obj.get(field)
            if val and isinstance(val, str) and 'NCERT' in val.upper():
                cleaned, n = strip_ncert_from_html(val)
                if n > 0:
                    obj[field] = cleaned
                    total_stripped += n

        # passageQuestion sub-questions
        if qtype == "passageQuestion":
            passage = obj.get("passage") or obj.get("stem")
            if passage and isinstance(passage, str) and 'NCERT' in passage.upper():
                cleaned, n = strip_ncert_from_html(passage)
                if n > 0:
                    if "passage" in obj: obj["passage"] = cleaned
                    elif "stem" in obj: obj["stem"] = cleaned
                    total_stripped += n

            sub_qs = obj.get("questions", [])
            if isinstance(sub_qs, list):
                for sq in sub_qs:
                    if not isinstance(sq, dict): continue
                    for sub_type in ["scq", "mcq", "integerQuestion", "matchQuestion"]:
                        sub_obj = sq.get(sub_type)
                        if sub_obj and isinstance(sub_obj, dict):
                            for field in ["question", "ques", "solution", "sol"]:
                                val = sub_obj.get(field)
                                if val and isinstance(val, str) and 'NCERT' in val.upper():
                                    cleaned, n = strip_ncert_from_html(val)
                                    if n > 0:
                                        sub_obj[field] = cleaned
                                        total_stripped += n
                            break

        entry[qtype] = obj
        break

    return entry, total_stripped


def main():
    _safe_print("=" * 60)
    _safe_print("NCERT Reference Stripping")
    _safe_print("=" * 60)

    start = time.time()
    total = 0
    stripped_count = 0
    total_refs_removed = 0
    unchanged = 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            cleaned_entry, n_stripped = process_entry(entry)

            if n_stripped > 0:
                stripped_count += 1
                total_refs_removed += n_stripped
            else:
                unchanged += 1

            f_out.write(json.dumps(cleaned_entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - start

    _safe_print(f"\nDONE in {elapsed:.1f}s")
    _safe_print(f"  Total entries: {total}")
    _safe_print(f"  Entries with refs stripped: {stripped_count}")
    _safe_print(f"  Total refs removed: {total_refs_removed}")
    _safe_print(f"  Unchanged (flag-only or inline mention): {unchanged}")
    _safe_print(f"  Output: {OUTPUT_JSONL}")

    # Verify: show before/after for a few entries
    _safe_print(f"\n--- BEFORE/AFTER SAMPLES ---")
    with open(INPUT_JSONL, "r", encoding="utf-8") as f_orig, \
         open(OUTPUT_JSONL, "r", encoding="utf-8") as f_clean:
        shown = 0
        for orig_line, clean_line in zip(f_orig, f_clean):
            orig = json.loads(orig_line)
            clean = json.loads(clean_line)
            ref = orig.get("NCERTref", "")
            if "Page" not in ref: continue

            for qtype in QUESTION_TYPES:
                o_obj = orig.get(qtype, {})
                c_obj = clean.get(qtype, {})
                if not isinstance(o_obj, dict): continue
                o_q = str(o_obj.get("question", "") or o_obj.get("ques", ""))
                c_q = str(c_obj.get("question", "") or c_obj.get("ques", ""))
                if o_q != c_q:
                    # Find the diff region
                    # Strip HTML for display
                    o_text = re.sub(r'<[^>]+>', ' ', o_q)
                    o_text = re.sub(r'&[a-z]+;', ' ', o_text)
                    o_text = re.sub(r'\s+', ' ', o_text).strip()
                    c_text = re.sub(r'<[^>]+>', ' ', c_q)
                    c_text = re.sub(r'&[a-z]+;', ' ', c_text)
                    c_text = re.sub(r'\s+', ' ', c_text).strip()

                    _safe_print(f"\n  REF: {ref}")
                    _safe_print(f"  BEFORE: ...{o_text[-120:]}")
                    _safe_print(f"  AFTER:  ...{c_text[-120:]}")
                    shown += 1
                    break
            if shown >= 5: break


if __name__ == "__main__":
    main()
