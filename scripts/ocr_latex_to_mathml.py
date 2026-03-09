"""Convert OCR'd LaTeX to MathML + wrap into HTML.

Reads ocr_clean.jsonl (19,633 entries with LaTeX from MathPix OCR),
converts all LaTeX blocks to MathML using latex2mathml, wraps in HTML,
and outputs a ready-to-use JSONL for DB replacement.

MathPix delimiters: $ $ (inline) and $$ $$ (display).

Usage:
    python scripts/ocr_latex_to_mathml.py
    python scripts/ocr_latex_to_mathml.py --limit 10  # test with N entries
    python scripts/ocr_latex_to_mathml.py --preview 5  # preview N samples, no write
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from latex2mathml.converter import convert as latex_to_mathml
except ImportError:
    print("ERROR: pip install latex2mathml")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────
APP_ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = APP_ROOT / "data" / "ocr_cache"
INPUT_FILE = OCR_DIR / "ocr_clean.jsonl"
OUTPUT_FILE = OCR_DIR / "ocr_converted.jsonl"
FAILED_FILE = OCR_DIR / "latex_conversion_failures.json"
STATS_FILE = OCR_DIR / "conversion_stats.json"
LOG_FILE = OCR_DIR / "conversion_log.jsonl"


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs, flush=True)


# ── LaTeX→MathML conversion ─────────────────────────────────────────────

# Patterns: process longer delimiters first to avoid partial matches
# MathPix text output uses \(...\) for inline and \[...\] for display
# Also supports $...$ and $$...$$ from some outputs
LATEX_PATTERNS = [
    # $$...$$ (display math)
    (re.compile(r'\$\$([^$]+?)\$\$', re.DOTALL), True),
    # \[...\] (display math)
    (re.compile(r'\\\[(.+?)\\\]', re.DOTALL), True),
    # \(...\) (inline math)
    (re.compile(r'\\\((.+?)\\\)', re.DOTALL), False),
    # $...$ (inline math) — last, avoid matching $$ content
    (re.compile(r'(?<!\$)\$(?!\$)([^$]+?)\$(?!\$)'), False),
]


def convert_latex_block(raw_latex: str, is_display: bool) -> str:
    """Convert a single LaTeX block to MathML.

    Returns MathML string with proper xmlns, display attr, and mstyle wrapper.
    ALL math gets <mstyle displaystyle="true"> to avoid squeezed fractions/integrals
    (matches the app's production MathML format).
    Raises ValueError on conversion failure.
    """
    cleaned = raw_latex.strip()
    if not cleaned:
        raise ValueError("Empty LaTeX block")

    mathml = latex_to_mathml(cleaned)

    # Ensure xmlns
    if 'xmlns=' not in mathml:
        mathml = mathml.replace('<math', '<math xmlns="http://www.w3.org/1998/Math/MathML"', 1)

    if is_display:
        # Remove any existing display attr, add display="block"
        mathml = re.sub(r'display="[^"]*"', '', mathml, count=1)
        mathml = mathml.replace('<math', '<math display="block"', 1)
    else:
        # Inline math
        if 'display=' not in mathml:
            mathml = mathml.replace('<math', '<math display="inline"', 1)

    # ALL math gets <mstyle displaystyle="true"> wrapper — matches app production format
    # This prevents squeezed fractions, integrals, summations in inline context
    mathml = re.sub(
        r'(<math[^>]*>)(.*)(</math>)',
        r'\1<mstyle displaystyle="true">\2</mstyle>\3',
        mathml, count=1, flags=re.DOTALL,
    )

    return mathml


def convert_all_latex(text: str) -> Tuple[str, int, int, List[str]]:
    """Convert all LaTeX blocks in text to MathML.

    Returns (converted_text, success_count, fail_count, fail_details).
    """
    success = 0
    fail = 0
    fail_details = []

    for pattern, is_display in LATEX_PATTERNS:
        def _replacer(match):
            nonlocal success, fail
            raw = match.group(1)
            try:
                mathml = convert_latex_block(raw, is_display)
                success += 1
                return mathml
            except Exception as e:
                fail += 1
                fail_details.append(f"{raw[:80]}... -> {str(e)[:80]}")
                # Return original with delimiters intact
                delim = "$$" if is_display else "$"
                return f"{delim}{raw}{delim}"

        text = pattern.sub(_replacer, text)

    return text, success, fail, fail_details


def normalize_option_labels(text: str) -> str:
    """Normalize option labels to (A) (B) (C) (D) format.

    Handles: A) B) C) D), a) b) c) d), A. B. C. D., a. b. c. d.,
    (a) (b) (c) (d), and bare A B C D at line boundaries.

    Forked from app's stage4_metadata._normalize_options_to_ABCD().
    Only matches at line boundaries (<br/>, newline, start) to avoid
    corrupting content like "2.5 kg" or image filenames.
    """
    if not text:
        return text

    # Already has (A) format — skip
    if re.search(r'\(A\)', text) and re.search(r'\(B\)', text):
        return text

    # Map: letter variants → uppercase (A)
    LETTER_MAP = {
        'A': '(A)', 'B': '(B)', 'C': '(C)', 'D': '(D)',
        'a': '(A)', 'b': '(B)', 'c': '(C)', 'd': '(D)',
    }

    # Boundary prefix: <br/>, <br>, newline, or start of string
    boundary = r'(?:<br/>|<br>|\n|^)'

    for lower, upper in [('a', 'A'), ('b', 'B'), ('c', 'C'), ('d', 'D')]:
        label = f'({upper})'
        # X) or x) — letter followed by closing paren
        text = re.sub(
            rf'({boundary})\s*[{lower}{upper}]\)\s*',
            rf'\1{label} ',
            text, flags=re.MULTILINE,
        )
        # X. or x. — letter followed by period (only at boundary)
        text = re.sub(
            rf'({boundary})\s*[{lower}{upper}]\.\s*',
            rf'\1{label} ',
            text, flags=re.MULTILINE,
        )
        # (x) or (X) lowercase → uppercase
        if lower != upper:
            text = re.sub(
                rf'({boundary})\s*\(\s*{lower}\s*\)',
                rf'\1({upper})',
                text, flags=re.MULTILINE,
            )

    # Also handle numeric: 1. 2. 3. 4. and (1) (2) (3) (4) at boundaries
    for num, letter in [('1', 'A'), ('2', 'B'), ('3', 'C'), ('4', 'D')]:
        text = re.sub(
            rf'({boundary})\s*{num}\.\s*',
            rf'\1({letter}) ',
            text, flags=re.MULTILINE,
        )
        text = re.sub(
            rf'({boundary})\s*\(\s*{num}\s*\)',
            rf'\1({letter})',
            text, flags=re.MULTILINE,
        )

    return text


def ensure_option_line_breaks(text: str) -> str:
    """Ensure each option (A)/(B)/(C)/(D) starts on its own line.

    If (A) doesn't have <br/> before it, inject one. Handles cases like:
    "...field is: (A) foo (B) bar" → "...field is:<br/>(A) foo<br/>(B) bar"
    """
    if not text:
        return text

    # For (B), (C), (D): always ensure <br/> before them if not already there
    for letter in ['B', 'C', 'D']:
        text = re.sub(
            rf'(?<!<br/>)(?<!<br>)(?<!\n)\s*\({letter}\)',
            rf'<br/>({letter})',
            text,
        )

    # For (A): inject <br/> before it if preceded by text (not at start)
    # but only if there's actual content before it (not just whitespace)
    text = re.sub(
        r'(?<=\S)\s+\(A\)',
        r'<br/>(A)',
        text,
    )

    return text


def wrap_in_html(text: str) -> str:
    """Wrap converted text in HTML matching the app's cleaner format.

    Uses <br/> for newlines (same as Stage 3 _insert_line_breaks_safely).
    Does NOT use <p> tags — the app uses <br/> injection.
    Also normalizes option labels and ensures line breaks between options.
    """
    if not text:
        return text

    # If already has HTML block tags, leave as-is
    if re.search(r'<(?:p|div|h[1-6]|table|ul|ol)\b', text, re.IGNORECASE):
        return text

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Protect MathML blocks from <br/> injection and option normalization
    math_blocks = []
    def _protect(m):
        math_blocks.append(m.group(0))
        return f"__MATH_{len(math_blocks) - 1}__"
    protected = re.sub(r'<math[^>]*>.*?</math>', _protect, text, flags=re.DOTALL)

    # Double newlines → paragraph break, single → line break
    protected = re.sub(r'\n{2,}', '<br/><br/>', protected)
    protected = protected.replace('\n', '<br/>')

    # Normalize option labels: A) → (A), a. → (A), 1. → (A), etc.
    protected = normalize_option_labels(protected)

    # Ensure each option starts on its own line
    protected = ensure_option_line_breaks(protected)

    # Restore MathML blocks
    for i, block in enumerate(math_blocks):
        protected = protected.replace(f"__MATH_{i}__", block)

    return protected


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert OCR LaTeX to MathML + HTML")
    parser.add_argument("--limit", type=int, default=0, help="Limit entries (0=all)")
    parser.add_argument("--preview", type=int, default=0, help="Preview N samples, no write")
    args = parser.parse_args()

    _safe_print("=" * 60)
    _safe_print("OCR LaTeX -> MathML + HTML Conversion")
    _safe_print("=" * 60)

    if not INPUT_FILE.exists():
        _safe_print(f"ERROR: Input not found: {INPUT_FILE}")
        sys.exit(1)

    # Count input
    total_input = sum(1 for _ in open(INPUT_FILE, "r", encoding="utf-8"))
    _safe_print(f"  Input: {INPUT_FILE} ({total_input} entries)")

    if args.preview:
        _safe_print(f"\n[PREVIEW MODE] Showing {args.preview} samples:\n")
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.preview:
                    break
                entry = json.loads(line)
                qid = entry["_id"].get("$oid", "?")
                for field, data in entry.get("ocr_fields", {}).items():
                    text = data.get("ocr_text", "")
                    latex_raw = data.get("ocr_latex", "")
                    source = text if re.search(r'\\\(|\\\[|\$', text) else (r"\(" + latex_raw + r"\)" if latex_raw else "")
                    if not source or not re.search(r'\\\(|\\\[|\$', source):
                        continue
                    converted, sc, fc, _ = convert_all_latex(source)
                    html = wrap_in_html(converted)
                    _safe_print(f"--- {qid} | {field} | {sc} OK, {fc} fail ---")
                    _safe_print(f"  SRC:  {source[:150]}")
                    _safe_print(f"  HTML: {html[:150]}")
                    _safe_print()
        return

    # Process all entries
    start = time.time()
    total_entries = 0
    total_success = 0
    total_fail = 0
    total_latex_blocks = 0
    entries_with_failures = []
    conversion_log = []

    OCR_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout, \
         open(LOG_FILE, "w", encoding="utf-8") as flog:

        for line_num, line in enumerate(fin):
            if args.limit > 0 and line_num >= args.limit:
                break

            entry = json.loads(line)
            qid = entry["_id"].get("$oid", "?")
            total_entries += 1

            entry_success = 0
            entry_fail = 0
            entry_fail_details = []

            for field, data in entry.get("ocr_fields", {}).items():
                text = data.get("ocr_text", "")
                latex_raw = data.get("ocr_latex", "")

                if not text and not latex_raw:
                    continue

                # Primary: ocr_text has \(...\) and \[...\] delimiters mixed with plain text
                # Fallback: if ocr_text has no delimiters but latex_raw exists, wrap it
                source = text
                has_delimiters = bool(
                    re.search(r'\\\(|\\\[|\$', source)
                )

                if not has_delimiters and latex_raw:
                    # Bare latex — wrap as single inline math block
                    source = r"\(" + latex_raw + r"\)"

                if re.search(r'\\\(|\\\[|\$', source):
                    converted, sc, fc, details = convert_all_latex(source)
                    entry_success += sc
                    entry_fail += fc
                    entry_fail_details.extend(details)
                    total_latex_blocks += sc + fc
                else:
                    converted = source

                html = wrap_in_html(converted)

                # Store converted HTML, keep originals for reference
                data["html"] = html
                data["latex_conversion_success"] = entry_success
                data["latex_conversion_fail"] = entry_fail

            total_success += entry_success
            total_fail += entry_fail

            if entry_fail > 0:
                entries_with_failures.append({
                    "qid": qid,
                    "fail_count": entry_fail,
                    "details": entry_fail_details[:5],
                })

            # Log entry
            log_entry = {
                "qid": qid,
                "qtype": entry.get("qtype", ""),
                "fields_converted": list(entry.get("ocr_fields", {}).keys()),
                "latex_success": entry_success,
                "latex_fail": entry_fail,
            }
            flog.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            # Write converted entry
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if total_entries % 2000 == 0:
                _safe_print(
                    f"  [{total_entries}/{total_input}] "
                    f"LaTeX: {total_success} OK, {total_fail} fail"
                )

    elapsed = time.time() - start

    # Save failure report
    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total_entries_with_failures": len(entries_with_failures),
            "total_failed_blocks": total_fail,
            "entries": entries_with_failures,
        }, f, ensure_ascii=False, indent=2)

    # Save stats
    stats = {
        "total_entries": total_entries,
        "total_latex_blocks": total_latex_blocks,
        "total_success": total_success,
        "total_fail": total_fail,
        "success_rate": round(total_success / max(total_latex_blocks, 1) * 100, 2),
        "entries_with_failures": len(entries_with_failures),
        "entries_fully_converted": total_entries - len(entries_with_failures),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    _safe_print(f"\n{'=' * 60}")
    _safe_print(f"DONE in {elapsed:.1f}s")
    _safe_print(f"  Entries processed: {total_entries}")
    _safe_print(f"  LaTeX blocks: {total_latex_blocks}")
    _safe_print(f"    Success: {total_success} ({stats['success_rate']}%)")
    _safe_print(f"    Failed: {total_fail}")
    _safe_print(f"  Entries fully converted: {stats['entries_fully_converted']}")
    _safe_print(f"  Entries with failures: {len(entries_with_failures)}")
    _safe_print(f"\n  Output: {OUTPUT_FILE}")
    _safe_print(f"  Failures: {FAILED_FILE}")
    _safe_print(f"  Stats: {STATS_FILE}")
    _safe_print(f"  Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
