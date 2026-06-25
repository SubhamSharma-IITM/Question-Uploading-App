"""Stage 4: Metadata V2 Mapper — pipeline adapter.

FORKED FROM: CodeBase/PCMB_Metadata_v2_Mapper.py
CHANGES:
- Adapted for web app session-based architecture (BasePipelineStage)
- Auto-detects field names: question_content vs question (from production)
- Option normalization 1./2./3./4. -> (A)/(B)/(C)/(D) from embedded text (production)
- Markdown image -> <img> tag replacement (production)
- Reads from cleaned_output/ (or mapped_output/ as fallback)
- Generates QIDs from session test_code
- PYQ support: isPrevious, previousExam, previousExamYear

Key rules (from CLAUDE.md):
- Options embedded in question text as (A), (B), (C), (D)
- quesImages/solutionImages arrays intentionally empty (inline)
- questionType: scq or integerQuestion (for JEE Mains) + mcq (for JEE Advanced)
- All content wrapped in <h3> tags
- level, isPrevious, source fields populated
- taggedBy: AIT/NIT/AITAI/NITAI/NITRAI
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.pipeline.base import BasePipelineStage


# ── Constants ───────────────────────────────────────────────────────────────
IMAGE_STYLE = 'style="height:169px; width:350px"'


# ── PILOT bold conversion (markdown ** ** -> <strong> ... </strong>) ──────
# After stage3 has stripped noise, any leftover **X** is real bold per user
# direction (2026-05-01). Apply at HTML emission time. Does NOT touch math
# blocks (those use \\textbf or <math> tags, never markdown asterisks).
#
# Pattern: ``\*\*([^*\n]+?)\*\*`` — non-greedy, no newlines, no asterisks
# inside. Avoids matching across paragraph boundaries or eating multi-line
# stretches. Real-content asterisks (e.g. inside MathML <mtext>* like</mtext>)
# are extremely rare and would not be wrapped in **...** pairs.
_BOLD_MARKDOWN = re.compile(r'\*\*([^*\n]+?)\*\*')


def _convert_remaining_bold(html: str) -> str:
    """Convert leftover **X** markdown bold to <strong>X</strong>.

    Stage3 _strip_pilot_noise removes known noise (which is most of the bold
    asterisks we've seen). Whatever remains is treated as real bold and
    converted here at HTML emission time.
    """
    if not html:
        return html
    return _BOLD_MARKDOWN.sub(r'<strong>\1</strong>', html)


# ── PILOT match-solution canonical label conversion ──────────────────────
# The match-type SOLUTION text frequently contains source-label mappings like
# `(A → p,r,t)(B → q)(C → q,s)(D → r,s,t)` that need conversion to canonical
# `(1 → C,E)(2 → B)(3 → B,D)(4 → C,D,E)`.
#
# Generalization safety:
#   - REQUIRES the arrow `→` (or LaTeX `\rightarrow`/`\to`) — strong signal
#     this is a label mapping, not real content. Plain `(A) ... (B)` without
#     arrows is left untouched.
#   - Source labels are SINGLE chars from {A,B,C,D} on left, {p,q,r,s,t} on
#     right. Bounded character class prevents matching on multi-char terms.
#   - Operates only on the text between an opening `(` and closing `)` —
#     never crosses tag boundaries.
#
# Pattern accepts inline whitespace and optional spaces after commas.
# Examples handled (positive):
#   (A → p,r,t)        -> (1 → C,E)         [for source A,p,r,t -> 1,C,E... wait re-check map]
# Source-side: A,B,C,D map to canonical 1,2,3,4 (List-I)
# Source-side: p,q,r,s,t map to canonical A,B,C,D,E (List-II)

# Match a left-hand label cluster like `A` or `(A)`, an arrow, then comma-
# separated right-hand single-char labels. The arrow can be Unicode `→`
# or LaTeX `\to` / `\rightarrow`. Whitespace tolerant.
_ARROW_TOKEN = r'(?:→|\\to\b|\\rightarrow\b)'
_MATCH_SOLUTION_BLOCK = re.compile(
    r'\(\s*([A-Da-d])\s*' + _ARROW_TOKEN + r'\s*([p-tP-T](?:\s*,\s*[p-tP-T])*)\s*\)'
)


# LaTeX `\mathrm{X}` wrappers around single-char labels — collapse to bare X
# before canonicalization. Bounded char set ([A-Da-dp-tP-T]) ensures we never
# touch `\mathrm{cm}` or `\mathrm{kg}` or other real content.
_MATHRM_LABEL_WRAPPER = re.compile(r'\\mathrm\s*\{\s*([A-Da-dp-tP-T])\s*\}')


def _canonicalize_match_solution_text(text: str) -> str:
    """Convert raw `(A→p,r)` or `(\\mathrm{A} \\rightarrow \\mathrm{p},\\mathrm{r})`
    source-label mappings to canonical `(1→C,E)` form.

    Generalization safety:
    - Step 1 strips `\\mathrm{X}` ONLY for X in {A-D,p-t} (label chars). Real
      content like `\\mathrm{cm}`, `\\mathrm{kg}`, `\\mathrm{HCl}` is multi-char
      and will not match.
    - Step 2 requires the arrow `→` or LaTeX `\\to`/`\\rightarrow` plus
      single-char labels. Plain `(A) ... (B)` prose is untouched.
    - Operates on prose; never crosses opening/closing parens with newlines.
    """
    if not text:
        return text
    has_arrow = '→' in text or '\\to' in text or '\\rightarrow' in text
    if not has_arrow:
        return text

    # Step 1: collapse `\mathrm{X}` -> `X` only for single label chars
    cleaned = _MATHRM_LABEL_WRAPPER.sub(r'\1', text)

    # Step 2: canonicalize each `(A→p,r)` block
    def _repl(m):
        left = m.group(1).upper()
        rights = re.split(r'\s*,\s*', m.group(2))
        l_can = _LIST_I_LABEL_MAP.get(left, left)
        r_can = []
        for r in rights:
            r_norm = r.lower()
            r_can.append(_LIST_II_LABEL_MAP.get(r_norm, _LIST_II_LABEL_MAP.get(r.upper(), r)))
        return f'({l_can} → {",".join(r_can)})'

    return _MATCH_SOLUTION_BLOCK.sub(_repl, cleaned)

SUBJECT_PREFIX = {
    "Physics": "PHY",
    "Chemistry": "CHEM",
    "Mathematics": "MATH",
    "Biology": "BIO",
}

PREV_EXAM_MAP = {
    "MAINS": "MAINS",
    "ADVANCED": "ADVANCED",
    "NEET": "NEET",
    "MHCET": "MHCET",
    "JEE MAINS": "MAINS",
    "JEE ADVANCED": "ADVANCED",
    "MH CET": "MHCET",
}


# ── Option normalization (verbatim from production PCMB_Metadata_v2_Mapper.py) ──
def _normalize_options_to_ABCD(html: str) -> str:
    """Convert options from 1. 2. 3. 4. format to (A) (B) (C) (D) format.

    PCMB format: "1. option text<br/>2. option text..."
    Target format: "(A) option text<br/>(B) option text..."

    IMPORTANT: Only matches option markers that appear after <br/> tags or at
    line boundaries to avoid corrupting image filenames or math expressions.

    Forked from production PCMB_Metadata_v2_Mapper.py:normalize_options_to_ABCD()
    """
    if not html:
        return ""

    # Check if we already have (A),(B),(C),(D) format (uppercase)
    if re.search(r'\(A\)', html) and re.search(r'\(B\)', html):
        # Already uppercase — just ensure <br/> before (A) if missing
        if re.search(r'[^>]\(A\)', html) and not re.search(r'(<br/>|<br>)\s*\(A\)', html):
            html = re.sub(r'(?<!<br/>)(?<!<br>)\s*(\(A\))', r'<br/>\1', html, count=1)
        for letter in ("B", "C", "D"):
            if re.search(rf'[^>]\({letter}\)', html) and not re.search(rf'(<br/>|<br>)\s*\({letter}\)', html):
                html = re.sub(rf'(?<!<br/>)(?<!<br>)\s*(\({letter}\))', rf'<br/>\1', html, count=1)
        return html

    result = html

    # ── Handle inline horizontal options (convert to <br/>(Letter) format) ──
    # Handle (a), (b), (c), (d)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s*\(\s*a\s*\)', r'<br/>(A)', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s*\(\s*b\s*\)', r'<br/>(B)', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s*\(\s*c\s*\)', r'<br/>(C)', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s*\(\s*d\s*\)', r'<br/>(D)', result)

    # Handle 1., 2., 3., 4.
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+1\.\s*', r'<br/>(A) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+2\.\s*', r'<br/>(B) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+3\.\s*', r'<br/>(C) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+4\.\s*', r'<br/>(D) ', result)

    # Handle (1), (2), (3), (4)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s*\(\s*1\s*\)', r'<br/>(A)', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s*\(\s*2\s*\)', r'<br/>(B)', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s*\(\s*3\s*\)', r'<br/>(C)', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s*\(\s*4\s*\)', r'<br/>(D)', result)

    # Handle 1), 2), 3), 4)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+1\)\s*', r'<br/>(A) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+2\)\s*', r'<br/>(B) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+3\)\s*', r'<br/>(C) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+4\)\s*', r'<br/>(D) ', result)

    # Handle A), B), C), D)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+[Aa]\)\s*', r'<br/>(A) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+[Bb]\)\s*', r'<br/>(B) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+[Cc]\)\s*', r'<br/>(C) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+[Dd]\)\s*', r'<br/>(D) ', result)

    # Handle A. B. C. D.
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+[Aa]\.\s*', r'<br/>(A) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+[Bb]\.\s*', r'<br/>(B) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+[Cc]\.\s*', r'<br/>(C) ', result)
    result = re.sub(r'(?<!<br/>)(?<!<br>)(?<!^)\s+[Dd]\.\s*', r'<br/>(D) ', result)

    # ── Standard multiline/boundary replacements ──
    # Handle (a)(b)(c)(d) lowercase parenthesized → uppercase
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*a\s*\)', r'\1(A)', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*b\s*\)', r'\1(B)', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*c\s*\)', r'\1(C)', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*d\s*\)', r'\1(D)', result, flags=re.MULTILINE)

    # Pattern: <br/>N. or start-of-string followed by N.
    result = re.sub(r'(<br/>|<br>|^)1\.\s*', r'\1(A) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)2\.\s*', r'\1(B) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)3\.\s*', r'\1(C) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)4\.\s*', r'\1(D) ', result, flags=re.MULTILINE)

    # Also handle (1) (2) (3) (4) format if present
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*1\s*\)', r'\1(A)', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*2\s*\)', r'\1(B)', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*3\s*\)', r'\1(C)', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*4\s*\)', r'\1(D)', result, flags=re.MULTILINE)

    # Handle 1) 2) 3) 4) format (number-closing-paren without opening paren)
    result = re.sub(r'(<br/>|<br>|^)\s*1\)\s*', r'\1(A) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*2\)\s*', r'\1(B) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*3\)\s*', r'\1(C) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*4\)\s*', r'\1(D) ', result, flags=re.MULTILINE)

    # Handle A) B) C) D) format (without parentheses)
    result = re.sub(r'(<br/>|<br>|^)\s*[Aa]\)\s*', r'\1(A) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*[Bb]\)\s*', r'\1(B) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*[Cc]\)\s*', r'\1(C) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*[Dd]\)\s*', r'\1(D) ', result, flags=re.MULTILINE)

    # Handle A. B. C. D. format (letter-dot)
    result = re.sub(r'(<br/>|<br>|^)\s*[Aa]\.\s*', r'\1(A) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*[Bb]\.\s*', r'\1(B) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*[Cc]\.\s*', r'\1(C) ', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*[Dd]\.\s*', r'\1(D) ', result, flags=re.MULTILINE)

    # Final pass: ensure <br/> before (A) if it appears inline after question text
    if re.search(r'[^>]\(A\)', result) and not re.search(r'(<br/>|<br>)\s*\(A\)', result):
        result = re.sub(r'(?<!<br/>)(?<!<br>)\s*(\(A\))', r'<br/>\1', result, count=1)

    return result


# ── Image replacement (from production PCMB_Metadata_v2_Mapper.py) ──────────
def _basename_from_url(url: str) -> str:
    """Extract basename from a URL or path."""
    # Remove query params
    clean = url.split("?")[0]
    return clean.rsplit("/", 1)[-1] if "/" in clean else clean


def _replace_md_images_with_img(html: str) -> Tuple[str, List[str]]:
    """Replace markdown images with HTML <img> tags.

    PILOT: Keeps the ORIGINAL CDN URL so the image renders in Acadza CKEditor
    without needing Stage 6 (which we skip for the pilot).
    Production behavior was to use `./images/{base}` and rely on Stage 6 to
    upload + rewrite URLs with Acadza S3.

    Returns (html_with_img_tags, list_of_basenames_for_tracking).
    """
    if not html:
        return "", []

    found_images = []

    def _repl(m):
        raw_url = m.group(1).strip()
        # Remove any newlines, carriage returns, spaces, or tabs inside the URL
        clean_url = "".join(raw_url.split())
        # Strip any escaped ampersands from URL (MathPix uses \&)
        clean_url = clean_url.replace("\\&", "&")
        base = _basename_from_url(clean_url)
        alt = base.rsplit('.', 1)[0] if '.' in base else base
        found_images.append(base)
        # Pilot: keep absolute CDN URL so Acadza CKEditor renders it directly
        return f'<img src="{clean_url}" alt="{alt}" {IMAGE_STYLE} />'

    pat = r'!\[[^\]]*\]\(((?:[^()]+|\([^()]*\))+)\)'
    result = re.sub(pat, _repl, html, flags=re.DOTALL)
    return result, found_images


# ── HTML wrapping ───────────────────────────────────────────────────────────
def _wrap_h3(text: str) -> str:
    """Wrap text in <h3> tags if not already wrapped."""
    if not text:
        return "<h3></h3>"
    text = text.strip()
    if not text.startswith("<h3>"):
        text = f"<h3>{text}</h3>"
    return text


# ── Answer normalization ────────────────────────────────────────────────────
def _normalize_answer(answer: str, question_type: str) -> str:
    """Normalize answer value based on question type."""
    if not answer:
        return ""
    answer = str(answer).strip()

    if question_type == "integerQuestion":
        # Numeric answer — keep as-is
        return answer

    # SCQ/MCQ — normalize to letter(s)
    answer = answer.upper()
    if answer in ("A", "B", "C", "D"):
        return answer

    # Map numeric to letter
    num_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
    if answer in num_map:
        return num_map[answer]

    # MCQ: multi-letter like "AC" or "ABD"
    if all(c in "ABCD" for c in answer):
        return answer

    return answer


# ── Question type detection ─────────────────────────────────────────────────
def _detect_question_type(entry: Dict) -> str:
    """Detect question type from entry data."""
    # If explicitly set by LLM mapper
    qt = entry.get("questionType", "")
    if qt in ("scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"):
        return qt

    # Legacy format detection
    options = entry.get("options", {})
    if len(options) >= 4:
        return "scq"
    if len(options) == 0:
        return "integerQuestion"
    return "scq"


# ── PILOT: Source -> canonical label mapping for match-type ──────────────
_LIST_I_LABEL_MAP = {
    # old JEE Adv (FIITJEE): left side = A,B,C,D -> canonical 1,2,3,4
    "A": "1", "B": "2", "C": "3", "D": "4",
    # already canonical (newer JEE Adv): 1,2,3,4 stays
    "1": "1", "2": "2", "3": "3", "4": "4",
}
_LIST_II_LABEL_MAP = {
    # old JEE Adv: right side = p,q,r,s,t -> canonical A,B,C,D,E
    "p": "A", "q": "B", "r": "C", "s": "D", "t": "E",
    "P": "A", "Q": "B", "R": "C", "S": "D", "T": "E",
    # already canonical
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
}


def _parse_match_plain_text(content: str) -> Tuple[str, list, list]:
    """Parse match columns when they are formatted as plain text instead of LaTeX tabular.
    Handles:
      (A) at t = 0 (P) Total energy per unit length at x = 0 is 2μA²ω².
      or separate lists:
      (A) ...
      (P) ...
    """
    if not content:
        return "", [], []

    lines = content.splitlines()
    intro_lines = []
    list_i = []
    list_ii = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Look for inline columns: (A) some text (P) some text
        match_inline = re.match(
            r"^\s*[\(\$]*\s*([A-Ea-e1-4])\s*[\)\.\$]\s+(.*?)\s+[\(\$]*\s*([P-Tp-t])\s*[\)\.\$]\s+(.*)$",
            stripped
        )
        if match_inline:
            lbl_i, val_i, lbl_ii, val_ii = match_inline.groups()
            val_i = val_i.strip()
            val_ii = val_ii.strip()

            lbl_i_can = _LIST_I_LABEL_MAP.get(lbl_i.upper(), lbl_i)
            lbl_ii_can = _LIST_II_LABEL_MAP.get(lbl_ii.lower(), _LIST_II_LABEL_MAP.get(lbl_ii.upper(), lbl_ii))

            val_i_html, _ = _replace_md_images_with_img(val_i)
            val_ii_html, _ = _replace_md_images_with_img(val_ii)

            list_i.append({"label": lbl_i_can, "content": val_i_html})
            list_ii.append({"label": lbl_ii_can, "content": val_ii_html})
            continue

        # Check if it's a single item:
        # e.g., (A) at t = 0  or  (P) Total energy...
        match_single = re.match(r"^\s*[\(\$]*\s*([A-Ea-e1-4|P-Tp-t])\s*[\)\.\$]\s+(.*)$", stripped)
        if match_single:
            lbl, val = match_single.groups()
            lbl_upper = lbl.upper()
            val = val.strip()

            # Filter out numbered question headers/introductions (e.g. "2. Match the statements...")
            is_intro_header = False
            if lbl_upper in ("1", "2", "3", "4") and not list_i and not list_ii:
                if "match" in val.lower() or "following" in val.lower() or "statement" in val.lower() or not (stripped.startswith("(") or stripped.startswith("$(")):
                    is_intro_header = True

            if is_intro_header:
                if not list_i and not list_ii:
                    intro_lines.append(line)
                continue

            val_html, _ = _replace_md_images_with_img(val)

            if lbl_upper in ("A", "B", "C", "D", "E", "1", "2", "3", "4"):
                lbl_can = _LIST_I_LABEL_MAP.get(lbl.upper(), lbl)
                list_i.append({"label": lbl_can, "content": val_html})
            elif lbl_upper in ("P", "Q", "R", "S", "T"):
                lbl_can = _LIST_II_LABEL_MAP.get(lbl.lower(), _LIST_II_LABEL_MAP.get(lbl.upper(), lbl))
                list_ii.append({"label": lbl_can, "content": val_html})
            continue

        # Skip column/list header lines
        if "column" in stripped.lower() or "list-i" in stripped.lower() or "list-ii" in stripped.lower():
            continue

        # If it doesn't match either, and we haven't started parsing items, it's part of the intro
        if not list_i and not list_ii:
            intro_lines.append(line)

    intro_raw = "\n".join(intro_lines).strip()
    intro_html, _ = _replace_md_images_with_img(intro_raw)

    return intro_html, list_i, list_ii


# ── PILOT: Parse Stage-2 emitted LaTeX tabular -> list_i / list_ii ───────
def _parse_match_latex_tabular(content: str) -> Tuple[str, list, list]:
    """Parse a raw match-type question_content (containing a LaTeX `tabular` block)
    into (intro_html, list_i, list_ii) for the HTML table builder.

    Source format (FIITJEE typical):
        1. Match the following:
        \\begin{tabular}{|l|l|l|l|}
        \\hline \\multicolumn{2}{|c|}{Column - I} & \\multicolumn{2}{|r|}{Column - II} \\\\
        \\hline (A) & <left content> & (p) & <right content> \\\\
        \\hline (B) & <left content> & (q) & <right content> \\\\
        \\hline (C) & <left content> & (r) & <right content> \\\\
        \\hline (D) & <left content> & (s) & <right content> \\\\
        \\hline & & (t) & <right content> \\\\
        \\hline
        \\end{tabular}

    Returns (intro_html, list_i, list_ii) where:
      list_i = [{"label": "1", "content": "..."}, ...]   (canonical 1,2,3,4)
      list_ii = [{"label": "A", "content": "..."}, ...]  (canonical A,B,C,D[,E])

    Empty list_i cells (e.g. the (t) row with no left side) are skipped — only
    the List-II entry from that row is appended.

    If no `\\begin{tabular}` is found, returns (content, [], []) — caller falls
    back to direct list_i/list_ii fields if available.
    """
    import re

    if not content or "\\begin{tabular}" not in content:
        intro_html, list_i, list_ii = _parse_match_plain_text(content)
        if list_i and list_ii:
            return intro_html, list_i, list_ii
        return content or "", [], []

    # Split intro from tabular block
    tab_start = content.find("\\begin{tabular}")
    intro_html = content[:tab_start].strip()
    tab_end = content.find("\\end{tabular}", tab_start)
    if tab_end < 0:
        return intro_html, [], []
    body = content[tab_start:tab_end + len("\\end{tabular}")]

    # Strip header line `\begin{tabular}{|l|l|l|l|}` and capture column spec
    # so we can detect 5-column tabulars (with empty separator middle column).
    m = re.match(r"\\begin\{tabular\}\{([^}]*)\}", body)
    if not m:
        return intro_html, [], []
    col_spec = m.group(1)
    # Count actual column markers (l/c/r/p), ignoring vertical bars.
    n_cols = sum(1 for ch in col_spec if ch in "lcrp")
    body_inner = body[m.end():body.rfind("\\end{tabular}")].strip()

    # Split into rows by `\\` (row separator). Some rows may be on multiple lines.
    # Normalize: collapse newlines to spaces inside the tabular body.
    body_inner = body_inner.replace("\n", " ")

    # Row separator can be `\\` or `\\\n` — use regex to be robust
    raw_rows = re.split(r"\\\\(?:\s*\\hline)?", body_inner)

    list_i: list = []
    list_ii: list = []
    # Track body-row index for position-based label inference when label cell
    # is empty but content cell has text (M02 row C: `(C) & beta & & content`).
    body_row_idx = -1

    def _label_only(cell: str) -> str:
        """Pure label extraction from a label-only cell. Returns 'A' from '(A)' or 'A'.

        Lenient — also accepts:
          - `(p` (missing closing paren — M10 OCR artifact)
          - `p)` (missing opening paren)
          - `(p `, ` (p)`, etc. (whitespace variations)
          - `$(p)$` (LaTeX inline math wrapper)
        """
        cell = cell.strip()
        # Canonical: (X)
        m1 = re.match(r"^\s*\(\s*([A-Za-z0-9])\s*\)\s*$", cell)
        if m1:
            return m1.group(1)
        # Bare letter/digit
        m2 = re.match(r"^\s*([A-Za-z0-9])\s*$", cell)
        if m2:
            return m2.group(1)
        # LaTeX-math-wrapped: $(X)$
        m3 = re.match(r"^\s*\$\s*\(\s*([A-Za-z0-9])\s*\)\s*\$\s*$", cell)
        if m3:
            return m3.group(1)
        # Lenient: (X without closing paren — common OCR artifact (M10)
        m4 = re.match(r"^\s*\(\s*([A-Za-z0-9])\s*$", cell)
        if m4:
            return m4.group(1)
        # Lenient: X) without opening paren
        m5 = re.match(r"^\s*([A-Za-z0-9])\s*\)\s*$", cell)
        if m5:
            return m5.group(1)
        return ""

    # Source-label sequences for position-based recovery when label cell is empty
    # but content cell is non-empty (M02 row C: `(C) & beta-decay & & content...`).
    _SRC_LIST_I_SEQ = ["A", "B", "C", "D", "E"]
    _SRC_LIST_II_SEQ = ["p", "q", "r", "s", "t"]

    for raw_row in raw_rows:
        row = raw_row.replace("\\hline", "").strip()
        if not row:
            continue
        # Skip header rows (multicolumn)
        if "\\multicolumn" in row:
            continue
        # Split by `&` not preceded by backslash. Body rows are typically:
        #   4-col tabular: cells = [List-I label, List-I content, List-II label, List-II content]
        #   5-col tabular: cells = [List-I label, List-I content, <empty separator>, List-II label, List-II content]
        # M03 (CHEM) source uses 5-column form with empty middle separator —
        # without this branch the parser put the empty cell into the List-II
        # label slot and dropped every List-II row.
        cells = [c.strip() for c in re.split(r"(?<!\\)&", row)]
        if len(cells) < 4:
            continue
        if n_cols >= 5 and len(cells) >= 5:
            c0, c1, c2, c3 = cells[0], cells[1], cells[3], cells[4]
        else:
            c0, c1, c2, c3 = cells[0], cells[1], cells[2], cells[3]

        body_row_idx += 1

        # List-I (label + content cells). Skip empty rows like the 5th `& & (t) & ...`
        li_label_raw = _label_only(c0)
        if li_label_raw and c1.strip():
            li_label_can = _LIST_I_LABEL_MAP.get(li_label_raw, li_label_raw)
            li_content_html, _ = _replace_md_images_with_img(c1.strip())
            list_i.append({"label": li_label_can, "content": li_content_html})

        # List-II (label + content cells). Independent of List-I — supports the
        # 5th-row "(t)" case where List-I is empty.
        lii_label_raw = _label_only(c2)
        # POSITION-BASED RECOVERY: M02 has rows where c2 is empty (no source
        # label emitted) but c3 has substantive content (>5 chars). Source
        # likely dropped the (r) label between (q) and (s). Infer the source
        # label from body-row index, but ONLY when c3 looks like real content
        # (>5 chars, not whitespace/tabs/&nbsp;). Bounded to avoid corrupting
        # legitimate empty-content rows.
        if (not lii_label_raw
                and len(c3.strip()) > 5
                and 0 <= body_row_idx < len(_SRC_LIST_II_SEQ)):
            lii_label_raw = _SRC_LIST_II_SEQ[body_row_idx]
        if lii_label_raw and c3.strip():
            lii_label_can = _LIST_II_LABEL_MAP.get(
                lii_label_raw,
                _LIST_II_LABEL_MAP.get(lii_label_raw.upper(), lii_label_raw),
            )
            lii_content_html, _ = _replace_md_images_with_img(c3.strip())
            list_ii.append({"label": lii_label_can, "content": lii_content_html})

    return intro_html, list_i, list_ii


# ── PILOT: Build match-type table HTML ───────────────────────────────────
# ── PILOT: fill label-sequence gaps in list_i/list_ii ────────────────────
def _fill_label_gaps(items: list, label_seq: list) -> list:
    """Fill missing labels in `items` so the visible label sequence has no holes.

    Source PDFs sometimes emit only labels with non-empty content, leaving
    gaps like List-II = [A,B,D,E] (missing C — bug surfaced in M02). This
    function inserts placeholder rows ({"label": "C", "content": "&nbsp;"})
    so the rendered table shows (A)(B)(C)(D)(E) sequentially with the empty
    cell visually flagging the source-data gap.

    Only fills gaps WITHIN the existing range — does not extend beyond the
    last present label. Items without a label are kept in place.
    """
    if not items:
        return items
    present = {x.get("label") for x in items if x.get("label")}
    if not present:
        return items
    # Find the highest-rank present label
    max_idx = -1
    for idx, lbl in enumerate(label_seq):
        if lbl in present:
            max_idx = idx
    if max_idx < 0:
        return items
    # Build output by walking label_seq up to max_idx, falling back to items
    by_label = {}
    extras = []
    for x in items:
        lbl = x.get("label")
        if lbl in by_label:
            extras.append(x)
        elif lbl:
            by_label[lbl] = x
        else:
            extras.append(x)
    out = []
    for i in range(max_idx + 1):
        lbl = label_seq[i]
        if lbl in by_label:
            out.append(by_label[lbl])
        else:
            out.append({"label": lbl, "content": "&nbsp;"})
    return out + extras


def _build_match_table_html(intro_html: str, list_i: list, list_ii: list) -> str:
    """Build canonical match-type question HTML: optional intro + table.

    Schema (matches real DB sample):
    - Table outer: border="1" + inline CSS shorthand for CKEditor safety
    - Header row: two <th colspan="2"> with background-color:#e6e6e6
    - Body rows: max(|list_i|, |list_ii|) rows; missing cells use &nbsp;
    - Labels wrapped in <strong>(...)</strong>
    - Label cell uses ``white-space: nowrap`` so "(1)" / "(A)" never wraps onto
      two lines (M05/M06 parens-on-newline bug).
    - Every td/th uses inline `border: 1px solid #555555;` shorthand
    - PILOT: list_i/list_ii are gap-filled with sequential labels so source-
      data dropouts (e.g. M02 missing 'C') still render as (A)(B)(C)(D)(E)
      with a visually empty cell flagging the gap.

    Args:
      intro_html: optional question stem before the table (HTML or plain).
      list_i: [{"label": "1", "content": "<HTML>"}, ...] — left column.
      list_ii: [{"label": "A", "content": "<HTML>"}, ...] — right column (4 or 5 items).
    """
    parts = []
    if intro_html:
        # Strip leading question-number prefix like "1.", "2.", "1)" — these come
        # from raw mmd ("1. Match the following:") and shouldn't appear in the stem.
        cleaned_intro = re.sub(r"^\s*\d+[\.\)]\s+", "", intro_html, count=1)
        cleaned_intro = _convert_remaining_bold(cleaned_intro)
        intro = cleaned_intro if cleaned_intro.lstrip().startswith("<") else f"<p>{cleaned_intro}</p>"
        parts.append(intro)

    # PILOT: fill label-sequence gaps so visible sequence is contiguous
    list_i = _fill_label_gaps(list_i, ["1", "2", "3", "4"])
    list_ii = _fill_label_gaps(list_ii, ["A", "B", "C", "D", "E"])

    # Table open — both border="1" AND inline CSS for CKEditor reliability
    parts.append(
        '<table border="1" cellpadding="10" cellspacing="0" '
        'style="border-collapse: collapse; border: 1px solid #555555; width: 100%;">'
    )
    parts.append("<tbody>")

    # Header row
    parts.append("<tr>")
    parts.append(
        '<th colspan="2" style="background-color:#e6e6e6; text-align:center; '
        'border: 1px solid #555555;">List-I</th>'
    )
    parts.append(
        '<th colspan="2" style="background-color:#e6e6e6; text-align:center; '
        'border: 1px solid #555555;">List-II</th>'
    )
    parts.append("</tr>")

    # Body rows
    max_rows = max(len(list_i), len(list_ii))
    for i in range(max_rows):
        parts.append("<tr>")
        # List-I cells (label + content). nowrap on label cell prevents the
        # "(1" / ")" split-line bug (M05/M06).
        if i < len(list_i):
            li = list_i[i]
            parts.append(
                '<td style="border: 1px solid #555555; text-align:center; white-space: nowrap;">'
                f'<strong>({li.get("label", "")})</strong></td>'
            )
            parts.append(
                f'<td style="border: 1px solid #555555;">{_convert_remaining_bold(li.get("content", ""))}</td>'
            )
        else:
            parts.append('<td style="border: 1px solid #555555;">&nbsp;</td>')
            parts.append('<td style="border: 1px solid #555555;">&nbsp;</td>')

        # List-II cells (label + content)
        if i < len(list_ii):
            lii = list_ii[i]
            parts.append(
                '<td style="border: 1px solid #555555; text-align:center; white-space: nowrap;">'
                f'<strong>({lii.get("label", "")})</strong></td>'
            )
            parts.append(
                '<td style="border: 1px solid #555555; text-align:center;">'
                f'{_convert_remaining_bold(lii.get("content", ""))}</td>'
            )
        else:
            parts.append('<td style="border: 1px solid #555555;">&nbsp;</td>')
            parts.append('<td style="border: 1px solid #555555;">&nbsp;</td>')

        parts.append("</tr>")

    parts.append("</tbody>")
    parts.append("</table>")
    return "".join(parts)


# ── Build question HTML ────────────────────────────────────────────────────
def _build_question_html(entry: Dict) -> str:
    """Build question HTML with options embedded as (A)/(B)/(C)/(D).

    Handles both formats:
    - New format (question_content): options embedded in text, need normalization
    - Legacy format (question + options dict): options need to be combined
    """
    q_key = "question_content" if "question_content" in entry else "question"
    question_text = entry.get(q_key, "")

    if not question_text:
        return "<h3></h3>"

    # If legacy format with separate options dict, embed them
    options = entry.get("options", {})
    if options and isinstance(options, dict):
        parts = [question_text]
        for letter in ("A", "B", "C", "D"):
            if letter in options:
                parts.append(f"({letter}) {options[letter]}")
        question_text = "<br/>".join(parts)
    else:
        # New format: options already in question_content, normalize them
        question_text = _normalize_options_to_ABCD(question_text)

    # Replace markdown images with <img> tags
    question_text, _ = _replace_md_images_with_img(question_text)

    # PILOT: convert any leftover **X** markdown bold to <strong>X</strong>
    question_text = _convert_remaining_bold(question_text)

    return _wrap_h3(question_text)


# ── Build solution HTML ────────────────────────────────────────────────────
def _build_solution_html(entry: Dict) -> str:
    """Build solution HTML."""
    s_key = "solution_content" if "solution_content" in entry else "solution"
    solution_text = entry.get(s_key, "")

    # Prepend hint if present (legacy format)
    hint = entry.get("hint", "")
    if hint:
        solution_text = f"<b>Hint:</b> {hint}<br/>{solution_text}" if solution_text else f"<b>Hint:</b> {hint}"

    if not solution_text:
        return "<h3></h3>"

    # Replace markdown images
    solution_text, _ = _replace_md_images_with_img(solution_text)

    # PILOT: convert any leftover **X** markdown bold to <strong>X</strong>
    solution_text = _convert_remaining_bold(solution_text)

    return _wrap_h3(solution_text)


# ── Build metadata V2 entry ────────────────────────────────────────────────
def _build_metadata_v2(
    entry: Dict,
    source_name: str,
    test_code: str,
    tagged_by: str = "AITAI",
    is_pyq: bool = False,
    prev_exam: Optional[str] = None,
    pyq_year: Optional[int] = None,
) -> Dict:
    """Convert a single question entry into Metadata V2 schema.

    Forked from production PCMB_Metadata_v2_Mapper.py — adapted for both
    new format (question_content/solution_content) and legacy format
    (question/options/solution).
    """
    subject = entry.get("subject", "Physics")
    # Normalize sub-disciplines to parent subject
    _subj_map = {"Botany": "Biology", "Zoology": "Biology",
                 "Organic Chemistry": "Chemistry", "Inorganic Chemistry": "Chemistry",
                 "Physical Chemistry": "Chemistry"}
    subject = _subj_map.get(subject, subject)
    qtype = _detect_question_type(entry)
    answer = _normalize_answer(
        entry.get("answer_key", entry.get("answer", "")),
        qtype,
    )

    # Generate QID
    prefix = SUBJECT_PREFIX.get(subject, "UNK")
    test_short = test_code.replace("-", "")
    qnum = entry.get("question_number", 0)
    qid = entry.get("qid", "")
    if not qid or qid.startswith("TEMP_"):
        qid = f"{prefix}_{test_short}_Q{qnum}"

    # Build the active question type container
    if qtype == "passageQuestion":
        # ── PILOT FIX: Canonical passage schema matching real DB sample ──
        # Bugs fixed vs production:
        #   1. Add passageImages, passageSolutionImages at top level
        #   2. Always wrap inner sub-questions under "scq" (NOT their own qtype)
        #   3. Use field name "ques" (NOT "question")
        #   4. Add "passage" back-reference inside inner SCQ (uses parent meta qid
        #      as placeholder — DB upload step replaces with parent OID)
        #   5. Top-level solution = "" (NOT placeholder "<h3></h3>")
        #   6. Inner SCQ has only 4 fields: ques, solution, answer, passage
        passage_raw = entry.get("passage_content", "")
        if passage_raw:
            passage_raw, _ = _replace_md_images_with_img(passage_raw)
            # PILOT: convert any leftover **X** markdown bold to <strong>
            passage_raw = _convert_remaining_bold(passage_raw)
        passage_html = _wrap_h3(passage_raw)

        sub_questions_out = []
        for sq in entry.get("sub_questions", []):
            sq_type = sq.get("questionType", "integerQuestion")
            sq_answer = _normalize_answer(
                sq.get("answer_key", sq.get("answer", "")),
                sq_type,
            )
            sq_question_html = _build_question_html(sq)
            sq_solution_html = _build_solution_html(sq)
            # Canonical inner: always "scq" key, only 4 fields (no quesImages/solutionImages)
            sub_questions_out.append({
                "scq": {
                    "ques": sq_question_html,
                    "solution": sq_solution_html,
                    "answer": sq_answer,
                    "passage": qid,  # back-ref; replaced with parent OID at DB upload
                }
            })

        active_container = {
            "passage": passage_html,
            "solution": "",  # top-level empty — per-question solutions are inline
            "questions": sub_questions_out,
            "passageImages": [],
            "passageSolutionImages": [],
        }
        # No single answer for passage — answers are per sub-question
        answer = ""
    elif qtype == "matchQuestion":
        # ── PILOT: matchQuestion canonical schema ──
        # Schema (from real DB sample 68e747fa6e45ba37dcc0f889):
        #   {rows, quesImages, solutionImages, question, solution}
        # rows: list of lists for one-to-many support, e.g. [["B"],["D"],["A"],["E"]]
        # rows[i] = answers (List-II labels) for List-I item i+1
        # quesImages/solutionImages: null (NOT empty list) per real sample
        list_i = entry.get("list_i", [])
        list_ii = entry.get("list_ii", [])
        intro = entry.get("question_intro", "")

        # Stage 2 emits raw `question_content` (LaTeX tabular). Parse it into
        # list_i/list_ii if not already pre-structured.
        if (not list_i or not list_ii):
            raw_content = entry.get("question_content", "")
            parsed_intro, parsed_li, parsed_lii = _parse_match_latex_tabular(raw_content)
            if not intro:
                intro = parsed_intro
            if not list_i:
                list_i = parsed_li
            if not list_ii:
                list_ii = parsed_lii

        answer_mapping = entry.get("answer_mapping", entry.get("rows", []))

        question_html = _build_match_table_html(intro, list_i, list_ii)
        # PILOT: canonicalize raw `(A → p,r)` mappings in solution text BEFORE
        # the LaTeX → MathML conversion (stage 7). Operates on the raw
        # solution_content; stage7 will then encode the canonical form into
        # MathML. Safe: only fires on patterns containing arrows + label chars.
        s_key = "solution_content" if "solution_content" in entry else "solution"
        if s_key in entry and entry[s_key]:
            entry = dict(entry)
            entry[s_key] = _canonicalize_match_solution_text(entry[s_key])
        solution_html = _build_solution_html(entry)

        active_container = {
            "rows": answer_mapping,
            "quesImages": None,        # null in DB sample, not []
            "solutionImages": None,    # null in DB sample, not []
            "question": question_html,
            "solution": solution_html,
        }
        # rows array IS the answer for match — single-answer field empty
        answer = ""
    else:
        question_html = _build_question_html(entry)
        solution_html = _build_solution_html(entry)
        active_container = {
            "question": question_html,
            "solution": solution_html,
            "answer": answer,
            "quesImages": [],
            "solutionImages": [],
        }

    # Determine isNEET from taggedBy
    is_neet = tagged_by in ("NIT", "NITAI", "NITRAI")

    # Resolve previous exam
    resolved_prev_exam = ""
    prev_exam_year = 0
    is_mhcet = False
    if is_pyq and prev_exam:
        resolved_prev_exam = PREV_EXAM_MAP.get(prev_exam.upper(), prev_exam.upper())
        prev_exam_year = pyq_year or 0
        is_mhcet = prev_exam.upper() in ("MHCET", "MH CET")

    metadata = {
        "qid": qid,
        "questionType": qtype,
        "subject": subject,
        "chapter": "TBD",
        "concept": "TBD",
        "subConcept": "TBD",
        qtype: active_container,
        # Empty type slots
        **{t: {} for t in ("scq", "mcq", "integerQuestion", "matchQuestion",
                           "subjective", "passageQuestion") if t != qtype},
        "class": 12,
        "level": entry.get("level", "MEDIUM").upper(),
        "addedByAi": True,
        "taggedBy": tagged_by,
        "tagSubConcept": [],
        "error": 2,
        "isSolvedExample": False,
        "isNCERT": False,
        "isBoard": False,
        "isNEET": is_neet,
        "isMHCET": is_mhcet,
        "isBuffer": True,
        "isPrevious": is_pyq,
        "previousExam": resolved_prev_exam,
        "previousExamYear": prev_exam_year,
        "source": source_name,
    }

    return metadata


# ── Pipeline stage class ────────────────────────────────────────────────────
class MetadataMapperStage(BasePipelineStage):

    @property
    def stage_name(self) -> str:
        return "stage4_metadata"

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        # Try cleaned_output first, then mapped_output
        cleaned_dir = self.session_dir / "cleaned_output"
        mapped_dir = self.session_dir / "mapped_output"

        input_dir = cleaned_dir if any(cleaned_dir.glob("*.json")) else mapped_dir
        json_files = list(input_dir.glob("*.json"))
        count = 0
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                count += len(json.load(f))
        return {
            "files": len(json_files),
            "questions": count,
            "source_dir": input_dir.name,
            "message": f"Would create {count} metadata_v2 entries from {input_dir.name}/",
        }

    def _get_default_test_code(self) -> str:
        """Read test_code from session_config (set by Stage 2)."""
        config_path = self.session_dir / "session_config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                return config.get("test_code", "TEST")
            except Exception:
                pass
        return "TEST"

    def _find_input_files(self) -> List[Path]:
        """Find input JSON files — prefer cleaned_output, fallback to mapped_output."""
        cleaned_dir = self.session_dir / "cleaned_output"
        mapped_dir = self.session_dir / "mapped_output"

        # Prefer cleaned files
        cleaned_files = sorted(cleaned_dir.glob("*_cleaned.json")) if cleaned_dir.exists() else []
        if cleaned_files:
            self.log("info", f"Reading from cleaned_output/ ({len(cleaned_files)} files)")
            return cleaned_files

        # Fallback to mapped files
        mapped_files = sorted(mapped_dir.glob("*_mapped.json")) if mapped_dir.exists() else []
        if mapped_files:
            self.log("info", f"Reading from mapped_output/ ({len(mapped_files)} files, no cleaning applied)")
            return mapped_files

        return []

    def execute(self, **kwargs) -> Dict[str, Any]:
        source_name = kwargs.get("source", "")
        test_code = kwargs.get("test_code", "") or self._get_default_test_code()
        tagged_by = kwargs.get("tagged_by", "AITAI")
        is_pyq = kwargs.get("is_pyq", False)
        prev_exam = kwargs.get("prev_exam", None)
        pyq_year_from = kwargs.get("pyq_year_from", None)
        pyq_year_to = kwargs.get("pyq_year_to", None)

        out_dir = self.session_dir / "metadata_v2"
        out_dir.mkdir(parents=True, exist_ok=True)

        input_files = self._find_input_files()
        if not input_files:
            self.log("error", "No cleaned or mapped JSON files found")
            self.save_checkpoint("failed", {"error": "No input files"})
            return {"error": "No input files"}

        self.log("info", f"Tagged by: {tagged_by} | Test code: {test_code}")
        if is_pyq:
            self.log("info", f"PYQ mode: {prev_exam} ({pyq_year_from}-{pyq_year_to})")

        total = 0
        all_metadata = []

        for jf in input_files:
            self.log("info", f"Converting {jf.name} to metadata_v2...")

            with open(jf, "r", encoding="utf-8") as f:
                entries = json.load(f)

            metadata_entries = []
            for entry in entries:
                # For PYQ, use entry-level year if available, else start year
                pyq_year = None
                if is_pyq:
                    pyq_year = entry.get("pyq_year")
                    if not pyq_year and pyq_year_from:
                        pyq_year = pyq_year_from

                meta = _build_metadata_v2(
                    entry, source_name, test_code,
                    tagged_by=tagged_by,
                    is_pyq=is_pyq,
                    prev_exam=prev_exam,
                    pyq_year=pyq_year,
                )
                metadata_entries.append(meta)
                total += 1

                # Log entry summary
                qid = meta["qid"]
                qtype = meta["questionType"]
                container = meta.get(qtype, {})

                if qtype == "passageQuestion":
                    n_subs = len(container.get("questions", []))
                    has_p = bool(container.get("passage", "").replace("<h3></h3>", ""))
                    status = [f"P({'Y' if has_p else 'N'})", f"{n_subs}sub"]
                    self.log("info", f"  {qid}: {'+'.join(status)} | {qtype}")
                else:
                    has_q = bool(container.get("question", "").replace("<h3></h3>", ""))
                    has_a = bool(container.get("answer", ""))
                    has_s = bool(container.get("solution", "").replace("<h3></h3>", ""))
                    status = []
                    if has_q:
                        status.append("Q")
                    if has_a:
                        status.append("A")
                    if has_s:
                        status.append("S")
                    self.log("info",
                        f"  {qid}: {'+'.join(status)} | {qtype} | "
                        f"ans={container.get('answer', '?')}"
                    )

            # Save metadata_v2 file
            # Derive output name from input filename
            stem = jf.stem
            if stem.endswith("_cleaned"):
                out_name = stem.replace("_cleaned", "_metadata_v2") + ".json"
            elif stem.endswith("_mapped"):
                out_name = stem.replace("_mapped", "_metadata_v2") + ".json"
            else:
                out_name = f"{stem}_metadata_v2.json"

            out_file = out_dir / out_name
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(metadata_entries, f, ensure_ascii=False, indent=2)

            self.log("success", f"Saved {out_file.name} ({len(metadata_entries)} entries)")
            all_metadata.extend(metadata_entries)

        # Update question tracking
        from app.db import upsert_question

        def _meta_has_answer(m):
            qtype = m["questionType"]
            if qtype == "passageQuestion":
                subs = m.get(qtype, {}).get("questions", [])
                return any(sq.get(list(sq.keys())[0], {}).get("answer", "") for sq in subs if sq)
            return bool(m.get(qtype, {}).get("answer", ""))

        def _meta_has_solution(m):
            qtype = m["questionType"]
            if qtype == "passageQuestion":
                subs = m.get(qtype, {}).get("questions", [])
                return any(
                    sq.get(list(sq.keys())[0], {}).get("solution", "").replace("<h3></h3>", "")
                    for sq in subs if sq
                )
            return bool(m.get(qtype, {}).get("solution", "").replace("<h3></h3>", ""))

        def _meta_is_triplet(m):
            qtype = m["questionType"]
            if qtype == "passageQuestion":
                pq = m.get(qtype, {})
                has_p = bool(pq.get("passage", "").replace("<h3></h3>", ""))
                subs = pq.get("questions", [])
                if not has_p or not subs:
                    return False
                for sq in subs:
                    if not sq:
                        return False
                    sq_type = list(sq.keys())[0]
                    sq_data = sq[sq_type]
                    if not (sq_data.get("question", "").replace("<h3></h3>", "")
                            and sq_data.get("answer", "")):
                        return False
                return True
            qdata = m.get(qtype, {})
            return (bool(qdata.get("question", "").replace("<h3></h3>", ""))
                    and bool(qdata.get("answer", ""))
                    and bool(qdata.get("solution", "").replace("<h3></h3>", "")))

        for meta in all_metadata:
            upsert_question(
                self.session_id, meta["qid"],
                subject=meta["subject"],
                question_type=meta["questionType"],
                current_stage="stage4_metadata",
                has_answer=_meta_has_answer(meta),
                has_solution=_meta_has_solution(meta),
            )

        triplets = sum(1 for m in all_metadata if _meta_is_triplet(m))

        # Quarantine non-triplets (missing Q, A, or S)
        from app.db import quarantine_question
        non_triplet_qids = []
        for m in all_metadata:
            if not _meta_is_triplet(m):
                qtype = m["questionType"]
                if qtype == "passageQuestion":
                    missing = ["sub-question completeness"]
                else:
                    qdata = m.get(qtype, {})
                    missing = []
                    if not qdata.get("question", "").replace("<h3></h3>", ""): missing.append("question")
                    if not qdata.get("answer", ""): missing.append("answer")
                    if not qdata.get("solution", "").replace("<h3></h3>", ""): missing.append("solution")
                non_triplet_qids.append(m["qid"])
                quarantine_question(
                    self.session_id, m["qid"], "stage4_metadata",
                    reason="non_triplet",
                    data={"missing": missing},
                )

        if non_triplet_qids:
            self.log("warning", f"Quarantined {len(non_triplet_qids)} non-triplet questions (missing Q/A/S)")

        self.save_checkpoint("completed", {
            "total_entries": total,
            "triplets": triplets,
            "non_triplets_quarantined": len(non_triplet_qids),
            "tagged_by": tagged_by,
            "is_pyq": is_pyq,
            "output_dir": str(out_dir),
        })

        self.log("success", f"Metadata V2: {total} entries, {triplets} triplets, {len(non_triplet_qids)} non-triplets quarantined, tagged={tagged_by}")
        self.progress(100, "Metadata conversion complete")

        return {
            "total": total,
            "triplets": triplets,
            "non_triplets_quarantined": len(non_triplet_qids),
            "tagged_by": tagged_by,
        }
