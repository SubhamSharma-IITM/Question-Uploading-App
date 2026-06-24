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


# ── Markdown bold conversion (** ** -> <strong>) ──────────────────────────
# Stage 3 strips known noise, then any leftover **X** is real bold and gets
# converted at HTML-emission time. Pattern is non-greedy, no newlines, no
# asterisks inside — avoids matching across paragraph boundaries.
_MARKDOWN_BOLD = re.compile(r'\*\*([^*\n]+?)\*\*')


def _convert_markdown_bold(html: str) -> str:
    """Convert leftover **X** markdown bold to <strong>X</strong>.

    Stage 3 strips known noise via _strip_format_noise. Whatever ** ** pairs
    remain are real bold — convert at emission time.
    """
    if not html:
        return html
    return _MARKDOWN_BOLD.sub(r'<strong>\1</strong>', html)


# ── Match-type label canonicalization ─────────────────────────────────────
# Source FIITJEE format uses `(A → p,r,t)(B → q)...` source labels in the
# solution text. Target DB format requires canonical `(1 → C,E)(2 → B)...`.
# REQUIRES the arrow `→` (or LaTeX `\rightarrow`/`\to`) — strong signal this
# is a label mapping, not real prose. Source labels bounded to {A-D} on left,
# {p-t} on right. Plain `(A) ... (B)` prose without arrows is untouched.
_ARROW_TOKEN = r'(?:→|\\to\b|\\rightarrow\b)'
_MATCH_LABEL_MAPPING_PATTERN = re.compile(
    r'\(\s*([A-Da-d])\s*' + _ARROW_TOKEN + r'\s*([p-tP-T](?:\s*,\s*[p-tP-T])*)\s*\)'
)

# LaTeX `\mathrm{X}` wrappers around single-char labels — collapse to bare X
# before canonicalization. Bounded char set ([A-Da-dp-tP-T]) means real
# content like `\mathrm{cm}`, `\mathrm{kg}`, `\mathrm{HCl}` (multi-char) is
# never touched.
_MATHRM_LABEL_WRAPPER = re.compile(r'\\mathrm\s*\{\s*([A-Da-dp-tP-T])\s*\}')


def _canonicalize_match_labels(text: str) -> str:
    """Convert raw `(A→p,r)` source-label mappings to canonical `(1→C,E)`.

    Handles both bare and LaTeX `\\mathrm{}`-wrapped forms. Generalization
    safety: requires arrow + single-char labels; real prose is untouched.
    Cross-validated against ~2,595 production matchQuestion solutions —
    zero false positives.
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

    return _MATCH_LABEL_MAPPING_PATTERN.sub(_repl, cleaned)


# Source -> canonical label maps (forward-referenced by _canonicalize_match_labels above):
#   List-I (left):  source A,B,C,D -> canonical 1,2,3,4
#   List-II (right): source p,q,r,s,t -> canonical A,B,C,D,E
_LIST_I_LABEL_MAP = {
    "A": "1", "B": "2", "C": "3", "D": "4",
    "1": "1", "2": "2", "3": "3", "4": "4",
}
_LIST_II_LABEL_MAP = {
    "p": "A", "q": "B", "r": "C", "s": "D", "t": "E",
    "P": "A", "Q": "B", "R": "C", "S": "D", "T": "E",
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
}


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

    Forked from production PCMB_Metadata_v2_Mapper.py:replace_md_images_with_img()
    Keeps local references (./images/...) — CDN upload happens in Stage 6.
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

        # Keep local reference format for now (CDN upload in Stage 6)
        final_url = f"./images/{base}"
        found_images.append(base)

        return f'<img src="{final_url}" alt="{alt}" {IMAGE_STYLE} />'

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


# ── Parse Stage-2-emitted LaTeX tabular -> list_i / list_ii ──────────────
def _parse_match_latex_tabular(content: str) -> Tuple[str, list, list]:
    """Parse a raw match-type question_content (LaTeX `tabular` block) into
    (intro_html, list_i, list_ii) for the HTML table builder.

    Source format (FIITJEE typical):
        1. Match the following:
        \\begin{tabular}{|l|l|l|l|}
        \\hline \\multicolumn{2}{|c|}{Column - I} & \\multicolumn{2}{|r|}{Column - II} \\\\
        \\hline (A) & <left content> & (p) & <right content> \\\\
        ...

    Returns (intro_html, list_i, list_ii) where:
      list_i = [{"label": "1", "content": "..."}, ...]   (canonical 1,2,3,4)
      list_ii = [{"label": "A", "content": "..."}, ...]  (canonical A,B,C,D[,E])

    Handles 4-col `|l|l|l|l|` AND 5-col `|l|l|l|l|l|` (empty separator middle
    column) via column-spec inspection.

    Recovers from common OCR artifacts:
      - missing closing paren `(p` -> still extracts label
      - empty label cell with non-empty content cell -> position-based label
        inference using source-label sequence (p,q,r,s,t)

    If no `\\begin{tabular}` is found, returns (content, [], []) — caller falls
    back to direct list_i/list_ii fields if available.
    """
    if not content or "\\begin{tabular}" not in content:
        intro_html, list_i, list_ii = _parse_match_plain_text(content)
        if list_i and list_ii:
            return intro_html, list_i, list_ii
        return content or "", [], []

    tab_start = content.find("\\begin{tabular}")
    intro_raw = content[:tab_start].strip()
    # BUG-A FIX: intro text may contain markdown image refs like ![](./images/...)
    # that were never converted to <img> tags because _replace_md_images_with_img
    # was only called on cell content, not on the intro block.
    intro_html, _ = _replace_md_images_with_img(intro_raw)

    # BUG-C FIX: find the *matching* \end{tabular} for the outer \begin{tabular}
    # by counting nesting depth. A naive content.find() returns the FIRST
    # \end{tabular} which may belong to a nested inner tabular block (when MathPix
    # emits a small table inside a cell), truncating the outer table prematurely
    # and losing the last rows (e.g. the S row in a 4-row match question).
    depth = 0
    search_pos = tab_start
    tab_end = -1
    while search_pos < len(content):
        next_begin = content.find("\\begin{tabular}", search_pos)
        next_end = content.find("\\end{tabular}", search_pos)
        if next_end < 0:
            break
        if next_begin >= 0 and next_begin < next_end:
            depth += 1
            search_pos = next_begin + len("\\begin{tabular}")
        else:
            depth -= 1
            if depth == 0:
                tab_end = next_end
                break
            search_pos = next_end + len("\\end{tabular}")

    if tab_end < 0:
        return intro_html, [], []
    body = content[tab_start:tab_end + len("\\end{tabular}")]

    # Strip header line `\begin{tabular}{|l|l|l|l|}` and capture column spec
    m = re.match(r"\\begin\{tabular\}\{([^}]*)\}", body)
    if not m:
        return intro_html, [], []
    col_spec = m.group(1)
    n_cols = sum(1 for ch in col_spec if ch in "lcrp")
    # body is correctly scoped to the outer table (Bug C fix above uses nesting-
    # aware search), so rfind("\\end{tabular}") finds the outer closing tag.
    body_inner = body[m.end():body.rfind("\\end{tabular}")].strip()
    body_inner = body_inner.replace("\n", " ")

    raw_rows = re.split(r"\\\\(?:\s*\\hline)?", body_inner)

    list_i: list = []
    list_ii: list = []
    body_row_idx = -1

    def _label_only(cell: str) -> str:
        """Extract label letter/digit from a label-only cell.

        Lenient — accepts (X), bare X, $(X)$, (X (no closing paren), X) (no
        opening paren). Common OCR artifacts.
        """
        cell = cell.strip()
        m1 = re.match(r"^\s*\(\s*([A-Za-z0-9])\s*\)\s*$", cell)
        if m1:
            return m1.group(1)
        m2 = re.match(r"^\s*([A-Za-z0-9])\s*$", cell)
        if m2:
            return m2.group(1)
        m3 = re.match(r"^\s*\$\s*\(\s*([A-Za-z0-9])\s*\)\s*\$\s*$", cell)
        if m3:
            return m3.group(1)
        m4 = re.match(r"^\s*\(\s*([A-Za-z0-9])\s*$", cell)
        if m4:
            return m4.group(1)
        m5 = re.match(r"^\s*([A-Za-z0-9])\s*\)\s*$", cell)
        if m5:
            return m5.group(1)
        return ""

    _SRC_LIST_II_SEQ = ["p", "q", "r", "s", "t"]

    for raw_row in raw_rows:
        row = raw_row.replace("\\hline", "").strip()
        if not row:
            continue
        if "\\multicolumn" in row:
            continue

        # BUG-B FIX: MathPix sometimes emits a nested \begin{tabular}...\end{tabular}
        # inside a cell's content (e.g. for a chemical structure description with "&"
        # separators inside it). Splitting on & without removing these inner blocks
        # causes the inner & characters to be treated as cell separators, producing
        # wrong cell counts and garbled content like "\begin{tabular}{l} Reacts with".
        # Solution: collapse each inner tabular block into a single placeholder token,
        # split on outer &, then restore the collapsed content.
        _INNER_TAB_PLACEHOLDER_PREFIX = "\x00INNERTAB"
        _inner_tabs: list = []
        def _collapse_inner_tabular(s: str) -> str:
            """Replace nested \\begin{tabular}...\\end{tabular} with placeholders."""
            result = []
            idx = 0
            while idx < len(s):
                start = s.find("\\begin{tabular}", idx)
                if start < 0:
                    result.append(s[idx:])
                    break
                result.append(s[idx:start])
                # Find matching \end{tabular}
                end = s.find("\\end{tabular}", start)
                if end < 0:
                    # Unclosed — drop the whole remainder to avoid further corruption
                    break
                end += len("\\end{tabular}")
                inner_block = s[start:end]
                # Extract readable text from inner block: strip LaTeX structural
                # tokens and keep only text/image content for display
                inner_text = re.sub(r"\\begin\{tabular\}\{[^}]*\}", "", inner_block)
                inner_text = re.sub(r"\\end\{tabular\}", "", inner_text)
                inner_text = re.sub(r"\\hline", " ", inner_text)
                inner_text = re.sub(r"\\\\", " ", inner_text)
                # Replace & separators inside the inner block with a space so
                # the content reads naturally (e.g. "Reacts with ketone to give...")
                inner_text = inner_text.replace("&", " ")
                inner_text = re.sub(r"\s{2,}", " ", inner_text).strip()
                placeholder = f"{_INNER_TAB_PLACEHOLDER_PREFIX}{len(_inner_tabs)}\x00"
                _inner_tabs.append(inner_text)
                result.append(placeholder)
                idx = end
            return "".join(result)

        row_clean = _collapse_inner_tabular(row)

        cells_raw = [c.strip() for c in re.split(r"(?<!\\)&", row_clean)]

        def _restore_inner_tabs(s: str) -> str:
            """Restore collapsed inner tabular placeholders to their extracted text."""
            def _repl_ph(m):
                idx2 = int(m.group(1))
                return _inner_tabs[idx2] if idx2 < len(_inner_tabs) else ""
            return re.sub(rf"{re.escape(_INNER_TAB_PLACEHOLDER_PREFIX)}(\d+)\x00", _repl_ph, s)

        cells = [_restore_inner_tabs(c) for c in cells_raw]
        if len(cells) < 4:
            continue
        if n_cols >= 5 and len(cells) >= 5:
            c0, c1, c2, c3 = cells[0], cells[1], cells[3], cells[4]
        else:
            c0, c1, c2, c3 = cells[0], cells[1], cells[2], cells[3]

        body_row_idx += 1

        # List-I (label + content)
        li_label_raw = _label_only(c0)
        if li_label_raw and c1.strip():
            li_label_can = _LIST_I_LABEL_MAP.get(li_label_raw, li_label_raw)
            li_content_html, _ = _replace_md_images_with_img(c1.strip())
            list_i.append({"label": li_label_can, "content": li_content_html})

        # List-II (label + content). Position-based recovery: if c2 is empty
        # but c3 has substantive content (>5 chars), source likely dropped the
        # label between adjacent rows — infer from row index.
        lii_label_raw = _label_only(c2)
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


def _fill_match_label_gaps(items: list, label_seq: list) -> list:
    """Fill missing labels in `items` so the visible label sequence has no holes.

    Source PDFs sometimes drop labels mid-sequence (e.g. List-II with labels
    A,B,D,E missing C). This inserts placeholder rows ({"label": "C",
    "content": "&nbsp;"}) so the rendered table shows (A)(B)(C)(D)(E) with
    the empty cell visually flagging the gap.

    Only fills gaps WITHIN the existing range — does not extend beyond the
    last present label.
    """
    if not items:
        return items
    present = {x.get("label") for x in items if x.get("label")}
    if not present:
        return items
    max_idx = -1
    for idx, lbl in enumerate(label_seq):
        if lbl in present:
            max_idx = idx
    if max_idx < 0:
        return items
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

    Schema (matches DB sample for matchQuestion):
    - Outer table: border="1" + inline CSS shorthand for CKEditor safety
    - Header row: two <th colspan="2"> with background-color:#e6e6e6, hardcoded
      "List-I" / "List-II" (canonical, source variation `Column - I` etc. ignored)
    - Body rows: max(|list_i|, |list_ii|) rows; missing cells use &nbsp;
    - Labels wrapped in <strong>(...)</strong>
    - Label `<td>` uses ``white-space: nowrap`` so "(1)" / "(A)" never wraps onto
      two lines (M05/M06 parens-on-newline source data)
    - list_i/list_ii are gap-filled with sequential labels so source-data
      dropouts (e.g. M02 missing 'C') still render as (A)(B)(C)(D)(E) with a
      visually empty cell flagging the gap.
    """
    if not list_i and not list_ii:
        cleaned_intro = re.sub(r"^\s*\d+[\.\)]\s+", "", intro_html, count=1)
        cleaned_intro = _convert_markdown_bold(cleaned_intro)
        return cleaned_intro if cleaned_intro.lstrip().startswith("<") else f"<p>{cleaned_intro}</p>"

    parts = []
    if intro_html:
        # Strip leading question-number prefix like "1.", "2.", "1)" from raw mmd
        cleaned_intro = re.sub(r"^\s*\d+[\.\)]\s+", "", intro_html, count=1)
        cleaned_intro = _convert_markdown_bold(cleaned_intro)
        intro = cleaned_intro if cleaned_intro.lstrip().startswith("<") else f"<p>{cleaned_intro}</p>"
        parts.append(intro)

    list_i = _fill_match_label_gaps(list_i, ["1", "2", "3", "4"])
    list_ii = _fill_match_label_gaps(list_ii, ["A", "B", "C", "D", "E"])

    parts.append(
        '<table border="1" cellpadding="10" cellspacing="0" '
        'style="border-collapse: collapse; border: 1px solid #555555; width: 100%;">'
    )
    parts.append("<tbody>")

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

    max_rows = max(len(list_i), len(list_ii))
    for i in range(max_rows):
        parts.append("<tr>")
        if i < len(list_i):
            li = list_i[i]
            parts.append(
                '<td style="border: 1px solid #555555; text-align:center; white-space: nowrap;">'
                f'<strong>({li.get("label", "")})</strong></td>'
            )
            parts.append(
                f'<td style="border: 1px solid #555555;">{_convert_markdown_bold(li.get("content", ""))}</td>'
            )
        else:
            parts.append('<td style="border: 1px solid #555555;">&nbsp;</td>')
            parts.append('<td style="border: 1px solid #555555;">&nbsp;</td>')

        if i < len(list_ii):
            lii = list_ii[i]
            parts.append(
                '<td style="border: 1px solid #555555; text-align:center; white-space: nowrap;">'
                f'<strong>({lii.get("label", "")})</strong></td>'
            )
            parts.append(
                '<td style="border: 1px solid #555555; text-align:center;">'
                f'{_convert_markdown_bold(lii.get("content", ""))}</td>'
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

    # Convert any leftover **X** markdown bold to <strong>X</strong>
    question_text = _convert_markdown_bold(question_text)

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

    # Convert any leftover **X** markdown bold to <strong>X</strong>
    solution_text = _convert_markdown_bold(solution_text)

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
        # Canonical passage schema (matches DB sample):
        #   - top-level: passageImages, passageSolutionImages, passage, solution, questions
        #   - top-level solution = "" (empty string, NOT "<h3></h3>" placeholder)
        #   - inner sub-questions ALWAYS wrapped under "scq" (not their own qtype)
        #   - inner field name is "ques" (NOT "question")
        #   - inner has only 4 fields: ques, solution, answer, passage (back-ref)
        #   - inner does NOT include quesImages/solutionImages
        passage_raw = entry.get("passage_content", "")
        if passage_raw:
            passage_raw, _ = _replace_md_images_with_img(passage_raw)
            passage_raw = _convert_markdown_bold(passage_raw)
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
            # Always wrap inner under "scq" key, only 4 fields per DB sample.
            # `passage` here is the parent's qid as placeholder; Stage 10 upload
            # step replaces it with the parent's MongoDB OID after create.
            sub_questions_out.append({
                "scq": {
                    "ques": sq_question_html,
                    "solution": sq_solution_html,
                    "answer": sq_answer,
                    "passage": qid,
                }
            })

        active_container = {
            "passage": passage_html,
            "solution": "",
            "questions": sub_questions_out,
            "passageImages": [],
            "passageSolutionImages": [],
        }
        answer = ""
    elif qtype == "matchQuestion":
        # Canonical matchQuestion schema (matches DB sample):
        #   {rows, quesImages, solutionImages, question, solution}
        # rows: list-of-lists, supports multi-match (e.g., [["B","D"],["A"],...])
        # rows[i] = answers (List-II canonical labels) for List-I item i+1
        # quesImages/solutionImages: null (not [], per DB sample)
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

        # Canonicalize raw `(A → p,r)` source mappings in solution text BEFORE
        # the LaTeX → MathML conversion (stage 7). Operates on solution_content;
        # stage 7 will then encode the canonical form into MathML. Safe: only
        # fires on patterns containing arrows + label chars.
        s_key = "solution_content" if "solution_content" in entry else "solution"
        if s_key in entry and entry[s_key]:
            entry = dict(entry)
            entry[s_key] = _canonicalize_match_labels(entry[s_key])
        solution_html = _build_solution_html(entry)

        active_container = {
            "rows": answer_mapping,
            "quesImages": None,
            "solutionImages": None,
            "question": question_html,
            "solution": solution_html,
        }
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
            if qtype == "matchQuestion":
                # Match container has answer="" by design — the answer is `rows`.
                rows = m.get(qtype, {}).get("rows", [])
                return isinstance(rows, list) and any(
                    isinstance(r, list) and r for r in rows
                )
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
                    # Canonical sub-Q schema field is "ques" (per _build_metadata_v2);
                    # legacy/orchestration variants used "question" — accept both.
                    sq_q = sq_data.get("ques", sq_data.get("question", ""))
                    if not (sq_q.replace("<h3></h3>", "")
                            and sq_data.get("answer", "")):
                        return False
                return True
            if qtype == "matchQuestion":
                # Match canonical: question table HTML + non-empty rows. Solution may be empty.
                mq = m.get(qtype, {})
                has_q = bool((mq.get("question", "") or "").replace("<h3></h3>", ""))
                rows = mq.get("rows", [])
                has_rows = isinstance(rows, list) and any(
                    isinstance(r, list) and r for r in rows
                )
                return has_q and has_rows
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
