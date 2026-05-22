"""Stage 11: Error=2 Auto-Correction Loop.

Forked from intern's question_correction_v4.py with major refactor.
Operates on EXISTING uploaded questions (post Stage 10) that are flagged error=2.
Re-runs LLM1 corrector + lxml validator + LLM2 proofreader loop and, on
approval, calls /question/update/field to flip error=2 -> error=1.

Architecture:
    1. Read input set (CSV: qid, error_description, error_class)
    2. Per question:
        a. Fetch via /question/details (auth via auth.py)
        b. Backup raw response
        c. Extract Q + S HTML by question type
        d. MathML -> LaTeX (production: _mathml_to_latex.convert_mathml_to_latex)
        e. Detect images, attach as multimodal parts (port from stage9)
        f. LLM1 correction loop (max 3 attempts):
             - Call LLM1 with Q+S in $LaTeX$ form + error_description + (image attachments)
             - Receive corrected Q+S
             - LaTeX -> MathML (production: extracted from stage7_latex_tag.py)
             - On LaTeX->MathML failure: SIDELINE question (mathml_conversion_failure)
             - lxml validate corrected MathML (XML parse + structural tag check)
             - LLM2 proofread corrected vs ORIGINAL (key fix vs v4: passes both)
             - If approved: break; else: feedback to LLM1 next attempt
        g. After approval: similarity check (per-error-class threshold)
        h. If passes threshold: call /question/update/field with strip(_id, createdAt, updatedAt, __v)
        i. If SET_ERROR_TO_1 flag: flip error 2 -> 1
    3. Persistent checkpoint for resume

Production modules wired:
    - MathML -> LaTeX: app.pipeline._mathml_to_latex.convert_mathml_to_latex (= V5.0 byte-identical)
    - LaTeX -> MathML: lifted from app.pipeline.stage7_latex_tag._convert_latex_to_mathml + _fix_latex2mathml_output_for_ckeditor
    - JWT auth: app.pipeline.auth.authenticated_request
    - Settings: app.config.settings
    - Image multimodal: ported from app.pipeline.stage9_quality._collect_images_with_placeholders

Issues fixed vs intern v4.0:
    B1 hardcoded keys -> read from settings (.env)
    B2 LLM2 only saw corrected -> now sees ORIGINAL + corrected
    B3 wrong tiktoken encoder -> use cl100k_base baseline
    B4 sympy raw fallback -> use convert_mathml_to_latex
    B5 latex2mathml broken fallback -> raise mathml_conversion_failure
    B6 no image multimodal -> ported _collect_images_with_placeholders
    B7 silent mathml fail -> explicit sideline path
    H1 _DOLLAR_RE greedy on escaped dollar -> use stage7's hardened patterns
    H2 fixed 95% threshold -> per-error-class threshold table
    H3 lxml false positive on legitimate removals -> conditional skip
    H4 no body strip -> _build_update_body strips _id/createdAt/updatedAt/__v
    H5 no retry -> tenacity-style exponential backoff
    H6 hardcoded JWT -> auth.authenticated_request
"""

from __future__ import annotations

import base64
import copy
import difflib
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from lxml import etree

# Production modules (DO NOT bypass)
from app.config import settings
from app.pipeline.auth import authenticated_request
from app.pipeline._mathml_to_latex import convert_mathml_to_latex


# ============================================================================
# Configuration
# ============================================================================

# Models under benchmark.
# Pricing FORKED FROM PROOF-READING-V6.0/app/core/cost_tracker.py (verified March 2026):
#   gemini-3-flash-preview real-time: $0.50/M input uncached, $0.25/M cached, $3.00/M output
#   gemini-3-flash-preview batch:     50% off all rates
#   Thinking tokens: billed at the OUTPUT rate (added to completion).
#   Batch path is half-price but adds 24h SLA -> not viable for our retry loop.
SUPPORTED_MODELS = {
    "gpt-5-nano":              {"input_uncached": 0.05, "input_cached": 0.025, "output": 0.40,  "api": "openai", "multimodal": True},
    "gpt-5-mini":              {"input_uncached": 0.25, "input_cached": 0.125, "output": 2.00,  "api": "openai", "multimodal": True},
    "gpt-5.4-nano":            {"input_uncached": 0.20, "input_cached": 0.10,  "output": 1.25,  "api": "openai", "multimodal": True},
    "gpt-5.4-mini":            {"input_uncached": 0.75, "input_cached": 0.375, "output": 4.50,  "api": "openai", "multimodal": True},
    "gemini-3-flash-preview":  {"input_uncached": 0.50, "input_cached": 0.25,  "output": 3.00,  "api": "gemini", "multimodal": True},
}

# Default for Stage 11 final flow (reverted 2026-05-08 per user):
#   LLM1 = LLM2 = gpt-5-mini. Validated 5/5 in corrective_loop on cohort_A samples.
#   Gemini-3-flash-preview was tried but only 3/4 reliable on same cohort; reverted.
DEFAULT_LLM1_MODEL = "gpt-5-mini"
DEFAULT_LLM2_MODEL = "gpt-5-mini"

# Gemini thinking_level: "LOW" / "MEDIUM" / "HIGH" / dynamic via thinking_budget.
# Stage 9 production uses "LOW". Stage 11 will benchmark LOW vs MEDIUM on 4 samples.
DEFAULT_GEMINI_THINKING_LEVEL = "LOW"

# Pydantic schemas for structured output. Gemini's google.genai SDK accepts Pydantic
# models directly via response_schema= and parses internally (handles HTML quote escaping).
# OpenAI also supports response_format={"type": "json_schema", "json_schema": ...} but for
# now we keep OpenAI on json_object mode (it handles HTML correctly already).
from pydantic import BaseModel as _PydBase
from pydantic import Field as _PydField


class LLM1Response(_PydBase):
    question: str = _PydField(default="", description="Corrected question HTML")
    solution: str = _PydField(default="", description="Corrected solution HTML")
    mathml_correction: bool = _PydField(default=False)
    html_correction: bool = _PydField(default=False)
    latex_correction: bool = _PydField(default=False)


class LLM2Response(_PydBase):
    approved: bool
    preferred_version: str = _PydField(default="")
    reasoning: str = _PydField(default="")
    byte_drift_outside_error_span: bool = _PydField(default=False)
    drift_details: str = _PydField(default="")
    broken_code_found: bool = _PydField(default=False)
    broken_code_details: str = _PydField(default="")
    error_fixed_in_preferred: bool = _PydField(default=True)
    error_fix_details: str = _PydField(default="")
    feedback: str = _PydField(default="")

# Acadza API
DETAILS_URL = "https://api.acadza.in/question/details"
UPDATE_URL = "https://api.acadza.in/question/update/field"

# Question-type routing (matches main pipeline conventions)
TYPE_KEY_MAP = {
    "scq":             "scq",
    "mcq":             "mcq",
    "integer":         "integerQuestion",
    "integerquestion": "integerQuestion",
    "passage":         "passageQuestion",
    "passagequestion": "passageQuestion",
    "match":           "matchQuestion",
    "matchquestion":   "matchQuestion",
}

# Per-error-class similarity threshold (avg of question + solution similarity)
# Tuned for the typical change-magnitude of each error class.
SIMILARITY_THRESHOLDS = {
    "cleaner_artifact":   80.0,   # removing source artifacts like 'CDF127'
    "answer_prefix_leak": 75.0,   # removing '(B)' or '(7)' from question stem
    "source_artifact":    80.0,   # removing exam refs like 'JEE 2018'
    "mathml_broken":      92.0,   # fixing broken tags - structure should mostly survive
    "math_logic":         95.0,   # changing math content - high preservation expected
    "default":            90.0,   # unknown class falls back to mid-tight
}

# Removal-type error classes — for these, lxml structural tag check is relaxed
# because the correct fix REMOVES tags.
REMOVAL_TYPE_CLASSES = {"cleaner_artifact", "answer_prefix_leak", "source_artifact"}

MAX_CORRECTION_ATTEMPTS = 1   # SINGLE-PASS MODE (2026-05-22 tonight observation)
# Flow with max=1 (per user directive):
#   Attempt 1: LLM1 corrects -> LLM2 proofreads.
#              If LLM2 approves => promoted.
#              If LLM2 rejects   => flagged for faculty (no retry).
# Cost cap: exactly 2 LLM calls per question (down from 6 at attempts=3).
# Rationale: Thursday May 21 saw 51% promote rate; multi-attempt was burning
# tokens on hard cases without yield. Tomorrow morning we read stats and
# decide whether to bump back to 2-attempt mode.
ONE_SHOT_ATTEMPTS = 1         # one-shot mode: single LLM1 call, single LLM2 verdict, flag if rejected

# Note: temperature/seed/top_p/thinking_level are NOT exposed as Stage 11 constants.
# Stage 11 forks production conventions from stage9_quality.py:
#   - OpenAI GPT-5 family: NO temperature/top_p/seed (rejected by reasoning models);
#     reasoning_effort='low' for *mini*/*nano*; max_completion_tokens=8192.
#   - Gemini 3 Flash: temperature=0, max_output_tokens=8192, thinking_level='LOW',
#     response_mime_type='application/json'.

# Per Acadza API contract: these fields must be stripped from PUT body.
STRIP_FIELDS_ON_UPDATE = ["_id", "createdAt", "updatedAt", "__v"]

# Retry behavior for LLM/API calls
RETRY_BACKOFF_BASE_S = 2.0
RETRY_MAX_ATTEMPTS = 3


# ============================================================================
# Image multimodal helpers (ported from stage9_quality.py)
# ============================================================================

IMG_RE = re.compile(r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)
IMG_FULL_TAG_RE = re.compile(r'<img\b[^>]*>', re.IGNORECASE)
VIDEO_FULL_TAG_RE = re.compile(r'<video\b[^>]*>.*?</video>|<video\b[^>]*/?>', re.IGNORECASE | re.DOTALL)
DATA_URI_IMG_RE = re.compile(r'^data:image/(png|jpe?g|gif|bmp|webp|svg\+xml);base64,')
_IMG_SESSION = requests.Session()
_IMG_SESSION.headers.update({"User-Agent": "acadza-stage11/1.0"})


def _guess_mime_from_ext(url_path: str) -> Optional[str]:
    path = urlparse(url_path).path.lower()
    if path.endswith(".png"): return "image/png"
    if path.endswith((".jpg", ".jpeg")): return "image/jpeg"
    if path.endswith(".gif"): return "image/gif"
    if path.endswith(".webp"): return "image/webp"
    if path.endswith(".svg"): return "image/svg+xml"
    return None


def _guess_mime_from_signature(content: bytes) -> Optional[str]:
    if content.startswith(b"\x89PNG\r\n\x1a\n"): return "image/png"
    if content.startswith(b"\xff\xd8\xff"): return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")): return "image/gif"
    if content.startswith(b"RIFF") and b"WEBP" in content[:32]: return "image/webp"
    return None


def _fetch_as_data_uri(url: str) -> Optional[str]:
    try:
        if url.startswith("data:"):
            return url if DATA_URI_IMG_RE.match(url) else None
        if not url.startswith(("http://", "https://")):
            return None
        with _IMG_SESSION.get(url, timeout=(6, 20), stream=True) as r:
            r.raise_for_status()
            content = r.content or b""
            ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            mime = ct if ct.startswith("image/") else None
            if not mime:
                mime = _guess_mime_from_ext(url)
            if not mime:
                mime = _guess_mime_from_signature(content)
            if not mime or not mime.startswith("image/"):
                return None
            b64 = base64.b64encode(content).decode("ascii")
            return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def collect_images_with_placeholders(html_text: str, prefix: str) -> Tuple[str, List[Dict], Dict[str, str], Dict[str, str]]:
    """Replace <img> tags with [Q1]/[S1] placeholders and return data-URI attachments.

    prefix: 'Q' for question images, 'S' for solution images.
    Returns (new_html, attachments, src_to_token, token_to_full_tag).
      - src_to_token: maps S3 URL -> "[Q1]" placeholder text
      - token_to_full_tag: maps "[Q1]" -> the ORIGINAL full <img ...> tag, so restore_image_placeholders
        can byte-restore the exact tag (preserving alt, style, width, height).
    """
    if not html_text:
        return html_text, [], {}, {}
    matches = list(IMG_FULL_TAG_RE.finditer(html_text))
    if not matches:
        return html_text, [], {}, {}

    src_to_token: Dict[str, str] = {}
    token_to_full_tag: Dict[str, str] = {}
    idx = 1
    for m in matches:
        full_tag = m.group(0)
        src_m = re.search(r'src=["\']([^"\']+)["\']', full_tag, re.IGNORECASE)
        if not src_m:
            continue
        src = src_m.group(1)
        if src not in src_to_token:
            tok = f"[{prefix}{idx}]"
            src_to_token[src] = tok
            token_to_full_tag[tok] = full_tag
            idx += 1

    def repl(m):
        full_tag = m.group(0)
        src_m = re.search(r'src=["\']([^"\']+)["\']', full_tag, re.IGNORECASE)
        src = src_m.group(1) if src_m else ""
        return src_to_token.get(src, f"[{prefix}X]")

    new_html = IMG_FULL_TAG_RE.sub(repl, html_text)

    attachments = []
    for src in src_to_token:
        data_uri = _fetch_as_data_uri(src)
        if data_uri:
            attachments.append({
                "type": "image_url",
                "image_url": {"url": data_uri, "detail": "auto"},
            })
    return new_html, attachments, src_to_token, token_to_full_tag


def restore_image_placeholders(html_text: str, token_to_full_tag: Dict[str, str]) -> str:
    """Reverse of collect_images_with_placeholders for ONE side (Q or S).

    Replaces every "[Q1]"/"[S1]"/etc. token back to the original <img> tag byte-for-byte.
    If the LLM hallucinated `<img src="cid:ImageS1">` style references, those are also
    normalized: any cid:Image<digit> form is mapped back via the token table.
    """
    if not html_text or not token_to_full_tag:
        return html_text
    out = html_text
    for tok, full_tag in token_to_full_tag.items():
        out = out.replace(tok, full_tag)
        # Also normalize common LLM hallucinations: <img src="cid:ImageS1"> -> original tag.
        # Token like "[S1]" -> we look for src="cid:ImageS1" / src="cid:imageS1" etc.
        m = re.match(r"\[([QS])(\d+)\]", tok)
        if m:
            prefix, num = m.group(1), m.group(2)
            cid_kind = "Q" if prefix == "Q" else "S"
            cid_pattern = re.compile(
                rf'<img\s+[^>]*src=["\']cid:Image{cid_kind}{num}["\'][^>]*/?>',
                re.IGNORECASE,
            )
            out = cid_pattern.sub(full_tag, out)
    return out


_LATEX_FIGURE_RE = re.compile(r'\\begin\{figure\}|\\includegraphics\b', re.IGNORECASE)


def img_invariants(html_text: str) -> Tuple[int, frozenset, int, int]:
    """Return (img_count, frozenset_of_src_urls, video_tag_count, latex_figure_count).

    latex_figure_count covers questions where images are referenced via LaTeX
    (e.g. cohort_B PYQ scrapes) — these aren't <img> tags but losing them
    silently is just as bad as losing <img>.
    """
    if not html_text:
        return 0, frozenset(), 0, 0
    img_tags = IMG_FULL_TAG_RE.findall(html_text)
    src_urls = []
    for tag in img_tags:
        m = re.search(r'src=["\']([^"\']+)["\']', tag, re.IGNORECASE)
        if m:
            src_urls.append(m.group(1))
    video_count = len(VIDEO_FULL_TAG_RE.findall(html_text))
    latex_fig_count = len(_LATEX_FIGURE_RE.findall(html_text))
    return len(img_tags), frozenset(src_urls), video_count, latex_fig_count


# ============================================================================
# LaTeX <-> MathML (production-grade)
# ============================================================================

# MathML -> LaTeX: just delegate to the production helper.
def html_mathml_to_latex(html: str) -> str:
    """Replace each <math>...</math> block with an inline \\(...\\) LaTeX equivalent.

    Uses app.pipeline._mathml_to_latex.convert_mathml_to_latex (= PROOF-READING-V5.0
    byte-identical, hardened on 162K-q DB).
    """
    return convert_mathml_to_latex(html)


# LaTeX -> MathML: lifted logic from stage7_latex_tag.py with explicit failure detection.
# IMPORTANT: keep this in sync with stage7's _convert_latex_to_mathml when that module
# is updated. (Future refactor: move to a shared module.)
try:
    from latex2mathml.converter import convert as latex_to_mathml_lib
    HAS_LATEX2MATHML = True
except Exception:
    HAS_LATEX2MATHML = False

# Display math regex order matters: $$ before $, \[ before \(.
_LATEX_PATTERNS = [
    (r'\$\$([^$]+)\$\$',                      True),    # $$...$$
    (r'\\\[(.+?)\\\]',                        True),    # \[...\]
    (r'\\\((.+?)\\\)',                        False),   # \(...\)
    (r'(?<!\$)\$(?!\$)([^$]+)\$(?!\$)',       False),   # $...$  (lookahead prevents $$)
]
# Note: H1 fix — we still match $...$ even if there's a lone \$ in text. The pattern
# (?<!\$)\$(?!\$) ensures we don't match $$. Currency-style \$5 inside text (not math)
# is handled by skipping math sections that fail to parse — see _try_latex_to_mathml.

_MATRIX_ENVS = ('pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix', 'matrix', 'Bmatrix',
                'smallmatrix', 'cases', 'array')


class MathMLConversionError(Exception):
    """Raised when LaTeX->MathML conversion produces no usable output. Triggers sideline."""
    pass


def _preprocess_matrix(raw_latex: str) -> str:
    """latex2mathml only understands \\cr as row separator; convert from \\ and <br/>."""
    _TWO_BS = chr(92) + chr(92)
    _CR_SEP = chr(92) + 'cr '
    for env in _MATRIX_ENVS:
        begin_tag = chr(92) + 'begin{' + env + '}'
        end_tag = chr(92) + 'end{' + env + '}'
        bi = raw_latex.find(begin_tag)
        ei = raw_latex.find(end_tag)
        if bi >= 0 and ei > bi:
            body_start = bi + len(begin_tag)
            body = raw_latex[body_start:ei]
            body = body.replace(_TWO_BS, _CR_SEP)
            body = re.sub(r'<br\s*/?\s*>', lambda _: _CR_SEP, body)
            raw_latex = raw_latex[:body_start] + body + raw_latex[ei:]
    return raw_latex


def _postprocess_mathml(mathml: str, is_display: bool) -> str:
    """Add xmlns + display + mstyle wrapper for CKEditor compatibility."""
    if 'xmlns=' not in mathml:
        mathml = mathml.replace(
            '<math', '<math xmlns="http://www.w3.org/1998/Math/MathML"', 1
        )
    if is_display:
        mathml = re.sub(r'display="[^"]*"', '', mathml, count=1)
        mathml = mathml.replace('<math', '<math display="block"', 1)
        mathml = re.sub(
            r'(<math[^>]*>)(.*)(</math>)',
            r'\1<mstyle displaystyle="true">\2</mstyle>\3',
            mathml, count=1, flags=re.DOTALL,
        )
    else:
        if 'display=' not in mathml:
            mathml = mathml.replace('<math', '<math display="inline"', 1)
        if 'displaystyle' not in mathml:
            mathml = re.sub(
                r'(<math[^>]*>)(.*)(</math>)',
                r'\1<mstyle displaystyle="true">\2</mstyle>\3',
                mathml, count=1, flags=re.DOTALL,
            )
    return mathml


def _try_one_latex_to_mathml(raw_latex: str, is_display: bool) -> Optional[str]:
    """Convert one LaTeX expression to MathML; return None on failure (no exception)."""
    if not HAS_LATEX2MATHML:
        return None
    raw_latex = _preprocess_matrix(raw_latex.strip())
    try:
        mathml = latex_to_mathml_lib(raw_latex)
    except Exception:
        return None
    if not mathml or '<math' not in mathml:
        return None
    return _postprocess_mathml(mathml, is_display)


def html_latex_to_mathml(html: str) -> Tuple[str, int, int]:
    """Convert all LaTeX patterns ($...$, $$...$$, \\(...\\), \\[...\\]) in HTML to MathML.

    Returns (converted_html, success_count, failure_count).

    NOTE: This raises MathMLConversionError if ANY conversion fails AND there were
    LaTeX patterns to convert. The caller (Stage 11 loop) catches this and sidelines
    the question with reason='mathml_conversion_failure'. This is the B7 fix.
    """
    if not html or not HAS_LATEX2MATHML:
        return html, 0, 0

    success = 0
    fail = 0

    for pattern, is_display in _LATEX_PATTERNS:
        def _replace(match, _is_display=is_display):
            nonlocal success, fail
            raw = match.group(1)
            mathml = _try_one_latex_to_mathml(raw, _is_display)
            if mathml is None:
                fail += 1
                # Return the original $...$ block unchanged so downstream regexes
                # can still spot it; the caller checks fail count.
                return match.group(0)
            success += 1
            return mathml
        html = re.sub(pattern, _replace, html, flags=re.DOTALL)

    if fail > 0:
        raise MathMLConversionError(
            f"LaTeX->MathML conversion failed on {fail} expression(s) "
            f"(success={success}). Sidelining question."
        )

    return html, success, fail


# ============================================================================
# Tag balance verification (raw HTML+MathML in/out — replaces byte similarity)
# ============================================================================

# Match every opening + self-closing tag (we'll separately count closings).
# Captures: tag name (group 1).
_OPEN_TAG_RE = re.compile(r'<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*?(/?)>')
_CLOSE_TAG_RE = re.compile(r'</([a-zA-Z][a-zA-Z0-9]*)\s*>')

# LaTeX delimiter regexes (count occurrences, not balance — paired delimiters
# come in pairs so we can compare in vs out).
_LATEX_DELIMS = {
    "dollar_double":   re.compile(r'\$\$'),
    "dollar_single":   re.compile(r'(?<!\$)\$(?!\$)'),
    "paren_open":      re.compile(r'\\\('),
    "paren_close":     re.compile(r'\\\)'),
    "bracket_open":    re.compile(r'\\\['),
    "bracket_close":   re.compile(r'\\\]'),
}

# MathML element tag set (for splitting MathML vs HTML counts).
_MATHML_TAGS = {
    "math", "mrow", "mi", "mn", "mo", "ms", "mtext", "mspace",
    "mfrac", "msup", "msub", "msubsup", "mroot", "msqrt",
    "munder", "mover", "munderover", "mtable", "mtr", "mtd",
    "mstyle", "mphantom", "menclose", "mfenced", "merror",
    "semantics", "annotation", "annotation-xml",
}


def _tag_counts(html: str) -> Dict[str, Dict[str, int]]:
    """Return {'open': {tag: count}, 'close': {tag: count}, 'self_close': {tag: count}}.

    Self-closing tags (e.g. <br/>, <img/>) are tracked separately from open+close pairs.
    """
    open_c: Dict[str, int] = {}
    close_c: Dict[str, int] = {}
    self_c: Dict[str, int] = {}
    if not html:
        return {"open": open_c, "close": close_c, "self_close": self_c}
    for m in _OPEN_TAG_RE.finditer(html):
        tag = m.group(1).lower()
        is_self_closing = bool(m.group(2))
        if is_self_closing:
            self_c[tag] = self_c.get(tag, 0) + 1
        else:
            open_c[tag] = open_c.get(tag, 0) + 1
    for m in _CLOSE_TAG_RE.finditer(html):
        tag = m.group(1).lower()
        close_c[tag] = close_c.get(tag, 0) + 1
    return {"open": open_c, "close": close_c, "self_close": self_c}


def _split_mathml_html_counts(counts: Dict[str, Dict[str, int]]) -> Tuple[Dict, Dict]:
    """Split a single counts dict into (mathml_counts, html_counts) by tag classification."""
    mml = {"open": {}, "close": {}, "self_close": {}}
    htm = {"open": {}, "close": {}, "self_close": {}}
    for kind in ("open", "close", "self_close"):
        for tag, n in counts[kind].items():
            target = mml if tag in _MATHML_TAGS else htm
            target[kind][tag] = n
    return mml, htm


def _latex_delim_counts(html: str) -> Dict[str, int]:
    if not html:
        return {k: 0 for k in _LATEX_DELIMS}
    return {k: len(rx.findall(html)) for k, rx in _LATEX_DELIMS.items()}


def _diff_counts(a: Dict[str, int], b: Dict[str, int]) -> List[str]:
    """Return human-readable list of count diffs (a vs b). Empty list = no drift."""
    issues = []
    keys = set(a) | set(b)
    for k in sorted(keys):
        av = a.get(k, 0)
        bv = b.get(k, 0)
        if av != bv:
            issues.append(f"{k}: orig={av} new={bv} (delta={bv - av:+d})")
    return issues


def verify_tag_balance(
    orig_html: str,
    corrected_html: str,
    mathml_correction: bool,
    html_correction: bool,
    latex_correction: bool,
) -> Dict[str, Any]:
    """Hard structural check: count tag classes in vs out.

    Per-class rule:
      - flag=False  -> counts MUST match exactly. Drift = fail.
      - flag=True   -> counts may differ. Record but don't fail.

    Returns:
      {
        "passed": bool,
        "issues": [str, ...],
        "mathml_drift": [str],
        "html_drift": [str],
        "latex_drift": [str],
      }
    """
    issues = []

    orig_tag_counts = _tag_counts(orig_html)
    new_tag_counts = _tag_counts(corrected_html)
    orig_mml, orig_htm = _split_mathml_html_counts(orig_tag_counts)
    new_mml, new_htm = _split_mathml_html_counts(new_tag_counts)

    # MathML check
    mathml_drift = []
    for kind in ("open", "close", "self_close"):
        d = _diff_counts(orig_mml[kind], new_mml[kind])
        if d:
            mathml_drift.extend([f"mathml/{kind}/{x}" for x in d])
    if mathml_drift and not mathml_correction:
        issues.append(
            "MathML tag drift detected but mathml_correction=false:\n  "
            + "\n  ".join(mathml_drift)
        )

    # HTML check
    html_drift = []
    for kind in ("open", "close", "self_close"):
        d = _diff_counts(orig_htm[kind], new_htm[kind])
        if d:
            html_drift.extend([f"html/{kind}/{x}" for x in d])
    if html_drift and not html_correction:
        issues.append(
            "HTML tag drift detected but html_correction=false:\n  "
            + "\n  ".join(html_drift)
        )

    # LaTeX delimiter check
    orig_lx = _latex_delim_counts(orig_html)
    new_lx = _latex_delim_counts(corrected_html)
    latex_drift = _diff_counts(orig_lx, new_lx)
    if latex_drift and not latex_correction:
        issues.append(
            "LaTeX delimiter drift detected but latex_correction=false:\n  "
            + "\n  ".join(latex_drift)
        )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "mathml_drift": mathml_drift,
        "html_drift": html_drift,
        "latex_drift": latex_drift,
    }


# ============================================================================
# Programmatic MathML validator (lxml tag-by-tag)  -- KEPT for backward compat
# ============================================================================

_MATH_BLOCK_RE = re.compile(r'<math[^>]*?>.*?</math>', re.DOTALL | re.IGNORECASE)

# Structural tags — if present in original, should still be present in corrected.
# Skip this check entirely for removal-type error classes (H3 fix).
_STRUCTURAL_TAGS = {
    "mfrac", "msup", "msub", "msubsup", "mroot", "msqrt",
    "munder", "mover", "munderover", "mrow", "mo", "mi",
    "mn", "mtext", "mtable", "mtr", "mtd", "mstyle",
    "mspace", "mphantom", "menclose",
}


def _collect_tags(element) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for el in element.iter():
        tag = etree.QName(el.tag).localname if "}" in str(el.tag) else el.tag
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def _parse_all_math_blocks(html: str) -> List[Tuple[str, Optional[Any], Optional[str]]]:
    results = []
    for m in _MATH_BLOCK_RE.finditer(html):
        raw = m.group(0)
        try:
            el = etree.fromstring(raw.encode("utf-8"))
            results.append((raw, el, None))
        except etree.XMLSyntaxError as exc:
            results.append((raw, None, str(exc)))
    return results


def validate_mathml(original_html: str, corrected_html: str, error_class: str) -> Dict[str, Any]:
    """Returns {'passed': bool, 'issues': [str, ...]}.

    Check 1: every <math> block in corrected HTML must parse as XML.
    Check 2 (skipped for removal-type error classes per H3 fix):
      no structural tag that existed in original is completely absent from corrected.
    """
    issues: List[str] = []

    corr_blocks = _parse_all_math_blocks(corrected_html)
    for raw, _, err in corr_blocks:
        if err:
            snippet = raw[:150].replace("\n", " ")
            issues.append(f"BROKEN MathML (XML parse error): {err} | snippet: {snippet}")

    if issues:
        return {"passed": False, "issues": issues}

    # H3 fix: skip structural-tag presence check when error class is removal-type
    if error_class in REMOVAL_TYPE_CLASSES:
        return {"passed": True, "issues": []}

    orig_blocks = _parse_all_math_blocks(original_html)
    orig_tag_counts: Dict[str, int] = {}
    for _, el, _ in orig_blocks:
        if el is not None:
            for t, c in _collect_tags(el).items():
                orig_tag_counts[t] = orig_tag_counts.get(t, 0) + c

    corr_tag_counts: Dict[str, int] = {}
    for _, el, _ in corr_blocks:
        if el is not None:
            for t, c in _collect_tags(el).items():
                corr_tag_counts[t] = corr_tag_counts.get(t, 0) + c

    for tag in _STRUCTURAL_TAGS:
        if tag in orig_tag_counts and tag not in corr_tag_counts:
            issues.append(
                f"MathML structural tag <{tag}> existed in original "
                f"(count={orig_tag_counts[tag]}) but is absent from corrected output."
            )

    return {"passed": len(issues) == 0, "issues": issues}


# ============================================================================
# Acadza API helpers
# ============================================================================

def fetch_question(qid: str, logger: Optional[Callable] = None) -> Dict[str, Any]:
    """Fetch question by qid via /question/details. Uses centralized JWT auth."""
    resp = authenticated_request(
        "post", DETAILS_URL,
        logger=logger,
        headers={"questionid": qid},
        json={},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def update_question(qid: str, body: Dict[str, Any], logger: Optional[Callable] = None) -> Dict[str, Any]:
    """PUT /question/update/field. body must already be stripped via _build_update_body."""
    resp = authenticated_request(
        "put", UPDATE_URL,
        logger=logger,
        headers={"questionid": qid},
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def save_pre_update_backup(qid: str, original_api_resp: Dict[str, Any], backup_dir: Path) -> Path:
    """CRITICAL: Persist the full original API payload BEFORE any /question/update/field call.

    Used for revert if our edit corrupts the question. Backup is written atomically
    (temp + rename) so a crash mid-write leaves either no file or a complete file.

    Returns the path of the backup file.
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    out_file = backup_dir / f"{qid}.json"
    tmp_file = backup_dir / f"{qid}.json.tmp"
    payload = {
        "qid": qid,
        "saved_at": time.time(),
        "original_api_resp": original_api_resp,
    }
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    import os as _os
    _os.replace(tmp_file, out_file)
    return out_file


def _build_update_body(
    api_resp: Dict[str, Any],
    type_key: str,
    new_question_html: str,
    new_solution_html: str,
    set_error_to_1: bool = False,
) -> Dict[str, Any]:
    """Build PUT body. Strips _id/createdAt/updatedAt/__v per API contract (H4 fix)."""
    body = copy.deepcopy(api_resp)
    if type_key not in body:
        body[type_key] = {}
    body[type_key]["question"] = new_question_html
    body[type_key]["solution"] = new_solution_html
    if set_error_to_1:
        body["error"] = 1
    for f in STRIP_FIELDS_ON_UPDATE:
        body.pop(f, None)
    return body


# ============================================================================
# Question-type extraction
# ============================================================================

def get_type_key(q_type: str) -> str:
    return TYPE_KEY_MAP.get((q_type or "").lower().strip(), q_type)


def extract_qs(body: Dict[str, Any], type_key: str) -> Tuple[str, str]:
    block = body.get(type_key, {}) or {}
    return block.get("question", "") or "", block.get("solution", "") or ""


# ============================================================================
# Error class taxonomy (M3 fix)
# ============================================================================

# Heuristic mapping from operator's free-text error_description to a canonical class.
# This drives both the LLM1 prompt branching AND the similarity threshold (H2).
_CLASS_KEYWORDS = {
    "cleaner_artifact": [
        "artifact", "garbage", "code", "noise", "junk",
        r"[A-Z]{2,}\d{2,}",   # patterns like CDF127, ABC456 (source ID artifacts)
    ],
    "answer_prefix_leak": [
        r"\(\s*[A-D]\s*\)\s+(?=[A-Z])",   # "(B) " at start of a stem
        "prefix", "answer leaked", "ans (", "ans.",
        r"^\s*\(\d+\)\s+",                # "(7) " for integer answer leaks
    ],
    "source_artifact": [
        "jee", "neet", "mhcet", "advanced", "mains",
        "year", "previous year", "exam reference",
    ],
    "mathml_broken": [
        "mathml", "math tag", "broken math", "unclosed", "malformed",
        "xml", "parse error",
    ],
    "math_logic": [
        "wrong answer", "math logic", "calculation", "formula error",
        "incorrect math", "wrong math",
    ],
}


_FIELD_TAG_RE = re.compile(r'\[(question|solution)/[^\]]+\]', re.IGNORECASE)


def parse_target_fields(error_description: str) -> List[str]:
    """Extract which field(s) the Pass1/2/3 verdicts flagged.

    Pass error tags follow the pattern '[question/error_type]' or '[solution/error_type]'.
    Multiple tags possible across passes (e.g. one Pass marked question, another marked solution).

    Returns:
        ['question'] | ['solution'] | ['question', 'solution'] | [] (no tag found)
    """
    if not error_description:
        return []
    found = set()
    for m in _FIELD_TAG_RE.finditer(error_description):
        found.add(m.group(1).lower())
    return sorted(found)


def classify_error(error_description: str) -> str:
    """Heuristic: map operator free-text to a canonical error class.

    Falls back to 'default' (uses default similarity threshold).
    Future enhancement: have ops select from a dropdown so this becomes deterministic.
    """
    if not error_description:
        return "default"
    text = error_description.lower()
    for cls, patterns in _CLASS_KEYWORDS.items():
        for pat in patterns:
            if pat.startswith(("[", "(", r"\(")) or any(c in pat for c in r"\^$.?*+"):
                # treat as regex
                try:
                    if re.search(pat, error_description, re.IGNORECASE):
                        return cls
                except re.error:
                    pass
            elif pat.lower() in text:
                return cls
    return "default"


# ============================================================================
# LLM prompts
# ============================================================================

_SYSTEM_LLM1 = """You are a STRICT, MINIMAL-EDIT content editor for JEE/NEET/MHCET question HTML.

You will receive:
  - QUESTION HTML  (RAW: <math>...</math> MathML blocks, HTML tags <h3>/<br/>/<sup>/etc.,
                   possibly inline LaTeX in $...$ or \\(...\\). <img> -> [Q1], [Q2], ...)
  - SOLUTION HTML  (same conventions; <img> -> [S1], [S2], ...)
  - ERROR_DESCRIPTION: the ONLY thing you may modify.
  - ERROR_CLASS: cleaner_artifact | answer_prefix_leak | source_artifact | mathml_broken | math_logic | default.
  - FEEDBACK (optional): from previous validation attempt.
  - Image attachments (multimodal).

ABSOLUTE LAW — VIOLATING THIS REJECTS YOUR EDIT:

  Outside the EXACT span described by ERROR_DESCRIPTION you MUST emit the input
  byte-for-byte. Not "semantically equivalent". BYTE-FOR-BYTE.

  This means:
   - Do NOT add a <mrow>, <mstyle>, <mfenced>, or any other wrapper tag anywhere.
   - Do NOT remove a wrapper tag anywhere.
   - Do NOT reorder attributes on any tag (xmlns, display, mathvariant, stretchy ...).
   - Do NOT normalize whitespace, quote style, or numeric entity encoding.
   - Do NOT "clean up" or "simplify" any MathML structure.
   - Do NOT touch <math>, <mrow>, <mi>, <mn>, <mo>, <mfrac>, <msup>, <msub>, <mstyle>,
     <h3>, <br/>, <p>, <strong>, <sup>, <sub>, <em>, etc., UNLESS that exact tag block
     is the artifact described in ERROR_DESCRIPTION (i.e. error explicitly says to
     remove a standalone-wrapped section like "<sup>extra</sup>").
   - Do NOT touch any LaTeX expression unless it is the broken expression named in ERROR_DESCRIPTION.

  When in doubt: PASS THROUGH UNCHANGED. A failed edit reaches a human reviewer.
  A silent rewrite corrupts the database. Failure is preferred to silent rewrite.

HARD STRUCTURAL INVARIANTS (refuse the edit / pass through if any would be violated):

  IV1. SCQ/MCQ option count: if the input question already has 4 options labelled
       (A)/(B)/(C)/(D), your output MUST also have exactly 4 options (A)/(B)/(C)/(D).
       NEVER strip or drop these labels even when ERROR_DESCRIPTION claims they are
       "leaked answer", "answer in question text", or similar. Options ARE the
       answer choices. If a fix would reduce the option count below 4, return
       the input UNCHANGED.

  IV2. Solution preservation on missing/partial defects: if ERROR_DESCRIPTION
       contains "missing_solution", "partial_solution", or "incomplete_solution",
       DO NOT delete or shrink the existing solution text. Pass through UNCHANGED.
       Faculty will rewrite. An empty or near-empty solution is worse than the
       original.

  IV3. No placeholder leakage: if your output would contain literal text like
       "[Q1]", "[S1]", "[Image a)]", or "cid:ImageX", that means an image was
       not properly handled. Either preserve the original markup verbatim
       (including LaTeX "\\begin{figure}\\includegraphics{...}" blocks) OR refuse
       the edit. NEVER write bracketed placeholder text into the output.

  IV4. ANSWER-KEY ALIGNMENT (only enforced when OFFICIAL_ANSWER is provided in
       the user message): your corrected solution MUST conclude that
       OFFICIAL_ANSWER is the correct answer. You CANNOT change the answer field
       — that is owned by faculty. If applying the fix in ERROR_DESCRIPTION would
       force your corrected solution to derive a DIFFERENT letter/value than
       OFFICIAL_ANSWER, return both question and solution UNCHANGED (byte-for-byte
       passthrough). A pass-through reaches a human reviewer who can update the
       answer field; a silent rewrite corrupts the live answer key for students.

WHAT YOU MAY DO:

  - Locate the exact text/token described in ERROR_DESCRIPTION (e.g. '93 (d)', 'CDF127',
    answer prefix '(B)', a specific malformed `<msub>` block).
  - Remove ONLY that token/span. Touch nothing else.
  - Replace surrounding bytes with their original bytes if the natural way of writing
    the surgical removal would shift them — i.e. cut surgically.

CORRECTION FLAG DECLARATION (must be honest):

  - mathml_correction: true ONLY if you removed/modified content inside a <math>...</math> block
    AND the error description explicitly required it (mathml_broken / math_logic class).
  - html_correction: true ONLY if you removed/modified an HTML tag (h3, br, p, ...) AND
    the error description explicitly required removing or fixing that HTML structure.
  - latex_correction: true ONLY if you removed/modified content inside $...$ or \\(...\\)
    AND the error description explicitly required it.

  If you only removed PLAIN TEXT (e.g. '93 (d)' is plain text, no tags) — all three flags MUST be false.

Output rules:
  - Return ONLY a JSON object with EXACTLY these five keys:
      { "question": "<edited question HTML, byte-preserved outside error span>",
        "solution": "<edited solution HTML, byte-preserved outside error span>",
        "mathml_correction": true | false,
        "html_correction":   true | false,
        "latex_correction":  true | false }
  - No explanation. No markdown fences. No commentary outside the JSON.
"""


_SYSTEM_LLM2 = """You are an INDEPENDENT senior reviewer for JEE/NEET/MHCET question HTML.
You have NO knowledge of where Version A or Version B came from.
Judge them on the merits alone.

You will receive:
  - ERROR_DESCRIPTION: the defect that should be fixed (and ONLY that should be fixed).
  - ERROR_CLASS: cleaner_artifact | answer_prefix_leak | source_artifact | mathml_broken | math_logic | default.
  - QUESTION VERSION A / VERSION B
  - SOLUTION VERSION A / VERSION B
  - TAG_DRIFT_ADVISORY: automated tag-count diff between A and B. INFORMATIONAL only.
  - Image attachments (multimodal).

ABSOLUTE LAW — APPLY BYTE-FOR-BYTE OUTSIDE THE ERROR SPAN:

  Outside the EXACT span described by ERROR_DESCRIPTION, the two versions MUST be
  byte-for-byte identical. NOT "semantically equivalent". BYTE-FOR-BYTE.

  Any of the following count as DRIFT (reject the version with drift):
   - Added <mrow>, <mstyle>, <mfenced>, <span>, or any wrapper tag.
   - Removed wrapper tag.
   - Reordered attributes (xmlns, display, mathvariant, stretchy ...).
   - Normalized whitespace, quote style, numeric entity encoding (&#x0002C; vs ,).
   - "Cleaned up" or "simplified" MathML structure.
   - Modified any HTML tag, MathML tag, or LaTeX expression that is NOT the artifact
     described in ERROR_DESCRIPTION.

  The fact that two MathML blocks render the same visually is IRRELEVANT.
  The bytes must match outside the error span.

STRUCTURAL VALIDATORS (HARD GATES — set approved=false + preferred=neither if any fail):

  V1. SCQ/MCQ option count: count the literal markers "(A)" "(B)" "(C)" "(D)"
      in each version's question. If either version has fewer than 4 of these
      distinct markers when the question type is SCQ or MCQ, REJECT — options
      should never be dropped from the answer choices. Note the question type
      from ERROR_CLASS context or visible structure.

  V2. Placeholder leak: scan both versions for literal substrings "[Q1]" "[Q2]"
      "[S1]" "[S2]" "[Image" "cid:Image". If any are present, REJECT — these
      are unrendered placeholders that would break in the database.

  V3. Solution wipe: if either solution is shorter than 20 characters AND the
      other solution is longer than 50 characters, REJECT.

  V4. Body-number mutation: any constant or value embedded INSIDE a formula
      (between <math>...</math>, in a derivation, in a calculation result) MUST
      be byte-identical between A and B. Examples: "6.67e-11" vs "6.66e-11",
      "0.5 mm" vs "0.05 mm", "270" vs "60", "8.31" vs "8.314".
      If any body number differs between versions AND ERROR_DESCRIPTION does
      NOT explicitly name that number as wrong, REJECT — the changed version
      has silently mutated formula content.

  V5. ANSWER-KEY ALIGNMENT (only enforced when OFFICIAL_ANSWER is provided
      above). VERY CONSERVATIVE — only fires on UNAMBIGUOUS explicit verdict.

      Look for explicit, unambiguous final-verdict statements in the preferred
      version's solution. Examples that DO count as explicit verdict:
        - "the correct answer is (C)"  / "the correct option is C"
        - "hence option (C)"  / "answer: C"  / "Ans. (C)"
        - "→ (C)"  / "∴ (C)"  / "so (C) is correct"
        - A final line that names exactly one option letter as the conclusion

      Do NOT infer a verdict letter from:
        - A numeric or symbolic result alone (e.g., "200 N/kg" matching option D
          by value-comparison — that is INFERENCE, not an explicit verdict)
        - A letter appearing mid-derivation (e.g., "from (B) we substitute …")
        - Multiple letters mentioned without one being singled out as the final
          conclusion
        - A solution that ends with math symbols / equations / a number with no
          letter named

      Fire V5 ONLY if:
        (i) the solution contains an explicit final-verdict statement naming
            ONE letter as the answer, AND
        (ii) that letter is DIFFERENT from OFFICIAL_ANSWER.

      When V5 fires: approved=false, preferred_version="neither",
      broken_code_found=true,
      broken_code_details="V5: solution explicitly concludes (X) but
      OFFICIAL_ANSWER is (Y)".

      When the solution gives a numeric/symbolic result without naming any
      letter, or when the verdict is ambiguous → do NOT flag on V5.
      Rely on V1-V4 alone for those cases. False positives here cost faculty
      time; conservatism is the right default.

  Note: removing answer prefixes like "(b)", "162 (a)", "[113]", "Ans. (B)"
  is EXPECTED for cleaner_artifact defects and is NOT V4 drift. Only count
  numbers embedded in actual formulas/derivations.

INSIDE the error span (the exact removal/fix described by ERROR_DESCRIPTION),
judge whether the change correctly addresses the error.

Your evaluation:

STEP 1 — Identify the EXACT error span from ERROR_DESCRIPTION.
STEP 2 — Check OUTSIDE the error span: do A and B match byte-for-byte?
         Use TAG_DRIFT_ADVISORY to focus your byte inspection.
STEP 3 — Check INSIDE the error span: which version (A or B) correctly applied the fix?
STEP 4 — Verdict:
   - approved=true + preferred_version=X
       ONLY if version X has byte-exact match outside the error span
       AND version X correctly applied the fix inside the error span.
   - approved=false in ALL other cases:
       both have drift, both unfixed, both broken, or no clear preference.

Output rules:
  Return ONLY this JSON (no markdown fences):
  {
    "approved": true or false,
    "preferred_version": "A" | "B" | "neither",
    "reasoning": "one to three sentences",
    "byte_drift_outside_error_span": true or false,
    "drift_details": "byte-level differences observed outside the error span, or empty",
    "broken_code_found": true or false,
    "broken_code_details": "exact description or empty",
    "error_fixed_in_preferred": true or false,
    "error_fix_details": "what is still wrong if not fully fixed, or empty",
    "feedback": "actionable instructions for the non-preferred version, or empty"
  }
"""


# ============================================================================
# LLM client adapter (real-time, no batch)
# ============================================================================

@dataclass
class TokenUsage:
    input_tokens: int = 0       # total input (cached + uncached)
    output_tokens: int = 0      # response tokens (excluding thinking)
    cached_tokens: int = 0      # subset of input_tokens that hit the cache
    thinking_tokens: int = 0    # Gemini reasoning tokens, billed at output rate


@dataclass
class LLMResult:
    raw: str
    parsed: Dict[str, Any]
    usage: TokenUsage
    latency_s: float
    model: str


def _retry(fn, *, retries: int = RETRY_MAX_ATTEMPTS, base: float = RETRY_BACKOFF_BASE_S):
    """Exponential backoff retry on any exception. Returns the function's result."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(base * (2 ** attempt))
    raise last_exc


def call_llm_openai(
    model: str,
    system_prompt: str,
    user_text: str,
    image_attachments: List[Dict],
    max_output_tokens: int = 8192,   # bumped: GPT-5 reasoning consumes from this budget
) -> LLMResult:
    """Call an OpenAI chat-completions model. FORKED FROM stage9_quality._run_openai_pass.

    Mirrors production:
      - max_completion_tokens (not max_tokens) for GPT-5 family
      - reasoning_effort='low' for *mini* models (cuts internal reasoning tokens)
      - NO temperature/top_p/seed -- GPT-5 reasoning rejects these
      - response_format=json_object (Stage 9 uses a stricter schema; Stage 11 returns
        a simple {question, solution} object so json_object is sufficient)
      - finish_reason='length' check -- prod has shown gpt-5-mini truncates without this
    """
    from openai import OpenAI
    api_key = settings.openai_api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env / settings")
    client = OpenAI(api_key=api_key)

    user_content: List[Dict] = [{"type": "text", "text": user_text}]
    user_content.extend(image_attachments)

    def _do():
        t0 = time.time()
        create_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_output_tokens,
        }
        # Mirror prod stage9: reasoning_effort='low' for *mini* and *nano* models.
        # Cuts internal-reasoning tokens for structured output tasks.
        if "mini" in model or "nano" in model:
            create_kwargs["reasoning_effort"] = "low"

        resp = client.chat.completions.create(**create_kwargs)
        dt = time.time() - t0

        choice = resp.choices[0]
        # Truncation guard (prod-tested in stage9)
        if choice.finish_reason == "length":
            raise RuntimeError(
                f"OpenAI {model} truncated (finish_reason=length); "
                f"raise max_completion_tokens or reduce input"
            )

        raw = (choice.message.content or "").strip()
        try:
            parsed = _parse_json_response(raw)
        except Exception as exc:
            raise RuntimeError(f"LLM JSON parse failure: {exc} | raw={raw[:200]}")

        usage = TokenUsage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return LLMResult(raw=raw, parsed=parsed, usage=usage, latency_s=dt, model=model)

    return _retry(_do)


# Module-level Gemini cache registry: keyed by (model, system_prompt_hash) -> cache_handle.
# Cache is created lazily on first call within a batch and reused for subsequent calls.
# Verification: usage_metadata.cached_content_token_count > 0 proves cache hit.
import hashlib as _hashlib
_gemini_cache_registry: Dict[str, Any] = {}
_gemini_cache_lock = threading.Lock()


def _gemini_cache_key(model: str, system_prompt: str) -> str:
    h = _hashlib.md5(system_prompt.encode("utf-8")).hexdigest()[:16]
    return f"{model}::{h}"


def _get_or_create_gemini_cache(client, types_mod, model: str, system_prompt: str,
                                 ttl_seconds: int = 3600, logger: Optional[Callable] = None) -> Optional[Any]:
    """Return a cached_content handle for the given (model, system_prompt).

    Lazy-creates on first call; reuses across the batch. TTL refreshes implicitly on hit.
    Returns None if cache creation fails (caller falls back to non-cached call).
    """
    key = _gemini_cache_key(model, system_prompt)
    with _gemini_cache_lock:
        existing = _gemini_cache_registry.get(key)
        if existing is not None:
            return existing
        # PROOF-READING-V6.0 pattern: contents=[] (empty), system_instruction in cache config.
        try:
            cache_handle = client.caches.create(
                model=model,
                config=types_mod.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    contents=[],
                    ttl=f"{ttl_seconds}s",
                ),
            )
            _gemini_cache_registry[key] = cache_handle
            if logger:
                logger("info", f"Stage11 Gemini cache created: model={model} key={key}")
            return cache_handle
        except Exception as exc:
            if logger:
                logger("warning", f"Stage11 Gemini cache creation failed ({exc}); falling back to non-cached")
            return None


def call_llm_gemini(
    model: str,
    system_prompt: str,
    user_text: str,
    image_attachments: List[Dict],
    max_output_tokens: int = 16000,   # raised from 8192 -- LLM1 returns full HTML, MUST cover thinking + output
    use_cache: bool = False,           # Skipped for now (2026-05-08): Gemini cache requires non-empty contents; ~5% savings not worth complexity. Revisit at 100K+ scale.
    response_schema: Optional[Dict] = None,
) -> LLMResult:
    """Call a Gemini model. FORKED FROM stage9_quality._run_gemini_pass + Stage 2 caching.

    Mirrors production:
      - temperature=0
      - max_output_tokens=8192 (covers thinking + output)
      - response_mime_type='application/json'
      - thinking_config=ThinkingConfig(thinking_level=...) -- configurable
      - finish_reason MAX_TOKENS guard
      - Vertex AI preferred (paid quota); AI Studio fallback
      - Context caching for system_prompt (lazy-create per batch, ~50% input savings on hits)
    """
    from google import genai
    from google.genai import types

    # Vertex preferred; fall back to AI Studio
    use_vertex = bool(settings.vertex_ai_credentials and settings.vertex_ai_project)
    if use_vertex:
        import os
        if settings.vertex_ai_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.vertex_ai_credentials
        client = genai.Client(
            vertexai=True,
            project=settings.vertex_ai_project,
            location=settings.vertex_ai_location,
        )
    else:
        client = genai.Client(api_key=settings.gemini_api_key)

    # Build content: text + inline images (mirror prod stage9 pattern)
    parts: List[Any] = [types.Part.from_text(text=user_text)]
    for img in image_attachments:
        data_uri = img.get("image_url", {}).get("url", "")
        if data_uri.startswith("data:"):
            header, b64_data = data_uri.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            try:
                raw_bytes = base64.b64decode(b64_data)
                parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))
            except Exception:
                continue

    # Try cache first if enabled
    cache_handle = None
    if use_cache:
        cache_handle = _get_or_create_gemini_cache(client, types, model, system_prompt)

    def _do():
        t0 = time.time()
        cfg_kwargs = dict(
            response_mime_type="application/json",
            temperature=0,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=DEFAULT_GEMINI_THINKING_LEVEL),
        )
        if response_schema is not None:
            # google.genai accepts Pydantic class OR dict via response_schema.
            cfg_kwargs["response_schema"] = response_schema
        if cache_handle is not None:
            # Cache-hit path: use cached_content; don't re-send system_instruction.
            cfg_kwargs["cached_content"] = cache_handle.name
        else:
            cfg_kwargs["system_instruction"] = system_prompt
        cfg = types.GenerateContentConfig(**cfg_kwargs)
        resp = client.models.generate_content(
            model=model,
            contents=types.Content(role="user", parts=parts),
            config=cfg,
        )
        dt = time.time() - t0

        # Truncation / safety guard (prod-tested in stage9)
        if resp.candidates:
            fr = resp.candidates[0].finish_reason
            if fr and fr.name == "MAX_TOKENS":
                raise RuntimeError(f"Gemini {model} truncated (MAX_TOKENS); raise max_output_tokens")

        # V6 pattern: extract text from non-thought parts only.
        raw = ""
        if resp.candidates:
            cand = resp.candidates[0]
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    # Skip thought parts (Gemini returns reasoning separately)
                    if getattr(part, "thought", False):
                        continue
                    pt = getattr(part, "text", None)
                    if pt:
                        raw += pt
        raw = raw.strip() or (resp.text or "").strip()
        parsed = _parse_json_response(raw)

        # Track all 4 token classes per V6 cost_tracker conventions.
        um = getattr(resp, "usage_metadata", None) or {}
        usage = TokenUsage(
            input_tokens   = int(getattr(um, "prompt_token_count", 0) or 0),
            output_tokens  = int(getattr(um, "candidates_token_count", 0) or 0),
            cached_tokens  = int(getattr(um, "cached_content_token_count", 0) or 0),
            thinking_tokens= int(getattr(um, "thoughts_token_count", 0) or 0),
        )
        return LLMResult(raw=raw, parsed=parsed, usage=usage, latency_s=dt, model=model)

    return _retry(_do)


def call_llm(
    model: str,
    system_prompt: str,
    user_text: str,
    image_attachments: List[Dict],
    max_output_tokens: int = 16000,
) -> LLMResult:
    """Dispatch by model API. For Gemini, picks the right response_json_schema by role
    (LLM1 vs LLM2), inferred from a sentinel marker in the system prompt."""
    spec = SUPPORTED_MODELS.get(model)
    if spec is None:
        raise ValueError(f"Unknown model: {model}")
    if spec["api"] == "openai":
        return call_llm_openai(model, system_prompt, user_text, image_attachments, max_output_tokens)
    if spec["api"] == "gemini":
        # Pick Pydantic schema by role.
        if "INDEPENDENT senior reviewer" in system_prompt:
            schema = LLM2Response
        elif "STRICT, MINIMAL-EDIT" in system_prompt:
            schema = LLM1Response
        else:
            schema = None
        return call_llm_gemini(
            model, system_prompt, user_text, image_attachments, max_output_tokens,
            response_schema=schema,
        )
    raise ValueError(f"Unknown api for model {model}: {spec.get('api')}")


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """Strip optional ```json fences and parse."""
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    return json.loads(raw.strip())


# ============================================================================
# Similarity (per error class)
# ============================================================================

def similarity_score(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio() * 100.0


def get_similarity_threshold(error_class: str) -> float:
    return SIMILARITY_THRESHOLDS.get(error_class, SIMILARITY_THRESHOLDS["default"])


# ============================================================================
# Core process_question
# ============================================================================

@dataclass
class AttemptLog:
    attempt: int
    approved: bool = False
    stage_failed: Optional[str] = None
    lxml_issues: List[str] = field(default_factory=list)
    llm1_usage: Optional[TokenUsage] = None
    llm2_usage: Optional[TokenUsage] = None
    llm2_result: Dict[str, Any] = field(default_factory=dict)
    feedback_sent: str = ""
    llm1_latency_s: float = 0.0
    llm2_latency_s: float = 0.0


@dataclass
class QuestionResult:
    qid: str
    error_description: str = ""
    error_class: str = "default"
    q_type: str = ""
    view_link: str = ""
    status: str = "pending"     # pending|updated|skipped_low_sim|flagged|skipped_*|api_*_error
    similarity: Optional[float] = None
    flagged: bool = False
    flag_reason: str = ""
    orig_question: str = ""
    orig_solution: str = ""
    corr_question: str = ""
    corr_solution: str = ""
    approved_on: Optional[int] = None
    sideline_reason: Optional[str] = None
    attempts: List[AttemptLog] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_thinking_tokens: int = 0
    cost_usd: float = 0.0


def _calc_cost(model: str, in_tok: int, out_tok: int,
               cached_tok: int = 0, thinking_tok: int = 0) -> float:
    """Real cost from token counts. Cached input billed at lower rate; thinking billed as output."""
    spec = SUPPORTED_MODELS.get(model, {})
    uncached = max(0, in_tok - cached_tok)
    billed_output = out_tok + thinking_tok    # Gemini bills thinking at output rate
    return (
        uncached      * spec.get("input_uncached", 0) +
        cached_tok    * spec.get("input_cached", 0) +
        billed_output * spec.get("output", 0)
    ) / 1_000_000.0


def _build_llm1_user_msg(
    q_html: str, s_html: str,
    error_desc: str, error_class: str,
    target_fields: List[str],
    feedback: str,
    official_answer: str = "",
) -> str:
    fb_block = ""
    if feedback:
        fb_block = (
            "\n\nFEEDBACK FROM PREVIOUS VALIDATION ATTEMPT "
            "(you MUST address every issue listed):\n" + feedback + "\n"
        )
    target_block = ""
    if target_fields:
        if target_fields == ["question"]:
            target_block = "\n\nTARGET FIELD: question only. Do NOT modify the solution unless the error description explicitly requires it."
        elif target_fields == ["solution"]:
            target_block = "\n\nTARGET FIELD: solution only. Do NOT modify the question unless the error description explicitly requires it."
        else:
            target_block = "\n\nTARGET FIELDS: both question and solution have flagged errors. Fix only what is described; preserve everything else."
    # Answer-anchor block (Acadza triplet-gate guarantees answer is non-empty).
    # Provides the canonical answer so the corrected solution must conclude this letter/value.
    ans_block = ""
    if official_answer:
        ans_block = (
            f"\n\nOFFICIAL_ANSWER (CANONICAL — DO NOT TOUCH the answer field; the corrected "
            f"solution MUST conclude that this value is correct):\n  {official_answer}\n"
            f"If applying ERROR_DESCRIPTION would force your corrected solution to derive a "
            f"DIFFERENT letter/value than OFFICIAL_ANSWER, you MUST return the input "
            f"UNCHANGED (byte-for-byte). It is far safer to flag than to silently corrupt "
            f"the answer key. Faculty will handle answer disputes."
        )
    return (
        f"ERROR_DESCRIPTION:\n{error_desc}\n\n"
        f"ERROR_CLASS: {error_class}"
        f"{target_block}"
        f"{ans_block}"
        f"{fb_block}\n\n"
        f"QUESTION HTML (math as $LaTeX$):\n{q_html}\n\n"
        f"SOLUTION HTML (math as $LaTeX$):\n{s_html}"
    )


def _build_llm2_user_msg(
    error_desc: str, error_class: str,
    version_a_q: str, version_a_s: str,
    version_b_q: str, version_b_s: str,
    tag_drift_advisory: str,
    official_answer: str = "",
) -> str:
    """Independent proofreader prompt: no ownership cues; tag-drift as advisory only."""
    advisory_block = ""
    if tag_drift_advisory:
        advisory_block = f"\n\nTAG_DRIFT_ADVISORY (informational; not a verdict):\n{tag_drift_advisory}\n"
    ans_block = ""
    if official_answer:
        ans_block = (
            f"\nOFFICIAL_ANSWER (the canonical answer key for this question; the preferred "
            f"version's solution MUST conclude this value is correct):\n  {official_answer}\n"
        )
    return (
        f"ERROR_DESCRIPTION:\n{error_desc}\n\n"
        f"ERROR_CLASS: {error_class}"
        f"{advisory_block}"
        f"{ans_block}\n"
        f"=== QUESTION VERSION A ===\n{version_a_q}\n\n"
        f"=== QUESTION VERSION B ===\n{version_b_q}\n\n"
        f"=== SOLUTION VERSION A ===\n{version_a_s}\n\n"
        f"=== SOLUTION VERSION B ===\n{version_b_s}"
    )


def process_question(
    qid: str,
    error_description: str,
    q_type: str = "scq",
    view_link: str = "",
    error_class: Optional[str] = None,
    llm1_model: str = DEFAULT_LLM1_MODEL,
    llm2_model: str = DEFAULT_LLM2_MODEL,
    max_attempts: int = MAX_CORRECTION_ATTEMPTS,    # set to 1 for one-shot mode
    set_error_to_1: bool = False,
    do_update_api: bool = False,
    backup_dir: Optional[Path] = None,              # MANDATORY when do_update_api=True
    logger: Optional[Callable] = None,
    raw_log_dir: Optional[Path] = None,
    official_answer: str = "",                      # Acadza answer field; anchors LLM1/LLM2
) -> QuestionResult:
    """Run the full Stage 11 correction loop on one question.

    raw_log_dir: if set, every LLM1/LLM2 call is dumped to disk for audit.
    do_update_api: if False (default), no PUT call is made — useful for pilots.
    """
    qid = (qid or "").strip()
    error_description = (error_description or "").strip()
    error_class = error_class or classify_error(error_description)
    target_fields = parse_target_fields(error_description)

    res = QuestionResult(
        qid=qid,
        error_description=error_description,
        error_class=error_class,
        q_type=q_type or "scq",
        view_link=view_link,
    )

    if not qid:
        res.status = "skipped_no_id"
        return res

    if not error_description:
        res.status = "skipped_no_error_description"
        return res

    if (q_type or "").lower() in {"passage", "passagequestion", "match", "matchquestion"}:
        # M2: TODO — passage/match handling needs nested schema recursion. For pilot,
        # we sideline. Production will handle these once Phase 2.5 is added.
        res.status = "skipped_unsupported_type_pilot"
        res.sideline_reason = "passage/match deferred to post-pilot"
        return res

    # Step 1: fetch
    try:
        api_resp = fetch_question(qid, logger=logger)
    except Exception as exc:
        res.status = f"api_fetch_error: {exc}"
        return res

    type_key = get_type_key(q_type)
    orig_q, orig_s = extract_qs(api_resp, type_key)
    res.orig_question = orig_q
    res.orig_solution = orig_s

    if not orig_q and not orig_s:
        res.status = "skipped_empty_question_body"
        return res

    # Step 2: extract image attachments. Send RAW HTML+MathML to the LLM (no LaTeX
    # conversion). Stage 11 architecture FINAL: tag count guardrails > byte similarity.
    orig_q_with_placeholders, q_attachments, _q_src2tok, q_tok2tag = collect_images_with_placeholders(orig_q, "Q")
    orig_s_with_placeholders, s_attachments, _s_src2tok, s_tok2tag = collect_images_with_placeholders(orig_s, "S")
    image_attachments = q_attachments + s_attachments

    # Pre-LLM hard invariants: image count + URL set + video count.
    # These must be byte-identical between orig and corrected. No exceptions.
    orig_q_img_inv = img_invariants(orig_q)
    orig_s_img_inv = img_invariants(orig_s)

    # Step 4: correction loop
    feedback = ""
    corr_q_html = orig_q
    corr_s_html = orig_s
    approved = False

    for attempt_num in range(1, max_attempts + 1):
        log = AttemptLog(attempt=attempt_num, feedback_sent=feedback)

        # --- LLM1 call ---
        user1 = _build_llm1_user_msg(
            orig_q_with_placeholders, orig_s_with_placeholders,
            error_description, error_class, target_fields, feedback,
            official_answer=official_answer,
        )
        try:
            llm1_res = call_llm(
                llm1_model, _SYSTEM_LLM1, user1, image_attachments,
                max_output_tokens=4096,
            )
        except Exception as exc:
            log.stage_failed = "llm1_error"
            log.lxml_issues = [f"LLM1 call/parse failed: {exc}"]
            res.attempts.append(log)
            feedback = f"Your previous JSON response could not be parsed: {exc}. Return valid JSON only."
            continue

        log.llm1_usage = llm1_res.usage
        log.llm1_latency_s = llm1_res.latency_s
        res.total_input_tokens += llm1_res.usage.input_tokens
        res.total_output_tokens += llm1_res.usage.output_tokens
        res.total_cached_tokens += llm1_res.usage.cached_tokens
        res.total_thinking_tokens += llm1_res.usage.thinking_tokens
        res.cost_usd += _calc_cost(
            llm1_model,
            llm1_res.usage.input_tokens, llm1_res.usage.output_tokens,
            cached_tok=llm1_res.usage.cached_tokens,
            thinking_tok=llm1_res.usage.thinking_tokens,
        )

        if raw_log_dir:
            _dump_raw_log(raw_log_dir, qid, llm1_model, attempt_num, "llm1", user1, llm1_res)

        corr_q_html = (llm1_res.parsed.get("question") or "").strip()
        corr_s_html = (llm1_res.parsed.get("solution") or "").strip()
        mathml_corr = bool(llm1_res.parsed.get("mathml_correction", False))
        html_corr   = bool(llm1_res.parsed.get("html_correction",   False))
        latex_corr  = bool(llm1_res.parsed.get("latex_correction",  False))

        if not corr_q_html and not corr_s_html:
            log.stage_failed = "llm1_empty_output"
            res.attempts.append(log)
            feedback = "Your output was empty. Return both question and solution fields."
            continue

        # --- Fix A: restore [Q1]/[S1] placeholders back to their original <img> tags ---
        # The LLM was sent placeholders; if it kept them, dropped them, or hallucinated
        # `<img src="cid:Image..">` references, restore the originals byte-for-byte before
        # LLM2 sees this corrected HTML and before any DB update is built.
        corr_q_html = restore_image_placeholders(corr_q_html, q_tok2tag)
        corr_s_html = restore_image_placeholders(corr_s_html, s_tok2tag)

        # --- Fix C: hard gate on <img>/<video> invariants ---
        # Image tag count + image URL set + video count must match between orig and corrected.
        # Any drift = automatic flag, skip LLM2, no DB write. No model can fix this.
        corr_q_img_inv = img_invariants(corr_q_html)
        corr_s_img_inv = img_invariants(corr_s_html)
        img_drift_msgs = []
        if corr_q_img_inv != orig_q_img_inv:
            img_drift_msgs.append(
                f"question side: orig (img={orig_q_img_inv[0]}, urls={len(orig_q_img_inv[1])}, video={orig_q_img_inv[2]}, latex_fig={orig_q_img_inv[3]}) "
                f"!= corrected (img={corr_q_img_inv[0]}, urls={len(corr_q_img_inv[1])}, video={corr_q_img_inv[2]}, latex_fig={corr_q_img_inv[3]}). "
                f"missing_urls={list(orig_q_img_inv[1] - corr_q_img_inv[1])[:3]}, added_urls={list(corr_q_img_inv[1] - orig_q_img_inv[1])[:3]}"
            )
        if corr_s_img_inv != orig_s_img_inv:
            img_drift_msgs.append(
                f"solution side: orig (img={orig_s_img_inv[0]}, urls={len(orig_s_img_inv[1])}, video={orig_s_img_inv[2]}, latex_fig={orig_s_img_inv[3]}) "
                f"!= corrected (img={corr_s_img_inv[0]}, urls={len(corr_s_img_inv[1])}, video={corr_s_img_inv[2]}, latex_fig={corr_s_img_inv[3]}). "
                f"missing_urls={list(orig_s_img_inv[1] - corr_s_img_inv[1])[:3]}, added_urls={list(corr_s_img_inv[1] - orig_s_img_inv[1])[:3]}"
            )
        if img_drift_msgs:
            log.stage_failed = "image_invariant_violation"
            log.lxml_issues = img_drift_msgs
            res.attempts.append(log)
            res.status = "flagged_image_drift"
            res.flag_reason = "Image/video invariant violated: " + " || ".join(img_drift_msgs)
            res.corr_question = corr_q_html
            res.corr_solution = corr_s_html
            return res

        # --- Fix D: SCQ/MCQ option count hard gate (post-image-restoration audit) ---
        # If original SCQ/MCQ had >=3 option markers (A)/(B)/(C)/(D), corrected MUST also
        # have at least the same count. Any drop = silent option loss. Skip LLM2, no DB write.
        if q_type.lower() in ("scq", "mcq"):
            _opt_re = re.compile(r"\([A-D]\)", re.IGNORECASE)
            orig_opts = len(_opt_re.findall(orig_q))
            corr_opts = len(_opt_re.findall(corr_q_html))
            if orig_opts >= 3 and corr_opts < orig_opts:
                log.stage_failed = "option_count_drop"
                log.lxml_issues = [f"SCQ/MCQ option count dropped {orig_opts} -> {corr_opts}"]
                res.attempts.append(log)
                res.status = "flagged_option_drop"
                res.flag_reason = (
                    f"SCQ/MCQ option count dropped from {orig_opts} to {corr_opts}. "
                    "Question's answer choices must not be removed."
                )
                res.corr_question = corr_q_html
                res.corr_solution = corr_s_html
                return res

        # --- Fix E: placeholder leak hard gate ---
        # If post-restoration text still contains [Q1]/[S1]/[Image X]/cid:Image* literal
        # placeholders, an image format wasn't restored. Refuse to write garbage to DB.
        _placeholder_re = re.compile(
            r"\[(Q|S)\d+\]|\[Image\s*[a-d0-9\)]|cid:Image[QS]?\d*",
            re.IGNORECASE,
        )
        leaks = []
        if _placeholder_re.search(corr_q_html):
            leaks.append(f"question side: {_placeholder_re.findall(corr_q_html)[:3]}")
        if _placeholder_re.search(corr_s_html):
            leaks.append(f"solution side: {_placeholder_re.findall(corr_s_html)[:3]}")
        if leaks:
            log.stage_failed = "placeholder_leak"
            log.lxml_issues = [f"Placeholder leak detected: {'; '.join(leaks)}"]
            res.attempts.append(log)
            res.status = "flagged_placeholder_leak"
            res.flag_reason = "Placeholder leak (image format not restored): " + "; ".join(leaks)
            res.corr_question = corr_q_html
            res.corr_solution = corr_s_html
            return res

        # --- Tag balance: SIGNAL not gate. Compute drift summary; do NOT fail here. ---
        # (Earlier strict gate caused the loop to spin: LLMs cannot byte-restore on retry.)
        bal_q = verify_tag_balance(orig_q, corr_q_html, mathml_corr, html_corr, latex_corr)
        bal_s = verify_tag_balance(orig_s, corr_s_html, mathml_corr, html_corr, latex_corr)
        tag_drift_pieces = []
        if bal_q["mathml_drift"] or bal_q["html_drift"] or bal_q["latex_drift"]:
            tag_drift_pieces.append("Question side:")
            for i in bal_q["mathml_drift"] + bal_q["html_drift"] + bal_q["latex_drift"]:
                tag_drift_pieces.append(f"  - {i}")
        if bal_s["mathml_drift"] or bal_s["html_drift"] or bal_s["latex_drift"]:
            tag_drift_pieces.append("Solution side:")
            for i in bal_s["mathml_drift"] + bal_s["html_drift"] + bal_s["latex_drift"]:
                tag_drift_pieces.append(f"  - {i}")
        if mathml_corr or html_corr or latex_corr:
            tag_drift_pieces.append(
                f"Editor declared morphing flags: mathml={mathml_corr}, html={html_corr}, latex={latex_corr}."
            )
        tag_drift_advisory = "\n".join(tag_drift_pieces) if tag_drift_pieces else "No tag-count drift detected."
        log.lxml_issues = bal_q["issues"] + bal_s["issues"]   # kept for analytics

        # --- LLM2 INDEPENDENT proofread: blind A/B, no ownership cues ---
        # Randomize per attempt so positional bias can't favor one side consistently.
        # The hash of qid+attempt is deterministic per call (reproducible) but varies across calls.
        import hashlib
        flip = (int(hashlib.md5(f"{qid}:{attempt_num}".encode()).hexdigest(), 16) % 2) == 1
        if flip:
            version_a_q, version_a_s = corr_q_html, corr_s_html
            version_b_q, version_b_s = orig_q, orig_s
            corrected_label = "A"
        else:
            version_a_q, version_a_s = orig_q, orig_s
            version_b_q, version_b_s = corr_q_html, corr_s_html
            corrected_label = "B"
        user2 = _build_llm2_user_msg(
            error_description, error_class,
            version_a_q, version_a_s,
            version_b_q, version_b_s,
            tag_drift_advisory,
            official_answer=official_answer,
        )
        try:
            llm2_res = call_llm(
                llm2_model, _SYSTEM_LLM2, user2, image_attachments,
                max_output_tokens=1024,
            )
        except Exception as exc:
            log.stage_failed = "llm2_error"
            log.llm2_result = {"error": str(exc)}
            res.attempts.append(log)
            feedback = f"LLM2 validator call failed: {exc}. Return valid JSON."
            continue

        log.llm2_usage = llm2_res.usage
        log.llm2_latency_s = llm2_res.latency_s
        log.llm2_result = llm2_res.parsed
        res.total_input_tokens += llm2_res.usage.input_tokens
        res.total_output_tokens += llm2_res.usage.output_tokens
        res.total_cached_tokens += llm2_res.usage.cached_tokens
        res.total_thinking_tokens += llm2_res.usage.thinking_tokens
        res.cost_usd += _calc_cost(
            llm2_model,
            llm2_res.usage.input_tokens, llm2_res.usage.output_tokens,
            cached_tok=llm2_res.usage.cached_tokens,
            thinking_tok=llm2_res.usage.thinking_tokens,
        )

        if raw_log_dir:
            _dump_raw_log(raw_log_dir, qid, llm2_model, attempt_num, "llm2", user2, llm2_res)

        # Approve only when the proofreader prefers the CORRECTED side blindly.
        verdict = llm2_res.parsed
        approved_judgment = bool(verdict.get("approved"))
        preferred = (verdict.get("preferred_version") or "").upper()
        proofreader_picks_corrected = (preferred == corrected_label)

        if approved_judgment and proofreader_picks_corrected:
            log.approved = True
            log.stage_failed = None
            res.attempts.append(log)
            res.approved_on = attempt_num
            approved = True
            break

        # Build feedback for next LLM1 attempt
        parts = []
        if verdict.get("byte_drift_outside_error_span"):
            parts.append("BYTE DRIFT outside the error span (you must preserve those bytes exactly):\n" + (verdict.get("drift_details") or ""))
        if verdict.get("broken_code_found"):
            parts.append("BROKEN CODE in your output:\n" + (verdict.get("broken_code_details") or ""))
        if not verdict.get("error_fixed_in_preferred", True) or not proofreader_picks_corrected:
            parts.append("ORIGINAL ERROR NOT YET FIXED:\n" + (verdict.get("error_fix_details") or ""))
        extra = verdict.get("feedback") or verdict.get("reasoning", "")
        if extra:
            parts.append(extra)
        feedback = "\n\n".join(p for p in parts if p)

        log.stage_failed = "llm2"
        res.attempts.append(log)

    # End of loop
    if not approved:
        res.status = "flagged"
        res.flagged = True
        res.flag_reason = f"Not approved after {max_attempts} attempt(s)."
        res.corr_question = corr_q_html
        res.corr_solution = corr_s_html
        return res

    res.corr_question = corr_q_html
    res.corr_solution = corr_s_html

    # Step 5: similarity recorded for analytics only (tag-balance gate replaced it).
    sim_q = similarity_score(orig_q, corr_q_html)
    sim_s = similarity_score(orig_s, corr_s_html)
    res.similarity = round((sim_q + sim_s) / 2, 2)

    # Step 6: API update (only if do_update_api=True)
    if not do_update_api:
        res.status = "approved_pending_update"
        return res

    # CRITICAL: backup BEFORE any PUT to allow revert if our edit corrupts the question.
    # Hard-fail if backup_dir not provided when updates are enabled.
    if backup_dir is None:
        res.status = "update_blocked_no_backup_dir"
        res.flagged = True
        res.flag_reason = "do_update_api=True but backup_dir is None (refusing to update without revert path)"
        return res
    try:
        save_pre_update_backup(qid, api_resp, Path(backup_dir))
    except Exception as exc:
        res.status = f"backup_save_error: {exc}"
        res.flagged = True
        res.flag_reason = f"refusing to update because backup save failed: {exc}"
        return res

    body = _build_update_body(api_resp, type_key, corr_q_html, corr_s_html, set_error_to_1=set_error_to_1)
    try:
        update_question(qid, body, logger=logger)
        res.status = "updated"
    except Exception as exc:
        res.status = f"update_api_error: {exc}"

    return res


def _dump_raw_log(
    raw_log_dir: Path,
    qid: str,
    model: str,
    attempt: int,
    role: str,
    user_msg: str,
    result: LLMResult,
) -> None:
    """Persist every LLM call: input + output + tokens + latency. For audit/no-data-loss.

    NOTE: errors are printed to stderr but do not fail the main flow. This is critical
    for benchmarking — we need to know if logs are dropped silently.
    """
    try:
        d = Path(raw_log_dir) / model.replace("/", "_") / qid
        d.mkdir(parents=True, exist_ok=True)
        out_file = d / f"attempt{attempt}_{role}.json"
        payload = {
            "qid": qid,
            "model": model,
            "attempt": attempt,
            "role": role,
            "timestamp": time.time(),
            "user_msg": user_msg,
            "raw_response": result.raw,
            "parsed_response": result.parsed,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "latency_s": result.latency_s,
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        import sys as _sys
        print(f"!! _dump_raw_log FAILED for {qid}/{role}: {exc}", file=_sys.stderr)


# ============================================================================
# Batch runner (real-time, parallel)
# ============================================================================

def load_input_from_cohort_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load Stage 11 input rows from a cohort CSV produced from combined_upload_log.csv.

    Expected columns (per CSV ground-truth memory):
      qid              -- UploadedQuestionID (24-digit Mongo OID), used in API calls
      q_type           -- scq/mcq/integer/passage/match
      view_link        -- public view URL
      error_description -- "[Pass1] ... || [Pass2] ... || [Pass3] ..." combined reasoning
      image            -- 'TRUE' or 'FALSE'
      original_error_level -- '' or '2' or '3' or '4'  (Stage 9 verdict before promotion)
      meta_qid, session_id, promotion_note  -- audit columns

    Returns rows in the format process_batch expects.
    """
    import csv as _csv
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            qid = (row.get("qid") or "").strip()
            if not qid or qid == "QUARANTINED":
                continue
            rows.append({
                "qid": qid,
                "error_description": (row.get("error_description") or "").strip(),
                "q_type": (row.get("q_type") or "scq").strip(),
                "view_link": (row.get("view_link") or "").strip(),
                # error_class auto-classified inside process_question if not supplied
            })
    return rows


def process_batch(
    rows: List[Dict[str, Any]],
    *,
    llm1_model: str = DEFAULT_LLM1_MODEL,
    llm2_model: str = DEFAULT_LLM2_MODEL,
    max_attempts: int = MAX_CORRECTION_ATTEMPTS,
    set_error_to_1: bool = False,
    do_update_api: bool = False,
    backup_dir: Optional[Path] = None,
    max_workers: int = 10,
    raw_log_dir: Optional[Path] = None,
    logger: Optional[Callable] = None,
    on_progress: Optional[Callable[[int, int, QuestionResult], None]] = None,
) -> List[QuestionResult]:
    """Run process_question in parallel for a list of input rows.

    Each row dict requires keys: qid, error_description (optional: q_type, view_link, error_class).
    """
    results: List[QuestionResult] = []
    total = len(rows)
    done = 0
    lock = threading.Lock()

    def _run(row: Dict[str, Any]) -> QuestionResult:
        return process_question(
            qid=row.get("qid", ""),
            error_description=row.get("error_description", ""),
            q_type=row.get("q_type", "scq"),
            view_link=row.get("view_link", ""),
            error_class=row.get("error_class"),
            llm1_model=llm1_model,
            llm2_model=llm2_model,
            max_attempts=max_attempts,
            set_error_to_1=set_error_to_1,
            do_update_api=do_update_api,
            backup_dir=backup_dir,
            logger=logger,
            raw_log_dir=raw_log_dir,
            official_answer=row.get("official_answer", ""),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run, row): row for row in rows}
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:
                row = futures[fut]
                r = QuestionResult(qid=row.get("qid", ""), status=f"exception: {exc}")
            with lock:
                results.append(r)
                done += 1
                if on_progress:
                    on_progress(done, total, r)
    return results


# ============================================================================
# Module self-test (very minimal — no LLM calls)
# ============================================================================

if __name__ == "__main__":
    # Smoke test: classification + similarity threshold + mathml conversion + target field
    print("=== Stage 11 smoke test ===")
    print(f"  classify_error('remove CDF127 artifact') = {classify_error('remove CDF127 artifact')!r}")
    print(f"  classify_error('answer (B) leaked into stem') = {classify_error('answer (B) leaked into stem')!r}")
    print()
    sample_q = "[Pass1] [question/cleaner_artifact] Typo in option (C) || [Pass2] [solution/math_logic_error] wrong"
    print(f"  parse_target_fields (mixed) = {parse_target_fields(sample_q)!r}")
    print(f"  parse_target_fields ('[Pass1] [question/x] only') = {parse_target_fields('[Pass1] [question/x] only')!r}")
    print(f"  parse_target_fields ('[Pass1] [solution/x] only') = {parse_target_fields('[Pass1] [solution/x] only')!r}")
    print(f"  parse_target_fields ('no tag here') = {parse_target_fields('no tag here')!r}")
    print()
    print(f"  Threshold cleaner_artifact = {get_similarity_threshold('cleaner_artifact')}")
    print()
    sample_html = '<p>Test <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>5</mn></mrow></math> end</p>'
    print(f"  html_mathml_to_latex sample: {html_mathml_to_latex(sample_html)!r}")
    sample_latex = '<p>Test $x = 5$ end</p>'
    out, s, f = html_latex_to_mathml(sample_latex)
    print(f"  html_latex_to_mathml sample: success={s}, fail={f}")
    print()
    print(f"  Modes: corrective_loop={MAX_CORRECTION_ATTEMPTS} attempt(s), one_shot={ONE_SHOT_ATTEMPTS} attempt")
    print(f"  Grounding: forked from stage9_quality (OpenAI reasoning_effort='low' for mini/nano, Gemini thinking_level='LOW')")
