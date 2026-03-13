"""Stage 9: Quality Verifier — pipeline adapter.

3-model independent voting system:
  Pass 1: Gemini 3 Flash (via Gemini API)
  Pass 2: GPT-5-mini (via OpenAI)
  Pass 3: GPT-5.1 (via OpenAI)

Voting rules:
  3/3 clean -> upload directly (error=1) + CSV with details
  2/3 clean -> separate metadata with model-wise ledger (error=2, upload with error=yes) + CSV
  1/3 clean -> reject outright, quarantine
  0/3 clean -> reject outright, quarantine

Forked from: CodeBase/quality_verifier_v2/ (3-pass system)
"""

import base64
import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests

from app.pipeline.base import BasePipelineStage

# ─── Image handling (forked from question_checker.py) ───────────────────
IMG_RE = re.compile(r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)
DATA_URI_IMG_RE = re.compile(r'^data:image/(png|jpe?g|gif|bmp|webp|svg\+xml);base64,')
_IMG_SESSION = requests.Session()
_IMG_SESSION.headers.update({"User-Agent": "acadza-qa/1.0"})


def _guess_mime_from_ext(url_path: str) -> Optional[str]:
    path = urlparse(url_path).path.lower()
    if path.endswith(".png"): return "image/png"
    if path.endswith(".jpg") or path.endswith(".jpeg"): return "image/jpeg"
    if path.endswith(".gif"): return "image/gif"
    if path.endswith(".webp"): return "image/webp"
    if path.endswith(".svg"): return "image/svg+xml"
    return None


def _guess_mime_from_signature(content: bytes) -> Optional[str]:
    if content.startswith(b"\x89PNG\r\n\x1a\n"): return "image/png"
    if content.startswith(b"\xff\xd8\xff"): return "image/jpeg"
    if content.startswith(b"GIF87a") or content.startswith(b"GIF89a"): return "image/gif"
    if content.startswith(b"RIFF") and b"WEBP" in content[:32]: return "image/webp"
    return None


def _fetch_as_data_uri(url: str) -> Optional[str]:
    """Fetch image URL and return base64 data URI, or None on failure."""
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


def _collect_images_with_placeholders(html_text: str, prefix: str) -> tuple:
    """Replace <img> tags with placeholders, return (new_html, attachments).

    Attachments are dicts: {"type": "image_url", "image_url": {"url": "data:..."}}
    """
    if not html_text:
        return html_text, []
    urls = IMG_RE.findall(html_text)
    if not urls:
        return html_text, []

    idx_map = {}
    idx = 1
    for src in urls:
        if src not in idx_map:
            idx_map[src] = f"[{prefix}{idx}]"
            idx += 1

    def repl(m):
        src = m.group(1)
        return idx_map.get(src, f"[{prefix}X]")

    new_html = IMG_RE.sub(repl, html_text)

    attachments = []
    for src in idx_map:
        data_uri = _fetch_as_data_uri(src)
        if data_uri:
            # "detail": "low" = 85 tokens per image (OpenAI). Gemini ignores this field.
            # Sufficient for checking presence/completeness, avoids 1000+ token high-detail cost.
            attachments.append({"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}})

    return new_html, attachments

# Models used in each pass
# Pass 1 "api" is resolved at runtime: Vertex AI when credentials exist, else AI Studio.
# This avoids the AI Studio free-tier 429 problem entirely — Vertex AI has proper paid quota.
PASS_MODELS = {
    "pass1": {"name": "Gemini Flash", "model_id": "gemini-3-flash-preview", "api": "gemini"},
    "pass2": {"name": "GPT-5-mini", "model_id": "gpt-5-mini", "api": "openai"},
    "pass3": {"name": "GPT-5.1", "model_id": "gpt-5.1", "api": "openai"},
}


def _resolve_gemini_api() -> str:
    """Decide whether to use Vertex AI or AI Studio for Gemini batches.

    Priority: Vertex AI (paid quota, no 429) > AI Studio (free tier, hits 429).
    Returns 'vertex' or 'gemini'.
    """
    import os
    from app.config import settings
    creds = settings.vertex_ai_credentials
    if creds and os.path.exists(creds) and settings.vertex_ai_project:
        return "vertex"
    return "gemini"

QUESTION_EDIT_URL = {
    "scq": "https://www.acadza.com/question/edit-question/single-question/",
    "mcq": "https://www.acadza.com/question/edit-question/multi-question/",
    "passage": "https://www.acadza.com/question/edit-question/passage-question/",
    "integer": "https://www.acadza.com/question/edit-question/integer-question/",
    "match": "https://www.acadza.com/question/edit-question/match-question/",
}
QUESTION_VIEW_URL = "https://www.acadza.com/question/view-question/"

# ─── The INSPECTOR Prompt (v2 — military-grade, forked from production) ──────
# Token budget (per request):
#   System prompt (base):       ~2,800 tokens
#   Concept tree (per subject): ~9,000-12,000 tokens
#   MMD ground truth:           variable (full file, no truncation)
#   Per-question user content:  ~1,500-3,000 tokens
#   Images (low detail each):   ~85 tokens (OpenAI), ~258 tokens (Gemini)
#   Output (max):               2,048 tokens
#   All models (Gemini 1M, GPT-5-mini 128K, GPT-5.1 128K): SAFE

INSPECTOR_SYSTEM_PROMPT = """
================================================================================
SYSTEM ROLE: THE INSPECTOR — INDEPENDENT FORENSIC AUDITOR (PCMB EDITION v2)
================================================================================

You are a deterministic, zero-tolerance Quality Assurance engine for high-stakes
Indian entrance exams (JEE Mains, JEE Advanced, NEET).

You are NOT a creative writer. You are NOT a tutor. You are an AUTOMATED
CHECKLIST EXECUTOR that produces binary verdicts backed by provable evidence.

PRIME DIRECTIVE: Find every REAL error. Never flag a non-error.
- Zero false negatives: every genuine defect MUST be caught.
- Zero false positives: if you cannot PROVE a rule is violated, output has_error=false.

================================================================================
SECTION 0: MATCH-TYPE GATE (EXECUTE FIRST — BEFORE ALL OTHER CHECKS)
================================================================================

BEFORE any other inspection, scan the question text for match-type content.
If detected, REJECT IMMEDIATELY. Do NOT evaluate any other dimension.

MATCH-TYPE DETECTION PATTERNS (flag if ANY pattern is present):
- "Match List-I with List-II" or any "Match List-* with List-*"
- "Match the following" or "Match the columns"
- "Column-I" / "Column-II" or "Column I" / "Column II" table structure
- "List-I" / "List-II" paired with mapping options
- Tabular grids with (A)→(I), (B)→(II), (P)→(1) style correspondence mappings
- "Choose the correct match" or "correct matching" or "correctly matched"
- Any structure where two parallel lists must be cross-mapped
- Options presented as combinations like "(A)-(IV), (B)-(III), (C)-(I), (D)-(II)"

IF DETECTED:
  Return: {"qid": "<qid>", "has_error": true, "errors": [
    {"field": "question", "type": "match_format_content",
     "detail": "Match-type question detected: <quote the triggering phrase>"}
  ]}
  STOP HERE. No further checks.

IF NOT DETECTED:
  Proceed to Section 1.

================================================================================
SECTION 1: ACCEPT DATA STRUCTURE AS-IS
================================================================================

The question metadata structure is CORRECT by design. Do NOT flag these as errors:
- Options embedded in question text as (A), (B), (C), (D) — this is the INTENDED format
- Image placeholders like [ImageQ1], [ImageS1] — these are NORMAL
- S3/CDN URLs or inline base64 image references — NORMAL
- <h3> wrapping tags — NORMAL
- HTML entities (&amp;, &lt;, etc.) — NORMAL
- <br/> or <br /> tags between options — NORMAL formatting
- Thin spaces, non-breaking spaces, Unicode math symbols — cosmetic, NOT errors
- Empty string vs null in non-required fields — NOT an error

Your job: verify CONTENT QUALITY against the rules below. Not structure. Not style.

================================================================================
SECTION 2: QUESTION TYPE SCHEMAS (STRICT FIELD VALIDATION)
================================================================================

Each question type has EXACT requirements. Validate the question against its
declared questionType. If ANY required field is missing, empty, or malformed, flag it.

─── SCQ (Single Correct Question) ───
REQUIRED FIELDS:
  question: Non-empty HTML. MUST contain exactly 4 options labeled (A), (B), (C), (D).
  answer:   Exactly ONE letter from {A, B, C, D}. Case-insensitive but must be single letter.
  solution: Non-empty. Must contain reasoning or derivation (not just restating the answer).

VALIDATION RULES:
  - Count options: scan for (A), (B), (C), (D) patterns. All 4 MUST be present.
  - If ANY option is missing → flag {field:"question", type:"missing_options",
    detail:"Option (X) missing from SCQ"}
  - If answer is not exactly one of A/B/C/D → flag {field:"answer", type:"invalid_option"}
  - If solution is empty/whitespace-only → flag {field:"solution", type:"missing_solution"}
  - If solution ends mid-sentence or mid-derivation → flag {field:"solution", type:"partial_solution"}

─── MCQ (Multiple Correct Question) ───
REQUIRED FIELDS:
  question: Non-empty HTML. MUST contain exactly 4 options (A), (B), (C), (D).
  answer:   Valid letter combination from {A,B,C,D}. Examples: "AB", "ACD", "ABCD", "BC".
            Must contain 1-4 UNIQUE letters, each from {A,B,C,D}. Order does not matter.
  solution: Non-empty with reasoning.

VALIDATION RULES:
  - Same option count check as SCQ.
  - Answer must match pattern: 1-4 unique letters from {A,B,C,D}. No digits, no spaces,
    no punctuation. "ABD" is valid. "A,B,D" is invalid. "125" is invalid.
  - If answer format invalid → flag {field:"answer", type:"invalid_option"}

─── integerQuestion (Integer/Numerical) ───
REQUIRED FIELDS:
  question: Non-empty HTML. Should NOT contain (A)/(B)/(C)/(D) option labels.
  answer:   A single numeric value (integer or decimal). NOT a range.
  solution: Non-empty with derivation leading to the numeric answer.

VALIDATION RULES:
  - Answer MUST be parseable as a number. "42", "-3", "0.75", "3.14" are valid.
  - Answer MUST NOT be a range: "2 to 5", "2-5", "[2,5]" are all INVALID.
    → flag {field:"answer", type:"invalid_option", detail:"Range answer not allowed for integerQuestion"}
  - If question text contains (A), (B), (C), (D) option labels, it is likely mistyped as
    integerQuestion → flag {field:"question", type:"wrong_question_type",
    detail:"integerQuestion contains MCQ/SCQ options — likely wrong questionType"}

─── passageQuestion (Passage/Comprehension) ───
REQUIRED FIELDS:
  question: Non-empty. Contains passage text and sub-question(s).
  answer:   Non-empty.
  solution: Non-empty.

VALIDATION RULES:
  - question/answer/solution are flat fields (NOT nested arrays) — this is by design.
  - All three fields must be non-empty. Flag missing_answer/missing_solution if empty.

─── subjective ───
REQUIRED FIELDS:
  question: Non-empty.
  answer:   Non-empty.
  solution: Non-empty.

VALIDATION RULES:
  - All three fields must be non-empty.

================================================================================
SECTION 3: EXAM-SPECIFIC RULES
================================================================================

JEE MAINS (sources: PCMB, BRM, NRY, CGT):
  - Allowed questionTypes: scq, integerQuestion
  - MCQ is NOT allowed in JEE Mains.
  - If questionType = "mcq" → flag {field:"question", type:"wrong_question_type",
    detail:"MCQ not allowed in JEE Mains"}

JEE ADVANCED (source: NWT):
  - Allowed questionTypes: scq, mcq, integerQuestion, passageQuestion
  - All types permitted.

NEET (source: D6C):
  - Allowed questionTypes: scq, subjective
  - Biology questions follow NEET taxonomy.

Flag wrong_question_type ONLY if the type is invalid for the exam source.

================================================================================
SECTION 4: IMAGE VALIDATION (MULTIMODAL — YOU RECEIVE ATTACHED IMAGES)
================================================================================

Images from the question and solution HTML have been:
1. Extracted from <img> tags
2. Fetched from CDN
3. Converted to base64
4. ATTACHED to this message as multimodal image content

Text placeholders like [ImageQ1], [ImageQ2], [ImageS1] correspond to attached
images IN ORDER. [ImageQ1] = first question image, [ImageS1] = first solution image.

YOUR OBLIGATIONS:
  a) EXAMINE each attached image visually.
  b) Verify the image is COMPLETE (not cut off, not blank, not corrupted).
  c) Verify the image is RELEVANT to the surrounding text context.
  d) Verify diagrams, graphs, chemical structures, circuit diagrams are readable.

FLAG THESE:
  - Text says "as shown in the figure", "from the diagram", "refer to the figure",
    "in the given circuit", "the graph shows" BUT no [ImageQn] placeholder exists
    AND no image is attached → {field:"question", type:"missing_image",
    detail:"Question references a figure but no image is present"}
  - Text mentions "Figure 1" and "Figure 2" but only 1 image attached
    → {field:"question", type:"missing_image", detail:"Text references N figures but only M images attached"}
  - Attached image is completely blank (solid color, no content) → {field:"question", type:"broken_image"}
  - Attached image is visibly cut off or incomplete → {field:"question", type:"partial_image"}

SMILES / CHEMICAL STRUCTURE ERRORS:
  - If text still contains raw `<smiles>...</smiles>` tags (not rendered to image),
    flag: {field:"question", type:"broken_chemical_structure",
    detail:"Unrendered <smiles> tag — chemical structure not converted to image"}
  - If a chemical structure image appears blank, distorted, or shows an error
    placeholder → {field:"question", type:"broken_chemical_structure",
    detail:"Chemical structure image is blank/corrupted"}

DO NOT FLAG:
  - Questions that are purely text/math with no figure references — no image needed.
  - Valid image placeholders with corresponding valid attached images — this is NORMAL.
  - Solution images that show worked derivations — NORMAL.
  - Chemical structure images, circuit diagrams, geometry figures — NORMAL content.

================================================================================
SECTION 5: LATEX / MATHML VALIDATION
================================================================================

All math in the pipeline has been converted from LaTeX to MathML. Raw LaTeX
remaining in the final output is a CONVERSION FAILURE.

─── Raw LaTeX Detection (flag: latex_not_converted) ───
Flag if you find ANY of these patterns in question or solution text:
  - \\frac{...}{...}
  - \\sqrt{...} or \\sqrt[n]{...}
  - \\mathrm{...}, \\text{...}, \\mathbf{...}
  - $...$ (dollar-sign delimited math)
  - \\(...\\) (parenthesis-delimited math)
  - \\begin{...}...\\end{...} (environment blocks)
  - \\sum, \\int, \\lim, \\prod, \\infty as standalone tokens outside <math> tags
  - \\alpha, \\beta, \\gamma etc. as standalone tokens outside <math> tags

Exception: LaTeX-like tokens INSIDE <math>...</math> MathML are NOT errors (some
MathML generators use \\mathrm inside <mi> elements — this is valid).

─── LaTeX Artifacts (flag: latex_artifacts) ───
Flag ONLY these specific patterns:
  - Orphaned backslash sequences not part of any expression: lone "\\", "\\\\", "\\,"
  - Stray \\text{} or \\mathrm{} with empty braces
  - Broken \\begin{} without matching \\end{}
  - Random LaTeX command fragments that produce NO meaningful output

Do NOT flag:
  - Backslashes inside MathML tags
  - HTML escape sequences that look like LaTeX

─── MathML Validation ───
  - malformed_mathml: MathML TAGS structurally broken → renders WRONG visual output.
    ONLY for TAG-LEVEL problems: <mi> wrapping <mrow>, triple-nested empty
    <mrow><mrow><mrow></mrow></mrow></mrow>, missing closing tags within <math>.
    Do NOT flag complex-but-correct MathML (deeply nested fractions, matrices are fine if
    they render the correct math expression).
    IMPORTANT: Do NOT use malformed_mathml for mathematical LOGIC errors (wrong
    inequality direction, incorrect formula, wrong derivation step). Those belong
    under math_logic_error below.
  - mathml_parsing_error: Unclosed <math> tags. Missing </math> closing. Unmatched opening/closing
    tags. MathML entities corrupted into garbage text.

─── Mathematical Logic Errors ───
  - math_logic_error: The mathematical CONTENT is logically wrong — inequality reversed,
    incorrect formula application, wrong substitution, derivation step contradicts itself,
    bounds applied incorrectly, sign error in calculation, missing negative sign.
    Use this for errors in the MATHEMATICS itself, NOT in the MathML tag structure.
    Example: Solution says "since x²>x, we get e^{x²} < e^{x}" — the inequality direction
    is reversed. This is math_logic_error, NOT malformed_mathml.
    Example: "∫₀¹ f(t)dt ≤ ∫₀¹ t dt = 1" when upper bound should be different — math_logic_error.

  Do NOT flag:
  - Valid HTML/XML numeric character references like &#x00063; (= 'c'), &#x003B1; (= 'α'),
    &#x0221A; (= '√'), etc. These are valid Unicode code points and render correctly
    in browsers and CKEditor. They are NOT malformed MathML or encoding errors.
  - Plain text (letters, commas, simple expressions) wrapped in MathML <math><mi>...</mi></math>.
    This is unnecessary but harmless — it renders identically to plain text. Not an error.
  - Mathematical approaches you disagree with but that are logically valid.

================================================================================
SECTION 6: SOLUTION COMPLETENESS
================================================================================

  - missing_solution: Solution field is empty, null, contains only whitespace, or contains
    only a restated answer with no derivation.
  - partial_solution: Solution is cut off mid-sentence (ends with "therefore...",
    "hence the answer is...", "so we get..." without the final result). Solution
    references "from equation (3) above" but equation (3) is not shown. Solution
    has numbered steps 1-5 but only steps 1-3 are present.

Do NOT flag:
  - Short but complete solutions ("Direct application of Pythagorean theorem: 3^2+4^2=5^2, answer=5")
  - Solutions that are terse but logically complete
  - Solutions for assertion-reason questions that simply state which option is correct with brief reasoning

================================================================================
SECTION 6B: CLEANER ARTIFACTS (flag: cleaner_artifact)
================================================================================

Flag when OCR/extraction artifacts survive the cleaning pipeline. These are
formatting residues that should have been stripped before reaching QA.

QUESTION FIELD — flag if the question text starts with or contains:
  - Question number prefixes: "Q. 1", "Q1 ", "Ques. 3)", "Question 4:"
  - Bare number-dot prefixes: "93. " at the very start (number + dot + space)
  - Parenthesized question numbers: "(1) " at start when NOT an options list
  - Section headers: \\section*{Q. N}, \\section*{Chemistry Section 1}, etc.
  - Coaching institute boilerplate: "Sri Chaitanya IIT Academy"

SOLUTION FIELD — flag if the solution text starts with or contains:
  - Solution prefixes: "Sol 1.", "Sol 2. Answer: C", "Soln. 3"
  - Answer leak lines: "Answer: C" or "Correct Answer: D" as a standalone line
  - Question prefixes in solutions: "Q. 1" at start of solution text
  - Section headers: \\section*{Sol 4. Answer: B}, \\section*{SOLUTION Ans: C}
  - "SOLUTION Ans: C" text not inside a section header

Do NOT flag:
  - Bold text like **CONCEPT HINT** (these are intentionally preserved)
  - Numbers that are part of the actual content (stoichiometric "2H₂O", "100 K")
  - Option labels "(A)", "(B)", "(1)", "(2)" that are part of an options list

================================================================================
SECTION 7: ANSWER VALIDATION
================================================================================

  - missing_answer: Answer field is empty, null, or blank.
  - invalid_option: Answer format does not match questionType requirements (see Section 2).
  - wrong_answer: Answer CONTRADICTS the solution's own conclusion. Example: solution
    derives "velocity = 5 m/s, which is option C" but answer field says "B".
    Flag ONLY when the solution explicitly arrives at a different answer than what is recorded.
    Do NOT flag if solution is ambiguous or if you cannot verify the physics/chemistry/math.

  IMPORTANT — Number-Letter Mapping:
    Source solutions often use (1)/(2)/(3)/(4) numbering instead of (A)/(B)/(C)/(D).
    The standard mapping is: (1)=A, (2)=B, (3)=C, (4)=D.
    If the answer field is "B" and the solution says "Answer: (2)", this is CORRECT.
    If the answer field is "D" and the solution says "Answer: (4)", this is CORRECT.
    Do NOT flag these as wrong_answer — they are equivalent notations.

    NOTE: The pipeline automatically converts "Answer : (1)" to "Answer : (A)" etc.
    in solution text. If you see mixed notation (some converted, some not), this is
    expected — do NOT flag it as an error. Both formats are equivalent.

================================================================================
SECTION 8: CONCEPT TREE TAG VALIDATION
================================================================================

If a CONCEPT TREE is provided in the system context below, validate the question's tags:

TAG STRUCTURE (each question has these fields):
  subject:    Must match one of the subjects in the concept tree
  chapter:    Must be an exact chapter name from the concept tree for that subject
  concept:    Must be an exact concept name under that chapter
  subConcept: Must be an exact subConcept name under that concept

VALIDATION RULES:
  1. Verify chapter EXISTS in the concept tree under the given subject.
  2. Verify concept EXISTS under that chapter.
  3. Verify subConcept EXISTS under that concept.
  4. Verify the tag is RELEVANT — the question's actual content should relate to the tagged
     concept. A kinematics question tagged under "Thermodynamics" is wrong_tagging even if
     the tag path exists in the tree.

Flag {field:"tags", type:"wrong_tagging", detail:"<specific reason>"} if:
  - Tag path does not exist in concept tree
  - Tag exists but is clearly irrelevant to the question's content

Do NOT flag:
  - Tags that are correct but could be more specific (we accept any valid relevant tag)
  - Questions where concept tree is not provided (skip this section entirely)

================================================================================
SECTION 9: MMD GROUND TRUTH VERIFICATION
================================================================================

If GROUND TRUTH MMD content is provided in the system context below, use it as a
reference to verify the question content was not corrupted during processing.

VERIFICATION RULES:
  - Compare the mathematical expressions in the question against the source MMD.
  - If a key formula, number, or variable has been CHANGED (not just reformatted) → flag
    {field:"question", type:"content_mismatch", detail:"<what changed vs source>"}
  - If answer in source MMD differs from the processed answer → flag
    {field:"answer", type:"wrong_answer", detail:"Source MMD says X but answer field says Y"}

TOLERANCE:
  - Formatting differences (LaTeX→MathML conversion artifacts) are NOT content mismatches.
  - Minor whitespace, tag wrapping, entity encoding differences are NOT mismatches.
  - Only flag when the MATHEMATICAL or TEXTUAL CONTENT is objectively different.

If no MMD content is provided, skip this section entirely.

================================================================================
SECTION 10: THE COMPLETE AUDIT CHECKLIST (REFERENCE TABLE)
================================================================================

| FIELD    | ERROR TYPE             | WHEN TO FLAG                                        |
|----------|------------------------|-----------------------------------------------------|
| question | match_format_content   | ANY match-type content (Section 0 — gate check)     |
| question | missing_options        | SCQ/MCQ missing one or more of (A),(B),(C),(D)      |
| question | malformed_mathml       | MathML TAGS structurally broken → wrong visual render |
| question | math_logic_error       | Mathematical logic wrong (inequality, formula, sign)  |
| question | mathml_parsing_error   | Unclosed/unmatched <math> tags                       |
| question | latex_not_converted    | Raw LaTeX remains outside <math> tags                |
| question | latex_artifacts        | Orphaned LaTeX fragments producing no math output    |
| question | incomplete_question    | Text truncated mid-sentence or contains other Qs     |
| question | cleaner_artifact       | OCR prefix/header artifacts surviving cleanup         |
| question | missing_image          | Text references figure but no image present          |
| question | broken_image           | Attached image is blank or corrupted                 |
| question | partial_image          | Attached image is cut off or incomplete              |
| question | broken_chemical_structure | Unrendered SMILES tag or corrupted structure image |
| question | wrong_question_type    | Type invalid for exam (JEE Mains: no mcq)           |
| question | content_mismatch       | Content differs from source MMD ground truth         |
| solution | missing_solution       | Solution field empty/null/whitespace-only            |
| solution | partial_solution       | Solution truncated mid-derivation or incomplete      |
| solution | malformed_mathml       | Same as question — broken MathML TAGS in solution    |
| solution | math_logic_error       | Math logic error in solution derivation/reasoning     |
| solution | mathml_parsing_error   | Same as question — unclosed tags in solution         |
| solution | latex_not_converted    | Raw LaTeX syntax in solution                         |
| solution | latex_artifacts        | Orphaned LaTeX fragments in solution                 |
| solution | cleaner_artifact       | Sol/Answer prefixes or section headers in solution   |
| answer   | missing_answer         | Answer field empty/null/blank                        |
| answer   | wrong_answer           | Answer contradicts solution's conclusion or MMD      |
| answer   | invalid_option         | Answer format wrong for questionType                 |
| tags     | wrong_tagging          | Tags irrelevant to content or not in concept tree    |

================================================================================
SECTION 11: DECISION SAFETY CHECKS (MANDATORY PRE-OUTPUT VERIFICATION)
================================================================================

Before producing your JSON output, run these checks IN ORDER:

CHECK 1 — MATCH GATE: Did you detect match-type content?
  YES → Output match_format_content error ONLY. Skip all other checks.
  NO  → Continue.

CHECK 2 — FALSE POSITIVE FILTER: For EACH error you are about to report:
  a) Can you quote the EXACT text/tag/token that violates the rule? If NO → drop it.
  b) Does the error map to EXACTLY ONE row in the audit table above? If NO → drop it.
  c) Is this a formatting/cosmetic issue? If YES → drop it.
  d) Did you note "correct" or "no issue" for this field earlier in your analysis?
     If YES → drop it (contradicts your own finding).

CHECK 3 — UNCERTAINTY DEFAULT: If you are not >95% confident an error is real,
  do NOT include it. Err on the side of has_error=false.

CHECK 4 — CLEAN QUESTION CONFIRMATION: If after all checks you have zero errors,
  you MUST output has_error=false with an empty errors array. Do NOT invent errors
  to justify calling something flagged. Clean questions are the EXPECTED majority.

================================================================================
SECTION 12: OUTPUT FORMAT (STRICT JSON — NO MARKDOWN, NO COMMENTARY)
================================================================================

Return ONLY valid JSON. No explanation text. No markdown code blocks. No preamble.

IF NO ERRORS FOUND:
{"qid": "<question_id>", "has_error": false, "errors": []}

IF ERRORS FOUND:
{
  "qid": "<question_id>",
  "has_error": true,
  "errors": [
    {
      "field": "question|solution|answer|tags",
      "type": "exact_type_from_audit_table",
      "detail": "Specific, quotable evidence of the error"
    }
  ]
}

RULES FOR "detail" FIELD:
  - MUST quote the offending text/token/tag when possible.
  - MUST be specific enough that a human reviewer can locate the error.
  - Keep under 200 characters.
  - No generic statements like "there seems to be an issue."

================================================================================
SECTION 13: CALIBRATION EXAMPLES
================================================================================

EXAMPLE 1 — CLEAN SCQ (has_error = false):
  Input: SCQ, Physics, 4 options present, answer="B", solution derives v=gt=10*2=20 m/s
         which matches option (B), MathML well-formed, no LaTeX residue.
  Output: {"qid":"EX1","has_error":false,"errors":[]}
  Why: All fields valid, solution supports answer, no rule violations.

EXAMPLE 2 — MISSING OPTION + PARTIAL SOLUTION:
  Input: SCQ, Chemistry, options (A)(B)(C) present but NO (D), solution ends
         "Therefore the equilibrium constant K = ..." (truncated), MathML clean.
  Output: {"qid":"EX2","has_error":true,"errors":[
    {"field":"question","type":"missing_options","detail":"Option (D) missing from SCQ — only (A),(B),(C) found"},
    {"field":"solution","type":"partial_solution","detail":"Solution truncated mid-derivation at 'K = ...'"}
  ]}

EXAMPLE 3 — MATCH TYPE (IMMEDIATE REJECTION):
  Input: questionType="scq", question text contains "Match List-I with List-II"
         followed by a mapping table.
  Output: {"qid":"EX3","has_error":true,"errors":[
    {"field":"question","type":"match_format_content","detail":"Match-type detected: 'Match List-I with List-II'"}
  ]}
  Why: Match gate triggered. No other checks performed.

EXAMPLE 4 — RAW LATEX + WRONG ANSWER:
  Input: integerQuestion, Physics, question contains "\\frac{mv^2}{r}" outside any
         <math> tag. Solution derives answer=42 but answer field says "35".
  Output: {"qid":"EX4","has_error":true,"errors":[
    {"field":"question","type":"latex_not_converted","detail":"Raw LaTeX '\\frac{mv^2}{r}' outside <math> tags"},
    {"field":"answer","type":"wrong_answer","detail":"Solution concludes 42 but answer field is 35"}
  ]}

EXAMPLE 5 — INTEGER WITH RANGE ANSWER:
  Input: integerQuestion, answer="2 to 5", solution present and complete.
  Output: {"qid":"EX5","has_error":true,"errors":[
    {"field":"answer","type":"invalid_option","detail":"Range answer '2 to 5' not allowed for integerQuestion — must be single numeric value"}
  ]}

EXAMPLE 6 — MISSING IMAGE:
  Input: SCQ, Physics, question says "A particle moves along the path shown in the
         figure below" but no [ImageQ1] placeholder and no image attached.
  Output: {"qid":"EX6","has_error":true,"errors":[
    {"field":"question","type":"missing_image","detail":"Question references 'shown in the figure below' but no image is present or attached"}
  ]}

EXAMPLE 7 — COMPLEX BUT CORRECT MATHML (DO NOT FLAG):
  Input: SCQ, Math, question contains deeply nested MathML with <mfrac><mrow><msup>...
         that correctly renders a triple-nested fraction. Looks complex but renders right.
  Output: {"qid":"EX7","has_error":false,"errors":[]}
  Why: Complex MathML is NOT malformed if it renders the correct expression.

EXAMPLE 8 — WRONG TAGGING:
  Input: Physics question about projectile motion, but tags say
         chapter="Thermodynamics", concept="Carnot Cycle".
  Output: {"qid":"EX8","has_error":true,"errors":[
    {"field":"tags","type":"wrong_tagging","detail":"Question is about projectile motion but tagged under Thermodynamics > Carnot Cycle"}
  ]}
""".strip()

# ─── OpenAI Structured Outputs JSON Schema (strict enforcement) ─────────
# The model CANNOT produce output that violates this schema.
OPENAI_QA_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "qa_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "qid": {"type": "string"},
                "has_error": {"type": "boolean"},
                "errors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "type": {"type": "string"},
                            "detail": {"type": "string"},
                        },
                        "required": ["field", "type", "detail"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["qid", "has_error", "errors"],
            "additionalProperties": False,
        },
    },
}


class QualityVerifierStage(BasePipelineStage):

    # Per-session stop flags (class-level dict, keyed by session_id)
    _stop_flags: Dict[str, bool] = {}

    def request_stop(self):
        """Set stop flag for this session."""
        QualityVerifierStage._stop_flags[self.session_id] = True

    def _is_stopped(self) -> bool:
        """Check if stop has been requested for this session."""
        return QualityVerifierStage._stop_flags.get(self.session_id, False)

    def _clear_stop(self):
        """Clear stop flag at start of processing."""
        QualityVerifierStage._stop_flags.pop(self.session_id, None)

    @property
    def stage_name(self) -> str:
        return "stage9_quality"

    def _get_metadata_dir(self) -> Path:
        d1 = self.session_dir / "metadata_v2_output"
        if d1.exists():
            return d1
        return self.session_dir / "metadata_v2"

    def _get_qa_dir(self) -> Path:
        d = self.session_dir / "quality_check"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_ledger_dir(self) -> Path:
        d = self._get_qa_dir() / "ledgers"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_csv_dir(self) -> Path:
        d = self._get_qa_dir() / "csv"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_all_metadata(self, filter_quarantine: bool = True) -> List[Dict]:
        """Load all metadata_v2 entries from flat or subdirectory layout."""
        meta_dir = self._get_metadata_dir()
        if not meta_dir.exists():
            return []

        all_entries = []
        for sub in sorted(meta_dir.iterdir()):
            if sub.is_dir():
                for f in sub.glob("*_metadata_v2.json"):
                    if "backup" in f.name:
                        continue
                    try:
                        with open(f, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        if isinstance(data, list):
                            all_entries.extend(data)
                    except Exception:
                        continue
            elif sub.suffix == ".json" and "metadata_v2" in sub.name and "backup" not in sub.name:
                try:
                    with open(sub, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        all_entries.extend(data)
                except Exception:
                    continue

        if filter_quarantine and all_entries:
            from app.db import get_quarantined_qids
            q_qids = get_quarantined_qids(self.session_id)
            if q_qids:
                before = len(all_entries)
                all_entries = [e for e in all_entries if e.get("qid") not in q_qids]
                skipped = before - len(all_entries)
                if skipped:
                    self.log("info", f"Filtered {skipped} quarantined questions from input")

        return all_entries

    def _build_entry_index(self) -> Dict[str, Dict]:
        """Build a QID -> full metadata entry index for CSV/upload use."""
        entries = self._load_all_metadata(filter_quarantine=False)
        return {e.get("qid", ""): e for e in entries if e.get("qid")}

    def scan(self) -> Dict[str, Any]:
        """Scan metadata and determine if QA is needed."""
        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata_v2 files found -- run Stage 4 first"}

        tagged_by_counts = {}
        needs_qa = 0
        skip_qa = 0
        for entry in entries:
            tb = entry.get("taggedBy", "unknown")
            tagged_by_counts[tb] = tagged_by_counts.get(tb, 0) + 1
            if tb in ("AITAI", "NITAI"):
                needs_qa += 1
            else:
                skip_qa += 1

        qa_dir = self._get_qa_dir()
        ledger_dir = self._get_ledger_dir()
        existing_ledgers = list(ledger_dir.glob("qa_pass*.json"))

        from app.config import settings
        has_openai = bool(settings.openai_api_key)
        has_gemini = bool(settings.gemini_api_key)

        by_subject = {}
        for entry in entries:
            subj = entry.get("subject", "Unknown")
            by_subject[subj] = by_subject.get(subj, 0) + 1

        self.log("info", f"Scanned {len(entries)} questions: {needs_qa} need QA, {skip_qa} skip QA")
        return {
            "total_questions": len(entries),
            "needs_qa": needs_qa,
            "skip_qa": skip_qa,
            "tagged_by": tagged_by_counts,
            "by_subject": by_subject,
            "existing_ledgers": len(existing_ledgers),
            "qa_needed": needs_qa > 0,
            "can_skip": needs_qa == 0,
            "credentials": {"openai": has_openai, "gemini": has_gemini},
            "pass_models": PASS_MODELS,
        }

    def build_qa_batch(self, extra_context: str = "", strict_mode: bool = False,
                       check_latex: bool = True) -> Dict[str, Any]:
        """Build QA batch files for all 3 passes."""
        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata_v2 files found"}

        qa_entries = [e for e in entries if e.get("taggedBy") in ("AITAI", "NITAI")]
        if not qa_entries:
            self.log("info", "No questions need QA (taggedBy is not AITAI/NITAI)")
            return {"qa_needed": False, "message": "All questions tagged as AIT/NIT -- QA not required"}

        qa_dir = self._get_qa_dir()
        qa_config = {"extra_context": extra_context, "strict_mode": strict_mode, "check_latex": check_latex}
        with open(qa_dir / "qa_config.json", "w", encoding="utf-8") as f:
            json.dump(qa_config, f, indent=2)

        batch_counts = {}
        for pass_key, pass_info in PASS_MODELS.items():
            batch_path = qa_dir / f"{pass_key}_batch.jsonl"
            count = 0
            with open(batch_path, "w", encoding="utf-8") as f:
                for entry in qa_entries:
                    qid = entry.get("qid", "?")
                    qtype = entry.get("questionType", "")
                    qdata = entry.get(qtype, {}) if qtype else {}

                    request = {
                        "custom_id": f"qa-{pass_key}-{qid}",
                        "qid": qid,
                        "subject": entry.get("subject", ""),
                        "questionType": qtype,
                        # Full question/solution — no premature truncation.
                        # Truncation happens in _build_user_content after image extraction.
                        "question": (qdata.get("question", "") or ""),
                        "answer": qdata.get("answer", ""),
                        "solution": (qdata.get("solution", "") or ""),
                        # Tags for concept tree validation
                        "chapter": entry.get("chapter", ""),
                        "concept": entry.get("concept", ""),
                        "subConcept": entry.get("subConcept", ""),
                        "model": pass_info["model_id"],
                    }
                    f.write(json.dumps(request) + "\n")
                    count += 1
            batch_counts[pass_key] = count
            self.log("info", f"Built {pass_key} batch: {count} requests ({pass_info['name']})")

        self.save_checkpoint("qa_batch_built", batch_counts)
        return {
            "qa_needed": True,
            "qa_questions": len(qa_entries),
            "batches": batch_counts,
            "total_requests": sum(batch_counts.values()),
            "qa_dir": str(qa_dir),
            "config": qa_config,
        }

    def _build_qa_system_prompt(self) -> str:
        """Build the full INSPECTOR system prompt with concept tree + MMD ground truth.

        Injection points:
          - Concept tree: filtered to the session's subject(s), appended after main prompt
          - MMD ground truth: loaded from mathpix_output, full content (no truncation)
          - Extra context: user-provided additional instructions
        """
        qa_dir = self._get_qa_dir()
        config_path = qa_dir / "qa_config.json"
        extra_context = ""
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                extra_context = cfg.get("extra_context", "")
            except Exception:
                pass

        prompt = INSPECTOR_SYSTEM_PROMPT

        # ── Inject concept tree (filtered by session subjects) ──────────────
        try:
            from app.config import settings
            ct_path = Path(settings.concept_tree_path)
            if ct_path.exists():
                with open(ct_path, "r", encoding="utf-8") as f:
                    full_tree = json.load(f)

                # Determine which subjects are in this session
                entries = self._load_all_metadata(filter_quarantine=True)
                session_subjects = set()
                for e in entries:
                    subj = e.get("subject", "")
                    if subj:
                        session_subjects.add(subj)

                # Filter concept tree to relevant subjects only
                # Tree structure: {exam: {subject: {chapter: {concept: [subConcepts]}}}}
                # Subject mapping: "Biology" → include both "Botany" and "Zoology"
                #                  "Chemistry" → include "Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry"
                SUBJECT_EXPANSION = {
                    "biology": {"botany", "zoology", "biology"},
                    "chemistry": {"chemistry", "organic chemistry", "inorganic chemistry", "physical chemistry"},
                    "mathematics": {"mathematics", "math"},
                    "math": {"mathematics", "math"},
                }
                filtered_tree = {}
                session_subjects_lower = {s.lower() for s in session_subjects}
                # Expand session subjects (e.g., Biology → Botany + Zoology)
                expanded_lower = set()
                for sl in session_subjects_lower:
                    expanded_lower.update(SUBJECT_EXPANSION.get(sl, {sl}))

                for exam_key, subjects in full_tree.items():
                    if not isinstance(subjects, dict):
                        continue
                    for subj_key, chapters in subjects.items():
                        if subj_key in session_subjects or subj_key.lower() in expanded_lower:
                            if exam_key not in filtered_tree:
                                filtered_tree[exam_key] = {}
                            filtered_tree[exam_key][subj_key] = chapters

                if filtered_tree:
                    ct_json = json.dumps(filtered_tree, indent=1, ensure_ascii=False)
                    prompt += f"""

================================================================================
CONCEPT TREE REFERENCE (for tag validation — Section 8)
================================================================================

Structure: {{exam: {{subject: {{chapter: {{concept: [subConcepts]}}}}}}}}
Use this to verify tags exist and are relevant.

{ct_json}
"""
        except Exception:
            pass  # Concept tree is optional — do not fail QA if unavailable

        # ── Inject MMD ground truth (full file, no truncation) ────────────
        try:
            mmd_dir = self.session_dir / "mathpix_output"
            if mmd_dir.exists():
                mmd_content_parts = []
                for f in sorted(mmd_dir.iterdir()):
                    if f.suffix == ".mmd" and f.is_file():
                        try:
                            text = f.read_text(encoding="utf-8")
                            if text.strip():
                                mmd_content_parts.append(f"--- {f.name} ---\n{text}")
                        except Exception:
                            continue

                if mmd_content_parts:
                    combined_mmd = "\n\n".join(mmd_content_parts)
                    prompt += f"""

================================================================================
MMD GROUND TRUTH (for content verification — Section 9)
================================================================================

Below is the original OCR'd source content. Use it to verify question content
has not been corrupted during processing. Focus on mathematical expressions,
numeric values, and answer keys.

{combined_mmd}
"""
        except Exception:
            pass  # MMD is optional — do not fail QA if unavailable

        # ── Inject extra user context ───────────────────────────────────────
        if extra_context:
            prompt += f"""

================================================================================
ADDITIONAL CONTEXT FROM USER
================================================================================

{extra_context}
"""

        return prompt

    def _build_user_content(self, req: Dict) -> tuple:
        """Build per-question user content with full metadata + tags + images.

        Returns (text_content: str, image_attachments: list[dict])
        Image attachments are base64 data URI dicts for multimodal API calls.
        """
        qid = req.get("qid", "?")
        subject = req.get("subject", "")
        qtype = req.get("questionType", "")
        question = req.get("question", "")
        answer = req.get("answer", "")
        solution = req.get("solution", "")

        # Tag data for concept tree validation
        chapter = req.get("chapter", "")
        concept = req.get("concept", "")
        sub_concept = req.get("subConcept", "")

        # Extract images — replace <img> tags with placeholders, collect base64 attachments
        q_text, q_attachments = _collect_images_with_placeholders(question, "ImageQ")
        s_text, s_attachments = _collect_images_with_placeholders(solution, "ImageS")
        # No truncation — all models have 128K-1M context, full content is essential for QA
        all_attachments = q_attachments + s_attachments

        # Build tag block (only if tags exist)
        tag_block = ""
        if chapter or concept or sub_concept:
            tag_block = f"""
TAGS (verify against concept tree if provided):
  Subject:    {subject}
  Chapter:    {chapter}
  Concept:    {concept}
  SubConcept: {sub_concept}
"""

        # Image inventory for the model
        image_note = ""
        if all_attachments:
            q_count = len(q_attachments)
            s_count = len(s_attachments)
            parts = []
            if q_count:
                parts.append(f"{q_count} question image(s) [ImageQ1..ImageQ{q_count}]")
            if s_count:
                parts.append(f"{s_count} solution image(s) [ImageS1..ImageS{s_count}]")
            image_note = f"""
ATTACHED IMAGES: {', '.join(parts)}
Placeholders in text correspond to attached images in order. EXAMINE each image for
completeness, relevance, and readability."""

        text = f"""================================================================================
TARGET QUESTION FOR AUDIT
================================================================================

QID: {qid}
Subject: {subject}
Question Type: {qtype}
{tag_block}
QUESTION HTML:
{q_text}

ANSWER: {answer}

SOLUTION HTML:
{s_text}
{image_note}
================================================================================
EXECUTE AUDIT: Run Sections 0-11. Return ONLY valid JSON.
================================================================================"""

        return text, all_attachments

    def submit_pass(self, pass_key: str) -> Dict[str, Any]:
        """Submit a QA pass.

        - Gemini (pass1): Real-time per-question calls (streaming progress)
        - OpenAI (pass2/pass3): OpenAI Batch API (async, poll for completion)
        """
        if pass_key not in PASS_MODELS:
            return {"error": f"Invalid pass: {pass_key}. Use pass1, pass2, or pass3"}

        pass_info = PASS_MODELS[pass_key]
        from app.config import settings

        # Credential check
        if pass_info["api"] == "gemini" and not settings.gemini_api_key:
            return {"error": "Gemini API key not configured"}
        if pass_info["api"] == "openai" and not settings.openai_api_key:
            return {"error": "OpenAI API key not configured"}

        qa_dir = self._get_qa_dir()
        batch_path = qa_dir / f"{pass_key}_batch.jsonl"
        if not batch_path.exists():
            return {"error": f"Batch file not found for {pass_key} -- run Build QA Batch first"}

        ledger_dir = self._get_ledger_dir()
        ledger_path = ledger_dir / f"qa_{pass_key}_{pass_info['name'].lower().replace(' ', '_')}.json"
        if ledger_path.exists():
            self.log("info", f"{pass_key} already completed -- results in ledger")
            return {"status": "already_completed", "pass": pass_key, "ledger": str(ledger_path)}

        # All passes now use Batch API
        if pass_info["api"] == "openai":
            try:
                result = self._submit_openai_batch(pass_key)
                return result
            except Exception as e:
                self.log("error", f"{pass_key} OpenAI batch submission failed: {e}")
                return {"error": f"Batch submission failed: {e}"}
        elif pass_info["api"] == "gemini":
            # Resolve whether to use Vertex AI (primary) or AI Studio (fallback)
            resolved_api = _resolve_gemini_api()

            if resolved_api == "vertex":
                # Vertex AI: primary path — paid quota, no 429 problem
                self.log("info", f"{pass_key} using Vertex AI (credentials found, paid quota)")
                try:
                    result = self._submit_vertex_batch(pass_key)
                    return result
                except Exception as e:
                    error_str = str(e)
                    self.log("error", f"{pass_key} Vertex AI batch failed: {error_str}")
                    # Fallback to AI Studio if Vertex AI fails
                    self.log("warning", f"{pass_key} falling back to AI Studio...")
                    try:
                        result = self._submit_gemini_batch(pass_key)
                        return result
                    except Exception as e2:
                        self.log("error", f"{pass_key} AI Studio fallback also failed: {e2}")
                        return {"error": f"Both Vertex AI and AI Studio failed: Vertex={error_str}, Studio={e2}"}
            else:
                # AI Studio: used when no Vertex AI credentials configured
                self.log("info", f"{pass_key} using AI Studio (no Vertex AI credentials)")
                try:
                    result = self._submit_gemini_batch(pass_key)
                    return result
                except Exception as e:
                    error_str = str(e)
                    self.log("error", f"{pass_key} AI Studio batch failed: {error_str}")
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                        self.log("warning", f"{pass_key} AI Studio hit quota limit. "
                                 "Configure VERTEX_AI_CREDENTIALS in .env to avoid this.")
                    return {"error": f"Batch submission failed: {e}"}

    def download_pass_results(self, pass_key: str = "") -> Dict[str, Any]:
        """Download batch results for completed passes (OpenAI or Gemini).

        If pass_key is specified, downloads just that pass. Otherwise downloads all
        passes that have batch_complete status.
        """
        results = {}
        passes_to_check = [pass_key] if pass_key else list(PASS_MODELS.keys())

        for pk in passes_to_check:
            qa_dir = self._get_qa_dir()
            status_path = qa_dir / f"{pk}_status.json"
            if not status_path.exists():
                results[pk] = {"status": "not_started"}
                continue

            with open(status_path, "r") as f:
                status_data = json.load(f)

            if status_data.get("status") == "completed":
                results[pk] = {"status": "already_completed"}
                continue

            if status_data.get("status") != "batch_complete":
                results[pk] = {"status": status_data.get("status", "unknown"),
                               "message": "Batch not yet complete -- poll first"}
                continue

            try:
                # Dispatch to correct downloader based on API type
                api_type = status_data.get("api", "openai")
                if api_type == "vertex":
                    result = self._download_vertex_batch_results(pk)
                elif api_type == "gemini":
                    result = self._download_gemini_batch_results(pk)
                else:
                    result = self._download_openai_batch_results(pk)
                results[pk] = result
            except Exception as e:
                self.log("error", f"Download {pk} results failed: {e}")
                results[pk] = {"error": str(e)}

        return results

    def _save_pass_status(self, status_path: Path, status: str, total: int,
                          processed: int, approved: int, rejected: int):
        """Save incremental pass status to disk for polling."""
        with open(status_path, "w") as f:
            json.dump({
                "status": status,
                "total": total,
                "processed": processed,
                "approved": approved,
                "rejected": rejected,
            }, f)

    # ─── OpenAI Batch API methods ──────────────────────────────────────────

    def _build_openai_batch_jsonl(self, requests_list: List[Dict], system_prompt: str,
                                   model: str, pass_key: str) -> Path:
        """Build API-ready JSONL for OpenAI Batch API.

        Each line is a complete chat completion request with multimodal content.
        """
        qa_dir = self._get_qa_dir()
        batch_path = qa_dir / f"{pass_key}_openai_batch.jsonl"

        count = 0
        with open(batch_path, "w", encoding="utf-8") as f:
            for req in requests_list:
                qid = req.get("qid", "?")
                text_content, image_attachments = self._build_user_content(req)

                # Build multimodal user content parts
                user_parts = [{"type": "text", "text": text_content}]
                for img in image_attachments:
                    user_parts.append(img)

                body = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_parts},
                    ],
                    "response_format": OPENAI_QA_RESPONSE_SCHEMA,
                    "max_completion_tokens": 8192,
                }
                # gpt-5-mini: reasoning_effort="low" for structured QA
                if "mini" in model:
                    body["reasoning_effort"] = "low"

                line = {
                    "custom_id": f"qa-{pass_key}-{qid}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1

        self.log("info", f"Built OpenAI batch JSONL for {pass_key}: {count} requests -> {batch_path.name}")
        return batch_path

    def _submit_openai_batch(self, pass_key: str) -> Dict[str, Any]:
        """Submit an OpenAI Batch API job for a QA pass.

        Builds API-ready JSONL from the internal batch file, uploads to OpenAI,
        creates the batch job, and saves the batch_id for polling.
        """
        pass_info = PASS_MODELS[pass_key]
        model = pass_info["model_id"]

        from app.config import settings
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        qa_dir = self._get_qa_dir()
        internal_batch_path = qa_dir / f"{pass_key}_batch.jsonl"

        if not internal_batch_path.exists():
            return {"error": f"Batch file not found for {pass_key} -- run Build QA Batch first"}

        # Load internal batch
        requests_list = []
        with open(internal_batch_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        requests_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not requests_list:
            return {"error": f"No requests in {pass_key} batch file"}

        # Build system prompt
        system_prompt = self._build_qa_system_prompt()

        # Build API-ready JSONL
        self.log("info", f"Building OpenAI batch JSONL for {pass_key} ({len(requests_list)} questions)...")
        batch_jsonl_path = self._build_openai_batch_jsonl(
            requests_list, system_prompt, model, pass_key
        )

        # Upload to OpenAI
        self.log("info", f"Uploading {pass_key} batch file to OpenAI...")
        with open(batch_jsonl_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        self.log("info", f"File uploaded: {file_obj.id}")

        # Create batch job
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"session_id": self.session_id, "pass": pass_key, "type": "qa"},
        )

        # Save batch status
        status_path = qa_dir / f"{pass_key}_status.json"
        with open(status_path, "w") as f:
            json.dump({
                "status": "batch_submitted",
                "batch_id": batch.id,
                "file_id": file_obj.id,
                "total": len(requests_list),
                "processed": 0,
                "approved": 0,
                "rejected": 0,
            }, f)

        self.log("success", f"{pass_key} batch submitted to OpenAI: {batch.id} ({len(requests_list)} requests)")
        return {
            "status": "batch_submitted",
            "pass": pass_key,
            "batch_id": batch.id,
            "total": len(requests_list),
        }

    def _poll_openai_batch(self, pass_key: str) -> Dict[str, Any]:
        """Poll OpenAI Batch API status for a QA pass."""
        qa_dir = self._get_qa_dir()
        status_path = qa_dir / f"{pass_key}_status.json"

        if not status_path.exists():
            return {"status": "not_started", "total": 0, "processed": 0, "approved": 0, "rejected": 0}

        with open(status_path, "r") as f:
            status_data = json.load(f)

        batch_id = status_data.get("batch_id")
        if not batch_id:
            return status_data

        # If already completed/stopped, return cached status
        if status_data.get("status") in ("completed", "stopped", "failed"):
            return status_data

        from app.config import settings
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        try:
            batch = client.batches.retrieve(batch_id)
            total = batch.request_counts.total if batch.request_counts else 0
            completed = batch.request_counts.completed if batch.request_counts else 0
            failed_count = batch.request_counts.failed if batch.request_counts else 0

            result = {
                "status": batch.status,
                "batch_id": batch_id,
                "total": total,
                "processed": completed + failed_count,
                "completed": completed,
                "failed": failed_count,
                "approved": status_data.get("approved", 0),
                "rejected": status_data.get("rejected", 0),
            }

            if batch.status == "completed" and batch.output_file_id:
                result["output_file_id"] = batch.output_file_id
                result["status"] = "batch_complete"

            if batch.status == "failed":
                error_detail = str(batch.errors) if batch.errors else "Batch failed (no error details)"
                result["error"] = error_detail
                self.log("error", f"{pass_key} OPENAI BATCH FAILED: {error_detail}")
                self.log("warning", f"{pass_key} failed — {failed_count} questions will be treated as abstentions in QA voting")

            # Update status file
            with open(status_path, "w") as f:
                json.dump(result, f)

            return result
        except Exception as e:
            self.log("error", f"{pass_key} batch poll error: {e}")
            return {**status_data, "error": str(e)}

    def _download_openai_batch_results(self, pass_key: str) -> Dict[str, Any]:
        """Download OpenAI Batch API results and build a ledger file.

        Called after polling shows batch_complete. Downloads the output file,
        parses each line's response, and saves the standard ledger format.
        """
        pass_info = PASS_MODELS[pass_key]
        qa_dir = self._get_qa_dir()
        status_path = qa_dir / f"{pass_key}_status.json"

        with open(status_path, "r") as f:
            status_data = json.load(f)

        output_file_id = status_data.get("output_file_id")
        if not output_file_id:
            return {"error": f"No output file for {pass_key} -- batch may not be complete"}

        from app.config import settings
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        # Download results
        self.log("info", f"Downloading {pass_key} batch results from OpenAI...")
        content = client.files.content(output_file_id)
        results_path = qa_dir / f"{pass_key}_batch_results.jsonl"
        results_path.write_bytes(content.content)
        self.log("info", f"Downloaded {pass_key} results: {results_path.name}")

        # Parse results into ledger format
        ledger = []
        errors_list = []
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue

                custom_id = result.get("custom_id", "")
                # Extract QID: "qa-pass2-JS_BIO_Q17" -> "JS_BIO_Q17"
                prefix = f"qa-{pass_key}-"
                qid = custom_id[len(prefix):] if custom_id.startswith(prefix) else custom_id

                response = result.get("response", {})
                status_code = response.get("status_code", 0)
                body = response.get("body", {})

                if status_code != 200:
                    # API error for this request
                    error_msg = body.get("error", {}).get("message", f"HTTP {status_code}")
                    ledger.append({
                        "qid": qid, "pass": pass_key, "model": pass_info["model_id"],
                        "has_error": True,
                        "errors": [{"type": "API_ERROR", "field": "metadata", "detail": error_msg}],
                    })
                    errors_list.append(f"{qid}: {error_msg}")
                    continue

                choices = body.get("choices", [])
                if not choices:
                    ledger.append({
                        "qid": qid, "pass": pass_key, "model": pass_info["model_id"],
                        "has_error": True,
                        "errors": [{"type": "API_ERROR", "field": "metadata", "detail": "No choices in response"}],
                    })
                    continue

                choice = choices[0]
                # Check truncation
                if choice.get("finish_reason") == "length":
                    ledger.append({
                        "qid": qid, "pass": pass_key, "model": pass_info["model_id"],
                        "has_error": True,
                        "errors": [{"type": "API_ERROR", "field": "metadata", "detail": "Output truncated (finish_reason=length)"}],
                    })
                    continue

                result_text = choice.get("message", {}).get("content", "")
                result_data = self._parse_qa_response(result_text, qid)
                ledger.append({
                    "qid": qid,
                    "pass": pass_key,
                    "model": pass_info["model_id"],
                    "has_error": result_data.get("has_error", len(result_data.get("errors", [])) > 0),
                    "errors": result_data.get("errors", []),
                })

        # Save ledger
        ledger_dir = self._get_ledger_dir()
        ledger_path = ledger_dir / f"qa_{pass_key}_{pass_info['name'].lower().replace(' ', '_')}.json"
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, ensure_ascii=False)

        approved = sum(1 for e in ledger if not e.get("has_error") and not e.get("errors"))
        rejected = len(ledger) - approved

        # Update status to completed
        with open(status_path, "w") as f:
            json.dump({
                "status": "completed",
                "batch_id": status_data.get("batch_id"),
                "total": len(ledger),
                "processed": len(ledger),
                "approved": approved,
                "rejected": rejected,
            }, f)

        if errors_list:
            self.log("warning", f"{pass_key} batch had {len(errors_list)} API errors")

        self.log("success", f"{pass_key} batch results parsed: {approved} approved, {rejected} rejected out of {len(ledger)}")
        return {
            "status": "completed",
            "pass": pass_key,
            "total": len(ledger),
            "approved": approved,
            "rejected": rejected,
            "api_errors": len(errors_list),
        }

    # ─── Gemini Batch API methods ────────────────────────────────────────

    GEMINI_API_VERSION = "v1beta"

    def _get_gemini_batch_client(self):
        """Get a google.genai Client configured for batch operations."""
        from app.config import settings
        from google import genai
        from google.genai import types
        return genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(api_version=self.GEMINI_API_VERSION)
        )

    def _build_gemini_batch_jsonl(self, requests_list: List[Dict], system_prompt: str,
                                   model: str, pass_key: str) -> Path:
        """Build Gemini Batch API JSONL file.

        Gemini batch format (per line):
        {
            "key": "<qid>",
            "request": {
                "system_instruction": {"parts": [{"text": "..."}]},
                "contents": [{"role": "user", "parts": [{"text": "..."}, ...]}],
                "generation_config": {"temperature": 0, "responseMimeType": "application/json", "maxOutputTokens": 8192}
            }
        }
        """
        qa_dir = self._get_qa_dir()
        batch_path = qa_dir / f"{pass_key}_gemini_batch.jsonl"

        count = 0
        with open(batch_path, "w", encoding="utf-8") as f:
            for req in requests_list:
                qid = req.get("qid", "?")
                text_content, image_attachments = self._build_user_content(req)

                # Build content parts: text + inline images
                user_parts = [{"text": text_content}]
                for img in image_attachments:
                    data_uri = img.get("image_url", {}).get("url", "")
                    if data_uri.startswith("data:"):
                        header, b64_data = data_uri.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        user_parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data,
                            }
                        })

                line = {
                    "key": qid,
                    "request": {
                        "system_instruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": user_parts}],
                        "generation_config": {
                            "temperature": 0,
                            "responseMimeType": "application/json",
                            "maxOutputTokens": 8192,
                            "thinkingConfig": {"thinkingBudget": 0},
                        }
                    }
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1

        self.log("info", f"Built Gemini batch JSONL for {pass_key}: {count} requests -> {batch_path.name}")
        return batch_path

    def _submit_gemini_batch(self, pass_key: str) -> Dict[str, Any]:
        """Submit a Gemini Batch API job for a QA pass."""
        pass_info = PASS_MODELS[pass_key]
        model = pass_info["model_id"]

        qa_dir = self._get_qa_dir()
        internal_batch_path = qa_dir / f"{pass_key}_batch.jsonl"

        if not internal_batch_path.exists():
            return {"error": f"Batch file not found for {pass_key} -- run Build QA Batch first"}

        # Load internal batch
        requests_list = []
        with open(internal_batch_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        requests_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not requests_list:
            return {"error": f"No requests in {pass_key} batch file"}

        # Build system prompt
        system_prompt = self._build_qa_system_prompt()

        # Build Gemini-format JSONL
        self.log("info", f"Building Gemini batch JSONL for {pass_key} ({len(requests_list)} questions)...")
        batch_jsonl_path = self._build_gemini_batch_jsonl(
            requests_list, system_prompt, model, pass_key
        )

        # Upload + submit via google.genai SDK
        from google.genai import types
        client = self._get_gemini_batch_client()

        self.log("info", f"Uploading {pass_key} batch file to Gemini API...")
        uploaded = client.files.upload(
            file=str(batch_jsonl_path),
            config=types.UploadFileConfig(
                display_name=f"qa-{pass_key}-{self.session_id[:8]}",
                mime_type="jsonl",
            )
        )
        self.log("info", f"File uploaded: {uploaded.name}")

        batch_job = client.batches.create(
            model=f"models/{model}",
            src=uploaded.name,
            config=types.CreateBatchJobConfig(
                display_name=f"qa-{pass_key}-{self.session_id[:8]}",
            )
        )
        self.log("success", f"{pass_key} Gemini batch submitted: {batch_job.name}")

        # Save status
        status_path = qa_dir / f"{pass_key}_status.json"
        with open(status_path, "w") as f:
            json.dump({
                "status": "batch_submitted",
                "batch_id": batch_job.name,
                "file_id": uploaded.name,
                "total": len(requests_list),
                "processed": 0,
                "approved": 0,
                "rejected": 0,
                "api": "gemini",
            }, f)

        return {
            "status": "batch_submitted",
            "pass": pass_key,
            "batch_id": batch_job.name,
            "total": len(requests_list),
        }

    # ── Vertex AI Batch (primary when credentials exist, AI Studio fallback) ──

    def _ensure_vertex_credentials(self):
        """Set GOOGLE_APPLICATION_CREDENTIALS env var for GCS and Vertex AI clients.

        Must be called BEFORE creating any google.cloud.storage or genai Client.
        """
        import os
        from app.config import settings
        creds_path = settings.vertex_ai_credentials
        if not creds_path or not os.path.exists(creds_path):
            raise RuntimeError(
                f"Vertex AI credentials not found at '{creds_path}'. "
                "Set VERTEX_AI_CREDENTIALS in .env to the service account JSON path."
            )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        return creds_path

    def _get_vertex_client(self):
        """Get a google.genai Client configured for Vertex AI."""
        from app.config import settings
        from google import genai
        from google.genai import types

        self._ensure_vertex_credentials()

        return genai.Client(
            vertexai=True,
            project=settings.vertex_ai_project,
            location=settings.vertex_ai_location,
            http_options=types.HttpOptions(api_version="v1beta1"),
        )

    def _build_vertex_batch_jsonl(self, requests_list: List[Dict], system_prompt: str,
                                   model: str, pass_key: str) -> Tuple[Path, Dict[str, str]]:
        """Build Vertex AI Batch JSONL — no 'key' field, track QIDs by line number.

        Returns (jsonl_path, qid_line_map) where qid_line_map = {line_idx: qid}.
        """
        qa_dir = self._get_qa_dir()
        batch_path = qa_dir / f"{pass_key}_vertex_batch.jsonl"

        qid_line_map = {}
        count = 0
        with open(batch_path, "w", encoding="utf-8") as f:
            for req in requests_list:
                qid = req.get("qid", "?")
                text_content, image_attachments = self._build_user_content(req)

                user_parts = [{"text": text_content}]
                for img in image_attachments:
                    data_uri = img.get("image_url", {}).get("url", "")
                    if data_uri.startswith("data:"):
                        header, b64_data = data_uri.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        user_parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data,
                            }
                        })

                # Vertex AI format: NO 'key' field
                line = {
                    "request": {
                        "system_instruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": user_parts}],
                        "generation_config": {
                            "temperature": 0,
                            "responseMimeType": "application/json",
                            "maxOutputTokens": 8192,
                            "thinkingConfig": {"thinkingBudget": 0},
                        }
                    }
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                qid_line_map[str(count)] = qid
                count += 1

        self.log("info", f"Built Vertex AI batch JSONL for {pass_key}: {count} requests")
        return batch_path, qid_line_map

    def _submit_vertex_batch(self, pass_key: str) -> Dict[str, Any]:
        """Submit a batch via Vertex AI (GCS-based, separate quota from AI Studio)."""
        pass_info = PASS_MODELS[pass_key]
        model = pass_info["model_id"]
        from app.config import settings

        qa_dir = self._get_qa_dir()
        internal_batch_path = qa_dir / f"{pass_key}_batch.jsonl"
        if not internal_batch_path.exists():
            return {"error": f"Batch file not found for {pass_key}"}

        # Load internal batch
        requests_list = []
        with open(internal_batch_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        requests_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not requests_list:
            return {"error": f"No requests in {pass_key} batch file"}

        system_prompt = self._build_qa_system_prompt()

        # Build Vertex-format JSONL + QID mapping
        self.log("info", f"Building Vertex AI batch for {pass_key} ({len(requests_list)} questions)...")
        batch_jsonl_path, qid_line_map = self._build_vertex_batch_jsonl(
            requests_list, system_prompt, model, pass_key
        )

        # Save QID line mapping for result download
        mapping_path = qa_dir / f"{pass_key}_vertex_qid_map.json"
        with open(mapping_path, "w") as f:
            json.dump(qid_line_map, f)

        # Upload JSONL to GCS (credentials must be set first)
        self._ensure_vertex_credentials()
        from google.cloud import storage as gcs_storage
        gcs_client = gcs_storage.Client()
        bucket = gcs_client.bucket(settings.vertex_ai_bucket)
        gcs_input_path = f"qa_sessions/{self.session_id[:8]}/{pass_key}_input.jsonl"
        gcs_output_path = f"qa_sessions/{self.session_id[:8]}/{pass_key}_output/"

        blob = bucket.blob(gcs_input_path)
        blob.upload_from_filename(str(batch_jsonl_path), timeout=600)
        self.log("info", f"Uploaded to GCS: gs://{settings.vertex_ai_bucket}/{gcs_input_path}")

        # Submit batch via Vertex AI
        client = self._get_vertex_client()
        batch_job = client.batches.create(
            model=f"models/{model}",
            src=f"gs://{settings.vertex_ai_bucket}/{gcs_input_path}",
            config={"dest": f"gs://{settings.vertex_ai_bucket}/{gcs_output_path}"},
        )
        self.log("success", f"{pass_key} Vertex AI batch submitted: {batch_job.name}")

        # Save status
        status_path = qa_dir / f"{pass_key}_status.json"
        with open(status_path, "w") as f:
            json.dump({
                "status": "batch_submitted",
                "batch_id": batch_job.name,
                "total": len(requests_list),
                "processed": 0,
                "approved": 0,
                "rejected": 0,
                "api": "vertex",
                "gcs_output": f"gs://{settings.vertex_ai_bucket}/{gcs_output_path}",
            }, f)

        return {
            "status": "batch_submitted",
            "pass": pass_key,
            "batch_id": batch_job.name,
            "total": len(requests_list),
            "api": "vertex",
        }

    def _poll_vertex_batch(self, pass_key: str) -> Dict[str, Any]:
        """Poll Vertex AI batch status."""
        qa_dir = self._get_qa_dir()
        status_path = qa_dir / f"{pass_key}_status.json"

        if not status_path.exists():
            return {"status": "not_started"}

        with open(status_path, "r") as f:
            status_data = json.load(f)

        batch_id = status_data.get("batch_id")
        if not batch_id or status_data.get("api") != "vertex":
            return status_data

        if status_data.get("status") in ("completed", "stopped", "failed"):
            return status_data

        try:
            client = self._get_vertex_client()
            batch_job = client.batches.get(name=batch_id)
            state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)

            VERTEX_STATE_MAP = {
                "JOB_STATE_PENDING": "batch_submitted",
                "JOB_STATE_RUNNING": "running",
                "JOB_STATE_SUCCEEDED": "batch_complete",
                "JOB_STATE_FAILED": "failed",
                "JOB_STATE_CANCELLED": "stopped",
            }
            mapped_status = VERTEX_STATE_MAP.get(state, state.lower())

            result = {
                "status": mapped_status,
                "batch_id": batch_id,
                "total": status_data.get("total", 0),
                "processed": 0,
                "approved": status_data.get("approved", 0),
                "rejected": status_data.get("rejected", 0),
                "api": "vertex",
                "gcs_output": status_data.get("gcs_output", ""),
            }

            if batch_job.completion_stats:
                s = batch_job.completion_stats
                success = getattr(s, 'successful_count', 0) or 0
                fail = getattr(s, 'failed_count', 0) or 0
                result["processed"] = success + fail
                result["completed"] = success
                result["failed"] = fail

            if state == "JOB_STATE_SUCCEEDED":
                result["status"] = "batch_complete"
            elif state == "JOB_STATE_FAILED":
                result["status"] = "failed"
                error_detail = "Vertex AI batch job failed"
                if hasattr(batch_job, 'error') and batch_job.error:
                    error_detail = f"Vertex AI failed: {getattr(batch_job.error, 'message', str(batch_job.error))}"
                result["error"] = error_detail
                self.log("error", f"{pass_key} VERTEX AI BATCH FAILED: {error_detail}")

            with open(status_path, "w") as f:
                json.dump(result, f)

            return result
        except Exception as e:
            self.log("error", f"{pass_key} Vertex AI batch poll error: {e}")
            return {**status_data, "error": str(e)}

    def _download_vertex_batch_results(self, pass_key: str) -> Dict[str, Any]:
        """Download Vertex AI batch results from GCS, build ledger file."""
        pass_info = PASS_MODELS[pass_key]
        qa_dir = self._get_qa_dir()
        status_path = qa_dir / f"{pass_key}_status.json"

        with open(status_path, "r") as f:
            status_data = json.load(f)

        gcs_output = status_data.get("gcs_output", "")
        if not gcs_output:
            return {"error": f"No GCS output path for {pass_key}"}

        # Load QID line mapping
        mapping_path = qa_dir / f"{pass_key}_vertex_qid_map.json"
        if not mapping_path.exists():
            return {"error": f"QID line mapping not found for {pass_key}"}
        with open(mapping_path, "r") as f:
            qid_line_map = json.load(f)

        # Download results from GCS (credentials must be set first)
        from app.config import settings
        self._ensure_vertex_credentials()
        from google.cloud import storage as gcs_storage
        gcs_client = gcs_storage.Client()
        bucket = gcs_client.bucket(settings.vertex_ai_bucket)

        # Parse GCS path
        prefix = gcs_output.replace(f"gs://{settings.vertex_ai_bucket}/", "")
        blobs = list(bucket.list_blobs(prefix=prefix))

        all_lines = []
        for blob in sorted(blobs, key=lambda b: b.name):
            if blob.name.endswith(".jsonl"):
                content = blob.download_as_text()
                for line in content.strip().split("\n"):
                    if line.strip():
                        all_lines.append(line.strip())

        self.log("info", f"Downloaded {len(all_lines)} results from Vertex AI for {pass_key}")

        # Parse results and build ledger
        ledger = []
        approved = 0
        rejected = 0
        errors_list = []

        for line_idx, raw_line in enumerate(all_lines):
            qid = qid_line_map.get(str(line_idx), f"unknown_{line_idx}")
            try:
                data = json.loads(raw_line)
                # Vertex AI output format: {"response": {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}}
                response = data.get("response", {})
                candidates = response.get("candidates", [])
                if not candidates:
                    ledger.append({
                        "qid": qid, "pass": pass_key, "model": pass_info["model_id"],
                        "has_error": True,
                        "errors": [{"type": "API_ERROR", "field": "metadata", "detail": "No candidates in Vertex AI response"}],
                    })
                    errors_list.append(f"{qid}: No candidates")
                    continue

                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not text:
                    ledger.append({
                        "qid": qid, "pass": pass_key, "model": pass_info["model_id"],
                        "has_error": True,
                        "errors": [{"type": "API_ERROR", "field": "metadata", "detail": "Empty response text"}],
                    })
                    continue

                # Parse JSON response from model
                parsed = json.loads(text)
                has_error = parsed.get("has_error", False)
                model_errors = parsed.get("errors", [])

                ledger.append({
                    "qid": qid,
                    "pass": pass_key,
                    "model": pass_info["model_id"],
                    "has_error": has_error,
                    "errors": model_errors if has_error else [],
                })

                if has_error:
                    rejected += 1
                else:
                    approved += 1

            except Exception as e:
                ledger.append({
                    "qid": qid, "pass": pass_key, "model": pass_info["model_id"],
                    "has_error": True,
                    "errors": [{"type": "API_ERROR", "field": "metadata", "detail": str(e)}],
                })
                errors_list.append(f"{qid}: {e}")

        # Save ledger
        ledger_dir = self._get_ledger_dir()
        ledger_path = ledger_dir / f"qa_{pass_key}_{pass_info['name'].lower().replace(' ', '_')}.json"
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, ensure_ascii=False)

        self.log("success", f"{pass_key} Vertex AI results: {approved} approved, {rejected} rejected, {len(errors_list)} errors")

        # ── GCS cleanup: delete input + output files to avoid storage cost ──
        self._cleanup_gcs_files(pass_key, gcs_output, bucket)

        # Update status
        status_data["status"] = "completed"
        status_data["approved"] = approved
        status_data["rejected"] = rejected
        with open(status_path, "w") as f:
            json.dump(status_data, f)

        return {
            "status": "completed",
            "pass": pass_key,
            "approved": approved,
            "rejected": rejected,
            "errors": len(errors_list),
            "ledger": str(ledger_path),
        }

    def _cleanup_gcs_files(self, pass_key: str, gcs_output: str, bucket) -> None:
        """Delete input JSONL + output folder from GCS after results are downloaded.

        Called after successful download to avoid accumulating stale files and storage cost.
        GCS charges $0.02/GB/month — small but adds up over many sessions.
        """
        from app.config import settings
        cleaned = 0
        try:
            # Delete input JSONL
            input_path = f"qa_sessions/{self.session_id[:8]}/{pass_key}_input.jsonl"
            input_blob = bucket.blob(input_path)
            if input_blob.exists():
                input_blob.delete()
                cleaned += 1

            # Delete all output blobs (Vertex AI writes multiple files to output folder)
            output_prefix = gcs_output.replace(f"gs://{settings.vertex_ai_bucket}/", "")
            for blob in bucket.list_blobs(prefix=output_prefix):
                blob.delete()
                cleaned += 1

            self.log("info", f"{pass_key} GCS cleanup: deleted {cleaned} files from gs://{settings.vertex_ai_bucket}/")
        except Exception as e:
            # Non-fatal: files will just stay until manually cleaned or bucket lifecycle policy kicks in
            self.log("warning", f"{pass_key} GCS cleanup failed (non-fatal): {e}")

    def _poll_gemini_batch(self, pass_key: str) -> Dict[str, Any]:
        """Poll Gemini Batch API status for a QA pass."""
        qa_dir = self._get_qa_dir()
        status_path = qa_dir / f"{pass_key}_status.json"

        if not status_path.exists():
            return {"status": "not_started", "total": 0, "processed": 0, "approved": 0, "rejected": 0}

        with open(status_path, "r") as f:
            status_data = json.load(f)

        batch_id = status_data.get("batch_id")
        if not batch_id or status_data.get("api") != "gemini":
            return status_data

        if status_data.get("status") in ("completed", "stopped", "failed"):
            return status_data

        client = self._get_gemini_batch_client()

        try:
            batch_job = client.batches.get(name=batch_id)
            state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)

            # Map Gemini job states to our internal status names
            GEMINI_STATE_MAP = {
                "JOB_STATE_PENDING": "batch_submitted",
                "JOB_STATE_RUNNING": "running",
                "JOB_STATE_SUCCEEDED": "batch_complete",
                "JOB_STATE_FAILED": "failed",
                "JOB_STATE_CANCELLED": "stopped",
            }
            mapped_status = GEMINI_STATE_MAP.get(state, state.lower())

            result = {
                "status": mapped_status,
                "batch_id": batch_id,
                "total": status_data.get("total", 0),
                "processed": 0,
                "approved": status_data.get("approved", 0),
                "rejected": status_data.get("rejected", 0),
                "api": "gemini",
            }

            if batch_job.completion_stats:
                s = batch_job.completion_stats
                success = getattr(s, 'successful_count', 0) or 0
                fail = getattr(s, 'failed_count', 0) or 0
                result["processed"] = success + fail
                result["completed"] = success
                result["failed"] = fail

            if state == "JOB_STATE_SUCCEEDED":
                result["status"] = "batch_complete"
                # Save result file reference
                if batch_job.dest and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
                    result["result_file"] = batch_job.dest.file_name

            elif state == "JOB_STATE_FAILED":
                result["status"] = "failed"
                # Extract detailed error info from batch job
                error_detail = "Gemini batch job failed"
                if hasattr(batch_job, 'error') and batch_job.error:
                    err_obj = batch_job.error
                    code = getattr(err_obj, 'code', '')
                    msg = getattr(err_obj, 'message', '')
                    error_detail = f"Gemini batch failed: code={code}, message={msg}"
                elif hasattr(batch_job, 'metadata') and batch_job.metadata:
                    error_detail = f"Gemini batch failed: {batch_job.metadata}"
                result["error"] = error_detail
                # Log prominently — 429/quota failures were going silently
                self.log("error", f"{pass_key} BATCH FAILED: {error_detail}")
                self.log("warning", f"{pass_key} failed questions will be treated as abstentions in QA voting (not penalized)")
                if "429" in str(error_detail) or "RESOURCE_EXHAUSTED" in str(error_detail):
                    self.log("warning", f"{pass_key} failed due to quota exhaustion (429). "
                             "Consider using Vertex AI batch as fallback (separate quota pool)")

            with open(status_path, "w") as f:
                json.dump(result, f)

            return result
        except Exception as e:
            error_str = str(e)
            self.log("error", f"{pass_key} Gemini batch poll error: {error_str}")
            # Detect quota errors specifically
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                self.log("warning", f"{pass_key} hit quota limit (429 RESOURCE_EXHAUSTED). "
                         "Vertex AI batch API has separate quota — consider as fallback.")
            return {**status_data, "error": error_str}

    def _download_gemini_batch_results(self, pass_key: str) -> Dict[str, Any]:
        """Download Gemini Batch API results and build ledger file."""
        pass_info = PASS_MODELS[pass_key]
        qa_dir = self._get_qa_dir()
        status_path = qa_dir / f"{pass_key}_status.json"

        with open(status_path, "r") as f:
            status_data = json.load(f)

        result_file = status_data.get("result_file")
        batch_id = status_data.get("batch_id")

        client = self._get_gemini_batch_client()

        # Get result file if not cached
        if not result_file and batch_id:
            try:
                batch_job = client.batches.get(name=batch_id)
                if batch_job.dest and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
                    result_file = batch_job.dest.file_name
            except Exception as e:
                return {"error": f"Could not get result file: {e}"}

        if not result_file:
            return {"error": f"No result file for {pass_key}"}

        # Download results
        self.log("info", f"Downloading {pass_key} Gemini batch results: {result_file}")
        try:
            file_content = client.files.download(file=result_file)
            content = file_content.decode('utf-8') if isinstance(file_content, bytes) else str(file_content)
        except Exception as e:
            # REST API fallback
            self.log("warning", f"SDK download failed ({e}), trying REST fallback...")
            from app.config import settings
            file_id = result_file.replace("files/", "")
            url = f"https://generativelanguage.googleapis.com/{self.GEMINI_API_VERSION}/files/{file_id}:download?alt=media"
            resp = requests.get(url, headers={"x-goog-api-key": settings.gemini_api_key})
            if resp.status_code == 200:
                content = resp.text
            else:
                return {"error": f"Download failed: {resp.status_code}"}

        # Save raw results
        results_path = qa_dir / f"{pass_key}_gemini_results.jsonl"
        results_path.write_text(content, encoding="utf-8")

        # Parse into ledger format
        ledger = []
        errors_list = []
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                qid = raw.get("key", "")

                # Extract response text from Gemini batch format
                response_text = ""
                response = raw.get("response", {})
                if isinstance(response, dict):
                    candidates = response.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            response_text = parts[0].get("text", "")

                result_data = self._parse_qa_response(response_text, qid)
                ledger.append({
                    "qid": qid,
                    "pass": pass_key,
                    "model": pass_info["model_id"],
                    "has_error": result_data.get("has_error", len(result_data.get("errors", [])) > 0),
                    "errors": result_data.get("errors", []),
                })
            except json.JSONDecodeError:
                errors_list.append(f"Unparseable line: {line[:100]}")

        # Save ledger
        ledger_dir = self._get_ledger_dir()
        ledger_path = ledger_dir / f"qa_{pass_key}_{pass_info['name'].lower().replace(' ', '_')}.json"
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, ensure_ascii=False)

        approved = sum(1 for e in ledger if not e.get("has_error") and not e.get("errors"))
        rejected = len(ledger) - approved

        # Update status
        with open(status_path, "w") as f:
            json.dump({
                "status": "completed",
                "batch_id": status_data.get("batch_id"),
                "total": len(ledger),
                "processed": len(ledger),
                "approved": approved,
                "rejected": rejected,
                "api": "gemini",
            }, f)

        self.log("success", f"{pass_key} Gemini results: {approved} approved, {rejected} rejected out of {len(ledger)}")
        return {
            "status": "completed",
            "pass": pass_key,
            "total": len(ledger),
            "approved": approved,
            "rejected": rejected,
        }

    # ─── Real-time pass methods (Gemini pass1 + fallback) ────────────────

    def _run_openai_pass(self, requests_list: List[Dict], system_prompt: str,
                         model: str, pass_key: str) -> List[Dict]:
        """Run a QA pass via OpenAI chat completions with multimodal image support."""
        from app.config import settings
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        ledger = []
        qa_dir = self._get_qa_dir()
        status_path = qa_dir / f"{pass_key}_status.json"

        for i, req in enumerate(requests_list):
            # Check stop flag before each question
            if self._is_stopped():
                self.log("warning", f"{pass_key}: Stop requested — halting after {len(ledger)} questions")
                break

            qid = req.get("qid", "?")
            text_content, image_attachments = self._build_user_content(req)

            # Build multimodal message parts
            user_parts = [{"type": "text", "text": text_content}]
            for img in image_attachments:
                user_parts.append(img)

            try:
                # GPT-5-mini defaults reasoning ON (medium) — thinking tokens consume
                # from max_completion_tokens budget. Use "low" effort for structured QA
                # (checklist-based, not open-ended) + 8192 tokens to accommodate reasoning.
                # GPT-5.1 defaults reasoning OFF — 8192 is safe headroom.
                create_kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_parts},
                    ],
                    "response_format": OPENAI_QA_RESPONSE_SCHEMA,
                    "max_completion_tokens": 8192,
                }
                # Set reasoning effort: low for gpt-5-mini (checklist QA, not open-ended)
                # reasoning_effort is a top-level param in chat.completions.create
                if "mini" in model:
                    create_kwargs["reasoning_effort"] = "low"

                response = client.chat.completions.create(**create_kwargs)
                choice = response.choices[0]

                # Truncation detection: if finish_reason is "length", output was cut off
                if choice.finish_reason == "length":
                    self.log("warning", f"{pass_key} truncated output for {qid} — treating as API error, will not count as rejection")
                    ledger.append({
                        "qid": qid, "pass": pass_key, "model": model,
                        "has_error": True,
                        "errors": [{"type": "API_ERROR", "field": "metadata", "detail": "Output truncated (finish_reason=length)"}],
                    })
                else:
                    result_text = choice.message.content
                    result_data = self._parse_qa_response(result_text, qid)
                    ledger.append({
                        "qid": qid,
                        "pass": pass_key,
                        "model": model,
                        "has_error": result_data.get("has_error", len(result_data.get("errors", [])) > 0),
                        "errors": result_data.get("errors", []),
                    })
            except Exception as e:
                self.log("warning", f"{pass_key} error on {qid}: {e}")
                ledger.append({
                    "qid": qid, "pass": pass_key, "model": model,
                    "has_error": True,
                    "errors": [{"type": "API_ERROR", "field": "metadata", "detail": str(e)}],
                })

            # Incremental progress update
            approved = sum(1 for e in ledger if not e.get("has_error") and not e.get("errors"))
            rejected = len(ledger) - approved
            if (i + 1) % 3 == 0 or i == len(requests_list) - 1:
                self._save_pass_status(status_path, "running", len(requests_list), len(ledger), approved, rejected)
                self.log("info", f"{pass_key}: {len(ledger)}/{len(requests_list)} processed ({approved} approved, {rejected} rejected)")

        return ledger

    def _run_gemini_pass(self, requests_list: List[Dict], system_prompt: str,
                         model: str, pass_key: str) -> List[Dict]:
        """Run a QA pass via Gemini (google.genai SDK) with multimodal + structured output."""
        from app.config import settings
        from google import genai
        from google.genai import types

        # Flat JSON schema for Gemini structured output (no $ref/$defs)
        gemini_qa_schema = {
            "type": "object",
            "properties": {
                "qid": {"type": "string"},
                "has_error": {"type": "boolean"},
                "errors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "type": {"type": "string"},
                            "detail": {"type": "string"},
                        },
                        "required": ["field", "type", "detail"],
                    },
                },
            },
            "required": ["qid", "has_error", "errors"],
        }

        client = genai.Client(api_key=settings.gemini_api_key)
        ledger = []
        qa_dir = self._get_qa_dir()
        status_path = qa_dir / f"{pass_key}_status.json"

        for i, req in enumerate(requests_list):
            # Check stop flag before each question
            if self._is_stopped():
                self.log("warning", f"{pass_key}: Stop requested — halting after {len(ledger)} questions")
                break

            qid = req.get("qid", "?")
            text_content, image_attachments = self._build_user_content(req)

            # Build Gemini content parts: text + inline images
            parts = [types.Part.from_text(text=text_content)]
            for img in image_attachments:
                data_uri = img.get("image_url", {}).get("url", "")
                if data_uri.startswith("data:"):
                    header, b64_data = data_uri.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                    raw_bytes = base64.b64decode(b64_data)
                    parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))

            try:
                # Gemini 3 Flash defaults thinking ON at HIGH — thinking tokens consume
                # from max_output_tokens. Use LOW thinking for structured QA checklist
                # + 8192 tokens to accommodate thinking + output.
                response = client.models.generate_content(
                    model=model,
                    contents=types.Content(role="user", parts=parts),
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_json_schema=gemini_qa_schema,
                        temperature=0,
                        max_output_tokens=8192,
                        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                    ),
                )

                # Truncation / safety detection
                if response.candidates:
                    fr = response.candidates[0].finish_reason
                    if fr and fr.name == "MAX_TOKENS":
                        self.log("warning", f"{pass_key} truncated output for {qid}")
                        ledger.append({
                            "qid": qid, "pass": pass_key, "model": model,
                            "has_error": True,
                            "errors": [{"type": "API_ERROR", "field": "metadata", "detail": "Output truncated (MAX_TOKENS)"}],
                        })
                    elif fr and fr.name == "SAFETY":
                        self.log("warning", f"{pass_key} safety-blocked for {qid}")
                        ledger.append({
                            "qid": qid, "pass": pass_key, "model": model,
                            "has_error": True,
                            "errors": [{"type": "API_ERROR", "field": "metadata", "detail": "Blocked by safety filter"}],
                        })
                    else:
                        # Use parsed result if available, else fallback to text
                        if hasattr(response, 'parsed') and response.parsed is not None:
                            result = response.parsed
                            has_err = result.get("has_error", False) if isinstance(result, dict) else getattr(result, "has_error", False)
                            errs = result.get("errors", []) if isinstance(result, dict) else [e.model_dump() if hasattr(e, 'model_dump') else e for e in getattr(result, "errors", [])]
                            ledger.append({
                                "qid": qid, "pass": pass_key, "model": model,
                                "has_error": has_err,
                                "errors": errs,
                            })
                        else:
                            result_text = response.text
                            result_data = self._parse_qa_response(result_text, qid)
                            ledger.append({
                                "qid": qid, "pass": pass_key, "model": model,
                                "has_error": result_data.get("has_error", len(result_data.get("errors", [])) > 0),
                                "errors": result_data.get("errors", []),
                            })
                else:
                    ledger.append({
                        "qid": qid, "pass": pass_key, "model": model,
                        "has_error": True,
                        "errors": [{"type": "API_ERROR", "field": "metadata", "detail": "No candidates in response"}],
                    })
            except Exception as e:
                self.log("warning", f"{pass_key} error on {qid}: {e}")
                ledger.append({
                    "qid": qid, "pass": pass_key, "model": model,
                    "has_error": True,
                    "errors": [{"type": "API_ERROR", "field": "metadata", "detail": str(e)}],
                })

            # Incremental progress update
            approved = sum(1 for e in ledger if not e.get("has_error") and not e.get("errors"))
            rejected = len(ledger) - approved
            if (i + 1) % 3 == 0 or i == len(requests_list) - 1:
                self._save_pass_status(status_path, "running", len(requests_list), len(ledger), approved, rejected)
                self.log("info", f"{pass_key}: {len(ledger)}/{len(requests_list)} processed ({approved} approved, {rejected} rejected)")

        return ledger

    def _parse_qa_response(self, text: str, qid: str) -> Dict:
        """Parse JSON response from QA model, handling edge cases."""
        if not text:
            return {"has_error": False, "errors": []}
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                errors = data.get("errors", [])
                has_error = data.get("has_error", len(errors) > 0)
                return {"has_error": has_error, "errors": errors}
            return {"has_error": False, "errors": []}
        except json.JSONDecodeError:
            # Try to extract JSON from text
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                try:
                    data = json.loads(m.group())
                    errors = data.get("errors", [])
                    has_error = data.get("has_error", len(errors) > 0)
                    return {"has_error": has_error, "errors": errors}
                except json.JSONDecodeError:
                    pass
            return {"has_error": False, "errors": [], "raw": text[:200]}

    def cancel_batches(self, pass_key: str = "") -> Dict[str, Any]:
        """Cancel active batch jobs for one or all passes."""
        from app.config import settings
        qa_dir = self._get_qa_dir()
        cancelled = []
        errors = []

        passes_to_cancel = [pass_key] if pass_key else list(PASS_MODELS.keys())

        for pk in passes_to_cancel:
            status_path = qa_dir / f"{pk}_status.json"
            if not status_path.exists():
                continue

            try:
                with open(status_path, "r") as f:
                    sd = json.load(f)
            except Exception:
                continue

            batch_id = sd.get("batch_id")
            if not batch_id:
                continue

            # Skip already-terminal states
            if sd.get("status") in ("completed", "stopped", "failed"):
                continue

            api_type = sd.get("api", PASS_MODELS.get(pk, {}).get("api", ""))

            try:
                if api_type == "openai" and settings.openai_api_key:
                    from openai import OpenAI
                    client = OpenAI(api_key=settings.openai_api_key)
                    client.batches.cancel(batch_id)
                    cancelled.append(pk)
                    self.log("info", f"Cancelled OpenAI batch for {pk}: {batch_id}")
                elif api_type == "gemini" and settings.gemini_api_key:
                    client = self._get_gemini_batch_client()
                    try:
                        client.batches.delete(name=batch_id)
                    except Exception:
                        # Some Gemini SDK versions use cancel instead of delete
                        try:
                            client.batches.cancel(name=batch_id)
                        except Exception:
                            pass
                    cancelled.append(pk)
                    self.log("info", f"Cancelled Gemini batch for {pk}: {batch_id}")
                elif api_type == "vertex":
                    vertex_client = self._get_vertex_client()
                    if vertex_client:
                        try:
                            vertex_client.batches.delete(name=batch_id)
                        except Exception:
                            try:
                                vertex_client.batches.cancel(name=batch_id)
                            except Exception:
                                pass
                        cancelled.append(pk)
                        self.log("info", f"Cancelled Vertex AI batch for {pk}: {batch_id}")
                        # Cleanup GCS files on cancel
                        gcs_output = sd.get("gcs_output", "")
                        if gcs_output:
                            try:
                                self._ensure_vertex_credentials()
                                from google.cloud import storage as gcs_storage
                                gcs_client = gcs_storage.Client()
                                bucket = gcs_client.bucket(settings.vertex_ai_bucket)
                                self._cleanup_gcs_files(pk, gcs_output, bucket)
                            except Exception as ce:
                                self.log("warning", f"{pk} GCS cleanup on cancel failed: {ce}")
            except Exception as e:
                err_str = str(e)
                if "Cannot cancel" in err_str or "already" in err_str.lower():
                    self.log("info", f"{pk} batch already finished")
                else:
                    errors.append(f"{pk}: {err_str}")
                    self.log("error", f"Failed to cancel {pk} batch: {e}")

            # Update status file to stopped
            sd["status"] = "stopped"
            with open(status_path, "w") as f:
                json.dump(sd, f)

        return {
            "status": "cancelled" if cancelled else "stop_requested",
            "cancelled": cancelled,
            "errors": errors,
        }

    def poll_passes(self) -> Dict[str, Any]:
        """Poll status of all QA passes with detailed stats including live progress.

        For OpenAI passes (pass2/pass3): polls Batch API status.
        For Gemini pass1: reads local status file (real-time progress).
        """
        statuses = {}
        ledger_dir = self._get_ledger_dir()
        qa_dir = self._get_qa_dir()

        for pass_key, pass_info in PASS_MODELS.items():
            ledger_path = ledger_dir / f"qa_{pass_key}_{pass_info['name'].lower().replace(' ', '_')}.json"
            status_path = qa_dir / f"{pass_key}_status.json"

            pass_status = {
                "model": pass_info["name"],
                "status": "not_started",
                "total": 0, "results": 0, "approved": 0, "rejected": 0,
                "error_breakdown": None,
            }

            # If ledger exists, this pass is done — read from ledger
            if ledger_path.exists():
                try:
                    with open(ledger_path, "r") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        pass_status["results"] = len(data)
                        approved = sum(1 for e in data if not e.get("has_error") and not e.get("errors"))
                        pass_status["approved"] = approved
                        pass_status["rejected"] = len(data) - approved
                        pass_status["total"] = len(data)
                        error_counts = {}
                        for entry in data:
                            for err in entry.get("errors", []):
                                etype = err.get("type", "unknown") if isinstance(err, dict) else str(err)
                                error_counts[etype] = error_counts.get(etype, 0) + 1
                        pass_status["error_breakdown"] = error_counts
                        pass_status["status"] = "completed"
                except Exception:
                    pass
                statuses[pass_key] = pass_status
                continue

            # Poll batch status for passes with batch_id (OpenAI or Gemini)
            if status_path.exists():
                try:
                    with open(status_path, "r") as f:
                        sd = json.load(f)

                    if sd.get("batch_id") and sd.get("status") not in ("completed", "stopped", "failed"):
                        # Live poll from Batch API
                        api_type = sd.get("api", pass_info["api"])
                        if api_type == "vertex":
                            batch_result = self._poll_vertex_batch(pass_key)
                        elif api_type == "gemini":
                            batch_result = self._poll_gemini_batch(pass_key)
                        else:
                            batch_result = self._poll_openai_batch(pass_key)

                        pass_status["status"] = batch_result.get("status", "unknown")
                        pass_status["total"] = batch_result.get("total", 0)
                        pass_status["results"] = batch_result.get("processed", 0)
                        pass_status["approved"] = batch_result.get("approved", 0)
                        pass_status["rejected"] = batch_result.get("rejected", 0)
                        if batch_result.get("batch_id"):
                            pass_status["batch_id"] = batch_result["batch_id"]

                        # Auto-download if batch is complete
                        if pass_status["status"] == "batch_complete":
                            try:
                                if api_type == "vertex":
                                    dl_result = self._download_vertex_batch_results(pass_key)
                                elif api_type == "gemini":
                                    dl_result = self._download_gemini_batch_results(pass_key)
                                else:
                                    dl_result = self._download_openai_batch_results(pass_key)
                                pass_status["status"] = "completed"
                                pass_status["approved"] = dl_result.get("approved", 0)
                                pass_status["rejected"] = dl_result.get("rejected", 0)
                                pass_status["results"] = dl_result.get("total", 0)
                                pass_status["total"] = dl_result.get("total", 0)
                            except Exception as e:
                                self.log("warning", f"Auto-download failed for {pass_key}: {e}")
                                pass_status["status"] = "batch_complete"
                    else:
                        # Completed/failed/stopped — use cached status
                        pass_status["status"] = sd.get("status", "not_started")
                        pass_status["total"] = sd.get("total", 0)
                        pass_status["results"] = sd.get("processed", 0)
                        pass_status["approved"] = sd.get("approved", 0)
                        pass_status["rejected"] = sd.get("rejected", 0)
                except Exception:
                    pass

            statuses[pass_key] = pass_status

        all_done = all(s["status"] in ("completed", "stopped") for s in statuses.values())
        any_started = any(s["status"] != "not_started" for s in statuses.values())
        any_running = any(s["status"] in ("running", "batch_submitted", "in_progress", "validating", "finalizing") for s in statuses.values())
        # batch_complete means results ready but not yet downloaded into ledger
        any_batch_ready = any(s["status"] == "batch_complete" for s in statuses.values())
        return {
            "passes": statuses,
            "all_complete": all_done and any_started,
            "can_vote": all_done and any_started,
            "any_running": any_running,
            "any_batch_ready": any_batch_ready,
        }

    def merge_votes(self) -> Dict[str, Any]:
        """Merge votes from all 3 passes.

        Voting rules:
          3/3 clean -> error=1, upload directly, CSV with details
          2/3 clean -> error=2, separate metadata with model-wise ledger, CSV with reasons
          1/3 clean -> reject, quarantine
          0/3 clean -> reject, quarantine
        """
        ledger_dir = self._get_ledger_dir()

        # Load all available ledgers
        all_ledgers: Dict[str, List[Dict]] = {}
        for pass_key, pass_info in PASS_MODELS.items():
            ledger_path = ledger_dir / f"qa_{pass_key}_{pass_info['name'].lower().replace(' ', '_')}.json"
            if ledger_path.exists():
                try:
                    with open(ledger_path, "r", encoding="utf-8") as f:
                        all_ledgers[pass_key] = json.load(f)
                except Exception:
                    continue

        if not all_ledgers:
            return {"error": "No QA pass results found -- submit passes first"}

        completed_passes = len(all_ledgers)
        total_passes = len(PASS_MODELS)
        self.log("info", f"Merging votes from {completed_passes}/{total_passes} passes")

        # Build per-QID vote map with model-wise details
        # per_qid[qid] = {pass_key: {"clean": bool, "errors": [...], "model": "..."}}
        per_qid: Dict[str, Dict[str, Dict]] = {}

        for pass_key, ledger in all_ledgers.items():
            model_name = PASS_MODELS[pass_key]["name"]
            for entry in ledger:
                qid = entry.get("qid", "")
                if not qid:
                    continue
                errors = entry.get("errors", [])
                has_error = entry.get("has_error", len(errors) > 0)

                # API_ERROR = abstention, not rejection (transient failures shouldn't penalize)
                is_api_error = any(
                    isinstance(e, dict) and e.get("type") == "API_ERROR"
                    for e in errors
                )
                is_clean = not has_error and len(errors) == 0

                if qid not in per_qid:
                    per_qid[qid] = {}
                per_qid[qid][pass_key] = {
                    "clean": is_clean,
                    "api_error": is_api_error,
                    "errors": errors,
                    "model": model_name,
                    "model_id": PASS_MODELS[pass_key]["model_id"],
                }

        # Compute error level for each QID
        error_levels = {1: 0, 2: 0, 3: 0, 4: 0}
        qid_levels: Dict[str, int] = {}
        category_qids: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: []}

        for qid, pass_results in per_qid.items():
            # Exclude API_ERROR entries from vote count (treat as abstention)
            valid_votes = {k: v for k, v in pass_results.items() if not v.get("api_error")}
            clean_count = sum(1 for v in valid_votes.values() if v["clean"])
            vote_count = len(valid_votes)

            if vote_count >= 3:
                if clean_count == vote_count:
                    level = 1
                elif clean_count >= vote_count - 1:
                    level = 2
                elif clean_count == 1:
                    level = 3
                else:
                    level = 4
            elif vote_count == 2:
                if clean_count == 2:
                    level = 1
                elif clean_count == 1:
                    level = 2
                else:
                    level = 4
            elif vote_count == 1:
                level = 1 if clean_count == 1 else 4
            else:
                # All passes were API errors — don't penalize, mark for re-run
                level = 2  # safe default: review, not quarantine

            error_levels[level] += 1
            qid_levels[qid] = level
            category_qids[level].append(qid)

        # Update metadata_v2 files with error field
        meta_dir = self._get_metadata_dir()
        updated_files = 0
        all_meta_files = []
        for sub in sorted(meta_dir.iterdir()):
            if sub.is_dir():
                all_meta_files.extend(f for f in sub.glob("*_metadata_v2.json") if "backup" not in f.name)
            elif sub.suffix == ".json" and "metadata_v2" in sub.name and "backup" not in sub.name:
                all_meta_files.append(sub)

        for f in all_meta_files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, list):
                    continue
                modified = False
                for entry in data:
                    qid = entry.get("qid", "")
                    if qid in qid_levels:
                        entry["error"] = qid_levels[qid]
                        modified = True
                if modified:
                    with open(f, "w", encoding="utf-8") as fh:
                        json.dump(data, fh, indent=2, ensure_ascii=False)
                    updated_files += 1
            except Exception:
                continue

        # Quarantine 1/3 and 0/3 questions
        from app.db import quarantine_question
        qa_quarantined = 0
        for qid in category_qids[3] + category_qids[4]:
            level = qid_levels[qid]
            pass_results = per_qid.get(qid, {})
            all_errors = []
            for pr in pass_results.values():
                all_errors.extend(pr.get("errors", []))
            quarantine_question(
                self.session_id, qid, self.stage_name,
                reason="qa_failure",
                data={
                    "error_level": level,
                    "approval_ratio": f"{sum(1 for v in pass_results.values() if v['clean'])}/{completed_passes}",
                    "errors": all_errors[:10],
                },
            )
            qa_quarantined += 1

        if qa_quarantined:
            self.log("warning", f"Quarantined {qa_quarantined} questions (1/3 or 0/3 approval) -- rejected outright")

        # Build entry index for CSV generation
        entry_index = self._build_entry_index()

        # ── Generate 3/3 approved CSV ──
        csv_3of3_path = self._generate_approved_csv(category_qids[1], per_qid, entry_index, "3of3_approved")
        self.log("info", f"3/3 approved: {len(category_qids[1])} questions -> {csv_3of3_path.name}")

        # ── Generate 2/3 approved CSV with model-wise ledger ──
        csv_2of3_path = self._generate_review_csv(category_qids[2], per_qid, entry_index, "2of3_review")
        self.log("info", f"2/3 approved: {len(category_qids[2])} questions -> {csv_2of3_path.name}")

        # ── Generate comprehensive ALL questions CSV with error type splits ──
        all_qids = list(per_qid.keys())
        csv_all_path = self._generate_comprehensive_csv(all_qids, per_qid, qid_levels, entry_index, "qa_all_questions")
        self.log("info", f"All questions CSV: {len(all_qids)} questions -> {csv_all_path.name}")

        # ── Generate separate 2/3 metadata file with ledger reasons ──
        review_meta_path = self._generate_review_metadata(category_qids[2], per_qid, entry_index)
        self.log("info", f"2/3 review metadata saved: {review_meta_path.name}")

        # Save vote summary
        summary_path = self._get_qa_dir() / "vote_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "passes_used": completed_passes,
                "total_voted": len(per_qid),
                "error_levels": error_levels,
                "qid_levels": qid_levels,
                "qa_quarantined": qa_quarantined,
                "csv_3of3": str(csv_3of3_path),
                "csv_2of3": str(csv_2of3_path),
                "review_metadata": str(review_meta_path),
            }, f, indent=2)

        self.log("success", f"Votes merged for {len(per_qid)} questions: "
                 f"3/3={error_levels[1]}, 2/3={error_levels[2]}, "
                 f"1/3={error_levels[3]} (rejected), 0/3={error_levels[4]} (rejected)")

        self.save_checkpoint("votes_merged", {
            "total": len(per_qid),
            "error_1": error_levels[1],
            "error_2": error_levels[2],
            "error_3": error_levels[3],
            "error_4": error_levels[4],
            "quarantined": qa_quarantined,
        })

        return {
            "votes_merged": len(per_qid),
            "passes_used": completed_passes,
            "updated_files": updated_files,
            "qa_quarantined": qa_quarantined,
            "results": {
                "approved_3of3": error_levels[1],
                "review_2of3": error_levels[2],
                "rejected_1of3": error_levels[3],
                "rejected_0of3": error_levels[4],
            },
            "csv_3of3": str(csv_3of3_path),
            "csv_2of3": str(csv_2of3_path),
            "review_metadata": str(review_meta_path),
        }

    def _generate_approved_csv(self, qids: List[str], per_qid: Dict,
                               entry_index: Dict, filename: str) -> Path:
        """Generate CSV for 3/3 approved questions with full details."""
        csv_dir = self._get_csv_dir()
        csv_path = csv_dir / f"{filename}.csv"

        fieldnames = [
            "QID", "Subject", "Question Type", "Chapter", "Concept",
            "Pass1 (Gemini Flash)", "Pass2 (GPT-5-mini)", "Pass3 (GPT-5.1)",
            "Error Level", "View Link", "Edit Link",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for qid in sorted(qids):
                entry = entry_index.get(qid, {})
                qtype = (entry.get("questionType") or "").lower()
                qtype_key = qtype
                if qtype_key == "integerquestion":
                    qtype_key = "integer"

                pass_results = per_qid.get(qid, {})
                base_url = QUESTION_EDIT_URL.get(qtype_key, "")

                row = {
                    "QID": qid,
                    "Subject": entry.get("subject", ""),
                    "Question Type": entry.get("questionType", ""),
                    "Chapter": entry.get("chapter", ""),
                    "Concept": entry.get("concept", ""),
                    "Pass1 (Gemini Flash)": "APPROVED" if pass_results.get("pass1", {}).get("clean") else "REJECTED",
                    "Pass2 (GPT-5-mini)": "APPROVED" if pass_results.get("pass2", {}).get("clean") else "REJECTED",
                    "Pass3 (GPT-5.1)": "APPROVED" if pass_results.get("pass3", {}).get("clean") else "REJECTED",
                    "Error Level": 1,
                    "View Link": f"{QUESTION_VIEW_URL}{qid}",
                    "Edit Link": f"{base_url}{qid}" if base_url else "N/A",
                }
                writer.writerow(row)

        return csv_path

    def _generate_review_csv(self, qids: List[str], per_qid: Dict,
                             entry_index: Dict, filename: str) -> Path:
        """Generate CSV for 2/3 approved questions with model-wise error ledger."""
        csv_dir = self._get_csv_dir()
        csv_path = csv_dir / f"{filename}.csv"

        fieldnames = [
            "QID", "Subject", "Question Type", "Chapter", "Concept",
            "Pass1 Verdict", "Pass1 Errors",
            "Pass2 Verdict", "Pass2 Errors",
            "Pass3 Verdict", "Pass3 Errors",
            "Error Level", "View Link", "Edit Link",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for qid in sorted(qids):
                entry = entry_index.get(qid, {})
                qtype = (entry.get("questionType") or "").lower()
                qtype_key = qtype
                if qtype_key == "integerquestion":
                    qtype_key = "integer"

                pass_results = per_qid.get(qid, {})
                base_url = QUESTION_EDIT_URL.get(qtype_key, "")

                def _verdict_and_errors(pk: str) -> Tuple[str, str]:
                    pr = pass_results.get(pk, {})
                    if not pr:
                        return "N/A", ""
                    if pr.get("clean"):
                        return "APPROVED", ""
                    errors = pr.get("errors", [])
                    error_strs = []
                    for e in errors:
                        if isinstance(e, dict):
                            error_strs.append(f"[{e.get('type','')}] {e.get('detail','')}")
                        else:
                            error_strs.append(str(e))
                    return "REJECTED", " | ".join(error_strs)

                p1v, p1e = _verdict_and_errors("pass1")
                p2v, p2e = _verdict_and_errors("pass2")
                p3v, p3e = _verdict_and_errors("pass3")

                row = {
                    "QID": qid,
                    "Subject": entry.get("subject", ""),
                    "Question Type": entry.get("questionType", ""),
                    "Chapter": entry.get("chapter", ""),
                    "Concept": entry.get("concept", ""),
                    "Pass1 Verdict": p1v,
                    "Pass1 Errors": p1e,
                    "Pass2 Verdict": p2v,
                    "Pass2 Errors": p2e,
                    "Pass3 Verdict": p3v,
                    "Pass3 Errors": p3e,
                    "Error Level": 2,
                    "View Link": f"{QUESTION_VIEW_URL}{qid}",
                    "Edit Link": f"{base_url}{qid}" if base_url else "N/A",
                }
                writer.writerow(row)

        return csv_path

    def _generate_review_metadata(self, qids: List[str], per_qid: Dict,
                                  entry_index: Dict) -> Path:
        """Generate separate metadata file for 2/3 approved questions with model-wise ledger."""
        qa_dir = self._get_qa_dir()
        review_path = qa_dir / "review_2of3_metadata.json"

        review_entries = []
        for qid in sorted(qids):
            entry = entry_index.get(qid, {})
            if not entry:
                continue

            # Deep copy and add error=2
            review_entry = json.loads(json.dumps(entry))
            review_entry["error"] = 2

            # Add model-wise ledger
            pass_results = per_qid.get(qid, {})
            review_entry["qa_ledger"] = {}
            for pk in ["pass1", "pass2", "pass3"]:
                pr = pass_results.get(pk, {})
                if pr:
                    review_entry["qa_ledger"][pk] = {
                        "model": pr.get("model", ""),
                        "verdict": "approved" if pr.get("clean") else "rejected",
                        "errors": pr.get("errors", []),
                    }

            review_entries.append(review_entry)

        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review_entries, f, indent=2, ensure_ascii=False)

        return review_path

    def _generate_comprehensive_csv(self, qids: List[str], per_qid: Dict,
                                     qid_levels: Dict, entry_index: Dict,
                                     filename: str) -> Path:
        """Generate comprehensive CSV for ALL questions — every QID with full error detail."""
        csv_dir = self._get_csv_dir()
        csv_path = csv_dir / f"{filename}.csv"

        fieldnames = [
            "QID", "Subject", "Question Type", "Chapter", "Concept", "SubConcept",
            "Error Level", "Final Verdict",
            "Pass1 Model", "Pass1 Verdict", "Pass1 Error Types", "Pass1 Error Details",
            "Pass2 Model", "Pass2 Verdict", "Pass2 Error Types", "Pass2 Error Details",
            "Pass3 Model", "Pass3 Verdict", "Pass3 Error Types", "Pass3 Error Details",
            "Total Errors", "Unique Error Types", "View Link", "Edit Link",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for qid in sorted(qids):
                entry = entry_index.get(qid, {})
                qtype = (entry.get("questionType") or "").lower()
                qtype_key = qtype
                if qtype_key == "integerquestion":
                    qtype_key = "integer"

                pass_results = per_qid.get(qid, {})
                base_url = QUESTION_EDIT_URL.get(qtype_key, "")
                error_level = qid_levels.get(qid, 0)

                # Determine final verdict
                if error_level == 1:
                    final = "APPROVED (3/3)"
                elif error_level == 2:
                    final = "REVIEW (2/3)"
                elif error_level == 3:
                    final = "REJECTED (1/3)"
                elif error_level == 4:
                    final = "REJECTED (0/3)"
                else:
                    final = "UNKNOWN"

                # Per-pass details
                all_error_types = set()
                total_errors = 0
                pass_data = {}
                for pk in ["pass1", "pass2", "pass3"]:
                    pr = pass_results.get(pk, {})
                    if not pr:
                        pass_data[pk] = ("N/A", "N/A", "", "")
                        continue
                    model = pr.get("model", "")
                    verdict = "APPROVED" if pr.get("clean") else "REJECTED"
                    errors = pr.get("errors", [])
                    error_types = []
                    error_details = []
                    for e in errors:
                        if isinstance(e, dict):
                            etype = e.get("type", "unknown")
                            error_types.append(etype)
                            all_error_types.add(etype)
                            detail = e.get("detail", "")
                            field = e.get("field", "")
                            error_details.append(f"[{field}/{etype}] {detail}" if field else f"[{etype}] {detail}")
                        else:
                            error_types.append(str(e))
                            all_error_types.add(str(e))
                            error_details.append(str(e))
                    total_errors += len(errors)
                    pass_data[pk] = (model, verdict, ", ".join(error_types), " | ".join(error_details))

                row = {
                    "QID": qid,
                    "Subject": entry.get("subject", ""),
                    "Question Type": entry.get("questionType", ""),
                    "Chapter": entry.get("chapter", ""),
                    "Concept": entry.get("concept", ""),
                    "SubConcept": entry.get("subConcept", ""),
                    "Error Level": error_level,
                    "Final Verdict": final,
                    "Pass1 Model": pass_data["pass1"][0],
                    "Pass1 Verdict": pass_data["pass1"][1],
                    "Pass1 Error Types": pass_data["pass1"][2],
                    "Pass1 Error Details": pass_data["pass1"][3],
                    "Pass2 Model": pass_data["pass2"][0],
                    "Pass2 Verdict": pass_data["pass2"][1],
                    "Pass2 Error Types": pass_data["pass2"][2],
                    "Pass2 Error Details": pass_data["pass2"][3],
                    "Pass3 Model": pass_data["pass3"][0],
                    "Pass3 Verdict": pass_data["pass3"][1],
                    "Pass3 Error Types": pass_data["pass3"][2],
                    "Pass3 Error Details": pass_data["pass3"][3],
                    "Total Errors": total_errors,
                    "Unique Error Types": ", ".join(sorted(all_error_types)),
                    "View Link": f"{QUESTION_VIEW_URL}{qid}",
                    "Edit Link": f"{base_url}{qid}" if base_url else "N/A",
                }
                writer.writerow(row)

        return csv_path

    def export_csv(self, category: str = "3of3") -> str:
        """Return CSV content for a given category.

        Categories: '3of3', '2of3', 'all' (comprehensive with error splits).
        """
        csv_dir = self._get_csv_dir()
        if category == "3of3":
            path = csv_dir / "3of3_approved.csv"
        elif category == "2of3":
            path = csv_dir / "2of3_review.csv"
        elif category == "all":
            path = csv_dir / "qa_all_questions.csv"
        else:
            return ""
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def get_summary(self) -> Dict[str, Any]:
        """Get QA summary including pass statuses and vote results."""
        scan = self.scan()
        passes = self.poll_passes()
        return {**scan, **passes}

    def skip(self) -> Dict[str, Any]:
        """Skip QA verification."""
        self.log("info", "Quality verification skipped (taggedBy indicates faculty review path)")
        self.save_checkpoint("completed", {"skipped": True, "reason": "taggedBy not AITAI/NITAI"})
        return {"skipped": True}

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        return self.scan()

    def execute(self, **kwargs) -> Dict[str, Any]:
        return self.build_qa_batch()
