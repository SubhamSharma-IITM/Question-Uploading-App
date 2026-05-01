"""Stage 3: Advanced Cleaner — pipeline adapter.

FORKED FROM: CodeBase/PCMB_AdvancedCleanerV2.py
CHANGES:
- Adapted for web app session-based architecture (BasePipelineStage)
- Auto-detects field names: question_content vs question (from production)
- Math block protection during <br/> injection (from production)
- Reads from mapped_output/, writes to cleaned_output/
- Preserves \n -> <br/> conversion for downstream LaTeX/MathML

Operations (forked from production):
- Protect math blocks before any text manipulation
- Strip leading question/solution numbers
- Remove section headers (\\section*{...})
- Convert newlines to <br/> tags safely (with math protection)
- Normalize inline whitespace
- Fix broken LaTeX delimiters
- Clean MathPIX artifacts
- Restore math blocks after all transformations
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.pipeline.base import BasePipelineStage


# ── Math block protection (verbatim from production PCMB_AdvancedCleanerV2.py) ──
def _protect_math_blocks(text: str) -> Tuple[str, List[str]]:
    """Replace math blocks with placeholders to protect them during cleaning.

    Protected patterns (order matters — larger patterns first):
    - <math>...</math>       MathML blocks
    - \\begin{...}...\\end{...}  LaTeX environments
    - \\[...\\]              display LaTeX
    - $$...$$               display math
    - \\(...\\)              inline LaTeX
    - $...$                 inline math (single $)
    """
    if not text:
        return text, []

    placeholders = []

    def _make_placeholder(match):
        placeholders.append(match.group(0))
        return f"__MATH_BLOCK_{len(placeholders) - 1}__"

    patterns = [
        (r'<math[^>]*>.*?</math>', re.DOTALL | re.IGNORECASE),
        (r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', re.DOTALL),
        (r'\\\[.*?\\\]', re.DOTALL),
        (r'\$\$.*?\$\$', re.DOTALL),
        (r'\\\(.*?\\\)', re.DOTALL),
        (r'(?<!\$)\$(?!\$)(?:[^$\\]|\\.)+\$(?!\$)', 0),
    ]

    result = text
    for pattern, flags in patterns:
        result = re.sub(pattern, _make_placeholder, result, flags=flags)

    return result, placeholders


def _restore_math_blocks(text: str, placeholders: List[str]) -> str:
    """Restore math blocks from placeholders.

    IMPORTANT: Restore in REVERSE order to handle nested blocks correctly.
    (Verbatim from production)
    """
    result = text
    for i in range(len(placeholders) - 1, -1, -1):
        result = result.replace(f"__MATH_BLOCK_{i}__", placeholders[i])
    return result


# ── Question/solution number stripping (from production) ────────────────────
def _strip_question_number(text: str) -> str:
    """Strip leading question numbers like '61. Find' or '1 Find'.

    Forked from production PCMB_AdvancedCleanerV2.py:strip_question_number()

    SAFETY: All patterns require a digit to be present (preventing bare-letter
    matches like "Quantum") and use bounded digit counts where possible to
    avoid eating stoichiometric numbers (e.g., "2H₂O").
    """
    if not text:
        return text

    text = text.strip()

    # Special case: number inside math block at start (OCR captured Q# inside $)
    # Require 2+ digits — single digit like "$2 \times" is legitimate math content.
    if text.startswith('$'):
        text = re.sub(r'^\$\d{2,}\s*(?=\\[a-zA-Z])', '$', text)
        text = re.sub(r'^\$\d{2,}\s+(?=[a-zA-Z(])', '$', text)
        return text

    # Pattern 1: digit(s) + dot + optional space (e.g., "61. Find")
    # The dot makes this unambiguous — "2." is clearly a number prefix, not content.
    # Also handle full-width dot ．(U+FF0E) from OCR.
    text = re.sub(r'^\d+[.\uff0e]\s*', '', text)

    # Pattern 2: digit(s) + space + letter (e.g., "1 Find", "61 What")
    # Only strip if followed by a LETTER (not a digit/symbol) to avoid eating
    # stoichiometric numbers like "2 H₂O" or "4 atoms" when number < 4.
    # Require 2+ digits OR single digit followed by at least 2 spaces for safety.
    text = re.sub(r'^\d{2,}[\s\n]+(?=[A-Za-z])', '', text)

    # Pattern 2b: N) format — question number prefix (e.g., "5) Which", "1) Find")
    # SKIP if text contains 2) later — that signals a numbered options list (1) 2) 3) 4)
    # which Stage 4 normalizes to (A)(B)(C)(D). Same heuristic as Pattern 3 below.
    if not re.search(r'2\)', text[3:]):
        text = re.sub(r'^\d{1,3}\)\s*', '', text)

    # Pattern 3: (N) numbered questions — only 1-2 digit numbers (e.g., "(1) Find", "(23) What")
    # Don't match 3+ digits like "(102) ions" which is likely content.
    # SKIP if text contains (2) later — that signals a numbered options list, not a Q number.
    if not re.search(r'\(2\)', text[3:]):
        text = re.sub(r'^\(\d{1,2}\)\s*', '', text)

    # Pattern 4: Q/Ques/Question prefixes — MUST have a digit after to avoid matching
    # words like "Quantum", "Quickly", etc.
    # e.g., "Q1 scq", "Q. 2:", "Ques. 3)", "Question 4"
    text = re.sub(
        r'^(?:Question|Ques\.?|Q\.)\s*\d+\s*(?:scq|mcq|integer|numerical|passage)?\s*[.:)]\s*',
        '', text, flags=re.IGNORECASE,
    )
    # Variant without trailing punctuation (e.g., "Q1 scq Find...")
    # Requires type suffix to be present to avoid over-matching.
    text = re.sub(
        r'^(?:Question|Ques\.?|Q\.?)\s*\d+\s*(?:scq|mcq|integer|numerical|passage)\s+',
        '', text, flags=re.IGNORECASE,
    )
    # Bare "Q. 1" / "Q.1" — requires the dot to disambiguate from content starting with Q
    text = re.sub(r'^Q\.\s*\d+\s+', '', text, flags=re.IGNORECASE)
    # Bare "Q1 " only if followed by a capital letter (strong signal it's a prefix)
    text = re.sub(r'^Q\d+\s+(?=[A-Z])', '', text, flags=re.IGNORECASE)

    return text


def _strip_answer_number(text: str) -> str:
    """Strip leading question number from answer field.

    Handles patterns like:
    - '(54) Answer : (3)' -> 'Answer : (3)'
    - '54. (B)' -> '(B)'
    - '136. (A)' -> '(A)'
    """
    if not text:
        return text
    text = text.strip()
    # Pattern: (number) prefix with 2+ digits — e.g., "(54) Answer"
    # Don't strip single-digit (N) as those are option labels like "(3)"
    text = re.sub(r'^\(\d{2,}\)\s*', '', text)
    # Pattern: number. prefix — e.g., "54. (B)"
    text = re.sub(r'^\d+\.\s*', '', text)
    # Pattern: number) prefix — e.g., "54) (B)"
    text = re.sub(r'^\d+\)\s*', '', text)
    return text


def _strip_solution_number(text: str) -> str:
    """Strip leading solution numbers.

    Forked from production PCMB_AdvancedCleanerV2.py:strip_solution_number()
    """
    if not text:
        return text

    text = text.strip()

    # Case 1: Starts with image markdown — strip number AFTER image
    if text.startswith('!['):
        text = re.sub(
            r'(!\[[^\]]*\]\([^)]+\))\s*\n\d+[\)\.]?\s*\n?',
            r'\1\n',
            text,
        )
        return text

    # Case 2: "Sol N." / "Sol N. Answer: X" / "Q. N" prefixes in solutions
    # These appear when OCR captures solution headers inline
    # "Sol 1.\nAnswer: C\n\n..." → strip "Sol 1.\n" AND "Answer: C\n"
    text = re.sub(
        r'^(?:Sol|Soln)\.?\s*\d+\.?\s*(?:Answer|Ans)\s*:\s*\(?[A-Da-d0-9]+\)?\s*(?:\n|$)',
        '', text, flags=re.IGNORECASE,
    )
    # "Sol N." alone on its own or followed by newline
    text = re.sub(r'^(?:Sol|Soln)\.?\s*\d+\.?\s*\n', '', text, flags=re.IGNORECASE)
    # "Sol N. Answer: X" same-line with content after
    text = re.sub(
        r'^(?:Sol|Soln)\.?\s*\d+\.?\s*(?:Answer|Ans)\s*:\s*\(?[A-Da-d0-9]+\)?\s*\n?',
        '', text, flags=re.IGNORECASE,
    )
    # "Q. N\n" prefix in solutions (cross-numbered from questions)
    text = re.sub(r'^Q\.?\s*\d+\s*\n', '', text, flags=re.IGNORECASE)
    # "Q. N Solution:-" prefix
    text = re.sub(
        r'^Q\.?\s*\d+\s*(?:Solution|Soln?)[\s:.:-]+\s*\n?',
        '', text, flags=re.IGNORECASE,
    )

    # Case 3: "(N) Answer : (X)" full answer header line — strip entire line
    # e.g., "(130) Answer : (2)\n" or "(147) Answer : (4)\n"
    text = re.sub(
        r'^\(\d+\)\s*(?:Answer|Ans)\s*:\s*\(?[A-Da-d0-9]+\)?\s*(?:\n|$)',
        '', text, flags=re.IGNORECASE,
    )
    # Case 3b: "(N) Sol." or "(N)<br/>Sol." at start of solution -- common PYQ artifact
    text = re.sub(
        r'^\(\d+\)\s*(?:<br\s*/?>)?\s*(?:Sol|Soln)\.?\s*',
        '', text, flags=re.IGNORECASE,
    )
    # Case 3c: Starts with (number) with 2+ digits -- e.g., "(54) Answer"
    text = re.sub(r'^\(\d{2,}\)\s*', '', text)
    # Case 4: Starts with bare number (also handle full-width dot ．)
    text = re.sub(r'^\d+\)\s*', '', text)
    text = re.sub(r'^\d+[.\uff0e]\s*', '', text)

    # "Answer: X" standalone line that may remain after Sol N. stripping
    text = re.sub(
        r'^Answer\s*:\s*\(?[A-Da-d0-9]+\)?\s*\n',
        '', text, flags=re.IGNORECASE,
    )

    # Bare "Solution:" or "Solution:-" label on its own line (after answer line stripped)
    text = re.sub(r'^Solution\s*[:\-]+\s*\n', '', text, flags=re.IGNORECASE)

    # Case 5: "SOLUTION Ans: X" prefix (e.g., from \section*{SOLUTION Ans: C})
    # Requires end-of-line or next line boundary — don't eat "SOLUTION Ans: C is colorless"
    text = re.sub(
        r'^SOLUTION\s+Ans\s*:\s*\(?[A-Da-d0-9]+\)?\s*(?:\n|$)',
        '', text, flags=re.IGNORECASE,
    )

    # Case 5: "Ans:", "Soln:", "Sol." standalone prefixes — only when they're the ENTIRE
    # prefix (followed by whitespace or option letter, not running into content)
    text = re.sub(r'^(?:Ans|Soln|Sol)[\.:]\s*(?=\(|[A-D]\b|\s*$)', '', text, flags=re.IGNORECASE)

    # Case 6: "Answer: X" standalone line leaking into solution content
    # Only strip if the answer is a single letter/number on its own line
    text = re.sub(
        r'^Answer\s*:\s*\(?[A-Da-d0-9]+\)?\s*\n',
        '', text, flags=re.IGNORECASE,
    )

    return text


# ── Section header removal (from production) ────────────────────────────────
def _remove_section_headers(text: str, is_solution: bool = False) -> str:
    """Remove \\section*{...} headers and exam-specific boilerplate.

    Forked from production PCMB_AdvancedCleanerV2.py:remove_section_headers()
    """
    if not text:
        return text

    result = text

    # Remove common coaching headers
    result = re.sub(r'\\section\*?\{Sri Chaitanya[^}]*\}\s*', '', result, flags=re.IGNORECASE)
    result = re.sub(r'Sri Chaitanya IIT Academy\s*', '', result, flags=re.IGNORECASE)

    # Remove question headers
    result = re.sub(r'\\section\*?\{Question\s*\d*\}\s*', '', result, flags=re.IGNORECASE)
    result = re.sub(r'\\section\*?\{Q\.\s*\d+\}\s*', '', result, flags=re.IGNORECASE)

    # Remove section type headers
    result = re.sub(r'\\section\*?\{SINGLE CORRECT[^}]*\}\s*', '', result, flags=re.IGNORECASE)
    result = re.sub(r'\\section\*?\{NUMERICAL VALUE[^}]*\}\s*', '', result, flags=re.IGNORECASE)
    result = re.sub(r'\\section\*?\{MULTIPLE CORRECT[^}]*\}\s*', '', result, flags=re.IGNORECASE)

    # Remove Options header
    result = re.sub(r'\\section\*?\{Options:?\}\s*', '', result, flags=re.IGNORECASE)
    result = re.sub(r'^\s*Options:\s*', '', result, flags=re.MULTILINE | re.IGNORECASE)

    # Remove "SOLUTION Ans: X" section headers (e.g., \section*{SOLUTION Ans: C})
    result = re.sub(r'\\section\*?\{SOLUTION\s+Ans\s*:\s*[^}]*\}\s*', '', result, flags=re.IGNORECASE)

    # Remove subject/section headers (e.g., \section*{Chemistry Section 1}, \section*{SECTION A - ...})
    result = re.sub(
        r'\\section\*?\{(?:Chemistry|Physics|Mathematics|Biology|Botany|Zoology)\s+Section\s*\d*[^}]*\}\s*',
        '', result, flags=re.IGNORECASE,
    )
    result = re.sub(r'\\section\*?\{SECTION\s+[A-Z]\s*[-–:][^}]*\}\s*', '', result, flags=re.IGNORECASE)

    if is_solution:
        # \section*{(147) Answer : (4)} — numbered answer header with parenthesized number
        result = re.sub(r'\\section\*?\{\(\d+\)\s*(?:Answer|Ans)\s*:\s*[^}]*\}\s*', '', result, flags=re.IGNORECASE)
        # \section*{147. Answer : (4)} — dotted number answer header
        result = re.sub(r'\\section\*?\{\d+\.?\s*(?:Answer|Ans)\s*:\s*[^}]*\}\s*', '', result, flags=re.IGNORECASE)
        # \section*{Sol N. Answer: X} — full solution header with answer
        result = re.sub(r'\\section\*?\{Sol\.?\s*\d+\.?\s*(?:Answer|Ans)\s*:\s*[^}]*\}\s*', '', result, flags=re.IGNORECASE)
        # \section*{Sol N.} — solution number header
        result = re.sub(r'\\section\*?\{Sol\.?\s*\d+\.?\}\s*', '', result, flags=re.IGNORECASE)
        # \section*{(N)} — bare parenthesized number header
        result = re.sub(r'\\section\*?\{\(\d+\)\}\s*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\\section\*?\{Sol\.?\}\s*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\\section\*?\{Solution:?\}\s*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\\section\*?\{Answer:?\}\s*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\\section\*?\{SOLUTIONS[^}]*\}\s*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'^Sol\.\s*', '', result.strip(), flags=re.IGNORECASE)

    # Convert remaining \section*{...} to bold text
    result = re.sub(r'\\section\*?\{([^}]+)\}', r'**\1**', result, flags=re.DOTALL)

    return result


# ── Line break insertion with math protection (from production) ─────────────
def _insert_line_breaks_safely(text: str) -> str:
    """Convert \\n to <br/> with math block protection.

    Forked from production PCMB_AdvancedCleanerV2.py:insert_line_breaks_safely()
    This preserves the br injection logic from the original app.
    """
    if not text:
        return text

    # Step 1: Protect math blocks
    protected_text, placeholders = _protect_math_blocks(text)

    # Step 2: Normalize line endings
    protected_text = protected_text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 3: Convert newlines to <br/> tags
    # Double newlines → paragraph break
    protected_text = re.sub(r'\n{2,}', '<br/><br/>', protected_text)
    # Single newlines → line break
    protected_text = protected_text.replace('\n', '<br/>')

    # Step 4: Restore math blocks
    result = _restore_math_blocks(protected_text, placeholders)

    return result


# ── Format noise strip (paragraph headers, line ranges, rough work) ───────
# Patterns identified during v2 match/passage rollout. Each pattern is noise
# from the source PDF that has no semantic meaning to the database — the
# metadata schema already captures the question order, paper code, etc. Must
# run BEFORE bold conversion in stage4 so noise gets removed instead of
# converted to <strong>. See tests/integration/test_stage3_noise_strip.py.
#
# Range separator that allows "to", "&", LaTeX-escaped "\&", "-", "–", "and".
# Used inside paragraph header patterns. `\\?&` permits both `12 & 13` and
# `12 \& 13` (LaTeX-escaped ampersand from .mmd output).
_RANGE_SEP = r'(?:to|\\?&|-|–|and)'

_NOISE_PATTERNS = [
    # LaTeX `\section*{Paragraph for ...}` form. MUST run BEFORE the bare
    # bold/plain pattern below — otherwise the bare pattern strips the inner
    # content first and leaves the empty `\section*{}` shell behind.
    (r'\\section\*\s*\{\s*Paragraph\s+for\s+[Qq]uestion[s]?\s+(?:Nos?\.\s*)?\d+\s*' + _RANGE_SEP + r'\s*\d+\s*[:.]?\s*\}',
     re.IGNORECASE),
    # LaTeX `\section*{Read the (above|following) ...}` form. Same reason as above —
    # must precede bare Read pattern so the wrapper braces get consumed.
    (r'\\section\*\s*\{\s*Read\s+the\s+(?:above|following)\s+(?:paragraph|write\s*up)\s+carefully[^{}\n]{1,160}?\s*\}',
     re.IGNORECASE),
    # LaTeX `\section*{Read carefully ...}` form (generic, no paragraph/write-up).
    (r'\\section\*\s*\{\s*Read\s+carefully[^{}\n]{1,100}?\s*\}',
     re.IGNORECASE),
    # Paragraph header (with or without ** wrapper):
    #   **Paragraph for Question Nos. 9 to 10**
    #   **Paragraph for question Nos. 11 to 12**
    #   **Paragraph for Questions 12 & 13**, Paragraph for Questions 12 \& 13
    #   Paragraph for Question Nos. 14 to 16
    (r'\*{0,2}\s*Paragraph\s+for\s+[Qq]uestion[s]?\s+(?:Nos?\.\s*)?\d+\s*' + _RANGE_SEP + r'\s*\d+\s*[:.]?\s*\*{0,2}',
     re.IGNORECASE),
    # "Read the (above|following) (paragraph|write up) carefully ... :"
    # Requires terminator `:` or `.` (not optional) so lazy match doesn't stop
    # too early at "answer". Body bounded to 140 chars to limit blast radius.
    (r'\*{0,2}\s*Read\s+the\s+(?:above|following)\s+(?:paragraph|write\s*up)\s+carefully[^*\n]{1,140}?[:.]\s*\*{0,2}',
     re.IGNORECASE),
    # Generic "Read carefully and answer ..." (no paragraph/write-up):
    (r'\*{0,2}\s*Read\s+carefully[^*\n]{1,80}?[:.]\s*\*{0,2}',
     re.IGNORECASE),
    # "Then find the answer of following questions based on above paragraph."
    (r'Then\s+find\s+the\s+answer\s+of\s+(?:the\s+)?following\s+question[s]?\s+based\s+on\s+(?:the\s+)?above\s+paragraph\.?',
     re.IGNORECASE),
    # Printer artifact: "Space for Rough work" (with optional trailing punctuation)
    (r'Space\s+for\s+Rough\s+work[:.]?',
     re.IGNORECASE),
]

# Line-range numbering leak at start of solution text. Anchored — runs after
# pattern list. Removes "14-16. iBℓ=0.5" -> "iBℓ=0.5".
_LINE_RANGE_PREFIX = re.compile(r'^\s*\d+\s*[-–]\s*\d+\s*\.\s*')


def _strip_format_noise(text: str) -> str:
    """Remove formatting-noise patterns from source PDFs.

    Noise = formatting artifacts (paragraph headers, "Read carefully..."
    preambles, line-range markers, "Space for Rough work") that have no
    semantic meaning. Real bold and real content are never touched — every
    pattern is anchored to specific keywords + structural markers.

    Apply BEFORE math protection: noise is pure prose, never inside math.
    Apply BEFORE the stage4 ``**`` -> <strong> conversion so noise gets
    removed instead of bolded.
    """
    if not text:
        return text

    cleaned = text
    for pattern, flags in _NOISE_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=flags)

    cleaned = _LINE_RANGE_PREFIX.sub('', cleaned)
    return cleaned


# ── MathPIX artifact cleanup ────────────────────────────────────────────────
def _clean_mathpix_artifacts(text: str) -> str:
    """Clean common MathPIX OCR artifacts."""
    if not text:
        return text

    cleaned = text

    # Fix broken LaTeX: stray backslashes before common words
    cleaned = re.sub(r"\\(and|or|the|in|of|is|for|at|to|with)\b", r"\1", cleaned)

    # Fix LaTeX ampersand artifact: \& -> &
    cleaned = cleaned.replace("\\&", "&amp;")

    # Fix: \text {} -> \text{}
    cleaned = re.sub(r"\\(text|mathrm|mathbf)\s+\{", r"\\\1{", cleaned)

    # Remove stray MathPIX crop parameters
    cleaned = re.sub(r"\?height=\d+&width=\d+", "", cleaned)

    return cleaned


# ── Whitespace normalization ────────────────────────────────────────────────
def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces within lines, trim per-line."""
    if not text:
        return text

    # Collapse multiple spaces (but not newlines)
    cleaned = re.sub(r"[^\S\n]+", " ", text)

    # Strip per-line
    lines = cleaned.split("\n")
    lines = [line.strip() for line in lines]
    cleaned = "\n".join(lines)

    return cleaned.strip()


# ── Solution answer format conversion: (1)→(A), (2)→(B), etc. ──────────────
_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}

def _convert_solution_answer_format(text: str) -> str:
    """Convert numbered answer references to lettered format in solution text.

    Converts patterns like:
        Answer : (1)  ->  Answer : (A)
        Ans:(2)       ->  Ans:(B)
        Answer: (3)   ->  Answer: (C)
        option (1)    ->  option (A)

    DOES NOT touch:
        - Statement/assertion content like "statements (1) and (2) are correct"
        - Row identifiers inside LaTeX tabular environments
        - Numbered references in running prose like "from (1), we get"
    """
    if not text:
        return text

    result = text

    # Pattern 1: "Answer" / "Ans" followed by optional punctuation then (N)
    # e.g. "Answer : (1)", "Ans:(3)", "Answer is (2)", "Ans. (4)"
    def _replace_answer_ref(m):
        prefix = m.group(1)  # "Answer", "Ans", etc.
        mid = m.group(2)     # separators like " : ", ":", " is ", ". "
        num = m.group(3)     # "1", "2", "3", "4"
        letter = _NUM_TO_LETTER.get(num, num)
        return f"{prefix}{mid}({letter})"

    result = re.sub(
        r'((?:Answer|Ans))((?:\s*[.:]\s*|\.\s*|\s+is\s+))\(\s*([1-4])\s*\)',
        _replace_answer_ref, result, flags=re.IGNORECASE
    )

    # Pattern 2: "option (N)" / "Option (N)" -- standalone option references
    def _replace_option_ref(m):
        prefix = m.group(1)
        num = m.group(2)
        letter = _NUM_TO_LETTER.get(num, num)
        return f"{prefix}({letter})"

    result = re.sub(
        r'((?:option|Option|OPTION)\s*)\(\s*([1-4])\s*\)',
        _replace_option_ref, result
    )

    # Pattern 3: "Correct option is (N)" / "correct answer: (N)"
    def _replace_correct_option(m):
        prefix = m.group(1)
        mid = m.group(2)
        num = m.group(3)
        letter = _NUM_TO_LETTER.get(num, num)
        return f"{prefix}{mid}({letter})"

    result = re.sub(
        r'((?:correct|right)\s+(?:option|answer|choice))((?:\s+is\s+|\s*:\s*))\(\s*([1-4])\s*\)',
        _replace_correct_option, result, flags=re.IGNORECASE
    )

    return result


# ── Main cleaning function ──────────────────────────────────────────────────
def _clean_text(text: str, is_solution: bool = False) -> str:
    """Clean a single text field with full production cleaning pipeline.

    Order of operations:
    1. Remove section headers
    2. Strip question/solution numbers
    3. Clean MathPIX artifacts
    4. Normalize whitespace
    5. Insert line breaks safely (with math block protection)
    6. Clean up redundant <br/> at start/end
    """
    if not text:
        return text

    cleaned = text

    # Step 0: Remove "THIS QUESTION CONTAINS IMAGES" banner span
    cleaned = re.sub(
        r'<em><strong><span[^>]*>THIS QUESTION CONTAINS IMAGES</span></strong></em><br\s*/?>',
        '', cleaned, flags=re.IGNORECASE)

    # Step 0.4: Strip format-noise patterns (paragraph headers, "Read carefully"
    # preambles, line-range markers, "Space for Rough work"). Run BEFORE math
    # protection — noise is pure prose, never inside math.
    cleaned = _strip_format_noise(cleaned)

    # Step 0.5: Protect math blocks EARLY — before number stripping and header removal
    # This prevents regex from eating math content that starts with digits or "Q".
    # We'll restore them after steps 1-3, before br injection (which has its own protection).
    cleaned, early_placeholders = _protect_math_blocks(cleaned)

    # Step 1: Remove section headers
    cleaned = _remove_section_headers(cleaned, is_solution=is_solution)

    # Step 2: Strip leading numbers
    if is_solution:
        cleaned = _strip_solution_number(cleaned)
    else:
        cleaned = _strip_question_number(cleaned)

    # Step 3: Clean MathPIX artifacts
    cleaned = _clean_mathpix_artifacts(cleaned)

    # Step 3.1: Restore early-protected math blocks before br injection
    cleaned = _restore_math_blocks(cleaned, early_placeholders)

    # Step 3.5: Convert solution answer format (1)->(A) etc.
    if is_solution:
        cleaned = _convert_solution_answer_format(cleaned)

    # Step 4: Normalize whitespace (before br injection)
    cleaned = _normalize_whitespace(cleaned)

    # Step 5: Convert \n to <br/> with math block protection
    cleaned = _insert_line_breaks_safely(cleaned)

    # Step 6: Clean up redundant <br/> at start/end
    cleaned = re.sub(r'^(<br/?>)+', '', cleaned)
    cleaned = re.sub(r'(<br/?>)+$', '', cleaned)

    # Step 7: Strip trailing answer leak from question text (PYQ artifact)
    # Patterns: "Ans. (7)", "Ans (200)", "Answer: 3" at end of text
    # Must handle trailing HTML tags like </h3>
    if not is_solution:
        cleaned = re.sub(
            r'\s*(?:<br\s*/?>)*\s*Ans\.?\s*[\(\[]\s*\d+\s*[\)\]]\s*(?:</\w+>)*\s*$',
            '', cleaned, flags=re.IGNORECASE,
        )

    return cleaned.strip()


# ── Entry-level cleaning (auto-detects field names) ─────────────────────────
def _clean_question_entry(entry: Dict) -> Dict:
    """Clean all text fields in a question entry.

    Auto-detects field names: question_content vs question, solution_content vs solution.
    Forked from production PCMB_AdvancedCleanerV2.py field auto-detection.
    """
    cleaned = entry.copy()

    # Auto-detect field names (from production)
    q_key = "question_content" if "question_content" in entry else "question"
    s_key = "solution_content" if "solution_content" in entry else "solution"

    # matchQuestion: preserve raw LaTeX tabular question_content. _clean_text
    # would inject <br/> for newlines and mangle the tabular row structure.
    # Stage 4's _parse_match_latex_tabular parses the raw block directly. We
    # still strip noise patterns from the surrounding intro/header text.
    qtype = entry.get("questionType", "")
    if qtype == "matchQuestion":
        cleaned[q_key] = _strip_format_noise(entry.get(q_key, ""))
    else:
        cleaned[q_key] = _clean_text(entry.get(q_key, ""), is_solution=False)

    # Clean solution (safe for all types — no LaTeX tabular blocks expected)
    cleaned[s_key] = _clean_text(entry.get(s_key, ""), is_solution=True)

    # Clean options if present as separate dict (legacy format)
    if "options" in entry:
        cleaned["options"] = {
            k: _clean_text(v, is_solution=False)
            for k, v in entry["options"].items()
        }

    # Clean answer — strip leading question numbers like "(54)"
    for ans_key in ("answer", "answer_content"):
        if ans_key in entry and entry[ans_key]:
            cleaned[ans_key] = _strip_answer_number(entry[ans_key])

    # Clean passage_content for passageQuestion items
    if "passage_content" in entry and entry["passage_content"]:
        cleaned["passage_content"] = _clean_text(entry["passage_content"], is_solution=False)

    # Clean sub_questions for passageQuestion items
    if "sub_questions" in entry and isinstance(entry["sub_questions"], list):
        cleaned_subs = []
        for sq in entry["sub_questions"]:
            csq = dict(sq)
            if sq.get("question_content"):
                csq["question_content"] = _clean_text(sq["question_content"], is_solution=False)
            if sq.get("solution_content"):
                csq["solution_content"] = _clean_text(sq["solution_content"], is_solution=True)
            if sq.get("answer_key"):
                csq["answer_key"] = _strip_answer_number(sq["answer_key"])
            cleaned_subs.append(csq)
        cleaned["sub_questions"] = cleaned_subs

    # Clean hint if present (legacy format)
    if "hint" in entry:
        cleaned["hint"] = _clean_text(entry.get("hint", ""), is_solution=True)

    return cleaned


def _compute_diff(original: Dict, cleaned: Dict) -> List[str]:
    """Compute what changed between original and cleaned."""
    diffs = []

    q_key = "question_content" if "question_content" in original else "question"
    s_key = "solution_content" if "solution_content" in original else "solution"

    for key in (q_key, s_key, "hint"):
        orig_val = original.get(key, "")
        clean_val = cleaned.get(key, "")
        if not orig_val and not clean_val:
            continue
        if orig_val != clean_val:
            changes = []
            if "<br/>" in (clean_val or "") and "<br/>" not in (orig_val or ""):
                changes.append("newlines → <br/>")
            if len(clean_val or "") < len(orig_val or ""):
                changes.append(f"trimmed {len(orig_val or '') - len(clean_val or '')} chars")
            if not changes:
                changes.append("text normalized")
            label = "question" if key == q_key else ("solution" if key == s_key else key)
            diffs.append(f"{label}: {', '.join(changes)}")

    if original.get("options", {}) != cleaned.get("options", {}):
        diffs.append("options: text normalized")

    return diffs


# ── Pipeline stage class ────────────────────────────────────────────────────
class CleanerStage(BasePipelineStage):

    @property
    def stage_name(self) -> str:
        return "stage3_cleaning"

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        """Preview cleaning changes without applying."""
        mapped_dir = self.session_dir / "mapped_output"
        json_files = list(mapped_dir.glob("*_mapped.json"))

        total_questions = 0
        total_changes = 0
        preview = []

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                entries = json.load(f)
            total_questions += len(entries)

            for entry in entries:
                cleaned = _clean_question_entry(entry)
                diffs = _compute_diff(entry, cleaned)
                if diffs:
                    total_changes += 1
                    preview.append({
                        "qid": entry.get("qid", "?"),
                        "question_number": entry.get("question_number"),
                        "changes": diffs,
                    })

        self.log("info", f"Dry run: {total_changes}/{total_questions} questions would be modified")
        return {
            "files": len(json_files),
            "total_questions": total_questions,
            "questions_modified": total_changes,
            "preview": preview,
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Apply cleaning to all mapped output files."""
        mapped_dir = self.session_dir / "mapped_output"
        out_dir = self.session_dir / "cleaned_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(mapped_dir.glob("*_mapped.json"))
        if not json_files:
            self.log("error", "No mapped JSON files found")
            self.save_checkpoint("failed", {"error": "No mapped files"})
            return {"error": "No mapped files"}

        total_cleaned = 0
        total_changes = 0

        for jf in json_files:
            self.log("info", f"Cleaning {jf.name}...")

            # Backup before mutation (mandatory backup policy)
            self.backup(jf)

            with open(jf, "r", encoding="utf-8") as f:
                entries = json.load(f)

            cleaned_entries = []
            for entry in entries:
                cleaned = _clean_question_entry(entry)
                diffs = _compute_diff(entry, cleaned)
                if diffs:
                    total_changes += 1
                    qid = entry.get("qid", entry.get("question_number", "?"))
                    self.log("info", f"  {qid}: {'; '.join(diffs)}")
                cleaned_entries.append(cleaned)
                total_cleaned += 1

            # Save to cleaned_output
            out_name = jf.name.replace("_mapped.json", "_cleaned.json")
            out_file = out_dir / out_name
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(cleaned_entries, f, ensure_ascii=False, indent=2)

            self.log("success", f"Saved {out_file.name} ({len(cleaned_entries)} questions)")

        # Update question tracking
        from app.db import upsert_question
        for jf in out_dir.glob("*_cleaned.json"):
            with open(jf, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                qid = entry.get("qid", "")
                if qid:
                    upsert_question(self.session_id, qid, current_stage="stage3_cleaning")

        self.save_checkpoint("completed", {
            "files_processed": len(json_files),
            "questions_cleaned": total_cleaned,
            "fields_modified": total_changes,
        })

        self.log("success", f"Cleaning complete: {total_cleaned} questions, {total_changes} modified")
        self.progress(100, "Cleaning complete")

        return {
            "questions_cleaned": total_cleaned,
            "fields_modified": total_changes,
        }
