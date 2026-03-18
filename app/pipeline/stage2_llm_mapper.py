"""Stage 2: LLM Cached Mapper — pipeline adapter.

FORKED FROM: CodeBase/PCMB_Cached_Mapper.py
CHANGES:
- Adapted for web app session-based architecture (BasePipelineStage)
- Supports Gemini and Anthropic (Claude Haiku 4.5) providers
- Extra context from user wired into extraction prompt
- isPYQ config wired into prompt
- Simplified caching (inline content, no file upload) for web app use
- Session config integration for test_code derivation
- Output format matches production: question_content/solution_content/answer_key
- Dynamic jump capping, batch size adaptation, rate limiting from production
"""

import datetime
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.pipeline.base import BasePipelineStage
from app.config import settings


# ── Constants (matched to production PCMB_Cached_Mapper.py) ───────────────
BATCH_SIZE_DEFAULT = 15          # Initial questions per LLM call
MIN_BATCH_SIZE = 3               # Minimum batch size (shrinks on failure)
MAX_BATCH_SIZE = 20              # Maximum batch size (grows on success)
MAX_RETRIES = 3
GENERATION_TIMEOUT = 120         # seconds
OVERALL_TIMEOUT = 1800           # 30 min hard cap for entire extraction
SAFETY_OVERLAP_LINES = 2         # Re-scan overlap between batches (production=2)
MAX_QUESTION_SPAN = 30           # Max lines a single question can span (production=30)
JUMP_MULTIPLIER = 1.5            # Dynamic jump cap multiplier
MIN_JUMP_ALLOWANCE = 100         # Minimum allowed cursor jump
BLIND_JUMP_SIZE = 50             # Jump size on empty/failed batches
EOF_REMAINING_THRESHOLD = 250    # Accept EOF if fewer lines remain
INITIAL_AVG_LINES_PER_Q = 20    # Initial estimate for dynamic jump calc

# ── Token limits per model (for context window safety check) ──────────────
MODEL_CONTEXT_LIMITS = {
    "gemini": 1_000_000,        # Gemini Flash: 1M tokens
    "anthropic": 200_000,       # Claude Sonnet 4.6: 200K tokens
}
CHARS_PER_TOKEN_ESTIMATE = 4    # Conservative: ~4 chars per token
PROMPT_OVERHEAD_TOKENS = 3000   # System instruction + extraction prompt overhead
SAFETY_MARGIN_TOKENS = 10_000   # Buffer to avoid hitting limit


def _derive_test_code(pdf_filename: str) -> str:
    """Derive a short test code from the PDF filename.

    'Test_Mode_Quick_Check_Acadza_2nd_March_2026.pdf' -> 'TMQCA'
    'AIATS05_questions.mmd' -> 'AIATS05'
    """
    if not pdf_filename:
        return "TEST"
    stem = Path(pdf_filename).stem
    # Remove common suffixes
    for suffix in ("_questions", "_solutions", "_answer_key", "_combined"):
        stem = stem.replace(suffix, "")
    # If it looks like a code already (short, alphanumeric), use as-is
    clean = re.sub(r"[^a-zA-Z0-9]", "", stem)
    if len(clean) <= 10:
        return clean.upper()
    # Take uppercase first letter of each word
    words = re.split(r"[_\s\-]+", stem)
    words = [w for w in words if w and not re.match(r"^\d{4}$|^\d{1,2}(st|nd|rd|th)$", w, re.IGNORECASE)]
    initials = "".join(w[0].upper() for w in words[:6] if w)
    return initials or "TEST"


# ── Line decoration (verbatim from production) ─────────────────────────────
def _decorate_lines(raw_lines: List[str]) -> str:
    """Add Line_XXXXX: prefix to each line for LLM referencing."""
    parts = []
    for idx, line in enumerate(raw_lines):
        parts.append(f"Line_{idx:05d}: {line}")
    return "".join(parts)


# ── Content slicing (verbatim from production) ─────────────────────────────
def _slice_content(raw_lines: List[str], start: int, end: int) -> Optional[str]:
    """Slice content from raw lines by index range."""
    if start is None or end is None:
        return None
    start = max(0, start)
    end = min(end, len(raw_lines) - 1)
    if start > end:
        return None
    return "".join(raw_lines[start:end + 1]).strip()


# ── Subject normalization (verbatim from production) ────────────────────────
def _normalize_subject(subject: str) -> Optional[str]:
    """Normalize subject name to canonical form."""
    if not subject:
        return None
    s = subject.strip().lower()
    mapping = {
        "physics": "Physics",
        "chemistry": "Chemistry",
        "mathematics": "Mathematics",
        "maths": "Mathematics",
        "math": "Mathematics",
        "biology": "Biology",
        "bio": "Biology",
        "botany": "Biology",
        "zoology": "Biology",
    }
    return mapping.get(s)


# ── Garbage detection (matched to production) ──────────────────────────────
def _is_garbage(text: Optional[str]) -> bool:
    """Check if extracted text is garbage. Matched to production is_garbage()."""
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 10:
        return True
    # All digits / whitespace
    if re.match(r"^[\d\s.]+$", stripped):
        return True
    # Solution/answer key prefixes — NOT questions
    if re.match(r"^\s*(Sol\.|Ans\.|Solution|Explanation|Hint:|Key\s*Sheet|KEY\s*SHEET)",
                stripped, re.IGNORECASE):
        return True
    # Tabular answer rows (\\hline patterns from answer key tables)
    if re.match(r"^\s*(?:\\hline\s*)+\d+\s*&", stripped):
        return True
    if re.match(r"^\s*\\begin\{tabular\}", stripped):
        return True
    # Image-only lines (no question text)
    if re.match(r"^!\[.*?\]\(.*?\)\s*$", stripped):
        return True
    # Option-only fragments — all non-empty lines start with small numbers (1-4)
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if len(lines) >= 2 and all(re.match(r"^\d+[\.\)]\s", l) for l in lines):
        first_match = re.match(r"^(\d+)", lines[0])
        if first_match and int(first_match.group(1)) <= 4:
            return True
    # Fragment starting with options (A)/(B)/(1)/(2) without a question stem before them
    if lines and re.match(r"^\s*\(?[A-Da-d1-4]\s*[\.\)]\s", lines[0]):
        # Check if there's ANY question-like text before the first option
        first_option_idx = 0
        has_question_stem = False
        for i, l in enumerate(lines):
            if re.match(r"^\s*\(?[A-Da-d1-4]\s*[\.\)]", l):
                first_option_idx = i
                break
            if len(l) > 20:  # Some substantial text before options
                has_question_stem = True
        if not has_question_stem and first_option_idx == 0:
            return True
    return False


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def estimate_model_compatibility(mmd_text: str) -> Dict[str, Any]:
    """Check which models can handle the document within their context window.

    Returns dict with per-model safety info.
    """
    doc_tokens = _estimate_tokens(mmd_text)
    total_needed = doc_tokens + PROMPT_OVERHEAD_TOKENS + SAFETY_MARGIN_TOKENS

    result = {"document_tokens_est": doc_tokens}
    for provider, limit in MODEL_CONTEXT_LIMITS.items():
        safe = total_needed <= limit
        result[provider] = {
            "context_limit": limit,
            "tokens_needed": total_needed,
            "safe": safe,
            "headroom": limit - total_needed,
        }
    result["any_safe"] = any(result[p]["safe"] for p in MODEL_CONTEXT_LIMITS)
    result["needs_split"] = not result["any_safe"]
    return result


# ── Document pre-scanning (ported from production) ─────────────────────────
def _scan_subject_boundaries(raw_lines: List[str]) -> List[Dict]:
    """Scan document for subject headers to help the model identify subjects."""
    boundaries = []
    _SUBJECT_RE = r"PHYSICS|CHEMISTRY|MATH(?:EMATICS)?|BIOLOGY|BOTANY|ZOOLOGY|MATHS"
    patterns = [
        rf"\\section\*?\{{({_SUBJECT_RE})\s*:?\}}",
        rf"^##?\s+({_SUBJECT_RE})\s*:?\s*$",
        rf"^\s*({_SUBJECT_RE})\s*:?\s*$",
    ]
    # Normalize subject names to canonical form
    _SUBJECT_MAP = {
        "Maths": "Mathematics",
        "Botany": "Biology",
        "Zoology": "Biology",
    }
    for idx, line in enumerate(raw_lines):
        for pat in patterns:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                subj_raw = m.group(1).strip().title()
                subj_raw = _SUBJECT_MAP.get(subj_raw, subj_raw)
                boundaries.append({"line": idx, "subject": subj_raw})
                break
    return boundaries


def _scan_answer_key_location(raw_lines: List[str]) -> Optional[int]:
    """Scan document for answer key / key sheet section start line."""
    patterns = [
        r"(?:Answer\s*Key|KEY\s*SHEET|ANSWER\s*KEY)",
        r"\\section\*?\{.*(?:Answer|Key\s*Sheet).*\}",
    ]
    for idx, line in enumerate(raw_lines):
        for pat in patterns:
            if re.search(pat, line, re.IGNORECASE):
                return idx
    return None


# ── Prompt building (adapted from production PCMB_Cached_Mapper.py) ────────
def _build_system_instruction(extra_context: str = "") -> str:
    """Build system instruction for the LLM."""
    base = (
        "You are an expert exam paper parser.\n"
        "This document contains an exam paper which may have multiple subjects. "
        "Each subject section has questions (SCQ/MCQ/Numerical), possibly followed "
        "by an Answer Key and Solutions.\n"
        "Your job: extract questions from the Questions section. Use the Answer Key "
        "to fill answers. Map solutions from the Solutions section.\n"
        "CRITICAL RULES:\n"
        "- Subject must be one of: \"Physics\", \"Chemistry\", \"Mathematics\", \"Biology\". "
        "Never return \"Unknown\".\n"
        "- Detect subject from section headers or infer from question content.\n"
        "- Stop extracting when you reach the Key Sheet, Answer Key, or Solutions section "
        "(those are NOT questions)."
    )
    if extra_context:
        base += (
            f"\n\nADDITIONAL CONTEXT FROM USER (treat as ground truth, overrides any "
            f"conflicting info from the document):\n{extra_context}"
        )
    return base


def _build_extraction_prompt(
    start_line: int,
    batch_size: int,
    total_lines: int,
    extra_context: str = "",
    is_pyq: bool = False,
    prev_exam: str = "",
    pyq_years: str = "",
    subject_boundaries: Optional[List[Dict]] = None,
    answer_key_line: Optional[int] = None,
) -> str:
    """Build the extraction prompt — matched to production PCMB_Cached_Mapper.py."""

    pyq_context = ""
    if is_pyq and prev_exam:
        pyq_context = (
            f"\nIMPORTANT: These are Previous Year Questions (PYQ) from {prev_exam} "
            f"({pyq_years}).\n"
            f"- Set isPYQ: true for all questions\n"
            f"- Set previousExam: \"{prev_exam}\"\n"
            f"- Try to identify the year of each question if mentioned in the document\n"
        )

    user_context = ""
    if extra_context:
        user_context = (
            f"\nUSER-PROVIDED CONTEXT (treat as ground truth, overrides any conflicting "
            f"info from the document):\n{extra_context}\n"
        )

    # Subject boundary hints (from pre-scan)
    subject_hint = ""
    if subject_boundaries:
        subject_hint = "\nSUBJECT BOUNDARIES (from section headers in the document):\n"
        for b in subject_boundaries:
            subject_hint += f"- {b['subject']} questions start around Line_{b['line']:05d}\n"
        subject_hint += (
            "- Use these boundaries to determine the subject for each question.\n"
            "- A question after a subject header belongs to THAT subject, regardless of content.\n"
        )

    # Answer key hint (from pre-scan)
    answer_key_hint = ""
    if answer_key_line is not None:
        answer_key_hint = (
            f"\nANSWER KEY NOTE:\n"
            f"- An explicit Answer Key / Key Sheet section appears around Line_{answer_key_line:05d}.\n"
            f"- Look up each question's answer from that table when possible.\n"
            f"- If options are numbered 1/2/3/4, map them to A/B/C/D.\n"
            f"- If a question is not listed there, return answer_key as null.\n"
        )

    return f"""ROLE: You are a document labeler, not a content generator.

TASK: Label questions in this exam document using the fields we define.
DO NOT invent or alter any content. Only label what exists in the document.

START READING FROM: Line_{start_line:05d}
EXTRACT THE NEXT: {batch_size} questions
{pyq_context}{user_context}{subject_hint}{answer_key_hint}
DOCUMENT STRUCTURE:
- Subject headers may appear as: \\section*{{PHYSICS}}, \\section*{{CHEMISTRY}}, \\section*{{MATHS}}, ## Physics, etc.
- Questions may appear as: "N." or "N " or "Q.N" or "\\section*{{Q. N}}" at line start
- Each question may have: Options (1/2/3/4 or (A)/(B)/(C)/(D)), Answer, and Solution
- The document may have zones: Questions -> Answer Key -> Solutions

EXTRACTION RULES:
1. Identify the SUBJECT from section headers or question content. Subject MUST be one of: "Physics", "Chemistry", "Mathematics", "Biology".
   Never return "Unknown". If no header is nearby, infer from the question content.
2. Find question START line and END line (inclusive, before the next question starts or solution begins).
   CRITICAL: Each question must have UNIQUE start/end lines. Two different questions MUST NOT overlap or share line ranges.
   IMPORTANT: The question END line must NOT include answer lines like "Ans. (7)" or "Answer: B". If the answer appears on the same line or next line after the question, the question END should be the line BEFORE the answer. The answer value goes in "answer_key" only.
3. If a solution exists for this question, return its START line and END line. If no solution, return solution as null.
4. Extract the correct answer ONLY if explicitly present in the document (near question, in an answer key, or marked). Return as A/B/C/D for MCQ or numeric string for numerical.
   - If options are labeled 1/2/3/4, map: 1->A, 2->B, 3->C, 4->D
   - For numerical-type questions, return the numeric value as a string.
   - If no answer is explicitly present, return null (do NOT solve).
5. Detect question type from section context:
   - Under "SINGLE CORRECT ANSWER TYPE" or with 4 options, single correct -> "scq"
   - Under "NUMERICAL VALUE TYPE" -> "integerQuestion"
   - Multiple correct options -> "mcq"
   - Passage/comprehension with sub-questions -> "passageQuestion" (NOT "subjective" or "comprehension")
6. Do NOT skip questions. Return every question you encounter after the start line until you reach the requested count.
7. ACCURACY: Each JSON entry must correspond to EXACTLY ONE question. Do NOT combine multiple questions into one entry.
   If Q116 ends at line 500 and Q117 starts at line 501, they must be SEPARATE entries with distinct line ranges.

GARBAGE AVOIDANCE:
- Do NOT extract answer key rows, solution headers, key sheet tables, or blank sections as questions.
- Do NOT jump forward more than ~{batch_size * 25} lines from the start line. If you cannot find questions nearby, return EOF.
- If content is garbled, has OCR noise, or is non-question material, SKIP it — do not include it.

EOF DETECTION:
- If there are NO MORE QUESTIONS after the start line (only solutions, answer keys, key sheets, or blank content), you MUST return EOF immediately: {{"eof": true, "last_content_line": <last_line_with_content>}}
- Do NOT try to re-extract questions from earlier in the document. If the start line is past the questions section, return EOF.
- Only return EOF if you are CERTAIN no new questions remain. Do not return EOF if you simply cannot find {batch_size} questions — return whatever new questions you find.

OUTPUT FORMAT (JSON):
- If questions found, return JSON array:
[
    {{
        "subject": "Physics",
        "type": "scq",
        "answer_key": "B",
        "question": {{"start": 45, "end": 52}},
        "solution": {{"start": 530, "end": 545}}
    }}
]

- If NO questions remain (end of document), return:
{{"eof": true, "last_content_line": 1234}}

IMPORTANT:
- Use line INDEX numbers (from Line_XXXXX prefixes in the document)
- subject MUST be one of: "Physics", "Chemistry", "Mathematics", "Biology"
- If no explicit answer is present in the document, answer_key must be null
- Do NOT extract content from solutions or key sheets as questions
- Return EOF only when genuinely at end of questions, not when batch is partial
- Do NOT assign question IDs — only provide line references. IDs are assigned by our system."""


# ── LLM call wrappers ──────────────────────────────────────────────────────
def _call_gemini(
    decorated_content: str,
    prompt: str,
    system_instruction: str,
    model_name: str = "gemini-3-flash-preview",
) -> Optional[Any]:
    """Call Gemini API for extraction. Returns parsed JSON."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=[decorated_content, prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=GENERATION_TIMEOUT * 1000),
        ),
    )

    return json.loads(response.text)


def _call_gemini_cached(
    decorated_content: str,
    prompt: str,
    system_instruction: str,
    model_name: str = "gemini-3-flash-preview",
    cache_handle: Any = None,
) -> Tuple[Optional[Any], Any]:
    """Call Gemini API with context caching for cost savings.

    Returns (parsed_json, cache_handle) — pass cache_handle to subsequent calls.
    Cache saves ~90% on repeated calls with the same document content.
    Uses google.genai SDK (not deprecated google.generativeai).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)

    if cache_handle is None:
        # Create cache on first call
        try:
            cache_handle = client.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    contents=[types.Content(
                        role="user",
                        parts=[types.Part(text=decorated_content)],
                    )],
                    ttl=f"{60 * 60}s",  # 60 minutes in seconds
                ),
            )
        except Exception:
            # Fallback to non-cached if caching not supported
            result = _call_gemini(decorated_content, prompt, system_instruction, model_name)
            return result, None

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                cached_content=cache_handle.name,
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=GENERATION_TIMEOUT * 1000),
            ),
        )
        return json.loads(response.text), cache_handle
    except Exception as e:
        err_str = str(e)
        # Cache expired or not found — recreate
        if "not found" in err_str.lower() or "403" in err_str or "expired" in err_str.lower():
            try:
                cache_handle = client.caches.create(
                    model=model_name,
                    config=types.CreateCachedContentConfig(
                        system_instruction=system_instruction,
                        contents=[types.Content(
                            role="user",
                            parts=[types.Part(text=decorated_content)],
                        )],
                        ttl=f"{60 * 60}s",
                    ),
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cache_handle.name,
                        response_mime_type="application/json",
                        http_options=types.HttpOptions(timeout=GENERATION_TIMEOUT * 1000),
                    ),
                )
                return json.loads(response.text), cache_handle
            except Exception:
                pass
        raise


def _call_anthropic_cached(
    decorated_content: str,
    prompt: str,
    system_instruction: str,
    model_name: str = "claude-sonnet-4-6",
) -> Optional[Any]:
    """Call Anthropic API with prompt caching for 90% cost savings on cached reads.

    Caching strategy:
    - System instruction: cached (cache_control ephemeral, 5-min TTL, refreshes on hit)
    - Decorated document: cached as first user message block
    - Extraction prompt: NOT cached (changes per batch)

    Pricing (claude-sonnet-4-6):
    - Input: $3.00/MTok | Output: $15.00/MTok
    - Cache write: $3.75/MTok (25% surcharge, first call only)
    - Cache read: $0.30/MTok (90% discount, all subsequent calls)

    Context window: 200K tokens. Max output: 8192 tokens (sufficient for 15-Q batch JSON).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    response = client.messages.create(
        model=model_name,
        max_tokens=8192,
        system=[
            {
                "type": "text",
                "text": system_instruction,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"DOCUMENT:\n{decorated_content}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"EXTRACTION TASK (respond with JSON only):\n{prompt}",
                    },
                ],
            },
        ],
        timeout=GENERATION_TIMEOUT,
    )

    raw_text = response.content[0].text

    # Parse JSON from response — Sonnet 4.6 may add reasoning text before JSON
    # or wrap in markdown code blocks
    text = raw_text.strip()

    # Strip markdown code fence if present
    if "```" in text:
        import re as _re
        fence_match = _re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()

    # Try direct parse first
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Sonnet may add reasoning before JSON — extract JSON array or object
        import re as _re
        # Try to find JSON array
        arr_match = _re.search(r"(\[[\s\S]*\])", text)
        if arr_match:
            data = json.loads(arr_match.group(1))
        else:
            # Try to find JSON object (for EOF responses)
            obj_match = _re.search(r"(\{[\s\S]*\})", text)
            if obj_match:
                data = json.loads(obj_match.group(1))
            else:
                raise

    # Unwrap {"questions": [...]} if present
    if isinstance(data, dict):
        if "questions" in data:
            return data["questions"]
        if data.get("eof"):
            return data
    return data


def _call_llm(
    provider: str,
    decorated_content: str,
    prompt: str,
    system_instruction: str,
    cache_handle: Any = None,
) -> Tuple[Optional[Any], Any]:
    """Dispatch to the selected LLM provider.

    Returns (result, cache_handle). cache_handle is None for Anthropic.
    Anthropic uses automatic prompt caching via cache_control (no handle needed).
    """
    for attempt in range(MAX_RETRIES):
        try:
            if provider == "anthropic":
                return _call_anthropic_cached(decorated_content, prompt, system_instruction), None
            else:
                return _call_gemini_cached(
                    decorated_content, prompt, system_instruction,
                    cache_handle=cache_handle,
                )
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
    return None, cache_handle


# ── Item processing (adapted from production process_item) ──────────────────
def _process_item(
    item: Dict,
    raw_lines: List[str],
    processed_uids: set,
    accepted_ranges: List[Tuple[int, int]],
    cursor_line: int,
) -> Optional[Dict]:
    """Process a single LLM-returned item into the output format.

    Forked from production PCMB_Cached_Mapper.py:process_item()

    Args:
        accepted_ranges: list of (line_start, line_end) tuples for all
            previously accepted questions (any subject). Used to reject
            overlapping fragments that slip past the UID-based dedup.
    """
    q_range = item.get("question", {})
    s_range = item.get("solution") or {}

    q_start = q_range.get("start")
    q_end = q_range.get("end")

    if q_start is None or q_end is None:
        return None

    # Ensure integers
    q_start = int(q_start) if isinstance(q_start, (int, float, str)) else 0
    q_end = int(q_end) if isinstance(q_end, (int, float, str)) else 0

    q_start = max(0, q_start)
    q_end = min(q_end, len(raw_lines) - 1)

    # Dedup by start line (exact match)
    uid = f"L{q_start}"
    if uid in processed_uids:
        return None

    # Overlap rejection: reject if this question's line range overlaps with
    # any already-accepted question (regardless of subject). This catches:
    #   - Cross-subject boundary fragments (e.g. Physics Q42 overlapping Chemistry Q1)
    #   - Near-duplicate fragments (e.g. BIO Q70 at L1292 vs BIO Q71 at L1293)
    for (acc_start, acc_end) in accepted_ranges:
        # Two ranges [q_start, q_end] and [acc_start, acc_end] overlap
        # if q_start <= acc_end AND q_end >= acc_start
        if q_start <= acc_end and q_end >= acc_start:
            return None

    # Sanity: q_start should be near or after cursor (allow some backward tolerance)
    if q_start < cursor_line - SAFETY_OVERLAP_LINES - 10:
        return None

    # Sanity: single question shouldn't span too many lines
    if q_end - q_start > MAX_QUESTION_SPAN:
        return None

    # Skip blank leading lines
    while q_start <= q_end and not raw_lines[q_start].strip():
        q_start += 1
    if q_start > q_end:
        return None

    # Slice question content
    q_content = _slice_content(raw_lines, q_start, q_end)
    if _is_garbage(q_content):
        return None

    # Slice solution content (if provided)
    s_content = None
    s_start = s_range.get("start")
    s_end = s_range.get("end")
    if s_start is not None and s_end is not None:
        s_start = int(s_start) if isinstance(s_start, (int, float, str)) else None
        s_end = int(s_end) if isinstance(s_end, (int, float, str)) else None
        if s_start is not None and s_end is not None:
            # Solution should not overlap question
            if s_start > q_end:
                s_content = _slice_content(raw_lines, s_start, s_end)

    # Normalize subject
    subject = _normalize_subject(item.get("subject"))
    if subject is None:
        return None

    # Normalize answer
    answer_key = item.get("answer_key")
    if answer_key is not None:
        answer_key = str(answer_key).strip()
        if not answer_key:
            answer_key = None

    processed_uids.add(uid)
    accepted_ranges.append((q_start, q_end))

    return {
        "subject": subject,
        "questionType": item.get("type", "scq"),
        "answer_key": answer_key,
        "question_content": q_content,
        "solution_content": s_content,
        "line_start": q_start,
        "line_end": q_end,
        "sol_line_start": s_start if s_content else None,
        "sol_line_end": s_end if s_content else None,
    }


# ── Main stage class ───────────────────────────────────────────────────────
class LLMMapperStage(BasePipelineStage):
    _stop_requested: bool = False

    def request_stop(self):
        """Set the stop flag so the extraction loop terminates after the current batch."""
        self._stop_requested = True

    @property
    def stage_name(self) -> str:
        return "stage2_mapping"

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        max_q = kwargs.get("max_questions", 5)
        mmd_dir = self.session_dir / "mathpix_output"
        mmd_files = list(mmd_dir.glob("*.mmd")) if mmd_dir.exists() else []
        return {
            "mmd_files": [f.name for f in mmd_files],
            "max_questions": max_q,
        }

    def _get_test_code(self) -> str:
        """Read test_code from session_config, or derive from PDF filename."""
        config_path = self.session_dir / "session_config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if config.get("test_code"):
                    return config["test_code"]
                if config.get("pdf_filename"):
                    return _derive_test_code(config["pdf_filename"])
            except Exception:
                pass
        # Last resort: check mathpix_status.json for pdf_filename
        status_path = self.session_dir / "mathpix_output" / "mathpix_status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                pdf_name = status.get("pdf_filename", "")
                if pdf_name:
                    return _derive_test_code(pdf_name)
            except Exception:
                pass
        return "TEST"

    def _get_session_config(self) -> Dict:
        """Read full session config."""
        config_path = self.session_dir / "session_config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _find_mmd_files(self) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Find MMD files in mathpix_output directory.

        Returns (questions_mmd, solutions_mmd, answer_key_file).
        Falls back to any .mmd file if no keyword match.
        """
        mmd_dir = self.session_dir / "mathpix_output"
        if not mmd_dir.exists():
            return None, None, None

        q_mmd = None
        s_mmd = None
        key_file = None

        for f in mmd_dir.iterdir():
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if "question" in name_lower and f.suffix == ".mmd":
                q_mmd = f
            elif "solution" in name_lower and f.suffix == ".mmd":
                s_mmd = f
            elif "key" in name_lower and f.suffix in (".txt", ".mmd"):
                key_file = f

        # Fallback: if no keyword-matched questions file, use any .mmd file
        if not q_mmd:
            all_mmds = sorted([
                f for f in mmd_dir.iterdir()
                if f.suffix == ".mmd" and f.is_file()
            ])
            if all_mmds:
                q_mmd = all_mmds[0]
                self.log("info", f"Using MMD file: {q_mmd.name}")

        return q_mmd, s_mmd, key_file

    def _merge_part_results(self, out_dir: Path, by_subject: Dict, test_code: str,
                             part: int, subject_prefix: Dict) -> None:
        """In multi-part mode, merge current part results with previous parts.

        Each part writes its own intermediate file. When a later part completes,
        we merge all part files into the unified by_subject dict.
        """
        # Save current part's results as intermediate file
        part_dir = out_dir / "parts"
        part_dir.mkdir(exist_ok=True)
        for subject in ("Physics", "Chemistry", "Mathematics", "Biology"):
            items = by_subject.get(subject, {})
            if not items:
                continue
            part_file = part_dir / f"part{part}_{subject}.json"
            with open(part_file, "w", encoding="utf-8") as f:
                json.dump(list(items.values()), f, ensure_ascii=False, indent=2)

        # Now load ALL part results into by_subject for unified output
        by_subject.clear()
        for subject in ("Physics", "Chemistry", "Mathematics", "Biology"):
            by_subject[subject] = {}
            for pf in sorted(part_dir.glob(f"part*_{subject}.json")):
                try:
                    data = json.loads(pf.read_text(encoding="utf-8"))
                    for q in data:
                        uid = f"P{pf.name.split('_')[0]}_L{q.get('line_start', 0)}"
                        by_subject[subject][uid] = q
                except Exception:
                    pass

        merged_total = sum(len(v) for v in by_subject.values())
        self.log("info", f"Merged {merged_total} questions from all parts")

    def _save_partial_output(self, out_dir: Path, by_subject: Dict, test_code: str,
                              cursor_line: int, total_extracted: int) -> None:
        """Save partial output and checkpoint for resume support."""
        subject_prefix = {
            "Physics": "PHY", "Chemistry": "CHEM",
            "Mathematics": "MATH", "Biology": "BIO",
        }
        all_questions = []
        by_subj_counts = {}
        for subject in ("Physics", "Chemistry", "Mathematics", "Biology"):
            items = by_subject.get(subject, {})
            if not items:
                continue
            sorted_items = sorted(items.values(), key=lambda x: x["line_start"])
            prefix = subject_prefix.get(subject, "UNK")
            for idx, item in enumerate(sorted_items, start=1):
                item["qid"] = f"{test_code}_{prefix}_Q{idx}"
                item["question_number"] = idx
            all_questions.extend(sorted_items)
            by_subj_counts[subject] = len(sorted_items)

        # Write partial output file
        partial_path = out_dir / "partial_output.json"
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)

        # Save running checkpoint with cursor position
        self.save_checkpoint("running", {
            "total_extracted": total_extracted,
            "cursor_line": cursor_line,
            "by_subject": by_subj_counts,
        })

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute LLM-based question extraction.

        Forked from production PCMB_Cached_Mapper.py cursor-based extraction loop.
        """
        self._stop_requested = False
        max_q = kwargs.get("max_questions", 30)
        provider = kwargs.get("llm_provider", "gemini")
        is_pyq = kwargs.get("is_pyq", False)
        prev_exam = kwargs.get("prev_exam", "")
        pyq_year_from = kwargs.get("pyq_year_from")
        pyq_year_to = kwargs.get("pyq_year_to")
        extra_context = kwargs.get("extra_context", "")
        resume = kwargs.get("resume", False)

        out_dir = self.session_dir / "mapped_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Fresh start: clean previous output if not resuming ────────────
        if not resume:
            # Remove old mapped output files so fresh extraction overwrites cleanly
            for old_file in out_dir.glob("*.json"):
                old_file.unlink()
            # Also remove parts subdir if exists
            parts_dir = out_dir / "parts"
            if parts_dir.exists():
                for old_file in parts_dir.glob("*.json"):
                    old_file.unlink()
            self.log("info", "Cleared previous extraction output — starting fresh")

        test_code = self._get_test_code()
        self.log("info", f"Test code: {test_code} | Provider: {provider} | Max: {max_q}")

        # ── Validate API key ───────────────────────────────────────────────
        if provider == "gemini" and not settings.gemini_api_key:
            self.log("error", "Gemini API key not configured in .env")
            self.save_checkpoint("failed", {"error": "Missing Gemini API key"})
            return {"error": "Gemini API key not configured"}
        if provider == "anthropic" and not settings.anthropic_api_key:
            self.log("error", "Anthropic API key not configured in .env")
            self.save_checkpoint("failed", {"error": "Missing Anthropic API key"})
            return {"error": "Anthropic API key not configured"}

        # ── Find MMD files ─────────────────────────────────────────────────
        part = kwargs.get("part")
        if part:
            # Multi-part mode: use specific part file
            part_path = self.session_dir / "mathpix_output" / f"part{part}_combined.mmd"
            if not part_path.exists():
                self.log("error", f"Part {part} file not found: {part_path.name}")
                return {"error": f"Part {part} file not found"}
            q_mmd = part_path
            s_mmd = None
            key_file = None
            self.log("info", f"Multi-part mode: processing {part_path.name}")
        else:
            q_mmd, s_mmd, key_file = self._find_mmd_files()
        if not q_mmd:
            self.log("error", "No MMD file found in mathpix_output/")
            self.save_checkpoint("failed", {"error": "No MMD file found"})
            return {"error": "No MMD file found"}

        # ── Read and decorate ──────────────────────────────────────────────
        self.log("info", f"Reading {q_mmd.name}...")
        raw_text = q_mmd.read_text(encoding="utf-8")
        raw_lines = raw_text.splitlines(keepends=True)
        total_lines = len(raw_lines)
        self.log("info", f"Loaded {total_lines} lines from {q_mmd.name}")

        # If separate solutions/key files exist, append them
        if s_mmd and s_mmd != q_mmd:
            self.log("info", f"Appending solutions from {s_mmd.name}...")
            sol_text = s_mmd.read_text(encoding="utf-8")
            sol_lines = sol_text.splitlines(keepends=True)
            raw_lines.extend(["\n", "--- SOLUTIONS FILE ---\n"])
            raw_lines.extend(sol_lines)
            self.log("info", f"Appended {len(sol_lines)} solution lines")

        if key_file:
            self.log("info", f"Appending answer key from {key_file.name}...")
            key_text = key_file.read_text(encoding="utf-8")
            key_lines = key_text.splitlines(keepends=True)
            raw_lines.extend(["\n", "--- ANSWER KEY FILE ---\n"])
            raw_lines.extend(key_lines)
            self.log("info", f"Appended {len(key_lines)} answer key lines")

        total_lines = len(raw_lines)
        decorated_content = _decorate_lines(raw_lines)
        self.log("success", f"Decorated {total_lines} total lines")
        self.progress(10, "Content prepared")

        # ── Build prompts ──────────────────────────────────────────────────
        pyq_years = ""
        if pyq_year_from and pyq_year_to:
            pyq_years = f"{pyq_year_from}-{pyq_year_to}"
        elif pyq_year_from:
            pyq_years = str(pyq_year_from)

        system_instruction = _build_system_instruction(extra_context)
        cache_handle = None  # For Gemini context caching

        # ── Pre-scan document for subject headers and answer key ──────────
        # These are optional hints — if nothing detected, model works on its own
        subject_boundaries = _scan_subject_boundaries(raw_lines)
        answer_key_line = _scan_answer_key_location(raw_lines)
        if subject_boundaries:
            self.log("info", f"Detected {len(subject_boundaries)} subject headers: " +
                     ", ".join(f"{b['subject']}@L{b['line']}" for b in subject_boundaries))
        if answer_key_line is not None:
            self.log("info", f"Detected answer key section at line {answer_key_line}")

        # ── Resume support: load partial results if requested ─────────────
        cursor_line = 0
        batch_size = min(BATCH_SIZE_DEFAULT, max_q)
        processed_uids = set()
        accepted_ranges: List[Tuple[int, int]] = []  # (line_start, line_end) for overlap rejection
        by_subject: Dict[str, Dict[str, Dict]] = {
            "Physics": {}, "Chemistry": {}, "Mathematics": {}, "Biology": {},
        }
        consecutive_empty = 0
        total_extracted = 0

        # Dynamic jump capping state (from production)
        avg_lines_per_q = INITIAL_AVG_LINES_PER_Q
        total_questions_for_avg = 0
        last_capped_target = None
        cap_attempt_count = 0

        if resume:
            partial_path = out_dir / "partial_output.json"
            if partial_path.exists():
                try:
                    partial_data = json.loads(partial_path.read_text(encoding="utf-8"))
                    for q in partial_data:
                        subj = q.get("subject")
                        if subj and subj in by_subject:
                            uid = f"L{q.get('line_start', 0)}"
                            processed_uids.add(uid)
                            by_subject[subj][uid] = q
                            ls = q.get("line_start")
                            le = q.get("line_end")
                            if ls is not None and le is not None:
                                accepted_ranges.append((ls, le))
                    total_extracted = sum(len(items) for items in by_subject.values())
                    # Resume cursor from checkpoint
                    from app.db import get_checkpoint
                    cp = get_checkpoint(self.session_id, "stage2_mapping")
                    if cp and cp.get("data", {}).get("cursor_line"):
                        cursor_line = cp["data"]["cursor_line"]
                    self.log("info", f"Resumed from partial: {total_extracted} questions, cursor at line {cursor_line}")
                except Exception as e:
                    self.log("warning", f"Could not resume: {e}. Starting fresh.")
            else:
                self.log("info", "No partial output found. Starting fresh.")

        start_time = time.time()

        while total_extracted < max_q:
            # ── Stop checks ───────────────────────────────────────────────
            if self._stop_requested:
                self.log("warning", "Stop requested by user — saving partial results")
                self._save_partial_output(out_dir, by_subject, test_code, cursor_line, total_extracted)
                break

            if cursor_line >= total_lines:
                self.log("info", "Reached end of file")
                break

            if time.time() - start_time > OVERALL_TIMEOUT:
                self.log("warning", f"Overall timeout ({OVERALL_TIMEOUT}s) reached. Saving partial results.")
                self._save_partial_output(out_dir, by_subject, test_code, cursor_line, total_extracted)
                break

            remaining = max_q - total_extracted
            current_batch = min(batch_size, remaining)

            self.log("info", f"Extracting from line {cursor_line} (batch of {current_batch})...")

            prompt = _build_extraction_prompt(
                start_line=cursor_line,
                batch_size=current_batch,
                total_lines=total_lines,
                extra_context=extra_context,
                is_pyq=is_pyq,
                prev_exam=prev_exam,
                pyq_years=pyq_years,
                subject_boundaries=subject_boundaries,
                answer_key_line=answer_key_line,
            )

            # ── Call LLM ───────────────────────────────────────────────────
            try:
                batch_data, cache_handle = _call_llm(
                    provider, decorated_content, prompt, system_instruction,
                    cache_handle=cache_handle,
                )
            except Exception as e:
                err_msg = str(e)
                self.log("error", f"LLM call failed: {err_msg[:200]}")
                # Rate limit handling (from production)
                if "429" in err_msg:
                    self.log("warning", "Rate limit hit. Waiting 30s...")
                    time.sleep(30)
                # Blind jump on failure
                cursor_line += 100
                consecutive_empty += 1
                # Shrink batch size on failure (production pattern)
                batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                if consecutive_empty >= 3:
                    self.log("error", "3 consecutive failures. Stopping extraction.")
                    break
                continue

            # ── Handle EOF ─────────────────────────────────────────────────
            if isinstance(batch_data, dict) and batch_data.get("eof"):
                # Accept EOF only if we have enough or few lines remain
                lines_remaining = total_lines - cursor_line
                if total_extracted > 0 or lines_remaining < EOF_REMAINING_THRESHOLD:
                    self.log("info", f"LLM returned EOF at line {cursor_line}. Total: {total_extracted}")
                    break
                else:
                    # Skeptical of early EOF — blind jump and retry
                    self.log("warning", f"EOF with 0 questions and {lines_remaining} lines remaining. Jumping ahead.")
                    cursor_line += 100
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        self.log("info", "3 consecutive EOF/empty. Stopping.")
                        break
                    continue

            # ── Handle empty response ──────────────────────────────────────
            if not batch_data or (isinstance(batch_data, list) and len(batch_data) == 0):
                consecutive_empty += 1
                self.log("info", f"Empty batch #{consecutive_empty}")
                if consecutive_empty >= 2 and total_extracted > 0:
                    self.log("info", "Consecutive empty batches with results. Stopping.")
                    break
                if consecutive_empty >= 3:
                    self.log("info", "3 empty batches. Stopping.")
                    break
                cursor_line += BLIND_JUMP_SIZE
                continue

            if not isinstance(batch_data, list):
                self.log("error", f"Unexpected response type: {type(batch_data)}")
                cursor_line += BLIND_JUMP_SIZE
                consecutive_empty += 1
                continue

            # ── Process batch items (from production) ──────────────────────
            consecutive_empty = 0
            highest_end_line = cursor_line
            new_count = 0

            # Sort by question start line
            batch_data.sort(key=lambda x: (x.get("question") or {}).get("start", 999999))

            for item in batch_data:
                if total_extracted + new_count >= max_q:
                    break

                result = _process_item(item, raw_lines, processed_uids, accepted_ranges, cursor_line)
                if result is None:
                    continue

                subject = result["subject"]
                temp_qid = f"TEMP_L{result['line_start']}"
                result["qid"] = temp_qid
                result["source"] = test_code

                if temp_qid not in by_subject[subject]:
                    by_subject[subject][temp_qid] = result
                    new_count += 1

                q_end = result["line_end"]
                if q_end > highest_end_line:
                    highest_end_line = q_end

            total_extracted = sum(len(items) for items in by_subject.values())
            self.log("success", f"Batch: +{new_count} new, {total_extracted} total")

            # Progress update
            pct = min(90, 10 + int(80 * total_extracted / max_q))
            self.progress(pct, f"{total_extracted}/{max_q} questions")

            # Incremental save after each batch for resume support
            self._save_partial_output(out_dir, by_subject, test_code, cursor_line, total_extracted)

            # ── Advance cursor with dynamic jump capping (production) ──────
            next_cursor = highest_end_line - SAFETY_OVERLAP_LINES
            if next_cursor <= cursor_line:
                next_cursor = highest_end_line + 1

            # Update running average lines-per-question
            batch_count = len(batch_data) if isinstance(batch_data, list) else 0
            if batch_count > 0:
                batch_lines = highest_end_line - cursor_line
                batch_avg = batch_lines / batch_count
                if total_questions_for_avg == 0:
                    avg_lines_per_q = batch_avg
                else:
                    avg_lines_per_q = (
                        (avg_lines_per_q * total_questions_for_avg + batch_avg * batch_count)
                        / (total_questions_for_avg + batch_count)
                    )
                total_questions_for_avg += batch_count

            # Dynamic jump limit: cap large jumps to prevent skipping
            expected_max_jump = max(
                int(batch_count * avg_lines_per_q * JUMP_MULTIPLIER),
                MIN_JUMP_ALLOWANCE,
            )
            actual_jump = next_cursor - cursor_line

            is_repeated_jump = (
                last_capped_target is not None
                and abs(next_cursor - last_capped_target) < 50
            )

            if actual_jump > expected_max_jump:
                if is_repeated_jump:
                    cap_attempt_count += 1
                    if cap_attempt_count >= 2:
                        # Allow the jump after 2 cap attempts (garbage escape hatch)
                        self.log("warning",
                                 f"GARBAGE DETECTED: {cap_attempt_count}x to reach line {next_cursor}. Allowing jump.")
                        last_capped_target = None
                        cap_attempt_count = 0
                    else:
                        capped = cursor_line + expected_max_jump
                        self.log("info", f"Jump capped: {next_cursor} -> {capped} (attempt {cap_attempt_count})")
                        last_capped_target = next_cursor
                        next_cursor = capped
                else:
                    capped = cursor_line + expected_max_jump
                    self.log("info", f"Jump capped: {next_cursor} -> {capped}")
                    last_capped_target = next_cursor
                    cap_attempt_count = 1
                    next_cursor = capped
            else:
                last_capped_target = None
                cap_attempt_count = 0

            cursor_line = next_cursor

            # ── Batch size adaptation (production) ─────────────────────────
            if new_count >= current_batch - 2:
                batch_size = min(MAX_BATCH_SIZE, batch_size + 2)
            elif new_count == 0:
                batch_size = max(MIN_BATCH_SIZE, batch_size - 2)
                consecutive_empty += 1
                cursor_line = highest_end_line + BLIND_JUMP_SIZE
                if consecutive_empty >= 2 and total_extracted > 0:
                    break

        # ── Assign final QIDs and write output ─────────────────────────────
        self.log("info", "Assigning QIDs and writing output files...")

        subject_prefix = {
            "Physics": "PHY", "Chemistry": "CHEM",
            "Mathematics": "MATH", "Biology": "BIO",
        }

        # In multi-part mode, merge with existing part results
        if part:
            self._merge_part_results(out_dir, by_subject, test_code, part, subject_prefix)

        all_questions = []
        for subject in ("Physics", "Chemistry", "Mathematics", "Biology"):
            items = by_subject.get(subject, {})
            if not items:
                continue

            # Sort by line_start for sequential numbering
            sorted_items = sorted(items.values(), key=lambda x: x["line_start"])
            prefix = subject_prefix.get(subject, "UNK")

            subject_questions = []
            for idx, item in enumerate(sorted_items, start=1):
                item["qid"] = f"{test_code}_{prefix}_Q{idx}"
                item["question_number"] = idx
                subject_questions.append(item)

            # Save per-subject file (production format)
            out_file = out_dir / f"{test_code}_{subject}_mapped.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(subject_questions, f, ensure_ascii=False, indent=2)
            self.log("success", f"Saved {out_file.name} ({len(subject_questions)} questions)")

            all_questions.extend(subject_questions)

        # ── Summary stats ──────────────────────────────────────────────────
        total = len(all_questions)
        triplets = sum(
            1 for q in all_questions
            if q.get("question_content") and q.get("answer_key") and q.get("solution_content")
        )
        with_answer = sum(1 for q in all_questions if q.get("answer_key"))
        with_solution = sum(1 for q in all_questions if q.get("solution_content"))

        by_subj_counts = {}
        for q in all_questions:
            s = q["subject"]
            by_subj_counts[s] = by_subj_counts.get(s, 0) + 1

        self.log("success", f"Extraction complete: {total} questions")
        for s, c in by_subj_counts.items():
            self.log("info", f"  {s}: {c}")
        self.log("info", f"  Answers: {with_answer}/{total} | Solutions: {with_solution}/{total} | Triplets: {triplets}/{total}")

        # Log sample entries
        for q in all_questions[:3]:
            status = "Q"
            if q.get("answer_key"):
                status += "+A"
            if q.get("solution_content"):
                status += "+S"
            qtext = (q.get("question_content") or "")[:80]
            self.log("info", f"  {q['qid']}: {status} | {q['questionType']} | {qtext}...")

        self.progress(100, "Mapping complete")

        # ── Save test_code to session_config ───────────────────────────────
        config_path = self.session_dir / "session_config.json"
        try:
            config = self._get_session_config()
            config["test_code"] = test_code
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except Exception:
            pass

        # ── Track questions in DB ──────────────────────────────────────────
        from app.db import upsert_question
        for q in all_questions:
            upsert_question(
                self.session_id, q["qid"],
                subject=q["subject"],
                question_type=q["questionType"],
                current_stage="stage2_mapping",
                has_answer=bool(q.get("answer_key")),
                has_solution=bool(q.get("solution_content")),
            )

        self.save_checkpoint("completed", {
            "questions": total,
            "answers": with_answer,
            "solutions": with_solution,
            "triplets": triplets,
            "by_subject": by_subj_counts,
            "provider": provider,
        })

        return {
            "questions": total,
            "triplets": triplets,
            "by_subject": by_subj_counts,
            "provider": provider,
        }
