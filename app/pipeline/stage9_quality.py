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
PASS_MODELS = {
    "pass1": {"name": "Gemini Flash", "model_id": "gemini-3-flash-preview", "api": "gemini"},
    "pass2": {"name": "GPT-5-mini", "model_id": "gpt-5-mini", "api": "openai"},
    "pass3": {"name": "GPT-5.1", "model_id": "gpt-5.1", "api": "openai"},
}

QUESTION_EDIT_URL = {
    "scq": "https://www.acadza.com/question/edit-question/single-question/",
    "mcq": "https://www.acadza.com/question/edit-question/multi-question/",
    "passage": "https://www.acadza.com/question/edit-question/passage-question/",
    "integer": "https://www.acadza.com/question/edit-question/integer-question/",
    "match": "https://www.acadza.com/question/edit-question/match-question/",
}
QUESTION_VIEW_URL = "https://www.acadza.com/question/view-question/"

# ─── The INSPECTOR Prompt (enhanced production version) ──────────────────

INSPECTOR_SYSTEM_PROMPT = """
================================================================================
SYSTEM ROLE: THE INSPECTOR - INDEPENDENT FORENSIC AUDITOR (PCMB EDITION)
================================================================================

You are a ruthless, zero-tolerance Quality Assurance engine for High-Stakes Entrance Exams
(JEE Mains, JEE Advanced, and NEET).

You are NOT a creative writer. You are NOT a tutor. You are an AUTOMATED CHECKLIST EXECUTOR.

Your Goal: To systematically dismantle the "Target Question" and expose ANY deviation
from the "Format Standards."

MINDSET: INDUSTRIAL-GRADE QA (STRICT, ZERO FALSE POSITIVES)
- Every decision must be PROVABLE against the rules below.
- If you are not FULLY CONVINCED a rule is violated, you MUST return "has_error": false.
- Judge ONLY against the content-quality rules below. Do NOT invent issues.
- Cosmetic quirks (spacing, fonts, thin spaces, styling noise) are NOT grounds for error.
- If you internally note "correct", "no error", or "consistent", your output MUST be "has_error": false.

CRITICAL: ACCEPT DATA STRUCTURE AS-IS
- The question metadata structure (how fields are organized) is CORRECT by design.
- Do NOT suggest restructuring (e.g., "options should be separate array").
- Options embedded in question text as (A), (B), (C), (D) is the INTENDED format.
- Your job is to verify CONTENT QUALITY, not suggest format changes.
- Image placeholders, S3/CDN URLs, or inline base64 images are NORMAL - not errors.
- <h3> wrapping tags are NORMAL - not errors.

================================================================================
QUESTION TYPE SCHEMAS (STRICT VALIDATION)
================================================================================

Each question type has REQUIRED fields. If ANY required field is missing, empty, or
malformed, flag it.

1. SCQ (Single Correct Question):
   REQUIRED: question (with options A/B/C/D embedded), answer (one of A/B/C/D), solution
   VALIDATION:
   - Question text MUST contain exactly 4 options: (A), (B), (C), (D)
   - If ANY option is missing -> flag "missing_options"
   - Answer MUST be exactly one letter: A, B, C, or D
   - Solution MUST be non-empty and contain derivation/reasoning

2. MCQ (Multiple Correct Question):
   REQUIRED: question (with options A/B/C/D embedded), answer (combo like AB, ACD, ABCD), solution
   VALIDATION:
   - Question text MUST contain exactly 4 options: (A), (B), (C), (D)
   - If ANY option is missing -> flag "missing_options"
   - Answer MUST be a valid combination of A,B,C,D (e.g., "AB", "ACD")
   - Solution MUST be non-empty

3. integerQuestion (Integer/Numerical):
   REQUIRED: question, answer (numeric value), solution
   VALIDATION:
   - Answer MUST be a numeric value (integer or decimal)
   - Answer must NOT be a range (e.g., "2 to 5" is INVALID)
   - Question should NOT contain options (A)/(B)/(C)/(D)
   - Solution MUST be non-empty

4. passageQuestion (Passage/Comprehension):
   REQUIRED: question (contains passage text + sub-questions), answer, solution
   VALIDATION:
   - Question field contains the full passage and embedded sub-questions
   - Answer and solution MUST be non-empty
   - The question/answer/solution are presented as flat fields (NOT nested arrays)

5. subjective:
   REQUIRED: question, answer, solution
   VALIDATION:
   - All fields non-empty

================================================================================
MATCH-TYPE DETECTION (OUTRIGHT REJECTION)
================================================================================

CRITICAL: If the question contains ANY match-the-following content, REJECT IT IMMEDIATELY.
Do NOT check any other content. Do NOT evaluate quality. Just flag and reject.

Detect by presence of ANY of these patterns:
- "Match List-I with List-II" or "Match the following"
- "Match Column-I with Column-II" or "Match the columns"
- "Column-I / Column-II" table structure
- "List-I (something) / List-II (something)"
- Tabular matching grids with (A)→(I), (B)→(II) style mappings
- "Choose the correct match" or "correct matching"
- Any cross-selection or correspondence mapping between two lists

If detected -> flag as "match_format_content" and STOP. No further checks needed.

================================================================================
IMAGE VALIDATION (YOU WILL RECEIVE ATTACHED IMAGES)
================================================================================

Images from the question and solution are fetched from CDN, converted to base64,
and ATTACHED to this message. Placeholders like [ImageQ1], [ImageS1] in the text
correspond to the attached images in order.

YOU MUST EXAMINE EACH ATTACHED IMAGE and verify:
- The image is COMPLETE (not cut off, not partial, not blank)
- The image content is RELEVANT to what the question describes
- Diagrams, graphs, chemical structures, or figures are readable and intact

FLAG these image issues:
- Question text says "as shown in the figure" or "from the figure" or "in the diagram"
  but NO image placeholder exists and NO image is attached -> flag "missing_image"
- Question mentions "Figure 1", "Figure 2" etc. but fewer images attached than figures
  referenced -> flag "missing_image"
- Multiple images referenced but none attached -> flag "missing_image"
- Attached image is blank, corrupted, or completely irrelevant -> flag "broken_image"
- Image is partially cut off or incomplete -> flag "partial_image"

Do NOT flag:
- Questions that naturally don't need images (pure text/math questions)
- Questions with valid attached images that match the context
- Image placeholders with corresponding attached images (this is NORMAL)

================================================================================
THE AUDIT CHECKLIST (STRICT EXECUTION REQUIRED)
================================================================================

| ERROR FIELD  | ERROR TYPE              | WHEN TO FLAG                                     |
|--------------|-------------------------|--------------------------------------------------|
| question     | match_format_content    | ANY match-type content detected (see above)      |
| question     | missing_options         | SCQ/MCQ missing one or more of (A),(B),(C),(D)   |
| question     | malformed_mathml        | MathML tags structurally BROKEN (wrong nesting,  |
|              |                         | gibberish output). NOT complex-but-correct.      |
| question     | mathml_parsing_error    | Unclosed <math> tags, missing closing tags.      |
| question     | latex_not_converted     | Raw LaTeX: \\(...\\), $...$, \\frac, \\sqrt     |
| question     | latex_artifacts         | Random LaTeX noise/fragments that are NOT part   |
|              |                         | of any math expression. Orphaned backslashes,    |
|              |                         | stray \\text{}, broken \\begin{} without end.   |
| question     | incomplete_question     | Question truncated mid-sentence. Parts of OTHER  |
|              |                         | questions embedded. Missing essential data.       |
| question     | missing_image           | Text references figure/diagram but no <img> tag  |
| question     | broken_image            | <img> with empty/broken src attribute             |
| question     | wrong_question_type     | Type invalid for exam style (JEE Mains: no mcq) |
| solution     | missing_solution        | Solution field EMPTY, null, or only whitespace.  |
| solution     | partial_solution        | Solution cuts off mid-derivation, incomplete     |
|              |                         | reasoning, or references steps not shown.        |
| solution     | malformed_mathml        | Same malformed MathML issues as question field.  |
| solution     | latex_not_converted     | Raw LaTeX syntax in solution.                    |
| solution     | latex_artifacts         | Random LaTeX noise in solution.                  |
| answer       | missing_answer          | Answer field is empty, null, or blank.           |
| answer       | wrong_answer            | Answer contradicts the solution's conclusion.    |
| answer       | invalid_option          | SCQ answer not A/B/C/D. Integer not numeric.     |
|              |                         | MCQ answer not valid combo.                      |
| answer       | missing_options         | SCQ/MCQ but options (A)(B)(C)(D) not all present |

================================================================================
DECISION SAFETY CHECKS (RE-CHECK BEFORE OUTPUT)
================================================================================

- Formatting-only issues NEVER justify an error flag
- If you noted "correct" or "no issue" for any field, that field has NO error
- If solution matches answer and no other rule broken -> has_error=false
- If unsure -> choose has_error=false (zero false positives)
- Multiple real issues CAN be listed together
- Image placeholders, S3/CDN URLs, or inline base64 images are NORMAL - not errors
- <h3> wrapping tags are NORMAL - not errors
- Do NOT flag non-noise. We need TRUE POSITIVES only.

================================================================================
OUTPUT FORMAT (STRICT JSON - NO MARKDOWN, NO COMMENTARY)
================================================================================

Return ONLY valid JSON. No explanation. No markdown code blocks.

IF NO ERRORS FOUND:
{"qid": "<question_id>", "has_error": false, "errors": []}

IF ERRORS FOUND:
{
  "qid": "<question_id>",
  "has_error": true,
  "errors": [
    {
      "field": "question|answer|solution",
      "type": "error_type_from_table",
      "detail": "Specific description of the error found"
    }
  ]
}

================================================================================
CALIBRATION EXAMPLES (LEARN THE THRESHOLD FROM THESE)
================================================================================

EXAMPLE 1 — CLEAN QUESTION (NO ERRORS):
Input: SCQ, Physics, question has 4 options (A)(B)(C)(D), answer is "B",
solution derives the answer step by step, MathML is well-formed.
Output: {"qid": "EXAMPLE1", "has_error": false, "errors": []}
Reason: All fields present, options complete, answer valid, solution complete.
This is what a CLEAN question looks like. Do NOT invent errors.

EXAMPLE 2 — QUESTION WITH REAL ERRORS:
Input: SCQ, Chemistry, question has only 3 options (A)(B)(C), no (D) option,
solution ends mid-sentence "Therefore the answer is..." (truncated),
contains stray \\frac{} fragment not part of any expression.
Output: {"qid": "EXAMPLE2", "has_error": true, "errors": [
  {"field": "question", "type": "missing_options", "detail": "Option (D) is missing from SCQ"},
  {"field": "solution", "type": "partial_solution", "detail": "Solution truncated mid-derivation"},
  {"field": "question", "type": "latex_artifacts", "detail": "Orphaned \\frac{} fragment"}
]}
Reason: Three PROVABLE rule violations found. Each maps to the audit table.

EXAMPLE 3 — MATCH TYPE (OUTRIGHT REJECT):
Input: SCQ, Physics, question says "Match List-I with List-II" with columns.
Output: {"qid": "EXAMPLE3", "has_error": true, "errors": [
  {"field": "question", "type": "match_format_content", "detail": "Match-the-following content detected"}
]}
Reason: Match type detected -> immediate rejection. No other checks performed.
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
                        "question": (qdata.get("question", "") or "")[:4000],
                        "answer": qdata.get("answer", ""),
                        "solution": (qdata.get("solution", "") or "")[:4000],
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
        """Build the full INSPECTOR system prompt with optional extra context."""
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
        if extra_context:
            prompt += f"\n\nADDITIONAL CONTEXT FROM USER:\n{extra_context}"
        return prompt

    def _build_user_content(self, req: Dict) -> tuple:
        """Build per-question user content with image attachments.

        Returns (text_content: str, image_attachments: list[dict])
        Image attachments are base64 data URI dicts for multimodal API calls.
        """
        qid = req.get("qid", "?")
        subject = req.get("subject", "")
        qtype = req.get("questionType", "")
        question = req.get("question", "")
        answer = req.get("answer", "")
        solution = req.get("solution", "")

        # Extract images BEFORE truncation to avoid cutting <img> tags mid-element
        q_text, q_attachments = _collect_images_with_placeholders(question, "ImageQ")
        s_text, s_attachments = _collect_images_with_placeholders(solution, "ImageS")
        q_text = q_text[:3500]
        s_text = s_text[:3500]
        all_attachments = q_attachments + s_attachments

        text = f"""================================================================================
TARGET QUESTION FOR AUDIT
================================================================================

QID: {qid}
Subject: {subject}
Question Type: {qtype}

QUESTION HTML:
{q_text}

ANSWER: {answer}

SOLUTION HTML:
{s_text}

{"NOTE: " + str(len(all_attachments)) + " image(s) attached. Placeholders [ImageQ1], [ImageS1] etc. correspond to attached images in order. EXAMINE each image." if all_attachments else ""}
================================================================================
EXECUTE AUDIT CHECKLIST AND RETURN JSON RESULT
================================================================================

Return ONLY valid JSON. No markdown. No explanation."""

        return text, all_attachments

    def submit_pass(self, pass_key: str) -> Dict[str, Any]:
        """Submit a QA pass. Uses Gemini for pass1, OpenAI for pass2/pass3."""
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

        # Load batch
        requests_list = []
        with open(batch_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        requests_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if not requests_list:
            return {"error": f"No requests found in {pass_key} batch file"}

        system_prompt = self._build_qa_system_prompt()
        model = pass_info["model_id"]
        ledger = []

        self.log("info", f"Starting {pass_key} ({pass_info['name']}): {len(requests_list)} questions via {model}")

        status_path = qa_dir / f"{pass_key}_status.json"
        with open(status_path, "w") as f:
            json.dump({"status": "submitted", "total": len(requests_list)}, f)

        if pass_info["api"] == "gemini":
            ledger = self._run_gemini_pass(requests_list, system_prompt, model, pass_key)
        else:
            ledger = self._run_openai_pass(requests_list, system_prompt, model, pass_key)

        # Save ledger
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, ensure_ascii=False)

        with open(status_path, "w") as f:
            json.dump({"status": "completed", "total": len(requests_list), "processed": len(ledger)}, f)

        approved = sum(1 for e in ledger if not e.get("errors"))
        rejected = len(ledger) - approved
        self.log("success", f"{pass_key} complete: {approved} approved, {rejected} rejected out of {len(ledger)}")

        return {
            "status": "completed",
            "pass": pass_key,
            "model": pass_info["name"],
            "requests": len(requests_list),
            "processed": len(ledger),
            "approved": approved,
            "rejected": rejected,
        }

    def _run_openai_pass(self, requests_list: List[Dict], system_prompt: str,
                         model: str, pass_key: str) -> List[Dict]:
        """Run a QA pass via OpenAI chat completions with multimodal image support."""
        from app.config import settings
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        ledger = []

        for i, req in enumerate(requests_list):
            qid = req.get("qid", "?")
            text_content, image_attachments = self._build_user_content(req)

            # Build multimodal message parts
            user_parts = [{"type": "text", "text": text_content}]
            for img in image_attachments:
                user_parts.append(img)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_parts},
                    ],
                    response_format=OPENAI_QA_RESPONSE_SCHEMA,
                    max_completion_tokens=16000,
                )
                choice = response.choices[0]

                # Truncation detection: if finish_reason is "length", output was cut off
                if choice.finish_reason == "length":
                    self.log("warning", f"{pass_key} truncated output for {qid} — treating as API error, will not count as rejection")
                    ledger.append({
                        "qid": qid, "pass": pass_key, "model": model,
                        "has_error": True,
                        "errors": [{"type": "API_ERROR", "field": "metadata", "detail": "Output truncated (finish_reason=length)"}],
                    })
                    continue

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

            if (i + 1) % 5 == 0 or i == len(requests_list) - 1:
                self.log("info", f"{pass_key}: {i + 1}/{len(requests_list)} processed")

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

        for i, req in enumerate(requests_list):
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
                response = client.models.generate_content(
                    model=model,
                    contents=types.Content(role="user", parts=parts),
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_json_schema=gemini_qa_schema,
                        temperature=0,
                        max_output_tokens=2048,
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
                        continue
                    if fr and fr.name == "SAFETY":
                        self.log("warning", f"{pass_key} safety-blocked for {qid}")
                        ledger.append({
                            "qid": qid, "pass": pass_key, "model": model,
                            "has_error": True,
                            "errors": [{"type": "API_ERROR", "field": "metadata", "detail": "Blocked by safety filter"}],
                        })
                        continue

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
            except Exception as e:
                self.log("warning", f"{pass_key} error on {qid}: {e}")
                ledger.append({
                    "qid": qid, "pass": pass_key, "model": model,
                    "has_error": True,
                    "errors": [{"type": "API_ERROR", "field": "metadata", "detail": str(e)}],
                })

            if (i + 1) % 5 == 0 or i == len(requests_list) - 1:
                self.log("info", f"{pass_key}: {i + 1}/{len(requests_list)} processed")

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

    def poll_passes(self) -> Dict[str, Any]:
        """Poll status of all QA passes with detailed stats."""
        statuses = {}
        ledger_dir = self._get_ledger_dir()

        for pass_key, pass_info in PASS_MODELS.items():
            ledger_path = ledger_dir / f"qa_{pass_key}_{pass_info['name'].lower().replace(' ', '_')}.json"
            pass_status = {
                "model": pass_info["name"],
                "status": "completed" if ledger_path.exists() else "not_started",
                "results": 0, "approved": 0, "rejected": 0,
                "batch_id": None, "error_breakdown": None,
            }

            if ledger_path.exists():
                try:
                    with open(ledger_path, "r") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        pass_status["results"] = len(data)
                        approved = sum(1 for e in data if not e.get("has_error") and not e.get("errors"))
                        pass_status["approved"] = approved
                        pass_status["rejected"] = len(data) - approved
                        error_counts = {}
                        for entry in data:
                            for err in entry.get("errors", []):
                                etype = err.get("type", "unknown") if isinstance(err, dict) else str(err)
                                error_counts[etype] = error_counts.get(etype, 0) + 1
                        pass_status["error_breakdown"] = error_counts
                except Exception:
                    pass

            qa_dir = self._get_qa_dir()
            batch_status_path = qa_dir / f"{pass_key}_status.json"
            if batch_status_path.exists():
                try:
                    with open(batch_status_path, "r") as f:
                        bs = json.load(f)
                    pass_status["batch_id"] = bs.get("batch_id")
                    if pass_status["status"] == "not_started" and bs.get("status") == "submitted":
                        pass_status["status"] = "running"
                except Exception:
                    pass

            statuses[pass_key] = pass_status

        all_complete = all(s["status"] == "completed" for s in statuses.values())
        return {"passes": statuses, "all_complete": all_complete, "can_vote": all_complete}

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

    def export_csv(self, category: str = "3of3") -> str:
        """Return CSV content for a given category."""
        csv_dir = self._get_csv_dir()
        if category == "3of3":
            path = csv_dir / "3of3_approved.csv"
        elif category == "2of3":
            path = csv_dir / "2of3_review.csv"
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
