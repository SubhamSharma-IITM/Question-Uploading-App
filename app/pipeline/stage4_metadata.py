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
        return html

    result = html

    # ── Handle (a)(b)(c)(d) lowercase parenthesized → uppercase ──
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*a\s*\)', r'\1(A)', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*b\s*\)', r'\1(B)', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*c\s*\)', r'\1(C)', result, flags=re.MULTILINE)
    result = re.sub(r'(<br/>|<br>|^)\s*\(\s*d\s*\)', r'\1(D)', result, flags=re.MULTILINE)
    # Also catch (a) without preceding <br> (inline after question text)
    result = re.sub(r'(?<!<br/>)(?<!<br>)\s*\(\s*a\s*\)', r'<br/>(A)', result, count=1)

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
        base = _basename_from_url(raw_url)
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
    if qt in ("scq", "mcq", "integerQuestion", "passageQuestion"):
        return qt

    # Legacy format detection
    options = entry.get("options", {})
    if len(options) >= 4:
        return "scq"
    if len(options) == 0:
        return "integerQuestion"
    return "scq"


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
        # Build nested passage structure matching Stage 6's expected schema
        passage_raw = entry.get("passage_content", "")
        if passage_raw:
            passage_raw, _ = _replace_md_images_with_img(passage_raw)
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
            sub_questions_out.append({
                sq_type: {
                    "question": sq_question_html,
                    "solution": sq_solution_html,
                    "answer": sq_answer,
                    "quesImages": [],
                    "solutionImages": [],
                }
            })

        active_container = {
            "passage": passage_html,
            "solution": "<h3></h3>",  # overall solution (usually empty for JEE)
            "questions": sub_questions_out,
        }
        # No single answer for passage — answers are per sub-question
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
