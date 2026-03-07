"""Stage 7: LaTeX + Tagging Batch — pipeline adapter.

Builds JSONL batch files for:
  1. Concept tagging (subject -> chapter -> concept -> subConcept)
  2. LaTeX-to-MathML conversion (raw LaTeX in HTML -> MathML)

Submits batches to OpenAI Batch API, polls for completion, downloads results,
and merges tags + MathML back into metadata_v2 files.

Forked from: CodeBase/Gemini_Bulk_Batch_Processor.py + CodeBase/PCMB_Bulk_Post_Processor.py
CHANGES: Adapted for session-based pipeline, OpenAI batch API, actual latex2mathml conversion.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.pipeline.base import BasePipelineStage

try:
    from latex2mathml.converter import convert as latex_to_mathml
    HAS_LATEX2MATHML = True
except ImportError:
    HAS_LATEX2MATHML = False


class LatexTagStage(BasePipelineStage):

    @property
    def stage_name(self) -> str:
        return "stage7_batch"

    def _get_metadata_dir(self) -> Path:
        return self.session_dir / "metadata_v2"

    def _get_batch_dir(self) -> Path:
        d = self.session_dir / "batch_files"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_batch_results_dir(self) -> Path:
        d = self.session_dir / "batch_results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_all_metadata(self, filter_quarantine: bool = True) -> List[Dict]:
        """Load all metadata_v2 entries across all files.

        Args:
            filter_quarantine: If True, exclude quarantined QIDs (default).
        """
        meta_dir = self._get_metadata_dir()
        if not meta_dir.exists():
            return []

        all_entries = []
        for f in sorted(meta_dir.glob("*_metadata_v2.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
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

    def _count_latex_blocks(self, html: str) -> int:
        """Count LaTeX blocks ($...$, $$...$$, \\(...\\), \\[...\\]) in HTML."""
        if not html:
            return 0
        patterns = [
            r'\$\$[^$]+\$\$',
            r'\$[^$]+\$',
            r'\\\([^)]+\\\)',
            r'\\\[[^\]]+\\\]',
        ]
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, html))
        return count

    def _convert_latex_to_mathml(self, html: str) -> Tuple[str, int, int]:
        """Convert all LaTeX blocks in HTML to MathML.

        Finds all LaTeX patterns ($...$, $$...$$, \\(...\\), \\[...\\]),
        converts each to MathML using latex2mathml, and replaces in-place.

        Returns:
            (converted_html, success_count, fail_count)
        """
        if not html or not HAS_LATEX2MATHML:
            return html, 0, 0

        success_count = 0
        fail_count = 0

        # Define patterns in order: longer delimiters first to avoid partial matches
        # Each tuple: (regex_pattern, group_extractor_for_raw_latex)
        # We process $$...$$ before $...$ to avoid double-matching
        latex_patterns = [
            # $$...$$ (display math)
            (r'\$\$([^$]+)\$\$', True),
            # \[...\] (display math)
            (r'\\\[([^\]]+)\\\]', True),
            # \(...\) (inline math)
            (r'\\\(([^)]+)\\\)', False),
            # $...$ (inline math) — last to avoid matching $$ content
            (r'(?<!\$)\$(?!\$)([^$]+)\$(?!\$)', False),
        ]

        for pattern, is_display in latex_patterns:
            def _replace_match(match, _is_display=is_display):
                nonlocal success_count, fail_count
                raw_latex = match.group(1).strip()
                try:
                    mathml = latex_to_mathml(raw_latex)
                    # Ensure xmlns is present
                    if 'xmlns=' not in mathml:
                        mathml = mathml.replace(
                            '<math', '<math xmlns="http://www.w3.org/1998/Math/MathML"', 1
                        )
                    if _is_display:
                        # Display math: add display="block" + wrap in mstyle displaystyle
                        mathml = mathml.replace(
                            '<math', '<math display="block"', 1
                        )
                        # Wrap inner content in <mstyle displaystyle="true">
                        # Find closing </math> and insert mstyle wrapper
                        import re as _mre
                        mathml = _mre.sub(
                            r'(<math[^>]*>)(.*)(</math>)',
                            r'\1<mstyle displaystyle="true">\2</mstyle>\3',
                            mathml,
                            count=1,
                            flags=re.DOTALL,
                        )
                    else:
                        # Inline math
                        if 'display=' not in mathml:
                            mathml = mathml.replace(
                                '<math', '<math display="inline"', 1
                            )
                    success_count += 1
                    return mathml
                except Exception:
                    # If conversion fails, leave the original text intact
                    fail_count += 1
                    return match.group(0)

            html = re.sub(pattern, _replace_match, html)

        return html, success_count, fail_count

    def scan(self) -> Dict[str, Any]:
        """Scan metadata and report what batches would be built."""
        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata_v2 files found — run Stage 4 first"}

        # Count questions needing tagging
        needs_tagging = 0
        needs_latex = 0
        total_latex_blocks = 0
        by_subject = {}

        for entry in entries:
            subject = entry.get("subject", "Unknown")
            by_subject[subject] = by_subject.get(subject, 0) + 1

            # Check if needs tagging (chapter == "TBD")
            if entry.get("chapter") == "TBD" or entry.get("concept") == "TBD":
                needs_tagging += 1

            # Check if needs LaTeX conversion
            qtype = entry.get("questionType", "")
            qdata = entry.get(qtype, {})
            for field in ["question", "solution"]:
                html = qdata.get(field, "")
                latex_count = self._count_latex_blocks(html)
                if latex_count > 0:
                    needs_latex += 1
                    total_latex_blocks += latex_count
                    break  # count question once even if both fields have LaTeX

        batch_dir = self._get_batch_dir()
        existing_batches = list(batch_dir.glob("*.jsonl"))

        self.log("info", f"Scanned {len(entries)} questions: {needs_tagging} need tagging, {needs_latex} need LaTeX conversion")
        return {
            "total_questions": len(entries),
            "needs_tagging": needs_tagging,
            "needs_latex": needs_latex,
            "total_latex_blocks": total_latex_blocks,
            "by_subject": by_subject,
            "existing_batch_files": len(existing_batches),
            "batch_dir": str(batch_dir),
        }

    def test_mathml(self, count: int = 3) -> Dict[str, Any]:
        """Test MathML conversion on sample questions with LaTeX blocks.

        Finds questions containing LaTeX, converts to MathML using latex2mathml,
        and returns both the original text and the converted version with counts.
        """
        if not HAS_LATEX2MATHML:
            self.log("warning", "latex2mathml not installed — run: pip install latex2mathml")
            return {
                "error": "latex2mathml library not installed",
                "mathml_available": False,
                "samples": [],
                "message": "Install latex2mathml: pip install latex2mathml",
            }

        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata to test"}

        # Find questions with LaTeX
        samples = []
        total_success = 0
        total_fail = 0

        for entry in entries:
            if len(samples) >= count:
                break
            qtype = entry.get("questionType", "")
            qdata = entry.get(qtype, {})
            html = qdata.get("question", "")
            latex_count = self._count_latex_blocks(html)
            if latex_count > 0:
                converted_html, s_count, f_count = self._convert_latex_to_mathml(html)
                total_success += s_count
                total_fail += f_count

                samples.append({
                    "qid": entry.get("qid", "?"),
                    "latex_blocks": latex_count,
                    "success_count": s_count,
                    "fail_count": f_count,
                    "original_text": html[:300] + "..." if len(html) > 300 else html,
                    "mathml_result": converted_html,
                })

        self.log("info", f"Tested MathML on {len(samples)} samples: {total_success} converted, {total_fail} failed")
        return {
            "samples": samples,
            "mathml_available": True,
            "total_success": total_success,
            "total_fail": total_fail,
            "message": f"Converted {total_success} LaTeX blocks, {total_fail} failures across {len(samples)} samples",
        }

    def convert_all_mathml(self) -> Dict[str, Any]:
        """Convert ALL LaTeX blocks to MathML across all metadata files.

        Processes every question, applies successful conversions to metadata in-place,
        and tracks which questions still have unconverted LaTeX (failures).
        These failures are then candidates for the OpenAI LaTeX batch.

        Returns dict with counts + list of failed QIDs for batch processing.
        """
        if not HAS_LATEX2MATHML:
            self.log("warning", "latex2mathml not installed — cannot convert")
            return {"error": "latex2mathml library not installed", "failed_qids": []}

        meta_dir = self._get_metadata_dir()
        if not meta_dir.exists():
            return {"error": "No metadata_v2 files found", "failed_qids": []}

        total_questions = 0
        questions_with_latex = 0
        total_success = 0
        total_fail = 0
        fully_converted = 0
        partially_converted = 0
        failed_qids = []
        files_modified = 0

        for f in sorted(meta_dir.glob("*_metadata_v2.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, list):
                    continue
            except Exception:
                continue

            file_modified = False
            for entry in data:
                total_questions += 1
                qid = entry.get("qid", "?")
                qtype = entry.get("questionType", "")
                qdata = entry.get(qtype, {})
                if not qdata:
                    continue

                entry_success = 0
                entry_fail = 0

                # Convert both question and solution fields
                for field in ["question", "solution"]:
                    html = qdata.get(field, "")
                    if not html:
                        continue
                    latex_count = self._count_latex_blocks(html)
                    if latex_count > 0:
                        converted, s, f_cnt = self._convert_latex_to_mathml(html)
                        entry_success += s
                        entry_fail += f_cnt
                        if s > 0:
                            qdata[field] = converted
                            file_modified = True

                if entry_success > 0 or entry_fail > 0:
                    questions_with_latex += 1
                    total_success += entry_success
                    total_fail += entry_fail

                    if entry_fail > 0:
                        failed_qids.append(qid)
                        if entry_success > 0:
                            partially_converted += 1
                    else:
                        fully_converted += 1

            if file_modified:
                # Backup before writing
                self.backup(f)
                with open(f, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                files_modified += 1

        self.log("success",
                 f"MathML conversion complete: {total_success} blocks converted, {total_fail} failed "
                 f"across {questions_with_latex} questions ({fully_converted} fully converted, "
                 f"{len(failed_qids)} need batch LaTeX)")

        # Save failed QIDs for batch building
        failed_path = self._get_batch_dir() / "mathml_failed_qids.json"
        with open(failed_path, "w", encoding="utf-8") as fh:
            json.dump(failed_qids, fh)

        self.save_checkpoint("mathml_converted", {
            "total_success": total_success,
            "total_fail": total_fail,
            "failed_qids_count": len(failed_qids),
        })

        return {
            "total_questions": total_questions,
            "questions_with_latex": questions_with_latex,
            "total_success": total_success,
            "total_fail": total_fail,
            "fully_converted": fully_converted,
            "partially_converted": partially_converted,
            "failed_qids_count": len(failed_qids),
            "files_modified": files_modified,
            "failed_qids": failed_qids[:20],  # First 20 for display
        }

    def build_batch(self) -> Dict[str, Any]:
        """Build JSONL batch files for tagging and LaTeX conversion.

        LaTeX batch only includes questions where local MathML conversion failed
        (from convert_all_mathml). If no failed QIDs file exists, falls back to
        building for ALL questions with LaTeX blocks.
        """
        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata_v2 files found"}

        batch_dir = self._get_batch_dir()

        # Load failed QIDs from MathML conversion (if available)
        failed_qids_path = batch_dir / "mathml_failed_qids.json"
        failed_qids_set = set()
        if failed_qids_path.exists():
            try:
                with open(failed_qids_path, "r", encoding="utf-8") as fh:
                    failed_qids_set = set(json.load(fh))
                self.log("info", f"Loaded {len(failed_qids_set)} failed QIDs from MathML conversion — only these will get LaTeX batch")
            except Exception:
                pass

        tagging_lines = 0
        latex_lines = 0

        tagging_path = batch_dir / "tagging_batch.jsonl"
        latex_path = batch_dir / "latex_batch.jsonl"

        with open(tagging_path, "w", encoding="utf-8") as tag_f, \
             open(latex_path, "w", encoding="utf-8") as latex_f:

            for entry in entries:
                qid = entry.get("qid", "unknown")
                subject = entry.get("subject", "Unknown")
                qtype = entry.get("questionType", "")
                qdata = entry.get(qtype, {})
                question_text = qdata.get("question", "")

                # Tagging request — questions with chapter == "TBD"
                if entry.get("chapter") == "TBD":
                    tag_request = {
                        "custom_id": f"tag-{qid}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-5-mini",
                            "messages": [
                                {"role": "system", "content": f"Tag this {subject} question with chapter, concept, subConcept from the Acadza concept tree."},
                                {"role": "user", "content": question_text[:2000]},
                            ],
                            "max_completion_tokens": 500,
                            "response_format": {"type": "json_object"},
                        },
                    }
                    tag_f.write(json.dumps(tag_request) + "\n")
                    tagging_lines += 1

                # LaTeX conversion — ONLY for QIDs that failed local MathML conversion
                # If no failed_qids file exists, fall back to checking for LaTeX blocks
                needs_latex = False
                if failed_qids_set:
                    needs_latex = qid in failed_qids_set
                else:
                    needs_latex = self._count_latex_blocks(question_text) > 0

                if needs_latex:
                    # Send both question AND solution for LaTeX conversion
                    solution_text = qdata.get("solution", "")
                    combined = question_text
                    if solution_text and self._count_latex_blocks(solution_text) > 0:
                        combined = f"QUESTION:\n{question_text}\n\nSOLUTION:\n{solution_text}"

                    latex_request = {
                        "custom_id": f"latex-{qid}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-5-mini",
                            "messages": [
                                {"role": "system", "content": "Convert all LaTeX in the HTML to MathML. If input has QUESTION: and SOLUTION: sections, return both in the same format. Return the full HTML with MathML replacing LaTeX."},
                                {"role": "user", "content": combined[:6000]},
                            ],
                            "max_completion_tokens": 4000,
                        },
                    }
                    latex_f.write(json.dumps(latex_request) + "\n")
                    latex_lines += 1

        self.log("success", f"Built batch files: {tagging_lines} tagging + {latex_lines} LaTeX requests")
        if failed_qids_set and latex_lines == 0:
            self.log("info", "All LaTeX was converted locally via MathML — no LaTeX batch needed!")

        self.save_checkpoint("batch_built", {
            "tagging_lines": tagging_lines,
            "latex_lines": latex_lines,
        })

        return {
            "tagging_file": str(tagging_path),
            "latex_file": str(latex_path),
            "tagging_requests": tagging_lines,
            "latex_requests": latex_lines,
            "total_requests": tagging_lines + latex_lines,
        }

    def check_credentials(self) -> Dict[str, Any]:
        """Check if OpenAI API key is configured."""
        from app.config import settings
        has_key = bool(settings.openai_api_key)
        return {
            "configured": has_key,
            "openai_api_key": "sk-***" + settings.openai_api_key[-4:] if has_key else "(not set)",
            "message": "OpenAI API key configured" if has_key else "OpenAI API key not configured. Set OPENAI_API_KEY in .env",
        }

    def _get_openai_client(self):
        """Get an OpenAI client instance."""
        from app.config import settings
        from openai import OpenAI
        return OpenAI(api_key=settings.openai_api_key)

    def _get_batch_status_path(self) -> Path:
        return self._get_batch_dir() / "batch_status.json"

    def _load_batch_status(self) -> Dict[str, Any]:
        path = self._get_batch_status_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_batch_status(self, status: Dict[str, Any]):
        with open(self._get_batch_status_path(), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)

    def submit_batch(self) -> Dict[str, Any]:
        """Submit batch files to OpenAI Batch API."""
        creds = self.check_credentials()
        if not creds["configured"]:
            self.log("error", "OpenAI API key not configured")
            return {
                "error": "OpenAI API key not configured",
                "message": "Set OPENAI_API_KEY in .env file",
            }

        batch_dir = self._get_batch_dir()
        tagging_path = batch_dir / "tagging_batch.jsonl"
        latex_path = batch_dir / "latex_batch.jsonl"

        if not tagging_path.exists() and not latex_path.exists():
            return {"error": "No batch files found — run Build Batch first"}

        client = self._get_openai_client()
        batch_status = self._load_batch_status()
        results = {}

        # Submit tagging batch
        if tagging_path.exists() and tagging_path.stat().st_size > 0:
            if batch_status.get("tagging_batch_id"):
                self.log("info", f"Tagging batch already submitted: {batch_status['tagging_batch_id']}")
                results["tagging"] = {"batch_id": batch_status["tagging_batch_id"], "status": "already_submitted"}
            else:
                try:
                    self.log("info", "Uploading tagging batch file to OpenAI...")
                    with open(tagging_path, "rb") as f:
                        file_obj = client.files.create(file=f, purpose="batch")
                    self.log("info", f"File uploaded: {file_obj.id}")

                    batch = client.batches.create(
                        input_file_id=file_obj.id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                        metadata={"session_id": self.session_id, "type": "tagging"},
                    )
                    batch_status["tagging_batch_id"] = batch.id
                    batch_status["tagging_file_id"] = file_obj.id
                    results["tagging"] = {"batch_id": batch.id, "status": "submitted"}
                    self.log("success", f"Tagging batch submitted: {batch.id}")
                except Exception as e:
                    self.log("error", f"Tagging batch submit failed: {e}")
                    results["tagging"] = {"error": str(e)}

        # Submit LaTeX batch
        if latex_path.exists() and latex_path.stat().st_size > 0:
            if batch_status.get("latex_batch_id"):
                self.log("info", f"LaTeX batch already submitted: {batch_status['latex_batch_id']}")
                results["latex"] = {"batch_id": batch_status["latex_batch_id"], "status": "already_submitted"}
            else:
                try:
                    self.log("info", "Uploading LaTeX batch file to OpenAI...")
                    with open(latex_path, "rb") as f:
                        file_obj = client.files.create(file=f, purpose="batch")
                    self.log("info", f"File uploaded: {file_obj.id}")

                    batch = client.batches.create(
                        input_file_id=file_obj.id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                        metadata={"session_id": self.session_id, "type": "latex"},
                    )
                    batch_status["latex_batch_id"] = batch.id
                    batch_status["latex_file_id"] = file_obj.id
                    results["latex"] = {"batch_id": batch.id, "status": "submitted"}
                    self.log("success", f"LaTeX batch submitted: {batch.id}")
                except Exception as e:
                    self.log("error", f"LaTeX batch submit failed: {e}")
                    results["latex"] = {"error": str(e)}

        self._save_batch_status(batch_status)
        return results

    def poll_status(self) -> Dict[str, Any]:
        """Poll batch job status from OpenAI API."""
        batch_status = self._load_batch_status()

        if not batch_status.get("tagging_batch_id") and not batch_status.get("latex_batch_id"):
            return {
                "tagging_batch": {"id": None, "status": "not_submitted", "progress": 0},
                "latex_batch": {"id": None, "status": "not_submitted", "progress": 0},
                "message": "No batches submitted yet — run Submit first",
            }

        client = self._get_openai_client()
        result = {}

        for batch_type in ["tagging", "latex"]:
            batch_id = batch_status.get(f"{batch_type}_batch_id")
            if not batch_id:
                result[f"{batch_type}_batch"] = {"id": None, "status": "not_submitted", "progress": 0}
                continue

            try:
                batch = client.batches.retrieve(batch_id)
                total = batch.request_counts.total if batch.request_counts else 0
                completed = batch.request_counts.completed if batch.request_counts else 0
                failed = batch.request_counts.failed if batch.request_counts else 0
                progress = int((completed / total * 100)) if total > 0 else 0

                info = {
                    "id": batch_id,
                    "status": batch.status,
                    "progress": progress,
                    "total": total,
                    "completed": completed,
                    "failed": failed,
                }

                # Save output file ID if completed
                if batch.status == "completed" and batch.output_file_id:
                    batch_status[f"{batch_type}_output_file_id"] = batch.output_file_id
                    info["output_file_id"] = batch.output_file_id

                if batch.status == "failed":
                    info["error"] = str(batch.errors) if batch.errors else "Unknown error"

                result[f"{batch_type}_batch"] = info
                self.log("info", f"{batch_type.title()} batch: {batch.status} ({progress}% — {completed}/{total})")
            except Exception as e:
                result[f"{batch_type}_batch"] = {"id": batch_id, "status": "error", "error": str(e), "progress": 0}
                self.log("error", f"{batch_type.title()} batch poll error: {e}")

        self._save_batch_status(batch_status)

        # Check if both complete
        tag_done = result.get("tagging_batch", {}).get("status") in ("completed", "not_submitted")
        latex_done = result.get("latex_batch", {}).get("status") in ("completed", "not_submitted")
        result["all_complete"] = tag_done and latex_done

        return result

    def download_results(self) -> Dict[str, Any]:
        """Download batch results from OpenAI and merge into metadata_v2."""
        batch_status = self._load_batch_status()
        client = self._get_openai_client()
        results_dir = self._get_batch_results_dir()

        tags_merged = 0
        latex_merged = 0
        mathml_converted = 0
        errors = []
        empty_tag_qids = []      # QIDs where tagging returned empty/TBD
        still_failed_qids = []   # QIDs where MathML still fails after batch

        # Download and parse results for each batch type
        for batch_type in ["tagging", "latex"]:
            output_file_id = batch_status.get(f"{batch_type}_output_file_id")
            if not output_file_id:
                self.log("info", f"No output file for {batch_type} batch — skipping")
                continue

            try:
                self.log("info", f"Downloading {batch_type} results from OpenAI...")
                content = client.files.content(output_file_id)
                result_path = results_dir / f"{batch_type}_results.jsonl"
                result_path.write_bytes(content.content)
                self.log("success", f"Downloaded {batch_type} results: {result_path.name}")

                # Parse results
                results = []
                with open(result_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                results.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

                self.log("info", f"Parsed {len(results)} {batch_type} results")

                # Build lookup by custom_id
                result_lookup = {}
                for r in results:
                    custom_id = r.get("custom_id", "")
                    response = r.get("response", {})
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if choices:
                        content_text = choices[0].get("message", {}).get("content", "")
                        # Extract QID from custom_id (e.g., "tag-PHY_TEST01_Q1" -> "PHY_TEST01_Q1")
                        prefix = "tag-" if batch_type == "tagging" else "latex-"
                        qid = custom_id[len(prefix):] if custom_id.startswith(prefix) else custom_id
                        result_lookup[qid] = content_text

                # Merge into metadata_v2 files
                meta_dir = self._get_metadata_dir()
                for f in sorted(meta_dir.glob("*_metadata_v2.json")):
                    try:
                        with open(f, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        if not isinstance(data, list):
                            continue

                        modified = False
                        for entry in data:
                            qid = entry.get("qid", "")
                            if qid not in result_lookup:
                                continue

                            content_text = result_lookup[qid]

                            if batch_type == "tagging":
                                try:
                                    tag_data = json.loads(content_text)
                                    if tag_data.get("chapter"):
                                        entry["chapter"] = tag_data["chapter"]
                                    if tag_data.get("concept"):
                                        entry["concept"] = tag_data["concept"]
                                    if tag_data.get("subConcept"):
                                        entry["subConcept"] = tag_data["subConcept"]
                                    tags_merged += 1
                                    modified = True
                                    # Track empty tags — model returned JSON but no valid chapter
                                    if not tag_data.get("chapter") or tag_data["chapter"] == "TBD":
                                        empty_tag_qids.append(qid)
                                except json.JSONDecodeError:
                                    errors.append(f"Invalid JSON for tag-{qid}")
                                    empty_tag_qids.append(qid)
                            else:  # latex
                                qtype = entry.get("questionType", "")
                                qdata = entry.get(qtype, {})
                                if content_text and len(content_text) > 10:
                                    # Split QUESTION/SOLUTION if present
                                    q_part = content_text
                                    s_part = ""
                                    if "QUESTION:" in content_text and "SOLUTION:" in content_text:
                                        parts = content_text.split("SOLUTION:", 1)
                                        q_part = parts[0].replace("QUESTION:", "", 1).strip()
                                        s_part = parts[1].strip()

                                    qdata["question"] = q_part
                                    latex_merged += 1
                                    modified = True

                                    # Also do MathML conversion locally on question
                                    converted_q, sq, fq = self._convert_latex_to_mathml(q_part)
                                    if sq > 0:
                                        qdata["question"] = converted_q
                                        mathml_converted += sq
                                    if fq > 0:
                                        still_failed_qids.append(qid)

                                    # Process solution if we got one back
                                    if s_part:
                                        qdata["solution"] = s_part
                                        converted_s, ss, fs = self._convert_latex_to_mathml(s_part)
                                        if ss > 0:
                                            qdata["solution"] = converted_s
                                            mathml_converted += ss
                                        if fs > 0 and qid not in still_failed_qids:
                                            still_failed_qids.append(qid)

                        if modified:
                            with open(f, "w", encoding="utf-8") as fh:
                                json.dump(data, fh, indent=2, ensure_ascii=False)

                    except Exception as e:
                        errors.append(f"Error processing {f.name}: {e}")

            except Exception as e:
                self.log("error", f"Download {batch_type} results failed: {e}")
                errors.append(str(e))

        self.log("success", f"Merged: {tags_merged} tags + {latex_merged} LaTeX + {mathml_converted} MathML conversions")
        if errors:
            self.log("warning", f"{len(errors)} errors during merge")

        # Quarantine MathML failures and empty tags
        from app.db import quarantine_question
        quarantined_count = 0

        if still_failed_qids:
            self.log("warning", f"{len(still_failed_qids)} questions still have MathML failures after batch — quarantining")
            for qid in still_failed_qids:
                quarantine_question(
                    self.session_id, qid, self.stage_name,
                    reason="mathml_failure",
                    data={"source": "stage7_mathml_still_failed_after_batch"},
                )
                quarantined_count += 1

        if empty_tag_qids:
            self.log("warning", f"{len(empty_tag_qids)} questions have empty/TBD tags after batch — quarantining")
            for qid in empty_tag_qids:
                quarantine_question(
                    self.session_id, qid, self.stage_name,
                    reason="empty_tags",
                    data={"source": "stage7_tagging_empty_chapter"},
                )
                quarantined_count += 1

        # Save quarantine info for downstream stages
        quarantine_path = self._get_batch_dir() / "stage7_quarantine.json"
        quarantine_data = {
            "mathml_failures": still_failed_qids,
            "empty_tags": empty_tag_qids,
            "total_quarantined": quarantined_count,
        }
        with open(quarantine_path, "w", encoding="utf-8") as qf:
            json.dump(quarantine_data, qf, indent=2)

        return {
            "tags_merged": tags_merged,
            "latex_merged": latex_merged,
            "mathml_converted": mathml_converted,
            "errors": errors[:10] if errors else [],
            "quarantined": quarantined_count,
            "mathml_still_failed": len(still_failed_qids),
            "empty_tags": len(empty_tag_qids),
        }

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        """Preview what batches would be built."""
        scan = self.scan()
        creds = self.check_credentials()
        return {**scan, "credentials": creds}

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the full batch pipeline: build -> submit -> poll -> download."""
        # In real implementation, this would orchestrate the full flow
        # For now, just build the batch files
        return self.build_batch()
