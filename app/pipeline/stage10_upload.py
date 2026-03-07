"""Stage 10: Upload to DB — pipeline adapter.

FORKED FROM: PCM_Batches/post_processor_uploader.py
CHANGES:
- Adapted for web app session-based architecture (BasePipelineStage)
- Filters quarantined QIDs before upload
- Generates CSV upload log per file
- API: POST https://api.acadza.in/question/create with JSON payload
- Returns uploaded_id from response.question._id
- Rate limiting: 3 retries, 2s delay between retries
"""

import csv
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.pipeline.base import BasePipelineStage

# ── API Configuration ────────────────────────────────────────────────────
API_BASE_URL = "https://api.acadza.in"
QUESTION_API_URL = f"{API_BASE_URL}/question/create"
MAX_RETRIES = 3
RETRY_DELAY = 2

QUESTION_EDIT_URL = {
    "scq": "https://www.acadza.com/question/edit-question/single-question/",
    "mcq": "https://www.acadza.com/question/edit-question/multi-question/",
    "passage": "https://www.acadza.com/question/edit-question/passage-question/",
    "integer": "https://www.acadza.com/question/edit-question/integer-question/",
    "match": "https://www.acadza.com/question/edit-question/match-question/",
}
QUESTION_VIEW_URL = "https://www.acadza.com/question/view-question/"


class UploadStage(BasePipelineStage):

    @property
    def stage_name(self) -> str:
        return "stage10_upload"

    def _get_metadata_dir(self) -> Path:
        d1 = self.session_dir / "metadata_v2_output"
        if d1.exists():
            return d1
        return self.session_dir / "metadata_v2"

    def _get_upload_dir(self) -> Path:
        d = self.session_dir / "upload_logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_quarantined_qids(self) -> set:
        """Load all quarantined QIDs from the DB."""
        from app.db import get_quarantined_qids
        return get_quarantined_qids(self.session_id)

    def _load_all_metadata(self) -> Tuple[List[Tuple[Path, List[Dict]]], int]:
        """Load all metadata_v2 entries, grouped by file.

        Returns:
            (list of (filepath, entries), total_count)
        """
        meta_dir = self._get_metadata_dir()
        if not meta_dir.exists():
            return [], 0

        file_groups = []
        total = 0

        # Check for subdirectory structure
        for sub in sorted(meta_dir.iterdir()):
            if sub.is_dir():
                for f in sub.glob("*_metadata_v2.json"):
                    if "backup" in f.name:
                        continue
                    try:
                        with open(f, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        if isinstance(data, list):
                            file_groups.append((f, data))
                            total += len(data)
                    except Exception:
                        continue
            elif sub.suffix == ".json" and "metadata_v2" in sub.name and "backup" not in sub.name:
                try:
                    with open(sub, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        file_groups.append((sub, data))
                        total += len(data)
                except Exception:
                    continue

        return file_groups, total

    def scan(self) -> Dict[str, Any]:
        """Scan what's ready for upload."""
        file_groups, total = self._load_all_metadata()
        if total == 0:
            return {"error": "No metadata_v2 files found"}

        quarantined = self._load_quarantined_qids()

        # Count upload-ready questions
        ready = 0
        by_subject = {}
        by_type = {}
        error_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        for _, entries in file_groups:
            for entry in entries:
                qid = entry.get("qid", "")
                error_val = entry.get("error", 2)

                if qid in quarantined:
                    continue
                if isinstance(error_val, int) and error_val >= 3:
                    error_counts[error_val] = error_counts.get(error_val, 0) + 1
                    continue

                ready += 1
                subj = entry.get("subject", "Unknown")
                by_subject[subj] = by_subject.get(subj, 0) + 1
                qtype = entry.get("questionType", "unknown")
                by_type[qtype] = by_type.get(qtype, 0) + 1
                if isinstance(error_val, int):
                    error_counts[error_val] = error_counts.get(error_val, 0) + 1

        self.log("info", f"Upload scan: {total} total, {ready} ready, {len(quarantined)} quarantined")
        return {
            "total_questions": total,
            "upload_ready": ready,
            "quarantined_count": len(quarantined),
            "error_3_4_count": error_counts.get(3, 0) + error_counts.get(4, 0),
            "by_subject": by_subject,
            "by_type": by_type,
            "error_breakdown": error_counts,
            "files_count": len(file_groups),
        }

    def upload(self, dry_run: bool = False) -> Dict[str, Any]:
        """Upload ready questions to Acadza DB API.

        Args:
            dry_run: If True, skip actual API calls but generate CSV.

        Forked from: post_processor_uploader.py -> post_question_to_api()
        API: POST https://api.acadza.in/question/create
        Response: {"isSuccessful": true, "question": {"_id": "..."}}
        """
        import requests as req_lib

        file_groups, total = self._load_all_metadata()
        if total == 0:
            return {"error": "No metadata_v2 files found"}

        quarantined = self._load_quarantined_qids()
        upload_dir = self._get_upload_dir()

        total_posted = 0
        total_failed = 0
        total_skipped = 0
        all_csv_rows = []

        for filepath, entries in file_groups:
            file_name = filepath.stem
            csv_rows = []
            posted = 0
            failed = 0

            for entry in entries:
                qid = entry.get("qid", "?")
                qtype = (entry.get("questionType") or "").lower()
                error_val = entry.get("error", 2)

                view_link = f"{QUESTION_VIEW_URL}{qid}"

                # Skip quarantined
                if qid in quarantined:
                    total_skipped += 1
                    csv_rows.append({
                        "Question META ID": qid,
                        "UploadedQuestionID": "QUARANTINED",
                        "Question Type": qtype,
                        "View Link": view_link,
                        "Question Link": "N/A",
                        "Error Level": error_val,
                        "IMAGE": "FALSE",
                    })
                    continue

                # Skip error >= 3
                if isinstance(error_val, int) and error_val >= 3:
                    total_skipped += 1
                    csv_rows.append({
                        "Question META ID": qid,
                        "UploadedQuestionID": f"SKIPPED_ERROR_{error_val}",
                        "Question Type": qtype,
                        "View Link": view_link,
                        "Question Link": "N/A",
                        "Error Level": error_val,
                        "IMAGE": "FALSE",
                    })
                    continue

                # Check for images
                qdata = entry.get(entry.get("questionType", ""), {})
                q_html = qdata.get("question", "") or qdata.get("ques", "")
                s_html = qdata.get("solution", "")
                has_img = "<img" in q_html or "<img" in s_html

                # Map questionType for edit URL
                qtype_key = qtype
                if qtype_key == "integerquestion":
                    qtype_key = "integer"
                if qtype_key == "matchquestion":
                    qtype_key = "match"
                if qtype_key == "passagequestion":
                    qtype_key = "passage"

                uploaded_id = None
                link = "N/A"

                if dry_run:
                    status = "DRY_RUN"
                else:
                    # POST to API
                    ok, uploaded_id = self._post_question(req_lib, entry)
                    if ok and uploaded_id:
                        posted += 1
                        total_posted += 1
                        base = QUESTION_EDIT_URL.get(qtype_key, "")
                        link = f"{base}{uploaded_id}" if base else "N/A"
                        status = uploaded_id
                    else:
                        failed += 1
                        total_failed += 1
                        status = "FAILED"

                csv_rows.append({
                    "Question META ID": qid,
                    "UploadedQuestionID": status,
                    "Question Type": qtype_key,
                    "View Link": view_link,
                    "Question Link": link,
                    "Error Level": error_val,
                    "IMAGE": "TRUE" if has_img else "FALSE",
                })

            # Save per-file CSV
            csv_path = upload_dir / f"{file_name}_upload_log.csv"
            self._write_csv(csv_path, csv_rows)
            all_csv_rows.extend(csv_rows)

            if not dry_run:
                self.log("info", f"{file_name}: posted={posted}, failed={failed}")

            self.progress(
                min(95, len(all_csv_rows) / max(total, 1) * 100),
                f"Processed {len(all_csv_rows)}/{total}"
            )

        # Save combined CSV
        combined_path = upload_dir / "combined_upload_log.csv"
        self._write_csv(combined_path, all_csv_rows)

        mode = "DRY RUN" if dry_run else "UPLOAD"
        self.log("success",
                 f"{mode} complete: posted={total_posted}, failed={total_failed}, "
                 f"skipped={total_skipped}, CSV at {combined_path.name}")

        self.save_checkpoint("completed", {
            "posted": total_posted,
            "failed": total_failed,
            "skipped": total_skipped,
            "dry_run": dry_run,
        })

        return {
            "posted": total_posted,
            "failed": total_failed,
            "skipped": total_skipped,
            "total": total,
            "csv_path": str(combined_path),
            "dry_run": dry_run,
        }

    def _post_question(self, req_lib, entry: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Post a single question to the Acadza API.

        Forked verbatim from post_processor_uploader.py:post_question_to_api()
        """
        headers = {"Content-Type": "application/json"}
        qid = entry.get("qid")

        for attempt in range(MAX_RETRIES):
            try:
                payload = dict(entry)
                if "error" not in payload:
                    payload["error"] = 2

                resp = req_lib.post(QUESTION_API_URL, json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if data.get("isSuccessful"):
                    uploaded_id = (data.get("question") or {}).get("_id", "")
                    return True, uploaded_id

                self.log("error", f"API error for {qid}: {data.get('message', 'unknown')}")
                return False, None

            except Exception as e:
                self.log("error", f"Post failed for {qid} (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    return False, None

        return False, None

    def _write_csv(self, path: Path, rows: List[Dict]) -> None:
        """Write upload log CSV."""
        if not rows:
            return
        fieldnames = ["Question META ID", "UploadedQuestionID", "Question Type",
                       "View Link", "Question Link", "Error Level", "IMAGE"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def export_csv(self) -> str:
        """Return the combined upload CSV as a string."""
        combined = self._get_upload_dir() / "combined_upload_log.csv"
        if combined.exists():
            return combined.read_text(encoding="utf-8")

        # No combined yet — generate from metadata
        scan = self.scan()
        if "error" in scan:
            return ""

        result = self.upload(dry_run=True)
        if combined.exists():
            return combined.read_text(encoding="utf-8")
        return ""

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        return self.upload(dry_run=True)

    def execute(self, **kwargs) -> Dict[str, Any]:
        return self.upload(dry_run=False)
