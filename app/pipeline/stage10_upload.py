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
        qa_ledger = self._load_qa_ledger()
        if qa_ledger:
            self.log("info", f"Loaded QA ledger for {len(qa_ledger)} questions")

        total_posted = 0
        total_failed = 0
        total_skipped = 0
        all_csv_rows = []
        upload_map = {}  # meta_qid → uploaded MongoDB _id

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
                        upload_map[qid] = uploaded_id
                        base = QUESTION_EDIT_URL.get(qtype_key, "")
                        link = f"{base}{uploaded_id}" if base else "N/A"
                        status = uploaded_id
                    else:
                        failed += 1
                        total_failed += 1
                        status = "FAILED"

                # Build row with QA ledger columns
                row = {
                    "Question META ID": qid,
                    "UploadedQuestionID": status,
                    "Question Type": qtype_key,
                    "View Link": view_link,
                    "Question Link": link,
                    "Error Level": error_val,
                    "IMAGE": "TRUE" if has_img else "FALSE",
                }
                # Add model-wise QA verdicts + error descriptions
                qa = qa_ledger.get(qid, {})
                for pk in ("pass1", "pass2", "pass3"):
                    n = pk[-1]  # 1, 2, 3
                    row[f"Pass{n} Model"] = qa.get(f"{pk}_model", "")
                    row[f"Pass{n} Verdict"] = qa.get(f"{pk}_verdict", "")
                    row[f"Pass{n} Errors"] = qa.get(f"{pk}_errors", "")
                csv_rows.append(row)

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

        # Save upload ID mapping for absorption step
        if not dry_run and upload_map:
            map_path = self._save_upload_map(upload_map)
            self.log("info", f"Saved {len(upload_map)} upload ID mappings to {map_path.name}")

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

    def _save_upload_map(self, upload_map: Dict[str, str]) -> Path:
        """Save meta_qid → uploaded_id mapping for absorption step."""
        map_path = self._get_upload_dir() / "upload_id_map.json"
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(upload_map, f, ensure_ascii=False, indent=2)
        return map_path

    def _load_upload_map(self) -> Dict[str, str]:
        """Load meta_qid → uploaded_id mapping from previous upload."""
        map_path = self._get_upload_dir() / "upload_id_map.json"
        if not map_path.exists():
            return {}
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def absorb_to_local_db(self) -> Dict[str, Any]:
        """Absorb uploaded questions into local JSONL DB + update FAISS index.

        After Stage 10 upload, this:
        1. Reads upload_id_map.json (meta_qid → MongoDB _id)
        2. Loads metadata entries for those QIDs
        3. Builds proper DB entries with real MongoDB _id
        4. Appends to the local JSONL DB file
        5. Generates embeddings for new entries
        6. Adds vectors to the global FAISS index cache

        Only processes error=1 and error=2 entries (not quarantined).
        """
        from app.config import settings
        from datetime import datetime

        # Step 1: Load upload map
        upload_map = self._load_upload_map()
        if not upload_map:
            return {"error": "No upload_id_map.json found — run Upload first"}

        self.log("info", f"Step 1/4: Loaded {len(upload_map)} uploaded ID mappings")

        # Step 2: Load metadata entries for uploaded QIDs
        file_groups, _ = self._load_all_metadata()
        quarantined = self._load_quarantined_qids()
        entries_to_absorb = []

        for _, entries in file_groups:
            for entry in entries:
                qid = entry.get("qid", "")
                if qid not in upload_map:
                    continue
                if qid in quarantined:
                    continue
                error_val = entry.get("error", 2)
                if isinstance(error_val, int) and error_val >= 3:
                    continue
                entries_to_absorb.append((qid, upload_map[qid], entry))

        self.log("info", f"Step 2/4: Found {len(entries_to_absorb)} entries to absorb (error 1/2, non-quarantined)")

        if not entries_to_absorb:
            return {"absorbed": 0, "message": "No eligible entries to absorb"}

        # Step 3: Build DB entries and append to JSONL
        db_path = Path(settings.db_path)
        if not db_path.exists():
            return {"error": f"DB file not found: {db_path}"}

        now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
        absorbed = 0
        db_entries = []

        with open(db_path, "a", encoding="utf-8") as f:
            for meta_qid, mongo_id, entry in entries_to_absorb:
                db_entry = self._build_db_entry(entry, mongo_id, now_iso)
                line = json.dumps(db_entry, ensure_ascii=False)
                f.write(line + "\n")
                db_entries.append((meta_qid, mongo_id, db_entry))
                absorbed += 1

        self.log("success", f"Step 3/4: Appended {absorbed} entries to {db_path.name}")

        # Step 4: Generate embeddings and update FAISS index
        embedded = self._update_faiss_index(db_entries)

        self.log("success", f"Step 4/4: Added {embedded} vectors to FAISS index")

        # Merge absorption data with existing "completed" checkpoint — don't overwrite status
        from app.db import get_checkpoint
        existing = get_checkpoint(self.session_id, self.stage_name)
        existing_data = existing.get("data", {}) if existing else {}
        merged = {
            **existing_data,
            "absorbed": absorbed,
            "embedded": embedded,
            "db_path": str(db_path),
        }
        self.save_checkpoint("completed", merged)

        return {
            "absorbed": absorbed,
            "embedded": embedded,
            "db_path": str(db_path),
        }

    def _build_db_entry(self, metadata_entry: Dict, mongo_id: str, now_iso: str) -> Dict:
        """Build a proper JSONL DB entry from metadata + uploaded MongoDB _id.

        Matches the format of existing allque030326.json entries.
        """
        entry = {}

        # MongoDB _id in extended JSON format
        # API returns plain string, but handle dict {"$oid": "..."} if ever returned
        if isinstance(mongo_id, dict) and "$oid" in mongo_id:
            entry["_id"] = mongo_id
        else:
            entry["_id"] = {"$oid": str(mongo_id)}

        # Question type data — copy the type-specific dict
        qtype = metadata_entry.get("questionType", "scq")
        for qt in ("scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"):
            qdata = metadata_entry.get(qt)
            if isinstance(qdata, dict) and qdata:
                entry[qt] = dict(qdata)
            else:
                entry[qt] = {}

        # Core metadata fields
        entry["qid"] = metadata_entry.get("qid", "")
        entry["subject"] = metadata_entry.get("subject", "")
        entry["chapter"] = metadata_entry.get("chapter", "")
        entry["concept"] = metadata_entry.get("concept")
        entry["subConcept"] = metadata_entry.get("subConcept")
        entry["level"] = metadata_entry.get("level", "MEDIUM")
        entry["questionType"] = qtype
        entry["difficulty"] = metadata_entry.get("difficulty", 1)
        entry["error"] = metadata_entry.get("error", 2)

        # Tags
        entry["tagSubConcept"] = metadata_entry.get("tagSubConcept", [])

        # Boolean/numeric fields — set defaults matching DB schema
        entry["isBuffer"] = metadata_entry.get("isBuffer", False)
        entry["isNCERT"] = metadata_entry.get("isNCERT", False)
        entry["isPrevious"] = metadata_entry.get("isPrevious", False)
        entry["isBoard"] = metadata_entry.get("isBoard", False)
        entry["isNEET"] = metadata_entry.get("isNEET", None)
        entry["isLengthy"] = metadata_entry.get("isLengthy", 1)
        entry["smartTrick"] = metadata_entry.get("smartTrick", 1)
        entry["trap"] = metadata_entry.get("trap", 1)
        entry["sillyMistake"] = metadata_entry.get("sillyMistake", 1)
        entry["negativePick"] = metadata_entry.get("negativePick", False)
        entry["positivePick"] = metadata_entry.get("positivePick", False)
        entry["practiceToughQuestion"] = metadata_entry.get("practiceToughQuestion", False)
        entry["tagByAI"] = metadata_entry.get("tagByAI", True)
        entry["taggedByCoachingTeacher"] = metadata_entry.get("taggedByCoachingTeacher", False)

        # PYQ fields
        entry["previousExam"] = metadata_entry.get("previousExam")
        entry["previousExamYear"] = metadata_entry.get("previousExamYear")

        # Arrays
        entry["questionocrimages"] = metadata_entry.get("questionocrimages", [])
        entry["correctClickingSubconcept"] = []
        entry["incorrectClickingSubconcept"] = []

        # Numeric defaults
        entry["chapterTaggedCount"] = metadata_entry.get("chapterTaggedCount", 0)
        entry["quetionChapterOrder"] = 0

        # Timestamps
        entry["createdAt"] = {"$date": now_iso}
        entry["updatedAt"] = {"$date": now_iso}
        entry["__v"] = 0

        # Text fields
        entry["trickData"] = ""
        entry["trapData"] = ""

        return entry

    def _update_faiss_index(self, db_entries: List[Tuple[str, str, Dict]]) -> int:
        """Generate embeddings for absorbed entries and add to global FAISS index.

        Args:
            db_entries: List of (meta_qid, mongo_id, db_entry) tuples

        Returns:
            Number of vectors added to index
        """
        from app.config import settings

        # Check if FAISS + OpenAI are available
        try:
            import faiss
            import numpy as np
        except ImportError:
            self.log("warning", "FAISS not available — skipping embedding update")
            return 0

        openai_key = getattr(settings, "openai_api_key", None) or ""
        if not openai_key or len(openai_key) < 10:
            self.log("warning", "OpenAI API key not configured — skipping embedding update")
            return 0

        # Load global FAISS index
        global_dir = Path(getattr(settings, "db_index_dir", ""))
        index_path = global_dir / "db_reference" / "faiss.index"
        meta_path = global_dir / "db_reference" / "metadata.json"

        if not index_path.exists():
            self.log("warning", f"Global FAISS index not found at {index_path} — skipping")
            return 0

        self.log("info", "Loading global FAISS index for update...")
        index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        id_map = metadata.get("id_map", [])
        text_map = metadata.get("text_map", {})
        dimension = metadata.get("dimension", 1536)

        self.log("info", f"Current index: {index.ntotal} vectors, adding {len(db_entries)} new...")

        # Extract text for embedding (same method as Stage 8)
        from app.pipeline.stage8_duplicates import _strip_html_keep_images, MATHML_RE, HTML_TAG_RE, IMG_SRC_RE
        import re

        texts = []
        new_ids = []
        for meta_qid, mongo_id, db_entry in db_entries:
            # Extract question text (same logic as DuplicateCheckStage._extract_question_text)
            parts = []
            qtype = db_entry.get("questionType", "")
            qdata = db_entry.get(qtype, {})
            if isinstance(qdata, dict):
                q_text = qdata.get("question", "") or qdata.get("ques", "")
                if q_text:
                    parts.append(_strip_html_keep_images(q_text))

            text = " ".join(parts)
            if text and len(text) >= 10:
                texts.append(text[:8000])
                new_ids.append(mongo_id)

        if not texts:
            self.log("info", "No texts extracted for embedding")
            return 0

        # Generate embeddings
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

        BATCH_SIZE = 100
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            valid = [t if t.strip() else "empty" for t in batch]
            try:
                response = client.embeddings.create(model="text-embedding-3-small", input=valid)
                for item in response.data:
                    all_embeddings.append(np.array(item.embedding, dtype="float32"))
            except Exception as e:
                self.log("error", f"Embedding batch failed: {e}")
                for _ in batch:
                    all_embeddings.append(np.zeros(dimension, dtype="float32"))

        self.log("info", f"Generated {len(all_embeddings)} embeddings")

        # Add to FAISS index
        vectors = np.array(all_embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)

        # Update metadata
        for i, mongo_id in enumerate(new_ids):
            id_map.append(mongo_id)
            text_map[mongo_id] = texts[i][:500]

        # Save updated index back to global cache
        faiss.write_index(index, str(index_path))
        metadata["id_map"] = id_map
        metadata["text_map"] = text_map
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        self.log("success", f"FAISS index updated: {index.ntotal} total vectors (added {len(new_ids)})")
        return len(new_ids)

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

    def _load_qa_ledger(self) -> Dict[str, Dict]:
        """Load QA ledger data per QID — model-wise verdicts + error descriptions.

        Returns:
            {qid: {
                "pass1_model": "Gemini Flash",
                "pass1_verdict": "APPROVED"/"REJECTED",
                "pass1_errors": "error description...",
                "pass2_model": ..., "pass2_verdict": ..., "pass2_errors": ...,
                "pass3_model": ..., "pass3_verdict": ..., "pass3_errors": ...,
            }}
        """
        ledger_dir = self.session_dir / "quality_check" / "ledgers"
        if not ledger_dir.exists():
            return {}

        qa_data: Dict[str, Dict] = {}

        for ledger_file in sorted(ledger_dir.glob("qa_pass*.json")):
            try:
                with open(ledger_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception:
                continue

            if not isinstance(entries, list):
                continue

            for entry in entries:
                qid = entry.get("qid", "")
                if not qid:
                    continue

                pass_key = entry.get("pass", "")  # pass1, pass2, pass3
                model = entry.get("model", "")
                has_error = entry.get("has_error", False)
                errors = entry.get("errors", [])

                verdict = "REJECTED" if has_error else "APPROVED"
                # Build error description string
                err_descs = []
                for err in errors:
                    field = err.get("field", "")
                    etype = err.get("type", "")
                    detail = err.get("detail", "")
                    err_descs.append(f"[{field}/{etype}] {detail}")
                error_str = " | ".join(err_descs) if err_descs else ""

                if qid not in qa_data:
                    qa_data[qid] = {}

                qa_data[qid][f"{pass_key}_model"] = model
                qa_data[qid][f"{pass_key}_verdict"] = verdict
                qa_data[qid][f"{pass_key}_errors"] = error_str

        return qa_data

    def _write_csv(self, path: Path, rows: List[Dict]) -> None:
        """Write upload log CSV with model-wise QA ledger columns."""
        if not rows:
            return
        fieldnames = [
            "Question META ID", "UploadedQuestionID", "Question Type",
            "View Link", "Question Link", "Error Level", "IMAGE",
            "Pass1 Model", "Pass1 Verdict", "Pass1 Errors",
            "Pass2 Model", "Pass2 Verdict", "Pass2 Errors",
            "Pass3 Model", "Pass3 Verdict", "Pass3 Errors",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def export_csv(self, error_level: Optional[int] = None) -> str:
        """Return the combined upload CSV as a string, optionally filtered.

        Args:
            error_level: If set (1 or 2), only include rows with that Error Level.
                         None = return all rows.
        """
        combined = self._get_upload_dir() / "combined_upload_log.csv"
        if not combined.exists():
            # No combined yet — generate from metadata
            scan = self.scan()
            if "error" in scan:
                return ""
            self.upload(dry_run=True)

        if not combined.exists():
            return ""

        content = combined.read_text(encoding="utf-8")

        if error_level is None:
            return content

        # Filter by error level
        import csv as csv_mod
        reader = csv_mod.DictReader(io.StringIO(content))
        output = io.StringIO()
        writer = csv_mod.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            try:
                row_error = int(row.get("Error Level", 0))
            except (ValueError, TypeError):
                continue
            if row_error == error_level:
                writer.writerow(row)
        return output.getvalue()

    def generate_pipeline_stats_csv(self) -> str:
        """Generate comprehensive pipeline stats CSV with per-stage metrics.

        Includes: question flow per stage, removal counts, pass/fail rates,
        model-wise QA accuracy, image counts (SVG SMILES + actual), error type
        splits from Stage 9, and final accuracy metrics.
        """
        from app.db import get_all_checkpoints, get_quarantine

        # ── Gather checkpoint data per stage ──
        checkpoints = get_all_checkpoints(self.session_id)
        cp_map = {}
        for cp in checkpoints:
            stage = cp.get("stage", "")
            data = cp.get("data")
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    data = {}
            if not isinstance(data, dict):
                data = {}
            cp_map[stage] = {"status": cp.get("status", "unknown"), **data}

        # ── Quarantine breakdown by stage/reason ──
        quarantine_records = get_quarantine(self.session_id)
        quarantine_by_stage = {}
        quarantine_by_reason = {}
        for rec in quarantine_records:
            st = rec.get("stage", "unknown")
            reason = rec.get("reason", "unknown")
            quarantine_by_stage[st] = quarantine_by_stage.get(st, 0) + 1
            quarantine_by_reason[reason] = quarantine_by_reason.get(reason, 0) + 1

        # ── Load metadata for final state ──
        file_groups, total_meta = self._load_all_metadata()
        all_entries = []
        for _, entries in file_groups:
            all_entries.extend(entries)

        # Count by error level, subject, type
        by_error = {1: 0, 2: 0, 3: 0, 4: 0}
        by_subject = {}
        by_type = {}
        img_count = 0
        svg_count = 0
        for entry in all_entries:
            ev = entry.get("error", 2)
            if isinstance(ev, int) and ev in by_error:
                by_error[ev] += 1
            subj = entry.get("subject", "Unknown")
            by_subject[subj] = by_subject.get(subj, 0) + 1
            qt = entry.get("questionType", "unknown")
            by_type[qt] = by_type.get(qt, 0) + 1

            # Count images in question+solution HTML
            qdata = entry.get(entry.get("questionType", ""), {})
            if isinstance(qdata, dict):
                for field in ("question", "ques", "solution"):
                    html = qdata.get(field, "") or ""
                    img_count += html.count("<img")
                    svg_count += html.count("image/svg+xml")

        # ── QA vote summary ──
        vote_path = self.session_dir / "quality_check" / "vote_summary.json"
        vote_data = {}
        if vote_path.exists():
            try:
                with open(vote_path, "r") as f:
                    vote_data = json.load(f)
            except Exception:
                pass

        # ── QA error type splits from ledger files ──
        error_type_counts = {}
        ledger_dir = self.session_dir / "quality_check" / "ledgers"
        model_stats = {}  # model → {total: N, approved: N, rejected: N}
        if ledger_dir.exists():
            for lf in sorted(ledger_dir.glob("qa_pass*.json")):
                try:
                    with open(lf, "r", encoding="utf-8") as f:
                        entries_l = json.load(f)
                except Exception:
                    continue
                if not isinstance(entries_l, list):
                    continue
                for le in entries_l:
                    model = le.get("model", "Unknown")
                    if model not in model_stats:
                        model_stats[model] = {"total": 0, "approved": 0, "rejected": 0}
                    model_stats[model]["total"] += 1
                    if le.get("has_error"):
                        model_stats[model]["rejected"] += 1
                        for err in le.get("errors", []):
                            if isinstance(err, dict):
                                etype = err.get("type", "unknown")
                            else:
                                etype = str(err)
                            error_type_counts[etype] = error_type_counts.get(etype, 0) + 1
                    else:
                        model_stats[model]["approved"] += 1

        # ── CDN image stats ──
        cdn_map_path = self.session_dir / "images_cdn_map.json"
        cdn_images = 0
        if cdn_map_path.exists():
            try:
                with open(cdn_map_path, "r") as f:
                    cdn_data = json.load(f)
                cdn_images = len(cdn_data)
            except Exception:
                pass

        # ── Build stage flow data ──
        # Stage 2: extraction
        s2 = cp_map.get("stage2_mapping", {})
        extracted = s2.get("questions", 0)

        # Stage 3: cleaner
        s3 = cp_map.get("stage3_cleaning", {})
        cleaned = s3.get("questions_cleaned", 0)

        # Stage 4: metadata
        s4 = cp_map.get("stage4_metadata", {})
        meta_total = s4.get("total_entries", 0)
        meta_triplets = s4.get("triplets", 0)
        meta_non_triplets = s4.get("non_triplets_quarantined", 0)

        # Stage 5: watermark
        s5 = cp_map.get("stage5_watermark", {})
        wm_total = s5.get("total", 0)
        wm_processed = s5.get("processed", 0)
        smiles_success = s5.get("smiles_success", 0)
        smiles_fail = s5.get("smiles_fail", 0)
        smiles_quarantined = s5.get("smiles_quarantined", 0)

        # Stage 6: CDN images
        s6 = cp_map.get("stage6_images", {})
        cdn_uploaded = s6.get("uploaded", 0)
        cdn_cached = s6.get("cached", 0)
        cdn_failed = s6.get("failed", 0)

        # Stage 8: duplicates
        s8 = cp_map.get("stage8_duplicates", {})
        self_dupes = s8.get("self_duplicates", 0)
        cross_dupes = s8.get("cross_duplicates", 0)

        # Stage 9: QA
        s9 = cp_map.get("stage9_quality", {})
        qa_total = s9.get("total", 0)
        qa_e1 = s9.get("error_1", 0)
        qa_e2 = s9.get("error_2", 0)
        qa_e3 = s9.get("error_3", 0)
        qa_e4 = s9.get("error_4", 0)
        qa_quarantined = s9.get("quarantined", 0)

        # Stage 10: upload
        s10 = cp_map.get("stage10_upload", {})
        uploaded = s10.get("posted", 0)
        upload_failed = s10.get("failed", 0)

        # ── Total quarantine ──
        total_quarantined = len(quarantine_records)
        final_uploaded = uploaded
        final_ready = total_meta - by_error.get(3, 0) - by_error.get(4, 0)

        # ── Build CSV with multiple sections ──
        output = io.StringIO()

        # --- Section 1: Pipeline Flow Overview ---
        output.write("=== PIPELINE FLOW OVERVIEW ===\n")
        writer = csv.writer(output)
        writer.writerow(["Stage", "Description", "Input Count", "Output Count", "Removed", "Pass Rate %"])

        stages_flow = [
            ("Stage 1", "PDF → MMD (Mathpix OCR)", "—", extracted, "—", "—"),
            ("Stage 2", "LLM Extraction (Q/A/S)", "—", extracted, 0, "100%"),
            ("Stage 3", "Text Cleaning", extracted, cleaned, max(0, extracted - cleaned), f"{cleaned / max(extracted, 1) * 100:.1f}%"),
            ("Stage 4", "Metadata V2 Conversion", cleaned, meta_triplets, meta_non_triplets, f"{meta_triplets / max(meta_total, 1) * 100:.1f}%"),
            ("Stage 5", "Watermark + SMILES", meta_triplets, meta_triplets - smiles_quarantined, smiles_quarantined, f"{(meta_triplets - smiles_quarantined) / max(meta_triplets, 1) * 100:.1f}%"),
            ("Stage 6", "CDN Image Upload", cdn_uploaded + cdn_cached + cdn_failed, cdn_uploaded + cdn_cached, cdn_failed, f"{(cdn_uploaded + cdn_cached) / max(cdn_uploaded + cdn_cached + cdn_failed, 1) * 100:.1f}%"),
            ("Stage 7", "LaTeX→MathML + Tagging", "—", "—", 0, "—"),
            ("Stage 8", "Duplicate Detection", "—", "—", self_dupes + cross_dupes, f"dupes: {self_dupes} self + {cross_dupes} cross"),
            ("Stage 9", "QA Verification (3-model)", qa_total, qa_e1 + qa_e2, qa_e3 + qa_e4, f"{(qa_e1 + qa_e2) / max(qa_total, 1) * 100:.1f}%"),
            ("Stage 10", "DB Upload", final_ready, uploaded, upload_failed, f"{uploaded / max(final_ready, 1) * 100:.1f}%"),
        ]
        for row in stages_flow:
            writer.writerow(row)

        # --- Section 2: Quarantine Breakdown ---
        output.write("\n=== QUARANTINE BREAKDOWN BY STAGE ===\n")
        writer.writerow(["Stage", "Questions Quarantined", "Reasons"])
        for st in sorted(quarantine_by_stage.keys()):
            # Collect reasons for this stage
            stage_reasons = {}
            for rec in quarantine_records:
                if rec.get("stage") == st:
                    r = rec.get("reason", "unknown")
                    stage_reasons[r] = stage_reasons.get(r, 0) + 1
            reason_str = "; ".join(f"{r}={c}" for r, c in sorted(stage_reasons.items()))
            writer.writerow([st, quarantine_by_stage[st], reason_str])
        writer.writerow(["TOTAL", total_quarantined, ""])

        # --- Section 3: QA Model-wise Accuracy ---
        output.write("\n=== MODEL-WISE QA ACCURACY ===\n")
        writer.writerow(["Model", "Total Checked", "Approved", "Rejected", "Approval Rate %"])
        for model in sorted(model_stats.keys()):
            ms = model_stats[model]
            rate = f"{ms['approved'] / max(ms['total'], 1) * 100:.1f}%"
            writer.writerow([model, ms["total"], ms["approved"], ms["rejected"], rate])

        # --- Section 4: QA Error Type Splits ---
        output.write("\n=== QA ERROR TYPE DISTRIBUTION ===\n")
        writer.writerow(["Error Type", "Count", "% of Total Errors"])
        total_errors = sum(error_type_counts.values())
        for etype, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
            pct = f"{count / max(total_errors, 1) * 100:.1f}%"
            writer.writerow([etype, count, pct])
        writer.writerow(["TOTAL", total_errors, "100%"])

        # --- Section 5: Subject Distribution ---
        output.write("\n=== SUBJECT DISTRIBUTION ===\n")
        writer.writerow(["Subject", "Count", "% of Total"])
        for subj, count in sorted(by_subject.items(), key=lambda x: -x[1]):
            pct = f"{count / max(total_meta, 1) * 100:.1f}%"
            writer.writerow([subj, count, pct])

        # --- Section 6: Question Type Distribution ---
        output.write("\n=== QUESTION TYPE DISTRIBUTION ===\n")
        writer.writerow(["Question Type", "Count", "% of Total"])
        for qt, count in sorted(by_type.items(), key=lambda x: -x[1]):
            pct = f"{count / max(total_meta, 1) * 100:.1f}%"
            writer.writerow([qt, count, pct])

        # --- Section 7: Image & SMILES Statistics ---
        output.write("\n=== IMAGE & SMILES STATISTICS ===\n")
        writer.writerow(["Metric", "Count"])
        writer.writerow(["Total <img> tags in metadata", img_count])
        writer.writerow(["SVG/SMILES images (base64 svg+xml)", svg_count])
        writer.writerow(["Regular images (non-SVG)", max(0, img_count - svg_count)])
        writer.writerow(["CDN images uploaded", cdn_uploaded])
        writer.writerow(["CDN images cached (already existed)", cdn_cached])
        writer.writerow(["CDN upload failures", cdn_failed])
        writer.writerow(["SMILES tags rendered successfully", smiles_success])
        writer.writerow(["SMILES tags render failures", smiles_fail])
        writer.writerow(["SMILES quarantined (all tags failed)", smiles_quarantined])

        # --- Section 8: Error Level Distribution ---
        output.write("\n=== QA ERROR LEVEL DISTRIBUTION ===\n")
        writer.writerow(["Error Level", "Description", "Count", "% of Total"])
        error_descs = {
            1: "3/3 approved (clean)",
            2: "2/3 approved (review)",
            3: "1/3 approved (rejected)",
            4: "0/3 approved (rejected)",
        }
        for lvl in (1, 2, 3, 4):
            count = by_error.get(lvl, 0)
            pct = f"{count / max(total_meta, 1) * 100:.1f}%"
            writer.writerow([lvl, error_descs.get(lvl, ""), count, pct])

        # --- Section 9: Final Summary ---
        output.write("\n=== FINAL PIPELINE SUMMARY ===\n")
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Questions extracted (Stage 2)", extracted])
        writer.writerow(["Questions in metadata (final)", total_meta])
        writer.writerow(["Total quarantined (all stages)", total_quarantined])
        writer.writerow(["Ready for upload (error 1+2)", final_ready])
        writer.writerow(["Actually uploaded to DB", uploaded])
        writer.writerow(["Upload failures", upload_failed])
        pipeline_accuracy = f"{final_ready / max(extracted, 1) * 100:.1f}%"
        writer.writerow(["Pipeline pass-through rate", pipeline_accuracy])
        qa_accuracy = f"{(qa_e1 + qa_e2) / max(qa_total, 1) * 100:.1f}%"
        writer.writerow(["QA approval rate (error 1+2)", qa_accuracy])
        clean_accuracy = f"{qa_e1 / max(qa_total, 1) * 100:.1f}%"
        writer.writerow(["Clean rate (3/3 approved)", clean_accuracy])

        csv_content = output.getvalue()

        # Save to file
        stats_path = self._get_upload_dir() / "pipeline_stats.csv"
        with open(stats_path, "w", newline="", encoding="utf-8") as f:
            f.write(csv_content)
        self.log("info", f"Pipeline stats CSV saved to {stats_path.name}")

        return csv_content

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        return self.upload(dry_run=True)

    def execute(self, **kwargs) -> Dict[str, Any]:
        return self.upload(dry_run=False)
