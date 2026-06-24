"""Stage 1: MathPIX OCR — pipeline adapter.

For testing: copies pre-existing MMD files into session directory.
For production: submits PDF to MathPIX API and polls for results.

Forked from: CodeBase/PCMB_MathPIX_Uploader.py
CHANGES: Adapted for session-based pipeline, async-friendly polling.
"""

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.pipeline.base import BasePipelineStage

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# MathPIX API endpoints
MATHPIX_UPLOAD_URL = "https://api.mathpix.com/v3/pdf"
MATHPIX_STATUS_URL = "https://api.mathpix.com/v3/pdf/{pdf_id}"
MATHPIX_DOWNLOAD_URL = "https://api.mathpix.com/v3/pdf/{pdf_id}.mmd.zip"

MATHPIX_OPTIONS = {
    "conversion_formats": {"docx": True, "tex.zip": True},
    "rm_spaces": True,
    "math_inline_delimiters": ["$", "$"],
    "math_display_delimiters": ["$$", "$$"],
}


class MathPIXStage(BasePipelineStage):

    @property
    def stage_name(self) -> str:
        return "stage1_upload"

    def _get_credentials(self) -> Tuple[str, str]:
        """Get MathPIX credentials from config."""
        from app.config import settings
        return settings.mathpix_app_id, settings.mathpix_app_key

    def _get_mineru_token(self) -> str:
        """Get MinerU token from config."""
        from app.config import settings
        return settings.mineru_token

    def _get_status_path(self) -> Path:
        """Path to the status file for resume support."""
        return self.session_dir / "mathpix_output" / "mathpix_status.json"

    def _load_status(self) -> Dict:
        """Load processing status."""
        p = self._get_status_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_status(self, status: Dict) -> None:
        """Save processing status."""
        p = self._get_status_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(status, indent=2), encoding="utf-8")

    def check_credentials(self) -> Dict[str, Any]:
        """Check if OCR credentials are configured."""
        from app.config import settings
        engine = getattr(settings, "ocr_engine", "mathpix")
        if engine == "mineru":
            token = self._get_mineru_token()
            if not token:
                return {
                    "configured": False,
                    "message": "MinerU Token not set. Add MINERU_TOKEN to .env",
                }
            if not HAS_REQUESTS:
                return {
                    "configured": False,
                    "message": "Python 'requests' package not installed. Run: pip install requests",
                }
            return {
                "configured": True,
                "token": token[:10] + "...",
                "message": "MinerU credentials configured",
            }
        else:
            app_id, app_key = self._get_credentials()
            if not app_id or not app_key:
                return {
                    "configured": False,
                    "message": "MathPIX credentials not set. Add MATHPIX_APP_ID and MATHPIX_APP_KEY to .env",
                }
            if not HAS_REQUESTS:
                return {
                    "configured": False,
                    "message": "Python 'requests' package not installed. Run: pip install requests",
                }
            return {
                "configured": True,
                "app_id": app_id[:10] + "...",
                "message": "MathPIX credentials configured",
            }

    def upload_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Upload PDF to OCR engine for processing.

        Returns: {pdf_id, status} on success, {error} on failure.
        """
        if not HAS_REQUESTS:
            return {"error": "Python 'requests' package not installed"}

        from app.config import settings
        engine = getattr(settings, "ocr_engine", "mathpix")

        # Check for existing upload (resume support)
        status = self._load_status()
        if status.get("pdf_id"):
            self.log("info", f"Reusing existing OCR pdf_id: {status['pdf_id']}")
            return {
                "pdf_id": status["pdf_id"],
                "status": "already_uploaded",
                "message": f"PDF already submitted to {engine} — poll for status",
            }

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return {"error": f"PDF not found: {pdf_path}"}

        self.log("info", f"Uploading {pdf_file.name} ({pdf_file.stat().st_size / 1024 / 1024:.1f} MB) to {engine}...")

        if engine == "mineru":
            token = self._get_mineru_token()
            if not token:
                return {"error": "MinerU token not configured"}
            url = "https://mineru.net/api/v4/file-urls/batch"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            data = {
                "files": [
                    {"name": pdf_file.name, "data_id": f"ocr_{self.session_id[:8]}"}
                ],
                "model_version": "vlm"
            }
            try:
                resp = requests.post(url, headers=headers, json=data, timeout=60)
                if resp.status_code != 200:
                    return {"error": f"MinerU Batch API error: HTTP {resp.status_code}"}
                result = resp.json()
                if result.get("code") != 0:
                    return {"error": f"MinerU Batch API error: {result.get('msg')}"}
                batch_id = result["data"]["batch_id"]
                upload_urls = result["data"]["file_urls"]
                if not upload_urls:
                    return {"error": "No upload URL returned by MinerU"}
                
                # Upload the PDF via PUT
                self.log("info", "Uploading file content to MinerU storage...")
                with open(pdf_path, "rb") as f:
                    res_upload = requests.put(upload_urls[0], data=f, timeout=300)
                if res_upload.status_code != 200:
                    return {"error": f"MinerU file upload failed: HTTP {res_upload.status_code}"}
                
                status = {
                    "pdf_id": batch_id,
                    "upload_status": "uploaded",
                    "pdf_filename": pdf_file.name,
                }
                self._save_status(status)
                self.log("success", f"Upload OK — batch_id: {batch_id}")
                return {"pdf_id": batch_id, "status": "uploaded"}
            except Exception as e:
                return {"error": f"MinerU upload failed: {e}"}
        else:
            app_id, app_key = self._get_credentials()
            if not app_id or not app_key:
                return {"error": "MathPIX credentials not configured"}

            headers = {"app_id": app_id, "app_key": app_key}
            try:
                with open(pdf_path, "rb") as f:
                    files = {"file": (pdf_file.name, f, "application/pdf")}
                    data = {"options_json": json.dumps(MATHPIX_OPTIONS)}
                    resp = requests.post(
                        MATHPIX_UPLOAD_URL, headers=headers,
                        files=files, data=data, timeout=300,
                    )

                if resp.status_code == 200:
                    pdf_id = resp.json().get("pdf_id")
                    if pdf_id:
                        status = {
                            "pdf_id": pdf_id,
                            "upload_status": "uploaded",
                            "pdf_filename": pdf_file.name,
                        }
                        self._save_status(status)
                        self.log("success", f"Upload OK — pdf_id: {pdf_id}")
                        return {"pdf_id": pdf_id, "status": "uploaded"}
                    return {"error": f"No pdf_id in response: {resp.text[:200]}"}
                return {"error": f"MathPIX HTTP {resp.status_code}: {resp.text[:300]}"}
            except requests.exceptions.Timeout:
                return {"error": "MathPIX upload timed out (300s). Try again or use a smaller PDF."}
            except Exception as e:
                return {"error": f"Upload failed: {e}"}

    def poll_status(self) -> Dict[str, Any]:
        """Poll OCR engine for current processing status.

        Returns: {status, percent_done, num_pages, ...}
        """
        if not HAS_REQUESTS:
            return {"error": "Python 'requests' package not installed"}

        status = self._load_status()
        pdf_id = status.get("pdf_id")
        if not pdf_id:
            return {"error": "No OCR upload found — upload a PDF first"}

        from app.config import settings
        engine = getattr(settings, "ocr_engine", "mathpix")

        if engine == "mineru":
            token = self._get_mineru_token()
            url = f"https://mineru.net/api/v4/extract-results/batch/{pdf_id}"
            headers = {"Authorization": f"Bearer {token}"}
            try:
                resp = requests.get(url, headers=headers, timeout=60)
                if resp.status_code == 200:
                    res_data = resp.json()
                    if res_data.get("code") == 0:
                        extract_results = res_data.get("data", {}).get("extract_result", [])
                        if extract_results:
                            first_result = extract_results[0]
                            state = first_result.get("state")
                            
                            # Map MinerU state
                            if state == "done":
                                ocr_status = "completed"
                                percent_done = 100
                                # Save URLs for downloading
                                status["full_zip_url"] = first_result.get("full_zip_url")
                                status["md_url"] = first_result.get("md_url")
                            elif state == "failed":
                                ocr_status = "failed"
                                percent_done = 0
                            else:
                                ocr_status = state
                                percent_done = 50
                            
                            status["ocr_status"] = ocr_status
                            status["percent_done"] = percent_done
                            self._save_status(status)

                            if ocr_status == "completed":
                                self.log("success", "MinerU OCR complete")
                            else:
                                self.log("info", f"MinerU: {ocr_status} ({percent_done}% done)")

                            return {
                                "pdf_id": pdf_id,
                                "status": ocr_status,
                                "percent_done": percent_done,
                                "num_pages": 0,
                            }
                        return {"error": "No extract results found in response data"}
                    return {"error": f"MinerU API error: {res_data.get('msg')}"}
                return {"error": f"MinerU HTTP {resp.status_code}"}
            except Exception as e:
                return {"error": f"Poll failed: {e}"}
        else:
            app_id, app_key = self._get_credentials()
            headers = {"app_id": app_id, "app_key": app_key}
            url = MATHPIX_STATUS_URL.format(pdf_id=pdf_id)

            try:
                resp = requests.get(url, headers=headers, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    result = {
                        "pdf_id": pdf_id,
                        "status": data.get("status", "unknown"),
                        "percent_done": data.get("percent_done", 0),
                        "num_pages": data.get("num_pages_completed", 0),
                    }

                    # Persist status
                    status["ocr_status"] = result["status"]
                    status["percent_done"] = result["percent_done"]
                    self._save_status(status)

                    if result["status"] == "completed":
                        self.log("success", f"MathPIX OCR complete — {result['num_pages']} pages")
                    else:
                        self.log("info", f"MathPIX: {result['status']} ({result['percent_done']}% done)")

                    return result
                return {"error": f"MathPIX HTTP {resp.status_code}: {resp.text[:200]}"}
            except Exception as e:
                return {"error": f"Poll failed: {e}"}

    def download_results(self) -> Dict[str, Any]:
        """Download and extract results from OCR engine.

        Returns: {mmd_path, images_count, ...}
        """
        if not HAS_REQUESTS:
            return {"error": "Python 'requests' package not installed"}

        status = self._load_status()
        pdf_id = status.get("pdf_id")
        if not pdf_id:
            return {"error": "No OCR upload found"}

        from app.config import settings
        engine = getattr(settings, "ocr_engine", "mathpix")

        out_dir = self.session_dir / "mathpix_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / f"{pdf_id}.zip"

        if engine == "mineru":
            zip_url = status.get("full_zip_url")
            if not zip_url:
                return {"error": "No zip URL found in status. Check if polling finished successfully."}
            self.log("info", "Downloading MinerU results ZIP...")
            try:
                resp = requests.get(zip_url, stream=True, timeout=300)
                if resp.status_code != 200:
                    return {"error": f"Download failed: HTTP {resp.status_code}"}
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                self.log("success", f"Downloaded MinerU results: {zip_path.name} ({zip_path.stat().st_size / 1024:.0f} KB)")
            except Exception as e:
                return {"error": f"Download failed: {e}"}
        else:
            app_id, app_key = self._get_credentials()
            headers = {"app_id": app_id, "app_key": app_key}
            self.log("info", "Downloading MathPIX results...")
            url = MATHPIX_DOWNLOAD_URL.format(pdf_id=pdf_id)
            try:
                resp = requests.get(url, headers=headers, stream=True, timeout=300)
                if resp.status_code != 200:
                    return {"error": f"Download failed: HTTP {resp.status_code}"}
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                self.log("success", f"Downloaded MathPIX results: {zip_path.name} ({zip_path.stat().st_size / 1024:.0f} KB)")
            except Exception as e:
                return {"error": f"Download failed: {e}"}

        # Extract zip
        self.log("info", "Extracting OCR output and images...")
        try:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(out_dir))
        except Exception as e:
            return {"error": f"Extraction failed: {e}"}

        # Find .md or .mmd file (exclude backups or parts)
        mmd_path = None
        for f in out_dir.rglob("*"):
            if f.is_file() and f.suffix in (".md", ".mmd") and "backup" not in f.name and not f.name.startswith("part"):
                mmd_path = f
                break

        if not mmd_path:
            return {"error": "No markdown/MMD file found in downloaded zip"}

        # Move the main .md/.mmd file to the root of mathpix_output for Stage 2 mapping
        if mmd_path.parent != out_dir:
            dest_path = out_dir / mmd_path.name
            shutil.copy2(mmd_path, dest_path)
            mmd_path = dest_path

        # Count and copy extracted images recursively (handling different folder shapes)
        img_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        images_count = 0
        orig_dir = self.session_dir / "images_original"
        orig_dir.mkdir(parents=True, exist_ok=True)
        
        for f in out_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in img_exts and f.parent != orig_dir:
                shutil.copy2(f, orig_dir / f.name)
                images_count += 1

        if images_count > 0:
            self.log("info", f"Copied {images_count} images to images_original/")

        # Update status
        status["download_status"] = "completed"
        status["mmd_path"] = str(mmd_path)
        status["images_count"] = images_count
        self._save_status(status)

        # Save checkpoint
        self.save_checkpoint("completed", {
            "pdf_id": pdf_id,
            "mmd_path": str(mmd_path.relative_to(self.session_dir)),
            "images_count": images_count,
        })

        mmd_lines = mmd_path.read_text(encoding="utf-8", errors="replace").count("\n")
        self.log("success", f"OCR complete: {mmd_path.name} ({mmd_lines} lines), {images_count} images")

        return {
            "mmd_path": str(mmd_path.relative_to(self.session_dir)),
            "mmd_filename": mmd_path.name,
            "mmd_lines": mmd_lines,
            "images_count": images_count,
            "status": "completed",
        }

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        pdf_path = kwargs.get("pdf_path")
        if pdf_path:
            p = Path(pdf_path)
            return {"file": p.name, "size_bytes": p.stat().st_size if p.exists() else 0}
        return {"message": "No PDF provided"}

    def execute(self, **kwargs) -> Dict[str, Any]:
        """For testing: load pre-existing MMD files into session."""
        mmd_questions = kwargs.get("mmd_questions")
        mmd_solutions = kwargs.get("mmd_solutions")
        answer_key = kwargs.get("answer_key")

        out_dir = self.session_dir / "mathpix_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        files_copied = []

        if mmd_questions:
            src = Path(mmd_questions)
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            files_copied.append(dst.name)
            self.log("success", f"Loaded questions MMD: {src.name}")

        if mmd_solutions:
            src = Path(mmd_solutions)
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            files_copied.append(dst.name)
            self.log("success", f"Loaded solutions MMD: {src.name}")

        if answer_key:
            src = Path(answer_key)
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            files_copied.append(dst.name)
            self.log("success", f"Loaded answer key: {src.name}")

        self.save_checkpoint("completed", {"files": files_copied})
        return {"files": files_copied, "count": len(files_copied)}
