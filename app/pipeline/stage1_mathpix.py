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

    def _get_status_path(self) -> Path:
        """Path to the MathPIX status file for resume support."""
        return self.session_dir / "mathpix_output" / "mathpix_status.json"

    def _load_status(self) -> Dict:
        """Load MathPIX processing status."""
        p = self._get_status_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_status(self, status: Dict) -> None:
        """Save MathPIX processing status."""
        p = self._get_status_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(status, indent=2), encoding="utf-8")

    def check_credentials(self) -> Dict[str, Any]:
        """Check if MathPIX credentials are configured."""
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
        """Upload PDF to MathPIX for OCR processing.

        Returns: {pdf_id, status} on success, {error} on failure.
        """
        if not HAS_REQUESTS:
            return {"error": "Python 'requests' package not installed"}

        app_id, app_key = self._get_credentials()
        if not app_id or not app_key:
            return {"error": "MathPIX credentials not configured"}

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return {"error": f"PDF not found: {pdf_path}"}

        # Check for existing upload (resume support)
        status = self._load_status()
        if status.get("pdf_id"):
            self.log("info", f"Reusing existing MathPIX pdf_id: {status['pdf_id']}")
            return {
                "pdf_id": status["pdf_id"],
                "status": "already_uploaded",
                "message": "PDF already submitted to MathPIX — poll for status",
            }

        self.log("info", f"Uploading {pdf_file.name} ({pdf_file.stat().st_size / 1024 / 1024:.1f} MB) to MathPIX...")

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
        """Poll MathPIX for current processing status.

        Returns: {status, percent_done, num_pages, ...}
        """
        if not HAS_REQUESTS:
            return {"error": "Python 'requests' package not installed"}

        status = self._load_status()
        pdf_id = status.get("pdf_id")
        if not pdf_id:
            return {"error": "No MathPIX upload found — upload a PDF first"}

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
        """Download and extract MMD results from MathPIX.

        Returns: {mmd_path, images_count, ...}
        """
        if not HAS_REQUESTS:
            return {"error": "Python 'requests' package not installed"}

        status = self._load_status()
        pdf_id = status.get("pdf_id")
        if not pdf_id:
            return {"error": "No MathPIX upload found"}

        app_id, app_key = self._get_credentials()
        headers = {"app_id": app_id, "app_key": app_key}

        out_dir = self.session_dir / "mathpix_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / f"{pdf_id}.mmd.zip"

        # Download zip
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

            self.log("success", f"Downloaded {zip_path.name} ({zip_path.stat().st_size / 1024:.0f} KB)")
        except Exception as e:
            return {"error": f"Download failed: {e}"}

        # Extract zip
        self.log("info", "Extracting MMD and images...")
        try:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(out_dir))
        except Exception as e:
            return {"error": f"Extraction failed: {e}"}

        # Find .mmd file
        mmd_path = None
        for f in out_dir.rglob("*.mmd"):
            mmd_path = f
            break

        if not mmd_path:
            return {"error": "No .mmd file found in downloaded zip"}

        # Count extracted images
        images_dir = out_dir / "images"
        img_exts = {".png", ".jpg", ".jpeg", ".gif"}
        images_count = 0
        if images_dir.exists():
            images_count = sum(1 for f in images_dir.iterdir() if f.suffix.lower() in img_exts)

        # Also copy images to images_original/ for watermark stage
        if images_count > 0:
            orig_dir = self.session_dir / "images_original"
            orig_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for img in images_dir.iterdir():
                if img.suffix.lower() in img_exts:
                    shutil.copy2(img, orig_dir / img.name)
                    copied += 1
            self.log("info", f"Copied {copied} images to images_original/")

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
