"""Stage 6: Image CDN Upload — pipeline adapter.

Scans metadata_v2 files for image references, uploads cleaned images to
Acadza CDN via api.acadza.in/question/upload/androidbulk, computes display
dimensions with PIL, and replaces local file paths with CDN URLs in metadata.

Forked from: CodeBase/PCMB_Image_CDN_Uploader.py
CHANGES:
  - Session-based paths instead of global SOURCE_CONFIG
  - Uses images_cleaned/ first, images_original/ as fallback
  - Persistent CDN map stored in session dir as images_cdn_map.json
  - Timestamped backup before metadata modification
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.pipeline.base import BasePipelineStage

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cairosvg as _cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False

try:
    import requests as http_requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# CDN upload endpoint (same as reference PCMB_Image_CDN_Uploader.py)
IMAGE_UPLOAD_API = "https://api.acadza.in/question/upload/androidbulk"

# Display dimension limits (from reference)
MAX_DISPLAY_WIDTH = 350
MIN_DISPLAY_HEIGHT = 40

# Rate limiting
UPLOAD_DELAY_S = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_S = 2

# Regex for <img> tags
IMG_TAG_PATTERN = re.compile(
    r'<img\b[^>]*\bsrc="(?P<src>[^"]+)"[^>]*/?>',
    re.IGNORECASE,
)

# Regex for base64 data URI images (from SMILES rendering etc.)
BASE64_IMG_PATTERN = re.compile(
    r'<img\b[^>]*\bsrc="(?P<src>data:image/(?P<fmt>[a-z+]+);base64,(?P<b64>[A-Za-z0-9+/=]+))"[^>]*/?>',
    re.IGNORECASE,
)

import base64 as _b64
import hashlib as _hashlib


def compute_display_dimensions(width_px: int, height_px: int) -> Tuple[int, int]:
    """Compute display dimensions with max-width scaling and min-height clamp."""
    if width_px > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / width_px
        dw = MAX_DISPLAY_WIDTH
        dh = max(MIN_DISPLAY_HEIGHT, int(height_px * scale))
    else:
        dw = width_px
        dh = max(MIN_DISPLAY_HEIGHT, height_px)
    return dw, dh


def _guess_mime(path: str) -> str:
    """Guess MIME type from file extension."""
    ext = os.path.splitext(path)[1].lower()
    return {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "application/octet-stream")


def upload_single_image(local_path: str, timeout: int = 60, logger=None) -> Tuple[bool, Optional[str], Optional[str]]:
    """Upload one image to Acadza CDN.  Returns (success, cdn_url, error)."""
    if not os.path.exists(local_path):
        return False, None, f"File not found: {local_path}"

    from app.pipeline.auth import authenticated_request

    basename = os.path.basename(local_path)
    mime = _guess_mime(local_path)
    with open(local_path, "rb") as f:
        resp = authenticated_request(
            "post",
            IMAGE_UPLOAD_API,
            logger=logger,
            max_retries=1,  # outer retry handles retries
            files={"image": (basename, f, mime)},
            timeout=timeout,
        )
    resp.raise_for_status()
    data = resp.json()
    url = data.get("url") or data.get("URL")
    if url:
        return True, url, None
    return False, None, f"No URL in response: {data}"


def upload_with_retry(local_path: str, logger=None) -> Tuple[bool, Optional[str], Optional[str]]:
    """Upload with exponential backoff retry and auth handling."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            success, url, err = upload_single_image(local_path, logger=logger)
            if success:
                return True, url, None
            last_err = err
        except Exception as e:
            last_err = str(e)

        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF_BASE_S * (2 ** attempt)
            time.sleep(wait)

    return False, None, last_err or "Max retries exceeded"


# ── Image src extraction (verbatim from reference) ──

IMG_SRC_PATTERN = re.compile(r'<img[^>]*\bsrc="([^"]+)"', re.IGNORECASE)


def _extract_img_srcs_from_text(text: str) -> List[str]:
    if not text:
        return []
    return IMG_SRC_PATTERN.findall(text)


def _extract_base64_imgs_from_text(text: str) -> List[dict]:
    """Extract base64 data URI <img> tags from text.

    Returns list of {data_uri, fmt, b64_data, hash} for each base64 image found.
    """
    if not text or "data:image/" not in text:
        return []
    results = []
    for match in BASE64_IMG_PATTERN.finditer(text):
        data_uri = match.group("src")
        fmt = match.group("fmt")
        b64_data = match.group("b64")
        # Use content hash for dedup and filename
        content_hash = _hashlib.sha256(b64_data.encode("ascii")).hexdigest()[:16]
        ext = "svg" if "svg" in fmt else "png"
        results.append({
            "data_uri": data_uri,
            "fmt": fmt,
            "b64_data": b64_data,
            "hash": content_hash,
            "ext": ext,
            "filename": f"smiles_{content_hash}.{ext}",
        })
    return results


def _save_base64_to_file(b64_data: str, filepath: Path) -> bool:
    """Decode base64 data and save to file."""
    try:
        raw = _b64.b64decode(b64_data)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(raw)
        return True
    except Exception:
        return False


def _svg_to_png(svg_path: str) -> Optional[str]:
    """Convert an SVG file to PNG.  Returns PNG path on success, None on failure.

    Uses cairosvg for high-quality rasterisation.  The output PNG is written
    next to the original SVG with a .png extension.
    """
    if not HAS_CAIROSVG:
        return None
    png_path = os.path.splitext(svg_path)[0] + ".png"
    try:
        _cairosvg.svg2png(url=svg_path, write_to=png_path)
        if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
            return png_path
        return None
    except Exception:
        # Silently fail — caller will keep the base64 SVG as fallback
        if os.path.exists(png_path):
            try:
                os.remove(png_path)
            except OSError:
                pass
        return None


def _extract_img_srcs_from_question(q: dict) -> List[str]:
    """Extract all local image src values from a question entry."""
    srcs = []
    for container_key in ("scq", "mcq", "integerQuestion", "subjective"):
        container = q.get(container_key)
        if not isinstance(container, dict):
            continue
        for field in ("question", "solution"):
            srcs.extend(_extract_img_srcs_from_text(container.get(field, "")))

    pq = q.get("passageQuestion")
    if isinstance(pq, dict):
        srcs.extend(_extract_img_srcs_from_text(pq.get("passage", "")))
        srcs.extend(_extract_img_srcs_from_text(pq.get("solution", "")))
        for sub_q in pq.get("questions", []):
            if isinstance(sub_q, dict):
                for sub_key in ("scq", "mcq", "integerQuestion"):
                    sub_container = sub_q.get(sub_key)
                    if isinstance(sub_container, dict):
                        for field in ("ques", "question", "solution"):
                            srcs.extend(_extract_img_srcs_from_text(
                                sub_container.get(field, "")))

    return [s for s in srcs if s.startswith("./images/") or
            (not s.startswith(("http://", "https://", "//")) and "images/" in s)]


# ── Replacement helpers (verbatim from reference) ──

def _replace_img_tags_in_text(text: str, cdn_cache: Dict[str, Dict]) -> Tuple[str, int]:
    """Replace local <img> tags with CDN URL + actual dimensions."""
    if not text:
        return text, 0

    count = 0

    def _replacer(match):
        nonlocal count
        full_tag = match.group(0)
        src = match.group("src")

        entry = cdn_cache.get(src)
        if not entry or "cdn_url" not in entry:
            return full_tag

        cdn_url = entry["cdn_url"]
        dw = entry.get("display_width", MAX_DISPLAY_WIDTH)
        dh = entry.get("display_height", 169)

        count += 1
        return f'<img src="{cdn_url}" alt="" style="width:{dw}px; height:{dh}px" />'

    new_text = IMG_TAG_PATTERN.sub(_replacer, text)
    return new_text, count


def _replace_in_question(q: dict, cdn_cache: Dict[str, Dict]) -> int:
    """Replace image tags in a single question entry. Returns replacement count."""
    total = 0

    for container_key in ("scq", "mcq", "integerQuestion", "subjective"):
        container = q.get(container_key)
        if not isinstance(container, dict):
            continue
        for field in ("question", "solution"):
            text = container.get(field, "")
            if text:
                new_text, n = _replace_img_tags_in_text(text, cdn_cache)
                if n > 0:
                    container[field] = new_text
                    total += n

    pq = q.get("passageQuestion")
    if isinstance(pq, dict):
        for field in ("passage", "solution"):
            text = pq.get(field, "")
            if text:
                new_text, n = _replace_img_tags_in_text(text, cdn_cache)
                if n > 0:
                    pq[field] = new_text
                    total += n

        for sub_q in pq.get("questions", []):
            if isinstance(sub_q, dict):
                for sub_key in ("scq", "mcq", "integerQuestion"):
                    sub_container = sub_q.get(sub_key)
                    if isinstance(sub_container, dict):
                        for field in ("ques", "question", "solution"):
                            text = sub_container.get(field, "")
                            if text:
                                new_text, n = _replace_img_tags_in_text(text, cdn_cache)
                                if n > 0:
                                    sub_container[field] = new_text
                                    total += n

    return total


# ════════════════════════════════════════════════════════════════════
#  ImageCDNStage — pipeline adapter
# ════════════════════════════════════════════════════════════════════

class ImageCDNStage(BasePipelineStage):
    """Stage 6 adapter: scan metadata for image refs, upload to CDN, replace."""

    @property
    def stage_name(self) -> str:
        return "stage6_images"

    # ── Path helpers ──

    def _metadata_dir(self) -> Path:
        # Stage 4 writes to metadata_v2/, production pipeline uses metadata_v2_output/
        for name in ("metadata_v2", "metadata_v2_output"):
            d = self.session_dir / name
            if d.exists():
                return d
        return self.session_dir / "metadata_v2"

    def _images_cleaned_dir(self) -> Path:
        return self.session_dir / "images_cleaned"

    def _images_original_dir(self) -> Path:
        return self.session_dir / "images_original"

    def _images_map_path(self) -> Path:
        return self.session_dir / "images_map.json"

    def _cdn_cache_path(self) -> Path:
        return self.session_dir / "images_cdn_map.json"

    # ── CDN cache persistence ──

    def _load_cdn_cache(self) -> Dict[str, Dict]:
        p = self._cdn_cache_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cdn_cache(self, cache: Dict[str, Dict]) -> None:
        p = self._cdn_cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    # ── Images map (for page-prefixed filenames) ──

    def _load_images_map(self) -> Dict[str, str]:
        p = self._images_map_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ── Path resolution ──

    # Image extensions to try when exact match fails (pipeline converts to PNG)
    _IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

    def _resolve_local_path(self, img_src: str, images_map: Dict[str, str]) -> Optional[Path]:
        """Resolve ./images/X.jpg -> actual file. Checks cleaned/ then original/.

        Pipeline Stage 5 converts images to PNG during watermark removal,
        so cleaned/ has .png while metadata references .jpg. This method
        tries the exact filename first, then tries alternate extensions
        (prioritizing .png since that's what Stage 5 outputs).
        """
        bare = img_src
        if bare.startswith("./images/"):
            bare = bare[len("./images/"):]
        elif bare.startswith("images/"):
            bare = bare[len("images/"):]

        # Determine actual filename via images_map if present
        actual_name = images_map.get(bare, bare) if images_map else bare
        stem = Path(actual_name).stem

        # Try images_cleaned first (preferred: watermark-free), then images_original
        for img_dir in (self._images_cleaned_dir(), self._images_original_dir()):
            # Exact match
            candidate = img_dir / actual_name
            if candidate.exists():
                return candidate
            # Extension-agnostic: try .png first (Stage 5 output), then others
            for ext in self._IMG_EXTENSIONS:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate

        # Also try bare name without map
        if images_map and bare != actual_name:
            bare_stem = Path(bare).stem
            for img_dir in (self._images_cleaned_dir(), self._images_original_dir()):
                candidate = img_dir / bare
                if candidate.exists():
                    return candidate
                for ext in self._IMG_EXTENSIONS:
                    candidate = img_dir / f"{bare_stem}{ext}"
                    if candidate.exists():
                        return candidate

        return None

    # ── Metadata file enumeration ──

    def _iter_metadata_files(self) -> List[Path]:
        """Find *_metadata_v2.json files — handles both flat and subdirectory layouts."""
        md = self._metadata_dir()
        if not md.exists():
            return []
        files = []
        # Check for flat files directly in metadata dir (Stage 4 web app layout)
        for f in sorted(md.glob("*_metadata_v2.json")):
            if "backup" not in f.name:
                files.append(f)
        # Also check subdirectories (production pipeline layout: BRM-01/, NRY-01/, etc.)
        for sub_dir in sorted(md.iterdir()):
            if not sub_dir.is_dir():
                continue
            for f in sorted(sub_dir.glob("*_metadata_v2.json")):
                if "backup" not in f.name:
                    files.append(f)
        return files

    # ── Scan (Phase 1-2) ──

    def scan(self) -> Dict[str, Any]:
        """Scan metadata for image refs, resolve paths, report stats."""
        md_files = self._iter_metadata_files()
        if not md_files:
            return {
                "error": None,
                "total_questions": 0,
                "questions_with_images": 0,
                "referenced_images": 0,
                "unique_images": 0,
                "images_on_disk": 0,
                "missing_images": 0,
                "orphan_images": 0,
                "can_skip": True,
                "missing_list": [],
            }

        images_map = self._load_images_map()
        total_qs = 0
        qs_with_imgs = 0
        all_srcs = []
        unique_srcs = set()
        resolved = {}
        unresolved = []

        # Filter quarantined questions from image processing
        from app.db import get_quarantined_qids
        q_qids = get_quarantined_qids(self.session_id)

        for json_file in md_files:
            with open(json_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
            for q in questions:
                if q_qids and q.get("qid") in q_qids:
                    continue  # Skip quarantined questions
                total_qs += 1
                img_srcs = _extract_img_srcs_from_question(q)
                if img_srcs:
                    qs_with_imgs += 1
                    all_srcs.extend(img_srcs)
                    for src in img_srcs:
                        unique_srcs.add(src)
                        if src not in resolved and src not in unresolved:
                            lp = self._resolve_local_path(src, images_map)
                            if lp:
                                resolved[src] = str(lp)
                            else:
                                unresolved.append(src)

        # Count images on disk
        disk_images = set()
        for img_dir in (self._images_cleaned_dir(), self._images_original_dir()):
            if img_dir.exists():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.gif", "*.webp"):
                    for p in img_dir.glob(ext):
                        disk_images.add(p.name)

        referenced_names = set()
        for src in unique_srcs:
            bare = src
            for prefix in ("./images/", "images/"):
                if bare.startswith(prefix):
                    bare = bare[len(prefix):]
                    break
            referenced_names.add(bare)

        orphan = len(disk_images - referenced_names)

        # Count base64 data URI images (SMILES SVG, etc.)
        base64_count = 0
        for json_file in md_files:
            with open(json_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
            for q in questions:
                for ck in ("scq", "mcq", "integerQuestion", "subjective",
                           "passageQuestion", "matchQuestion"):
                    c = q.get(ck)
                    if not isinstance(c, dict):
                        continue
                    for fk in ("question", "ques", "solution", "sol", "passage"):
                        text = c.get(fk, "")
                        if text and "data:image/" in text:
                            base64_count += len(_extract_base64_imgs_from_text(text))

        # Check CDN cache — how many already uploaded
        cdn_cache = self._load_cdn_cache()
        already_cached = sum(1 for src in resolved if src in cdn_cache and "cdn_url" in cdn_cache[src])
        b64_cached = sum(1 for uri in cdn_cache if uri.startswith("data:image/"))

        self.save_checkpoint("scanned", {
            "total_questions": total_qs,
            "unique_images": len(unique_srcs),
            "resolved": len(resolved),
            "unresolved": len(unresolved),
            "already_cached": already_cached,
            "base64_images": base64_count,
        })

        return {
            "total_questions": total_qs,
            "questions_with_images": qs_with_imgs,
            "referenced_images": len(all_srcs),
            "unique_images": len(unique_srcs),
            "images_on_disk": len(resolved),
            "missing_images": len(unresolved),
            "orphan_images": orphan,
            "already_cached": already_cached,
            "base64_images": base64_count,
            "base64_cached": b64_cached,
            "can_skip": len(unique_srcs) == 0 and base64_count == 0,
            "missing_list": unresolved[:20],
        }

    # ── Credential check ──

    def check_credentials(self) -> Dict[str, Any]:
        """Check if the CDN API endpoint is reachable, auth works, and deps installed."""
        issues = []
        if not HAS_REQUESTS:
            issues.append("requests library not installed (pip install requests)")
        if not HAS_PIL:
            issues.append("Pillow not installed (pip install Pillow)")

        if issues:
            return {
                "configured": False,
                "message": "Missing dependencies: " + "; ".join(issues),
            }

        # Verify JWT auth works
        try:
            from app.pipeline.auth import ensure_valid_token
            ensure_valid_token(logger=self.log)
            auth_ok = True
        except Exception as e:
            return {
                "configured": False,
                "message": f"Auth failed: {e}",
            }

        # Quick connectivity test to the API
        try:
            resp = http_requests.head(IMAGE_UPLOAD_API, timeout=10)
            api_ok = resp.status_code < 500
        except Exception:
            api_ok = False

        if not api_ok:
            return {
                "configured": False,
                "message": f"CDN API not reachable: {IMAGE_UPLOAD_API}",
            }

        return {
            "configured": True,
            "endpoint": IMAGE_UPLOAD_API,
            "message": "Auth OK, CDN API endpoint reachable, dependencies OK",
        }

    # ── Upload (Phase 3 + 4 combined) ──

    def upload_images(self) -> Dict[str, Any]:
        """Upload all images to CDN and replace refs in metadata."""
        if not HAS_REQUESTS:
            return {"error": "requests library not installed"}
        if not HAS_PIL:
            return {"error": "Pillow not installed"}

        self.log("info", "Starting image CDN upload...")
        self.save_checkpoint("uploading", {})

        # Phase 1-2: Scan + resolve
        md_files = self._iter_metadata_files()
        if not md_files:
            self.log("warning", "No metadata_v2 files found")
            return {"error": "No metadata_v2 files found in session"}

        images_map = self._load_images_map()
        resolved = {}
        unresolved_srcs = []
        file_img_refs: Dict[str, List[Tuple[str, str]]] = {}

        # Filter quarantined questions
        from app.db import get_quarantined_qids
        q_qids = get_quarantined_qids(self.session_id)

        # Also collect base64 data URI images (e.g., SMILES SVG rendered in Stage 5)
        base64_images = {}  # data_uri -> {filename, b64_data, ext, ...}

        total_qs = 0
        for json_file in md_files:
            with open(json_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
            for q in questions:
                if q_qids and q.get("qid") in q_qids:
                    continue  # Skip quarantined questions
                total_qs += 1
                qid = q.get("qid", "unknown")
                img_srcs = _extract_img_srcs_from_question(q)
                for src in img_srcs:
                    file_img_refs.setdefault(str(json_file), []).append((qid, src))
                    if src not in resolved and src not in unresolved_srcs:
                        lp = self._resolve_local_path(src, images_map)
                        if lp:
                            resolved[src] = str(lp)
                        else:
                            unresolved_srcs.append(src)

                # Scan for base64 data URI images in all fields
                for container_key in ("scq", "mcq", "integerQuestion", "subjective",
                                      "passageQuestion", "matchQuestion"):
                    container = q.get(container_key)
                    if not isinstance(container, dict):
                        continue
                    for field in ("question", "ques", "solution", "sol", "passage"):
                        text = container.get(field, "")
                        if text and "data:image/" in text:
                            for b64_img in _extract_base64_imgs_from_text(text):
                                data_uri = b64_img["data_uri"]
                                if data_uri not in base64_images:
                                    base64_images[data_uri] = b64_img
                                    file_img_refs.setdefault(str(json_file), []).append(
                                        (qid, data_uri))
                    # Also check passageQuestion sub-questions
                    if container_key == "passageQuestion":
                        for sub_q in container.get("questions", []):
                            if not isinstance(sub_q, dict):
                                continue
                            for sub_key in ("scq", "mcq", "integerQuestion"):
                                sub_c = sub_q.get(sub_key)
                                if not isinstance(sub_c, dict):
                                    continue
                                for field in ("ques", "question", "solution"):
                                    text = sub_c.get(field, "")
                                    if text and "data:image/" in text:
                                        for b64_img in _extract_base64_imgs_from_text(text):
                                            data_uri = b64_img["data_uri"]
                                            if data_uri not in base64_images:
                                                base64_images[data_uri] = b64_img

        has_local = bool(resolved) or bool(unresolved_srcs)
        has_base64 = bool(base64_images)

        if not has_local and not has_base64:
            self.log("info", "No image references found — nothing to upload")
            return {
                "status": "no_images",
                "uploaded": 0,
                "cached": 0,
                "failed": 0,
                "replaced": 0,
                "message": "No image references found in metadata",
            }

        self.log("info", f"Found {len(resolved)} local images, {len(unresolved_srcs)} unresolved, "
                 f"{len(base64_images)} base64 images (SMILES/SVG)")

        # Phase 3: Upload
        cdn_cache = self._load_cdn_cache()
        already_cached = sum(1 for src in resolved if src in cdn_cache and "cdn_url" in cdn_cache[src])

        to_upload = [(src, lp) for src, lp in sorted(resolved.items())
                     if src not in cdn_cache or "cdn_url" not in cdn_cache[src]]

        self.log("info", f"Already cached: {already_cached} | To upload: {len(to_upload)}")

        uploaded = 0
        failed = 0
        failed_items = []
        start_time = time.time()
        total_to_upload = len(to_upload)

        for i, (img_src, local_path) in enumerate(to_upload, 1):
            basename = os.path.basename(local_path)
            pct = int(i / total_to_upload * 80) if total_to_upload else 80
            self.progress(pct, f"Uploading {i}/{total_to_upload}: {basename}")

            # Get dimensions via PIL
            try:
                with PILImage.open(local_path) as img:
                    w, h = img.size
                dw, dh = compute_display_dimensions(w, h)
            except Exception as e:
                failed += 1
                failed_items.append((img_src, str(e)))
                self.log("warning", f"PIL error for {basename}: {e}")
                continue

            # Upload with retry
            success, url, err = upload_with_retry(local_path, logger=self.log)

            if success:
                cdn_cache[img_src] = {
                    "cdn_url": url,
                    "local_path": local_path,
                    "width_px": w,
                    "height_px": h,
                    "display_width": dw,
                    "display_height": dh,
                }
                uploaded += 1
                if uploaded <= 3:
                    self.log("info", f"  [{i}] {basename} {w}x{h} -> {dw}x{dh} | {url[:60]}...")
            else:
                failed += 1
                failed_items.append((img_src, err))
                self.log("warning", f"  FAILED {basename}: {err}")

            # Save cache every 50 uploads
            if uploaded > 0 and uploaded % 50 == 0:
                self._save_cdn_cache(cdn_cache)

            # Rate limit
            if i < total_to_upload:
                time.sleep(UPLOAD_DELAY_S)

        # Final save after local image uploads
        self._save_cdn_cache(cdn_cache)
        elapsed_local = time.time() - start_time
        self.log("info", f"Local upload done: {uploaded} uploaded, {failed} failed, {elapsed_local:.1f}s")

        # Phase 3b: Upload base64 data URI images (SMILES SVG, etc.)
        b64_uploaded = 0
        b64_failed = 0
        if base64_images:
            b64_temp_dir = self.session_dir / "images_b64_temp"
            b64_temp_dir.mkdir(parents=True, exist_ok=True)

            # Filter: skip already-cached base64 images (uploaded or kept as base64)
            b64_to_upload = {uri: info for uri, info in base64_images.items()
                             if uri not in cdn_cache or (
                                 "cdn_url" not in cdn_cache[uri]
                                 and not cdn_cache[uri].get("kept_base64"))}

            if b64_to_upload:
                self.log("info", f"Uploading {len(b64_to_upload)} base64 images (SMILES/SVG)...")

            for uri, info in b64_to_upload.items():
                temp_path = b64_temp_dir / info["filename"]

                # Save base64 to temp file
                if not _save_base64_to_file(info["b64_data"], temp_path):
                    b64_failed += 1
                    failed_items.append((uri[:60] + "...", "Failed to decode base64"))
                    continue

                # For SVG files: convert to PNG before uploading.
                # CDN serves all files as application/octet-stream which breaks
                # SVG rendering in <img> tags.  PNGs render fine regardless.
                if info["ext"] == "svg":
                    png_path = _svg_to_png(str(temp_path))
                    if png_path:
                        # PNG conversion succeeded — upload PNG instead
                        upload_path = png_path
                        upload_filename = os.path.basename(png_path)
                        try:
                            with PILImage.open(png_path) as img_obj:
                                w, h = img_obj.size
                            dw, dh = compute_display_dimensions(w, h)
                        except Exception:
                            dw, dh = MAX_DISPLAY_WIDTH, 169
                            w, h = dw, dh
                        self.log("info", f"  SVG->PNG: {info['filename']} -> {upload_filename}")
                    else:
                        # PNG conversion failed — keep base64 SVG inline (don't upload)
                        cdn_cache[uri] = {
                            "kept_base64": True,
                            "source": "base64_smiles",
                            "reason": "svg_to_png_failed",
                        }
                        self.log("warning", f"  SVG->PNG failed for {info['filename']}, keeping base64 inline")
                        continue
                else:
                    upload_path = str(temp_path)
                    upload_filename = info["filename"]
                    try:
                        with PILImage.open(upload_path) as img_obj:
                            w, h = img_obj.size
                        dw, dh = compute_display_dimensions(w, h)
                    except Exception:
                        dw, dh = MAX_DISPLAY_WIDTH, 169
                        w, h = dw, dh

                # Upload
                success, url, err = upload_with_retry(upload_path, logger=self.log)
                if success:
                    cdn_cache[uri] = {
                        "cdn_url": url,
                        "source": "base64_smiles",
                        "width_px": w,
                        "height_px": h,
                        "display_width": dw,
                        "display_height": dh,
                    }
                    b64_uploaded += 1
                    uploaded += 1
                else:
                    if info["ext"] == "svg":
                        # Upload failed for converted PNG — keep base64 SVG inline
                        cdn_cache[uri] = {
                            "kept_base64": True,
                            "source": "base64_smiles",
                            "reason": "cdn_upload_failed",
                        }
                        self.log("warning", f"  CDN upload failed for {upload_filename}, keeping base64 SVG inline")
                    else:
                        b64_failed += 1
                        failed += 1
                        failed_items.append((info["filename"], err))

                time.sleep(UPLOAD_DELAY_S)

            if b64_uploaded > 0 or b64_failed > 0:
                self._save_cdn_cache(cdn_cache)
                self.log("info", f"Base64 upload: {b64_uploaded} uploaded, {b64_failed} failed")

            # Cleanup temp dir
            import shutil
            shutil.rmtree(b64_temp_dir, ignore_errors=True)

        elapsed = time.time() - start_time

        # Phase 4: Replace img tags in metadata
        self.progress(85, "Replacing image refs in metadata...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_modified = 0
        total_replacements = 0

        for json_path_str in sorted(file_img_refs.keys()):
            json_path = Path(json_path_str)

            with open(json_path, "r", encoding="utf-8") as f:
                questions = json.load(f)

            file_reps = 0
            for q in questions:
                n = _replace_in_question(q, cdn_cache)
                file_reps += n

            if file_reps == 0:
                continue

            # Timestamped backup
            self.backup(json_path)

            # Save updated file
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)

            files_modified += 1
            total_replacements += file_reps

        self.log("info", f"Replaced {total_replacements} refs in {files_modified} files")

        # Phase 4b: Detect leftover <smiles> tags — these are unrendered chemical
        # structures that survived Stage 5 (either skipped or partially failed).
        # Quarantine these questions so broken <smiles> tags don't reach downstream.
        import re as _re_mod
        _leftover_smiles_re = _re_mod.compile(r'<smiles>', _re_mod.IGNORECASE)
        smiles_quarantined = 0

        for json_path_str in sorted(file_img_refs.keys()):
            json_path = Path(json_path_str)
            with open(json_path, "r", encoding="utf-8") as f:
                questions = json.load(f)

            qids_to_remove = set()
            for q in questions:
                qid = q.get("qid", "?")
                has_leftover = False
                for container_key in ("scq", "mcq", "integerQuestion", "subjective",
                                      "passageQuestion", "matchQuestion"):
                    container = q.get(container_key)
                    if not isinstance(container, dict):
                        continue
                    for field in ("question", "ques", "solution", "sol", "passage"):
                        text = container.get(field, "")
                        if text and _leftover_smiles_re.search(text):
                            has_leftover = True
                            break
                    if has_leftover:
                        break

                if has_leftover:
                    from app.db import quarantine_question
                    quarantine_question(
                        self.session_id, qid,
                        stage="stage6_images",
                        reason="unrendered_smiles",
                        data={"message": "Leftover <smiles> tags found — SMILES not rendered to image"},
                    )
                    qids_to_remove.add(qid)
                    smiles_quarantined += 1
                    self.log("warning", f"Quarantined {qid}: leftover <smiles> tags in content")

            if qids_to_remove:
                questions[:] = [q for q in questions if q.get("qid") not in qids_to_remove]
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(questions, f, indent=2, ensure_ascii=False)

        if smiles_quarantined:
            self.log("warning", f"Quarantined {smiles_quarantined} questions with unrendered <smiles> tags")

        # Build sample entries for UI
        sample_entries = []
        for src, entry in list(cdn_cache.items())[:5]:
            if "cdn_url" in entry:
                sample_entries.append({
                    "src": src,
                    "cdn_url": entry["cdn_url"],
                    "dimensions": f'{entry.get("display_width", "?")}x{entry.get("display_height", "?")}',
                })

        self.progress(100, "Image CDN upload complete")
        self.save_checkpoint("completed", {
            "uploaded": uploaded,
            "cached": already_cached,
            "failed": failed,
            "replaced": total_replacements,
            "files_modified": files_modified,
        })

        return {
            "status": "completed",
            "uploaded": uploaded,
            "cached": already_cached,
            "failed": failed,
            "replaced": total_replacements,
            "files_modified": files_modified,
            "unresolved": len(unresolved_srcs),
            "base64_uploaded": b64_uploaded,
            "base64_failed": b64_failed,
            "elapsed_s": round(elapsed, 1),
            "sample_entries": sample_entries,
            "failed_items": [(s, e) for s, e in failed_items[:10]],
        }

    # ── Skip ──

    def skip(self) -> Dict[str, Any]:
        """Skip image CDN upload."""
        self.save_checkpoint("skipped", {"skipped": True})
        return {"status": "skipped", "message": "Image CDN upload skipped"}

    # ── BasePipelineStage interface ──

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        return self.scan()

    def execute(self, **kwargs) -> Dict[str, Any]:
        return self.upload_images()
