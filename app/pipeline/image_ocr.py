"""Image OCR utility for duplicate checking.

Downloads images (local or remote), runs OCR via pix2text (tier 1),
falls back to Gemini Flash (tier 2) for uncertain results.
Caches all OCR results persistently so images are never re-processed.

Usage:
    ocr = ImageOCRCache(cache_dir)
    ocr.load_cache()
    text = ocr.get_ocr_text(image_url_or_path, session_dir=...)
    ocr.save_cache()
"""

import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ────────────────────────────────────────────────────────────
ASYNC_MAX_CONCURRENT = 256
ASYNC_LIMIT_PER_HOST = 64
CONNECT_TIMEOUT = 6
READ_TIMEOUT = 20

IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
DATA_URI_RE = re.compile(r'^data:image/(png|jpeg|jpg|gif|bmp|webp|svg\+xml);base64,')

GEMINI_OCR_SYSTEM = """You are an image content extractor for educational question papers.
Extract ALL readable text, formulas, and equations from this image.

Rules:
- Return text content and math formulas in LaTeX format
- If the image is a DIAGRAM (circuit, graph, geometric figure) with NO readable text/formulas, return {"has_text": false, "content": ""}
- Text written ON diagrams as labels is NOT considered readable text
- Only extract formulas, equations, numerical values, and regular text
- Be concise — just the extracted content, no descriptions

Return STRICT JSON:
{"has_text": true/false, "content": "extracted text and $latex$ here"}"""

GEMINI_OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "has_text": {"type": "boolean"},
        "content": {"type": "string"}
    },
    "required": ["has_text", "content"]
}


def extract_image_urls(html: str) -> List[str]:
    """Extract all image src URLs from HTML."""
    if not html:
        return []
    return IMG_SRC_RE.findall(html)


def _guess_mime(url: str, content: bytes = b"") -> Optional[str]:
    """Guess MIME type from URL extension or binary signature."""
    ext_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
               '.gif': 'image/gif', '.bmp': 'image/bmp', '.webp': 'image/webp'}
    for ext, mime in ext_map.items():
        if url.lower().endswith(ext):
            return mime

    sig_map = [(b'\x89PNG', 'image/png'), (b'\xff\xd8\xff', 'image/jpeg'),
               (b'GIF8', 'image/gif'), (b'BM', 'image/bmp'), (b'RIFF', 'image/webp')]
    for sig, mime in sig_map:
        if content[:4].startswith(sig):
            return mime
    return None


class ImageOCRCache:
    """Persistent OCR cache for images."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "image_ocr_cache.json"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._dirty = False

    def load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def save_cache(self):
        """Save cache to disk if modified."""
        if not self._dirty:
            return
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)
        self._dirty = False

    def get_cached(self, image_key: str) -> Optional[str]:
        """Get cached OCR text for an image. Returns None if not cached."""
        entry = self._cache.get(image_key)
        if entry is None:
            return None
        return entry.get("ocr_text", "")

    def set_cached(self, image_key: str, ocr_text: str, source: str = "pix2text"):
        """Cache OCR result for an image."""
        self._cache[image_key] = {
            "ocr_text": ocr_text,
            "source": source,
            "ts": int(time.time()),
        }
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self._cache)

    # ── Image resolution ─────────────────────────────────────────────

    def resolve_image_path(self, src: str, session_dir: Optional[Path] = None) -> Optional[str]:
        """Resolve an image src to a local file path.

        Handles:
          - Relative paths (./images/foo.jpg) → session_dir/images/foo.jpg
          - Absolute paths
          - HTTP URLs → returns None (need to download)
        """
        if src.startswith("data:"):
            return None  # data URIs handled separately
        if src.startswith("http://") or src.startswith("https://"):
            return None  # remote — needs download
        # Relative path
        if session_dir:
            # Try both ./images/... and images/...
            clean = src.lstrip("./")
            local = session_dir / clean
            if local.exists():
                return str(local)
            # Try images_original and images_cleaned
            for subdir in ("images", "images_original", "images_cleaned"):
                alt = session_dir / subdir / Path(clean).name
                if alt.exists():
                    return str(alt)
        return None

    # ── Async image downloader (forked from question_checker) ────────

    async def _async_download_one(self, session, url: str, sem) -> Tuple[str, Optional[bytes]]:
        """Download single image asynchronously."""
        async with sem:
            try:
                if not (url.startswith("http://") or url.startswith("https://")):
                    return url, None
                import aiohttp
                timeout = aiohttp.ClientTimeout(connect=CONNECT_TIMEOUT, total=CONNECT_TIMEOUT + READ_TIMEOUT)
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status != 200:
                        return url, None
                    content = await resp.read()
                    return url, content
            except Exception:
                return url, None

    async def _async_download_batch(self, urls: List[str]) -> Dict[str, Optional[bytes]]:
        """Download images concurrently (256 max)."""
        import aiohttp
        connector = aiohttp.TCPConnector(
            limit=ASYNC_MAX_CONCURRENT,
            limit_per_host=ASYNC_LIMIT_PER_HOST,
            ttl_dns_cache=600,
            keepalive_timeout=120,
            enable_cleanup_closed=True,
            force_close=False,
        )
        sem = asyncio.Semaphore(ASYNC_MAX_CONCURRENT)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": "acadza-dup-check/1.0"}
        ) as session:
            tasks = [self._async_download_one(session, url, sem) for url in urls]
            results = await asyncio.gather(*tasks)
        return dict(results)

    def download_images_batch(self, urls: List[str]) -> Dict[str, Optional[bytes]]:
        """Batch download images. Returns {url: raw_bytes_or_None}."""
        if not urls:
            return {}
        try:
            import aiohttp  # noqa: F401
            return asyncio.run(self._async_download_batch(urls))
        except ImportError:
            # Sequential fallback
            import requests
            sess = requests.Session()
            sess.headers["User-Agent"] = "acadza-dup-check/1.0"
            result = {}
            for url in urls:
                try:
                    r = sess.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                    r.raise_for_status()
                    result[url] = r.content
                except Exception:
                    result[url] = None
            return result

    def save_downloaded_images(self, downloads: Dict[str, Optional[bytes]], save_dir: Path):
        """Save downloaded image bytes to local files. Returns {url: local_path}."""
        save_dir.mkdir(parents=True, exist_ok=True)
        url_to_path = {}
        for url, content in downloads.items():
            if content is None:
                continue
            # Generate filename from URL
            fname = url.split("/")[-1][:60]
            if not fname or len(fname) < 3:
                fname = base64.urlsafe_b64encode(url.encode()).decode()[:40]
            # Ensure extension
            mime = _guess_mime(url, content)
            if mime and "." not in fname[-5:]:
                ext = mime.split("/")[-1].replace("jpeg", "jpg")
                fname += f".{ext}"
            local_path = save_dir / fname
            local_path.write_bytes(content)
            url_to_path[url] = str(local_path)
        return url_to_path

    # ── OCR: Tier 1 — pix2text ──────────────────────────────────────

    def ocr_pix2text(self, image_path: str) -> str:
        """Run pix2text OCR on a local image file. Returns extracted text."""
        try:
            from pix2text import Pix2Text
            p2t = Pix2Text.from_config()
            result = p2t.recognize(image_path)
            # pix2text returns a string with text + LaTeX
            text = str(result).strip() if result else ""
            return text
        except Exception as e:
            return f"[OCR_ERROR:{e}]"

    # ── OCR: Tier 2 — Gemini Flash fallback ─────────────────────────

    def ocr_gemini(self, image_path: str) -> Tuple[bool, str]:
        """Use Gemini Flash to extract text from image.

        Returns (has_text, content_text).
        """
        try:
            from app.config import settings
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=settings.gemini_api_key)

            # Read image as base64
            img_bytes = Path(image_path).read_bytes()
            mime = _guess_mime(image_path, img_bytes) or "image/jpeg"

            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=img_bytes, mime_type=mime),
                            types.Part.from_text(
                                "Extract all text and formulas from this image. "
                                "If it's only a diagram with no text/formulas, return "
                                '{"has_text": false, "content": ""}. '
                                "Diagram labels are NOT text. Only formulas, equations, "
                                "numerical values, and regular text. "
                                'Return JSON: {"has_text": bool, "content": "text here"}'
                            ),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                ),
            )

            text = response.text.strip()
            data = json.loads(text)
            has_text = data.get("has_text", False)
            content = data.get("content", "").strip()
            return has_text, content

        except Exception as e:
            return False, f"[GEMINI_ERROR:{e}]"

    # ── Two-tier OCR pipeline ────────────────────────────────────────

    def ocr_image(self, image_path: str, image_key: str) -> str:
        """Run two-tier OCR: pix2text → Gemini fallback.

        Returns extracted text (may be empty for pure diagrams).
        Caches the result.
        """
        # Check cache first
        cached = self.get_cached(image_key)
        if cached is not None:
            return cached

        # Tier 1: pix2text
        text = self.ocr_pix2text(image_path)

        if text and not text.startswith("[OCR_ERROR") and len(text.strip()) > 3:
            # Got meaningful text from pix2text
            self.set_cached(image_key, text, source="pix2text")
            return text

        # Tier 2: Gemini fallback (uncertain — maybe poor image quality)
        has_text, content = self.ocr_gemini(image_path)

        if has_text and content and not content.startswith("[GEMINI_ERROR"):
            self.set_cached(image_key, content, source="gemini")
            return content

        # No text found (diagram or truly empty)
        self.set_cached(image_key, "", source="no_text")
        return ""

    # ── Batch OCR for a question set ─────────────────────────────────

    def ocr_question_images(self, html: str, image_key_prefix: str,
                            session_dir: Optional[Path] = None) -> str:
        """Extract and OCR all images from question HTML.

        Returns combined OCR text from all images (space-separated).
        """
        urls = extract_image_urls(html)
        if not urls:
            return ""

        ocr_parts = []
        for url in urls:
            image_key = url  # Use URL as cache key

            # Check cache
            cached = self.get_cached(image_key)
            if cached is not None:
                if cached:
                    ocr_parts.append(cached)
                continue

            # Resolve to local path
            local_path = self.resolve_image_path(url, session_dir)
            if local_path:
                text = self.ocr_image(local_path, image_key)
                if text:
                    ocr_parts.append(text)
            else:
                # Remote URL — needs download first
                # For batch efficiency, this should be pre-downloaded
                # Mark as pending
                self.set_cached(image_key, "", source="pending_download")

        return " ".join(ocr_parts)
