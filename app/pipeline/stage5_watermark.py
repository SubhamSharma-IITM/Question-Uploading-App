"""Stage 5: Watermark Removal — pipeline adapter.

Smart watermark removal that preserves black text/lines on white background.
Removes colored watermarks (blue, red, grey) by analyzing pixel distributions.

Approach: Multi-method grayscale conversion + adaptive thresholding
- Method 1: Saturation-based — desaturate colored pixels (removes colored watermarks)
- Method 2: Luminance threshold — push light greys to white (removes grey watermarks)
- Method 3: Adaptive contrast — auto-contrast + gamma for text preservation

Forked from: CodeBase/soft_fade_watermark.py (OpenCV-based)
"""

import base64
import io
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from app.pipeline.base import BasePipelineStage

try:
    from PIL import Image, ImageOps, ImageFilter, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _remove_red_watermark(img_path: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Pre-check: detect and remove red/dark-red watermark pixels.

    Uses HSV color space to target saturated red pixels (e.g., "bohring bot",
    "TG" labels) while preserving black text (low saturation) and diagram lines.

    Returns (temp_path, dilated_mask) or (None, None).
    The dilated mask is used AFTER grayscale processing to force those pixels
    back to white, preventing the sharpening/LUT from creating dark outlines.
    """
    if HAS_CV2:
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red in HSV wraps around 0/180: H ∈ [0,10] ∪ [170,180]
        # S > 50 (must be saturated — not grey/black text)
        # V > 50 (must have brightness — not dark text)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Also catch dark maroon/brown-red: H ∈ [0,15], S > 40, V > 30
        lower_maroon = np.array([0, 40, 30])
        upper_maroon = np.array([15, 255, 200])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv, lower_maroon, upper_maroon)
        red_mask = mask1 | mask2 | mask3

        red_pixel_count = np.count_nonzero(red_mask)
        total_pixels = img.shape[0] * img.shape[1]

        # Only apply if red pixels are < 15% of image (watermark, not a red diagram)
        if red_pixel_count < 10 or red_pixel_count > total_pixels * 0.15:
            return None, None

        # Dilate the mask by 3px to catch anti-aliased edge pixels around red text.
        # Without this, edge pixels (mixed red+white, low saturation) survive the
        # red detection but get darkened by the grayscale LUT/sharpening → black outlines.
        dilate_kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(red_mask, dilate_kernel, iterations=3)

        # Replace red pixels (and their edges) with white
        img[dilated_mask > 0] = [255, 255, 255]

        # Save to temp file
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, img)
        return tmp.name, dilated_mask

    elif HAS_PIL:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None, None

        arr = np.array(img, dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Simple red detection: R > 120, R > G*1.5, R > B*1.5
        red_mask = (r > 120) & (r > g * 1.5) & (r > b * 1.5)

        red_pixel_count = np.count_nonzero(red_mask)
        total_pixels = arr.shape[0] * arr.shape[1]

        if red_pixel_count < 10 or red_pixel_count > total_pixels * 0.15:
            return None, None

        # Manual dilation for PIL path (no cv2.dilate available)
        dilated = red_mask.copy()
        for _ in range(3):
            padded = np.pad(dilated, 1, mode='constant', constant_values=False)
            dilated = (padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
                       padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:] |
                       padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:])

        # Replace red pixels (and their edges) with white
        result = np.array(img)
        result[dilated] = [255, 255, 255]

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(result).save(tmp.name)
        # Convert bool mask to uint8 for downstream use
        return tmp.name, (dilated.astype(np.uint8) * 255)

    return None, None


def remove_watermark(img_path: str, method: str = "auto") -> Optional[bytes]:
    """Remove watermark from an image. Returns cleaned image as PNG bytes.

    Strategy:
    0. Pre-check: Remove red watermark pixels (if detected)
    1. Convert to grayscale via luminance (preserves black text)
    2. Analyze histogram to find the watermark/background boundary
    3. Push light pixels (watermark + background) to white
    4. Enhance contrast for remaining dark pixels (text/lines)
    5. Apply gentle sharpening to preserve detail
    6. Re-apply red mask to force formerly-red areas to white

    This works because:
    - Text/diagrams are dark (grayscale < 128)
    - Watermarks are light colored or grey (grayscale > 160)
    - Background is white/near-white (grayscale > 230)
    """
    # Step 0: Red watermark pre-check — returns dilated mask for re-application
    temp_path, red_mask = _remove_red_watermark(img_path)
    effective_path = temp_path if temp_path else img_path

    try:
        if HAS_CV2:
            return _remove_watermark_cv2(effective_path, red_mask=red_mask)
        elif HAS_PIL:
            return _remove_watermark_pil(effective_path, red_mask=red_mask)
        else:
            return None
    finally:
        # Clean up temp file
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _remove_watermark_cv2(img_path: str, red_mask: Optional[np.ndarray] = None) -> Optional[bytes]:
    """OpenCV-based watermark removal."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Analyze histogram to find watermark threshold
    # The key insight: text pixels cluster around 0-120, watermarks around 150-210,
    # background around 230-255. We want to push everything > threshold to white.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # Find the valley between text and watermark peaks
    # Use percentile-based approach: text is darkest 30%, rest gets lightened
    p5 = np.percentile(gray, 5)    # darkest pixels (text)
    p50 = np.percentile(gray, 50)  # median
    p90 = np.percentile(gray, 90)  # lightest (background)

    # Adaptive threshold: midpoint between text cluster and background
    threshold = int(min(max(p50 + 20, 160), 210))

    # Step 3: Push watermark pixels to white using a smooth curve
    # Create lookup table for smooth transition
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < threshold - 40:
            # Dark pixels (text): enhance contrast, make darker
            # Map [0, threshold-40] -> [0, ~80] with slight darkening
            ratio = i / max(threshold - 40, 1)
            lut[i] = int(ratio * min(i, 180))
        elif i < threshold:
            # Transition zone: gentle fade to white
            ratio = (i - (threshold - 40)) / 40.0
            dark_val = int(min(i, 180))
            lut[i] = int(dark_val * (1 - ratio) + 255 * ratio)
        else:
            # Light pixels (watermark + background): push to white
            lut[i] = 255

    cleaned = cv2.LUT(gray, lut)

    # Step 4: Auto-contrast stretch
    p_low = np.percentile(cleaned[cleaned < 250], 2) if np.any(cleaned < 250) else 0
    p_high = 255
    if p_low < p_high:
        cleaned = np.clip((cleaned.astype(float) - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)

    # Step 5: Final sharpening to preserve text edges
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    cleaned = cv2.filter2D(cleaned, -1, kernel)
    cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

    # Step 6: Re-apply red mask — force formerly-red areas to pure white.
    # The LUT + sharpening can create dark outlines at red/white boundaries
    # from anti-aliased edge pixels. The dilated mask covers those edges.
    if red_mask is not None and red_mask.shape == cleaned.shape:
        cleaned[red_mask > 0] = 255

    # Encode as PNG bytes
    success, buffer = cv2.imencode('.png', cleaned)
    return buffer.tobytes() if success else None


def _remove_watermark_pil(img_path: str, red_mask: Optional[np.ndarray] = None) -> Optional[bytes]:
    """Pillow-based watermark removal (fallback)."""
    try:
        img = Image.open(img_path)
    except Exception:
        return None

    # Convert to grayscale
    gray = img.convert("L")

    # Get pixel data as numpy array
    arr = np.array(gray, dtype=np.float32)

    # Analyze distribution
    p50 = np.percentile(arr, 50)
    threshold = min(max(p50 + 20, 160), 210)

    # Apply LUT: dark pixels stay, light pixels (watermark) -> white
    result = np.where(arr > threshold, 255,
                      np.where(arr > threshold - 40,
                               255 * (arr - (threshold - 40)) / 40.0 + arr * (1 - (arr - (threshold - 40)) / 40.0),
                               arr))
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Auto-contrast
    low = np.percentile(result[result < 250], 2) if np.any(result < 250) else 0
    if low < 255:
        result = np.clip((result.astype(float) - low) / (255 - low) * 255, 0, 255).astype(np.uint8)

    # Convert back to PIL and sharpen
    out_img = Image.fromarray(result)
    out_img = out_img.filter(ImageFilter.SHARPEN)

    # Re-apply red mask AFTER sharpening — force formerly-red areas to white.
    # Must be last step so the sharpen filter can't re-create dark edges.
    if red_mask is not None:
        result_final = np.array(out_img)
        if red_mask.shape == result_final.shape:
            result_final[red_mask > 0] = 255
            out_img = Image.fromarray(result_final)

    # Encode as PNG bytes
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()


def image_to_base64(img_bytes: bytes, fmt: str = "png") -> str:
    """Convert image bytes to base64 data URI."""
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/{fmt};base64,{b64}"


def file_to_base64(img_path: str) -> Optional[str]:
    """Read image file and return as base64 data URI."""
    try:
        with open(img_path, "rb") as f:
            data = f.read()
        ext = Path(img_path).suffix.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{base64.b64encode(data).decode('ascii')}"
    except Exception:
        return None


class WatermarkStage(BasePipelineStage):

    @property
    def stage_name(self) -> str:
        return "stage5_watermark"

    def _get_image_dir(self) -> Path:
        """Get the images directory for this session.

        Fallback logic:
        1. images_original/ — primary location (copied during load-test-data or OCR)
        2. If images_original/ is empty or missing, check mathpix_output/images/
           (where MathPIX OCR deposits images) and auto-copy them to images_original/
        3. If still empty, check images_cleaned/ (watermark already applied in prior run)
        """
        primary = self.session_dir / "images_original"
        exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

        # Check if primary dir has images
        if primary.exists() and any(f.suffix.lower() in exts for f in primary.iterdir()):
            return primary

        # Fallback 1: mathpix_output/images/ — auto-copy to images_original/
        mathpix_images = self.session_dir / "mathpix_output" / "images"
        if mathpix_images.exists():
            mathpix_imgs = [f for f in mathpix_images.iterdir() if f.suffix.lower() in exts]
            if mathpix_imgs:
                primary.mkdir(parents=True, exist_ok=True)
                copied = 0
                for img in mathpix_imgs:
                    shutil.copy2(img, primary / img.name)
                    copied += 1
                self.log("info", f"Auto-copied {copied} images from mathpix_output/images/ to images_original/")
                return primary

        # Fallback 2: images_cleaned/ — watermark already applied in a prior run
        cleaned = self.session_dir / "images_cleaned"
        if cleaned.exists() and any(f.suffix.lower() in exts for f in cleaned.iterdir()):
            self.log("info", "Using images_cleaned/ (watermark already applied)")
            return cleaned

        # Return primary even if empty — callers handle the empty case
        primary.mkdir(parents=True, exist_ok=True)
        return primary

    def _get_output_dir(self) -> Path:
        """Get the cleaned images output directory."""
        d = self.session_dir / "images_cleaned"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def scan_images(self) -> Dict[str, Any]:
        """Scan for images in the session and return stats."""
        img_dir = self._get_image_dir()
        exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

        images = []
        if img_dir.exists():
            for f in sorted(img_dir.iterdir()):
                if f.suffix.lower() in exts:
                    images.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "path": str(f),
                    })

        return {
            "total_images": len(images),
            "images": images,
            "dir": str(img_dir),
        }

    def preview_samples(self, count: int = 7, cleaned: bool = False) -> Dict[str, Any]:
        """Get random image samples as base64 for preview.

        If cleaned=False, shows original images.
        If cleaned=True, shows before/after pairs.
        """
        img_dir = self._get_image_dir()
        out_dir = self._get_output_dir()
        exts = {".png", ".jpg", ".jpeg"}

        all_images = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in exts] if img_dir.exists() else []

        if not all_images:
            return {"samples": [], "total": 0}

        samples = random.sample(all_images, min(count, len(all_images)))
        result = []

        for img_path in samples:
            entry = {"name": img_path.name, "size": img_path.stat().st_size}

            # Original as base64
            entry["original"] = file_to_base64(str(img_path))

            if cleaned:
                # Check if cleaned version exists
                cleaned_path = out_dir / img_path.name
                if cleaned_path.exists():
                    entry["cleaned"] = file_to_base64(str(cleaned_path))
                else:
                    # Process on the fly for preview
                    cleaned_bytes = remove_watermark(str(img_path))
                    if cleaned_bytes:
                        entry["cleaned"] = image_to_base64(cleaned_bytes)

            result.append(entry)

        return {"samples": result, "total": len(all_images)}

    def process_single(self, filename: str) -> Dict[str, Any]:
        """Process a single image and return before/after base64."""
        img_dir = self._get_image_dir()
        img_path = img_dir / filename

        if not img_path.exists():
            return {"error": f"Image not found: {filename}"}

        original_b64 = file_to_base64(str(img_path))
        cleaned_bytes = remove_watermark(str(img_path))

        if cleaned_bytes is None:
            return {"error": "Failed to process image (missing cv2/PIL?)"}

        return {
            "name": filename,
            "original": original_b64,
            "cleaned": image_to_base64(cleaned_bytes),
        }

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        """Scan images and report what would be processed."""
        scan = self.scan_images()
        self.log("info", f"Found {scan['total_images']} images in {scan['dir']}")
        return scan

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Batch process all images for watermark removal."""
        img_dir = self._get_image_dir()
        out_dir = self._get_output_dir()
        exts = {".png", ".jpg", ".jpeg"}

        all_images = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in exts] if img_dir.exists() else []

        if not all_images:
            self.log("warning", "No images found to process")
            return {"processed": 0, "failed": 0, "skipped": 0}

        self.log("info", f"Processing {len(all_images)} images...")

        processed = 0
        failed = 0

        for i, img_path in enumerate(all_images):
            try:
                cleaned_bytes = remove_watermark(str(img_path))
                if cleaned_bytes:
                    out_path = out_dir / img_path.with_suffix(".png").name
                    with open(out_path, "wb") as f:
                        f.write(cleaned_bytes)
                    processed += 1
                else:
                    # Copy original if processing fails
                    shutil.copy2(img_path, out_dir / img_path.name)
                    failed += 1
                    self.log("warning", f"Failed to clean {img_path.name}, copied original")
            except Exception as e:
                shutil.copy2(img_path, out_dir / img_path.name)
                failed += 1
                self.log("error", f"Error processing {img_path.name}: {e}")

            # Progress
            pct = (i + 1) / len(all_images) * 100
            if (i + 1) % 10 == 0 or (i + 1) == len(all_images):
                self.progress(pct, f"{i + 1}/{len(all_images)} images")
                self.log("info", f"Progress: {i + 1}/{len(all_images)} ({pct:.0f}%)")

        self.save_checkpoint("completed", {
            "total": len(all_images),
            "processed": processed,
            "failed": failed,
        })

        self.log("success", f"Watermark removal complete: {processed} cleaned, {failed} failed")
        return {
            "total": len(all_images),
            "processed": processed,
            "failed": failed,
        }
