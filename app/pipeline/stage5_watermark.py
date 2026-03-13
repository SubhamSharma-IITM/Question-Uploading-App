"""Stage 5: Watermark Removal — pipeline adapter.

Smart watermark removal that preserves black text/lines on white background.
Removes colored watermarks (blue, red, grey) using a multi-step CV pipeline.

Pipeline (forked from Colab watermark_removal.ipynb "Final Code"):
1. Red watermark removal via HSV masking + cv2.inpaint (TELEA algorithm)
2. Grayscale conversion
3. Bilateral denoise (edge-preserving noise removal)
4. Adaptive threshold (local region binarization)
5. Dilation (fill dotted gaps in lines)
6. CLAHE local contrast enhancement
7. Unsharp mask sharpening
8. Gamma correction (push whites whiter, blacks blacker)

Forked from: watermark_removal.ipynb (Colab), watermark-removal/src/remove_watermark.py
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


# ── Watermark removal tuning parameters ──────────────────────────────────
# Tuned via Colab experiments on 120 sample images (wm_001 – wm_120).

# Red watermark HSV ranges (catches solid + faded/outline red, e.g. TG @bohring_bot)
_LOWER_RED1 = np.array([0, 100, 100]) if HAS_NUMPY else None     # solid red
_UPPER_RED1 = np.array([15, 255, 255]) if HAS_NUMPY else None
_LOWER_RED2 = np.array([160, 30, 30]) if HAS_NUMPY else None     # faded/outline red
_UPPER_RED2 = np.array([180, 255, 255]) if HAS_NUMPY else None

_INPAINT_RADIUS = 10       # how far inpaint looks for fill pixels
_RED_DILATE_ITER = 2        # expand red mask to catch anti-aliased edges

# Processing pipeline parameters
_DENOISE_STRENGTH = 15      # bilateral filter d — higher = smoother bg
_ADAPTIVE_BLOCK = 21        # local threshold region (must be odd)
_ADAPTIVE_C = 8             # darkness offset for adaptive threshold
_CLAHE_CLIP = 4.0           # CLAHE contrast clip limit
_CLAHE_GRID = (16, 16)      # CLAHE tile grid size
_SHARPEN_STRENGTH = 8       # unsharp mask centre weight
_GAMMA = 1.3                # gamma correction (>1 = whiter whites, blacker blacks)


def _remove_red_watermark(img_path: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Detect and remove red watermark pixels via inpainting.

    Uses HSV color space to target saturated red pixels (e.g., "TG @bohring_bot",
    "bohring bot" labels) and fills them using cv2.inpaint (TELEA algorithm) which
    reconstructs the area from surrounding pixels — much better than white-fill.

    Returns (temp_path, None) or (None, None).
    Unlike the old approach, inpainting fills naturally so no red mask re-application
    is needed after grayscale processing.
    """
    if HAS_CV2:
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Two HSV ranges for red (wraps around 0/180 in hue):
        # Range 1: H [0,15], S>100, V>100 — solid bright red
        # Range 2: H [160,180], S>30, V>30 — faded/outline/dark red
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, _LOWER_RED1, _UPPER_RED1),
            cv2.inRange(hsv, _LOWER_RED2, _UPPER_RED2),
        )

        red_pixel_count = np.count_nonzero(mask_red)
        total_pixels = img.shape[0] * img.shape[1]

        # Safety: skip if red > 15% of image (likely red content, not watermark)
        # or < 10 pixels (noise)
        if red_pixel_count < 10 or red_pixel_count > total_pixels * 0.15:
            return None, None

        # Dilate mask to catch anti-aliased edge pixels around red text
        kernel = np.ones((3, 3), np.uint8)
        mask_red = cv2.dilate(mask_red, kernel, iterations=_RED_DILATE_ITER)

        # Inpaint: fill masked area using surrounding pixels (TELEA algorithm)
        # This is far superior to white-fill — preserves background texture
        img = cv2.inpaint(img, mask_red, _INPAINT_RADIUS, cv2.INPAINT_TELEA)

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, img)
        return tmp.name, None

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
        for _ in range(_RED_DILATE_ITER):
            padded = np.pad(dilated, 1, mode='constant', constant_values=False)
            dilated = (padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
                       padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:] |
                       padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:])

        # PIL has no inpainting — fall back to white fill
        result = np.array(img)
        result[dilated] = [255, 255, 255]

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(result).save(tmp.name)
        return tmp.name, (dilated.astype(np.uint8) * 255)

    return None, None


def remove_watermark(img_path: str, method: str = "auto") -> Optional[bytes]:
    """Remove watermark from an image. Returns cleaned image as PNG bytes.

    Pipeline (forked from Colab watermark_removal.ipynb):
    0. Pre-check: Remove red watermark via HSV masking + inpainting
    1. Grayscale conversion
    2. Bilateral denoise (edge-preserving background smoothing)
    3. Adaptive threshold (local binarization — handles variable brightness)
    4. Dilation (fill dotted line gaps)
    5. CLAHE local contrast enhancement
    6. Unsharp mask sharpening
    7. Gamma correction (push whites whiter, blacks blacker)
    """
    # Step 0: Red watermark pre-check (inpainting for cv2, white-fill for PIL)
    temp_path, red_mask = _remove_red_watermark(img_path)
    effective_path = temp_path if temp_path else img_path

    try:
        if HAS_CV2:
            return _remove_watermark_cv2(effective_path)
        elif HAS_PIL:
            return _remove_watermark_pil(effective_path, red_mask=red_mask)
        else:
            return None
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _remove_watermark_cv2(img_path: str) -> Optional[bytes]:
    """OpenCV watermark removal — Colab pipeline.

    Steps: grayscale → bilateral denoise → adaptive threshold → dilation →
    CLAHE contrast → sharpen → gamma correction.

    No red mask re-application needed because inpainting (Step 0) fills
    naturally — no dark outlines to fix.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Bilateral denoise — smooths flat background (kills stripes, noise,
    # watermark texture) while preserving hard edges (text, arrows, lines).
    denoised = cv2.bilateralFilter(gray, d=_DENOISE_STRENGTH, sigmaColor=75, sigmaSpace=75)

    # Step 3: Adaptive threshold — looks at LOCAL regions to decide content vs bg.
    # "Is this pixel darker than its neighbours?" YES → black (content), NO → white.
    # Works even when watermark and content have similar global brightness
    # (where the old fixed-threshold LUT approach would fail).
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=_ADAPTIVE_BLOCK,
        C=_ADAPTIVE_C,
    )

    # Step 4: Dilation — spreads dark pixels outward slightly, fills small gaps
    # in dotted/thin lines that adaptive threshold may have broken.
    binary = cv2.dilate(binary, np.ones((1, 1), np.uint8), iterations=1)

    # Step 5: CLAHE — local contrast enhancement. Boosts dark content darker
    # and light background lighter, region by region.
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_GRID)
    enhanced = clahe.apply(binary)

    # Step 6: Sharpen — unsharp mask to crisp up edges after all the smoothing.
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, _SHARPEN_STRENGTH, -1],
        [0, -1, 0],
    ], dtype=np.float32)
    sharp = cv2.filter2D(enhanced, -1, sharpen_kernel)

    # Step 7: Gamma correction — final push. Whites become whiter, blacks blacker.
    lut = np.array([
        np.clip(((i / 255.0) ** _GAMMA) * 255, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)
    final = cv2.LUT(sharp, lut)

    success, buffer = cv2.imencode('.png', final)
    return buffer.tobytes() if success else None


def _remove_watermark_pil(img_path: str, red_mask: Optional[np.ndarray] = None) -> Optional[bytes]:
    """Pillow-based watermark removal (fallback when cv2 is unavailable).

    Approximates the cv2 pipeline using PIL equivalents:
    grayscale → denoise (median filter) → threshold → contrast → sharpen → gamma.
    """
    try:
        img = Image.open(img_path)
    except Exception:
        return None

    # Step 1: Grayscale
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # Step 2: Denoise — median filter (PIL's closest to bilateral)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    arr = np.array(gray, dtype=np.float32)

    # Step 3: Adaptive threshold approximation — since PIL has no adaptive threshold,
    # use a local mean approach: compare each pixel to its local neighbourhood.
    # Simulate with box blur as local mean, then threshold.
    from PIL import ImageFilter as IF
    local_mean = gray.filter(IF.BoxBlur(radius=_ADAPTIVE_BLOCK // 2))
    local_arr = np.array(local_mean, dtype=np.float32)
    # Pixel is content if it's significantly darker than local mean
    binary = np.where(arr < local_arr - _ADAPTIVE_C, 0, 255).astype(np.uint8)

    # Step 4: Contrast enhancement (CLAHE approximation)
    out_img = Image.fromarray(binary)
    out_img = ImageOps.autocontrast(out_img, cutoff=1)

    # Step 5: Sharpen
    out_img = out_img.filter(ImageFilter.SHARPEN)
    out_img = out_img.filter(ImageFilter.SHARPEN)

    # Step 6: Gamma correction
    arr_final = np.array(out_img, dtype=np.float32)
    arr_final = np.clip(((arr_final / 255.0) ** _GAMMA) * 255, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(arr_final)

    # Re-apply red mask for PIL path (uses white-fill, not inpainting)
    if red_mask is not None:
        result_final = np.array(out_img)
        if red_mask.shape == result_final.shape:
            result_final[red_mask > 0] = 255
            out_img = Image.fromarray(result_final)

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


# ── SMILES → SVG rendering ────────────────────────────────────────────────

import re as _re

_SMILES_TAG_RE = _re.compile(r'<smiles>(.*?)</smiles>', _re.IGNORECASE | _re.DOTALL)
_BR_TAG_RE = _re.compile(r'<br\s*/?>')

def _check_rdkit_available() -> bool:
    try:
        from rdkit import Chem
        return True
    except ImportError:
        return False

HAS_RDKIT = _check_rdkit_available()


def _compute_smiles_dimensions(mol) -> Tuple[int, int]:
    """Compute SVG dimensions based on molecule complexity.

    Small molecules (< 10 atoms): 200x150
    Medium molecules (10-25 atoms): 300x200
    Large molecules (> 25 atoms): 400x250
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 10:
        return 200, 150
    elif num_atoms <= 25:
        return 300, 200
    else:
        return 400, 250


def smiles_to_svg(smiles_str: str, width: int = 0, height: int = 0) -> Optional[Tuple[str, int, int]]:
    """Render a SMILES string to SVG XML using RDKit.

    Returns (svg_string, width, height) or None on failure.
    Width/height auto-computed from molecule complexity if not specified.
    """
    if not HAS_RDKIT:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        cleaned = smiles_str.strip()
        cleaned = _BR_TAG_RE.sub('', cleaned)
        if not cleaned:
            return None

        mol = Chem.MolFromSmiles(cleaned)
        if not mol:
            return None

        # Auto-compute dimensions from molecule complexity
        if width == 0 or height == 0:
            width, height = _compute_smiles_dimensions(mol)

        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText(), width, height
    except Exception:
        return None


def smiles_to_svg_data_uri(smiles_str: str) -> Tuple[str, int, int]:
    """Render SMILES to base64 SVG data URI.

    Returns (data_uri, width, height). Empty string + (0,0) on failure.
    """
    result = smiles_to_svg(smiles_str)
    if not result:
        return "", 0, 0
    svg, w, h = result
    b64 = base64.b64encode(svg.encode()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}", w, h


def replace_smiles_in_html(html: str) -> Tuple[str, int, int]:
    """Replace all <smiles>...</smiles> tags in HTML with rendered SVG <img> tags.

    Returns (updated_html, success_count, fail_count).
    Dimensions are auto-computed based on molecule complexity.
    """
    success = 0
    fail = 0

    def _replacer(match):
        nonlocal success, fail
        raw_smiles = match.group(1)
        svg_uri, w, h = smiles_to_svg_data_uri(raw_smiles)
        if svg_uri:
            success += 1
            return (
                f'<img src="{svg_uri}" alt="SMILES: {raw_smiles.strip()}" '
                f'style="vertical-align: middle; width:{w}px; height:{h}px;" />'
            )
        else:
            fail += 1
            return match.group(0)  # Keep original on failure

    result = _SMILES_TAG_RE.sub(_replacer, html)
    return result, success, fail


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

    # ── SMILES detection + rendering ──────────────────────────────────────

    def _get_metadata_files(self) -> List[Path]:
        """Get all metadata_v2 JSON files in the session.

        Same lookup as Stage 6: checks metadata_v2/ first, then metadata_v2_output/.
        Handles both flat files and subdirectory layouts.
        """
        for dir_name in ("metadata_v2", "metadata_v2_output"):
            meta_dir = self.session_dir / dir_name
            if meta_dir.exists():
                break
        else:
            return []

        files = []
        for f in sorted(meta_dir.glob("*_metadata_v2.json")):
            if "backup" not in f.name:
                files.append(f)
        for sub_dir in sorted(meta_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            for f in sorted(sub_dir.glob("*_metadata_v2.json")):
                if "backup" not in f.name:
                    files.append(f)
        return files

    def scan_smiles(self) -> Dict[str, Any]:
        """Scan session metadata for <smiles> tags.

        Returns count of entries and fields containing SMILES, plus rdkit availability.
        """
        meta_files = self._get_metadata_files()
        if not meta_files:
            return {
                "has_smiles": False,
                "rdkit_available": HAS_RDKIT,
                "total_entries": 0,
                "entries_with_smiles": 0,
                "total_smiles_tags": 0,
                "message": "No metadata files found",
            }

        entries_with_smiles = 0
        total_smiles = 0
        total_entries = 0
        sample_qids = []

        QUESTION_TYPES = ["scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"]

        for mf in meta_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            questions = data if isinstance(data, list) else data.get("questions", [])
            for q in questions:
                total_entries += 1
                qid = q.get("qid", "?")
                found_smiles = False

                for qt in QUESTION_TYPES:
                    obj = q.get(qt)
                    if not obj or not isinstance(obj, dict):
                        continue
                    for field_key in ["question", "ques", "solution", "sol"]:
                        val = str(obj.get(field_key, "") or "")
                        tags = _SMILES_TAG_RE.findall(val)
                        if tags:
                            total_smiles += len(tags)
                            found_smiles = True

                if found_smiles:
                    entries_with_smiles += 1
                    if len(sample_qids) < 5:
                        sample_qids.append(qid)

        has = entries_with_smiles > 0
        self.log("info",
                 f"SMILES scan: {entries_with_smiles} entries with {total_smiles} tags"
                 if has else "SMILES scan: no <smiles> tags found")

        return {
            "has_smiles": has,
            "rdkit_available": HAS_RDKIT,
            "total_entries": total_entries,
            "entries_with_smiles": entries_with_smiles,
            "total_smiles_tags": total_smiles,
            "sample_qids": sample_qids,
            "message": (
                f"Found {total_smiles} SMILES tags in {entries_with_smiles} entries"
                if has else "No SMILES tags detected — step inactive"
            ),
        }

    def render_smiles(self) -> Dict[str, Any]:
        """Replace all <smiles> tags in session metadata with rendered SVG <img> tags.

        Modifies metadata_v2 files in-place. Only runs if RDKit is available and
        SMILES tags are detected.

        Questions where ALL SMILES tags fail to render are quarantined and removed
        from metadata so they don't pass downstream with broken <smiles> tags.
        """
        if not HAS_RDKIT:
            return {"error": "RDKit not installed — pip install rdkit", "processed": 0}

        meta_files = self._get_metadata_files()
        if not meta_files:
            return {"error": "No metadata files found", "processed": 0}

        from app.db import quarantine_question

        QUESTION_TYPES = ["scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"]
        total_success = 0
        total_fail = 0
        entries_modified = 0
        files_modified = 0
        quarantined_qids = set()

        for mf in meta_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            questions = data if isinstance(data, list) else data.get("questions", [])
            file_modified = False
            questions_to_remove = set()

            for q in questions:
                entry_modified = False
                entry_success = 0
                entry_fail = 0
                qid = q.get("qid", "?")

                for qt in QUESTION_TYPES:
                    obj = q.get(qt)
                    if not obj or not isinstance(obj, dict):
                        continue
                    for field_key in ["question", "ques", "solution", "sol"]:
                        val = obj.get(field_key)
                        if not val or not isinstance(val, str):
                            continue
                        if "<smiles>" not in val.lower():
                            continue

                        new_val, sc, fc = replace_smiles_in_html(val)
                        obj[field_key] = new_val
                        entry_success += sc
                        entry_fail += fc
                        total_success += sc
                        total_fail += fc
                        if sc > 0 or fc > 0:
                            entry_modified = True

                # If ALL SMILES tags failed (zero success, some failures),
                # quarantine this question — it has unrenderable chemical structures
                if entry_fail > 0 and entry_success == 0:
                    quarantine_question(
                        self.session_id, qid,
                        stage="stage5_watermark",
                        reason="smiles_render_failure",
                        data={
                            "failed_tags": entry_fail,
                            "message": f"All {entry_fail} SMILES tag(s) failed to render via RDKit",
                        },
                    )
                    quarantined_qids.add(qid)
                    questions_to_remove.add(qid)
                    self.log("warning", f"Quarantined {qid}: all {entry_fail} SMILES failed to render")

                if entry_modified:
                    entries_modified += 1
                    file_modified = True

            # Remove fully-failed SMILES questions from metadata
            if questions_to_remove:
                if isinstance(data, list):
                    data[:] = [q for q in data if q.get("qid") not in questions_to_remove]
                elif "questions" in data:
                    data["questions"] = [q for q in data["questions"]
                                         if q.get("qid") not in questions_to_remove]
                file_modified = True

            if file_modified:
                files_modified += 1
                self.backup(mf)
                with open(mf, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        self.log("success" if total_success > 0 else "info",
                 f"SMILES rendering: {total_success} rendered, {total_fail} failed, "
                 f"{entries_modified} entries in {files_modified} files"
                 + (f", {len(quarantined_qids)} quarantined" if quarantined_qids else ""))

        # Save as "completed" — not "smiles_rendered" — so the advance gate works.
        # Merge with existing watermark checkpoint data if present.
        from app.db import get_checkpoint
        existing = get_checkpoint(self.session_id, self.stage_name)
        existing_data = existing.get("data", {}) if existing else {}
        merged = {
            **existing_data,
            "smiles_success": total_success,
            "smiles_fail": total_fail,
            "smiles_quarantined": len(quarantined_qids),
            "smiles_entries_modified": entries_modified,
        }
        self.save_checkpoint("completed", merged)

        return {
            "total_success": total_success,
            "total_fail": total_fail,
            "entries_modified": entries_modified,
            "files_modified": files_modified,
            "quarantined": list(quarantined_qids),
        }
