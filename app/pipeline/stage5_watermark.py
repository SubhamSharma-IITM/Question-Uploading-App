"""Stage 5: Watermark Removal — pipeline adapter.

V3 Architecture (2026-03-30):
  Images -> LAB Chroma Detection (threshold=12, min_pct=0.3%)
    - No colored watermark -> SKIP (keep original unchanged)
    - Has colored watermark -> Fix1 pipeline (red HSV inpaint + binarize)

Fix1 pipeline (for watermarked images only):
1. Red watermark removal via HSV masking + cv2.inpaint (TELEA algorithm)
2. Grayscale conversion
3. Bilateral denoise (edge-preserving noise removal)
4. Adaptive threshold (local region binarization)
5. Dilation (fill dotted gaps in lines)
6. CLAHE local contrast enhancement
7. Unsharp mask sharpening
8. Gamma correction (push whites whiter, blacks blacker)

Gray/black watermarks are NOT caught by chroma detection (known limitation).
They pass through as originals for human review in the preview pane.

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

# Post-sharpen watermark detection parameters (connected component blob filter)
# Catches watermark remnants that survive the initial red HSV removal pass.
_WM_DARK_THRESH = 220       # pixels below this intensity are WM candidates
_WM_DILATE = 4              # expand detected WM mask edges
_WM_THICKNESS_FRAC = 0.03   # min thickness as fraction of image min dimension
_WM_AREA_FRAC = 0.002       # min area as fraction of total image area
_WM_FILL_DENSITY = 0.25     # min fill ratio (area / bounding box) for WM blobs
_WM_MAX_ASPECT = 5.0        # reject blobs with aspect ratio above this (thin lines = content)

# Red mask blob filter parameters (prevents removing legitimate red content)
_RED_MIN_THICKNESS = 20     # min blob min-side in pixels
_RED_MIN_AREA = 800         # min blob area in pixels
_RED_FILL_DENSITY = 0.40    # min fill ratio (0-1) for red blobs

# Chroma detection parameters (LAB color space)
_CHROMA_THRESHOLD = 10      # min chroma distance from gray axis for "colored" pixel
_CHROMA_MIN_PCT = 0.3       # min % of pixels that must be colored to flag watermark


def detect_colored_watermark(
    img_bgr: "np.ndarray",
    chroma_threshold: float = _CHROMA_THRESHOLD,
    min_colored_pct: float = _CHROMA_MIN_PCT,
) -> Tuple[bool, Optional["np.ndarray"], Dict[str, Any]]:
    """Detect colored watermark via LAB chroma distance.

    Converts to LAB color space and measures chroma = sqrt(a^2 + b^2) for
    each pixel. Colored watermarks (red, blue, mixed) have high chroma;
    grayscale diagram content (lines, text, hatching) has chroma ~ 0.

    Returns:
        has_watermark: True if colored watermark detected
        mask: binary mask of colored pixels (or None if no watermark)
        stats: dict with chroma_mean, chroma_max, colored_pct
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)

    colored_mask = (chroma > chroma_threshold).astype(np.uint8) * 255

    # Morphological cleanup — close small gaps, remove noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel)
    colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_OPEN, kernel)

    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    colored_count = int(np.count_nonzero(colored_mask))
    colored_pct = colored_count / total_pixels * 100

    stats = {
        "chroma_mean": round(float(np.mean(chroma)), 2),
        "chroma_max": round(float(np.max(chroma)), 2),
        "colored_pct": round(colored_pct, 3),
    }

    has_watermark = colored_pct >= min_colored_pct
    return has_watermark, colored_mask if has_watermark else None, stats


def _filter_thick_blobs(
    mask: np.ndarray,
    min_thickness: int,
    min_area: int,
    fill_density: float,
) -> np.ndarray:
    """Filter a binary mask to keep only thick, dense blobs.

    Removes small spots and thin lines that might be actual content
    (arrows, circles, annotations) rather than watermark text.

    Uses connected component analysis to evaluate each blob's geometry.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        min_side = min(w, h)
        density = area / (w * h + 1e-5)
        if min_side >= min_thickness and density >= fill_density and area >= min_area:
            filtered[labels == i] = 255
    return filtered


def _remove_post_sharpen_watermark(sharp: np.ndarray) -> np.ndarray:
    """Second-pass watermark removal after sharpening.

    Detects remaining watermark blobs that survived the initial red HSV
    removal pass. Uses connected component analysis on dark pixels to find
    thick, dense blobs (watermark text patterns) and sets them to white.

    This catches grey/faded watermarks that aren't red-colored and thus
    aren't caught by the HSV-based pre-processing step.
    """
    h, w = sharp.shape
    img_area = h * w
    min_thickness = max(5, int(min(h, w) * _WM_THICKNESS_FRAC))
    min_area = max(100, int(img_area * _WM_AREA_FRAC))

    # Find dark pixels (potential watermark content)
    dark_mask = cv2.inRange(sharp, 0, _WM_DARK_THRESH)

    # Analyze each connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_mask, connectivity=8
    )
    thick_mask = np.zeros_like(dark_mask)
    for i in range(1, num_labels):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        min_side = min(bw, bh)
        fill_density = area / (bw * bh + 1e-5)
        aspect_ratio = max(bw, bh) / (min_side + 1e-5)
        if (
            min_side >= min_thickness
            and area >= min_area
            and fill_density >= _WM_FILL_DENSITY
            and aspect_ratio <= _WM_MAX_ASPECT
        ):
            thick_mask[labels == i] = 255

    # Dilate to cover anti-aliased edges
    wm_mask = cv2.dilate(
        thick_mask, np.ones((3, 3), np.uint8), iterations=_WM_DILATE
    )

    # Set watermark pixels to white
    result = sharp.copy()
    result[wm_mask == 255] = 255
    return result


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

        # Filter red mask: keep only thick, dense blobs (watermark text).
        # Rejects thin red lines and small spots that might be actual
        # diagram content (arrows, circles, annotations).
        mask_red = _filter_thick_blobs(
            mask_red, _RED_MIN_THICKNESS, _RED_MIN_AREA, _RED_FILL_DENSITY
        )

        # Re-check after filtering — might have removed everything
        if np.count_nonzero(mask_red) < 10:
            return None, None

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

    # Step 6b: Post-sharpen watermark detection — DISABLED.
    # This step used connected component analysis to find "thick, dense blobs"
    # and remove them as watermarks. However, after binarization (step 3),
    # grey watermarks are already eliminated. Step 6b was instead destroying
    # real content: hatched diagram regions, large chemistry text (CH3CH2MgBr),
    # DNA sequences, and dense labeling — all misclassified as watermarks.
    # Red watermarks are handled by step 0 (HSV inpainting); no post-sharpen
    # detection is needed. Disabled 2026-03-30 after analysis of 539 images
    # showed 15.2% false positive rate (82/539 images had content destroyed).
    # See experiments/watermark_lab/ for full diagnostic data.

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
        """Batch process all images with chroma detection gate.

        For each image:
        1. Run LAB chroma detection (7ms/image)
        2. If no colored watermark -> copy original to images_cleaned/ (SKIP)
        3. If colored watermark -> run Fix1 pipeline -> save to images_cleaned/

        Saves per-image decision log to watermark_decisions.json for the
        preview pane to display chroma labels and side-by-side comparisons.
        """
        img_dir = self._get_image_dir()
        out_dir = self._get_output_dir()
        exts = {".png", ".jpg", ".jpeg"}

        all_images = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in exts] if img_dir.exists() else []

        if not all_images:
            self.log("warning", "No images found to process")
            return {"processed": 0, "failed": 0, "skipped": 0}

        self.log("info", f"Processing {len(all_images)} images (chroma detection + Fix1 pipeline)...")

        processed = 0
        skipped = 0
        failed = 0
        decisions = []  # Per-image log for preview pane

        for i, img_path in enumerate(all_images):
            decision = {"filename": img_path.name, "action": "unknown"}
            try:
                if HAS_CV2:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        raise ValueError("cv2.imread returned None")

                    # Chroma detection gate
                    has_wm, _mask, chroma_stats = detect_colored_watermark(img_bgr)
                    decision.update(chroma_stats)

                    if has_wm:
                        # Colored watermark detected -> run Fix1 binarization pipeline
                        cleaned_bytes = remove_watermark(str(img_path))
                        if cleaned_bytes:
                            out_path = out_dir / img_path.with_suffix(".png").name
                            with open(out_path, "wb") as f:
                                f.write(cleaned_bytes)
                            processed += 1
                            decision["action"] = "cleaned"
                        else:
                            shutil.copy2(img_path, out_dir / img_path.name)
                            failed += 1
                            decision["action"] = "failed"
                            self.log("warning", f"Fix1 failed on {img_path.name}, copied original")
                    else:
                        # No colored watermark -> pass through original unchanged
                        shutil.copy2(img_path, out_dir / img_path.name)
                        skipped += 1
                        decision["action"] = "skipped"
                else:
                    # No cv2 available — copy original
                    shutil.copy2(img_path, out_dir / img_path.name)
                    skipped += 1
                    decision["action"] = "skipped"
                    decision["reason"] = "cv2_unavailable"

            except Exception as e:
                shutil.copy2(img_path, out_dir / img_path.name)
                failed += 1
                decision["action"] = "failed"
                decision["error"] = str(e)
                self.log("error", f"Error processing {img_path.name}: {e}")

            decisions.append(decision)

            # Progress
            pct = (i + 1) / len(all_images) * 100
            if (i + 1) % 10 == 0 or (i + 1) == len(all_images):
                self.progress(pct, f"{i + 1}/{len(all_images)} images")
                self.log("info",
                         f"Progress: {i + 1}/{len(all_images)} ({pct:.0f}%) "
                         f"-- cleaned={processed} skipped={skipped} failed={failed}")

        # Save per-image decision log for preview pane
        decisions_path = self.session_dir / "watermark_decisions.json"
        try:
            with open(decisions_path, "w", encoding="utf-8") as f:
                json.dump(decisions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log("warning", f"Could not save decisions log: {e}")

        result_data = {
            "total": len(all_images),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "watermark_step": "completed",
        }
        self.save_checkpoint("in_progress", result_data)

        self.log("success",
                 f"Watermark removal complete: {processed} cleaned, "
                 f"{skipped} skipped (no watermark), {failed} failed")
        return result_data

    # ── Preview pane: paginated image list with chroma decisions ─────────

    def get_preview_images(self, offset: int = 0, limit: int = 20) -> Dict[str, Any]:
        """Return paginated image list with original + cleaned base64 and chroma labels.

        Reads watermark_decisions.json (written by execute()) to determine:
        - Which images were cleaned (show original left + cleaned right)
        - Which were skipped (show original only, label "No watermark detected")

        Returns base64 data URIs for both original and cleaned (if applicable).
        """
        img_dir = self._get_image_dir()
        out_dir = self._get_output_dir()

        # Load decisions log
        decisions_path = self.session_dir / "watermark_decisions.json"
        decisions_map = {}
        if decisions_path.exists():
            try:
                with open(decisions_path, "r", encoding="utf-8") as f:
                    decisions_list = json.load(f)
                decisions_map = {d["filename"]: d for d in decisions_list}
            except Exception:
                pass

        # Get all images from original dir
        exts = {".png", ".jpg", ".jpeg"}
        all_images = sorted(
            [f for f in img_dir.iterdir() if f.suffix.lower() in exts],
            key=lambda p: p.name,
        ) if img_dir.exists() else []

        total = len(all_images)
        page = all_images[offset:offset + limit]

        items = []
        for img_path in page:
            fname = img_path.name
            decision = decisions_map.get(fname, {})
            action = decision.get("action", "unknown")

            entry = {
                "filename": fname,
                "action": action,
                "colored_pct": decision.get("colored_pct", 0),
                "chroma_label": (
                    "Watermark removed" if action == "cleaned"
                    else "No watermark detected" if action == "skipped"
                    else "Processing failed" if action == "failed"
                    else "Not processed"
                ),
                "original": file_to_base64(str(img_path)),
            }

            # Add cleaned image if watermark was removed
            if action == "cleaned":
                # Cleaned file may be .png even if original was .jpg
                cleaned_path = out_dir / img_path.with_suffix(".png").name
                if not cleaned_path.exists():
                    cleaned_path = out_dir / fname
                if cleaned_path.exists():
                    entry["cleaned"] = file_to_base64(str(cleaned_path))

            items.append(entry)

        return {
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total,
        }

    # ── Image-level quarantine ────────────────────────────────────────────

    def quarantine_images(self, qids_to_quarantine: List[str]) -> Dict[str, Any]:
        """Quarantine questions by QID — removes them from metadata_v2 files.

        Called after user reviews preview pane and clicks "Quarantine Unselected".
        Quarantined questions are:
        1. Recorded in the quarantine DB table
        2. Removed from all metadata_v2 files (so they don't flow downstream)

        Args:
            qids_to_quarantine: List of QIDs to quarantine

        Returns:
            Dict with counts of quarantined and remaining questions.
        """
        if not qids_to_quarantine:
            return {"quarantined": 0, "remaining": 0, "message": "Nothing to quarantine"}

        from app.db import quarantine_question

        qid_set = set(qids_to_quarantine)
        quarantined_count = 0
        remaining_count = 0

        # Record each in quarantine DB
        for qid in qids_to_quarantine:
            quarantine_question(
                self.session_id, qid,
                stage="stage5_watermark",
                reason="user_image_review",
                data={"message": "Quarantined via Stage 5 image review"},
            )
            quarantined_count += 1

        # Remove from metadata_v2 files
        meta_files = self._get_metadata_files()
        files_modified = 0
        for mf in meta_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            questions = data if isinstance(data, list) else data.get("questions", [])
            original_len = len(questions)
            filtered = [q for q in questions if q.get("qid") not in qid_set]

            if len(filtered) < original_len:
                if isinstance(data, list):
                    data[:] = filtered
                elif "questions" in data:
                    data["questions"] = filtered

                self.backup(mf)
                with open(mf, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                files_modified += 1
                remaining_count += len(filtered)
            else:
                remaining_count += len(questions)

        self.log("info",
                 f"Quarantined {quarantined_count} questions from image review, "
                 f"{remaining_count} remaining in {files_modified} files modified")

        # Update checkpoint with quarantine data
        cp_data = self._load_checkpoint_data()
        cp_data["quarantine_step"] = "completed"
        cp_data["quarantined_qids"] = list(qid_set)
        cp_data["quarantined_count"] = quarantined_count
        cp_data["remaining_count"] = remaining_count
        self.save_checkpoint("in_progress", cp_data)

        return {
            "quarantined": quarantined_count,
            "remaining": remaining_count,
            "files_modified": files_modified,
        }

    def get_image_question_map(self) -> Dict[str, str]:
        """Map image filenames to question QIDs by scanning metadata_v2 HTML.

        Scans all <img src="./images/filename"> tags in question/solution HTML
        and maps each filename to the QID of the entry containing it.

        Returns: {filename: qid}
        """
        import re
        img_src_re = re.compile(r'<img[^>]+src=["\']\.?/?images/([^"\']+)["\']', re.IGNORECASE)
        mapping = {}

        meta_files = self._get_metadata_files()
        QUESTION_TYPES = ["scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion", "subjective"]

        for mf in meta_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            questions = data if isinstance(data, list) else data.get("questions", [])
            for q in questions:
                qid = q.get("qid", "")
                for qt in QUESTION_TYPES:
                    obj = q.get(qt)
                    if not obj or not isinstance(obj, dict):
                        continue
                    for field in ("question", "ques", "solution", "sol"):
                        val = str(obj.get(field, "") or "")
                        for match in img_src_re.finditer(val):
                            mapping[match.group(1)] = qid

        return mapping

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
