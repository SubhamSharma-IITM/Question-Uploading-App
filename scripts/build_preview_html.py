"""Build rich HTML preview for OCR -> MathML conversion results.

Shows before (original image) and after (rendered HTML + diagram if present)
for each field separately. Question and solution properly mapped.

Usage:
    python scripts/build_preview_html.py
"""

import base64
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

APP_ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = APP_ROOT / "data" / "ocr_cache"
IMG_CACHE_DIR = APP_ROOT / "data" / "image_cache" / "images"
CONVERTED_FILE = OCR_DIR / "ocr_converted.jsonl"
QUESTION_OCR_MAP = OCR_DIR / "question_ocr_map.json"
QUESTION_IMAGE_MAP = APP_ROOT / "data" / "image_cache" / "question_image_map.json"
IMAGE_CACHE_MAP = APP_ROOT / "data" / "image_cache" / "image_cache_map.json"
DB_PATH = Path(r"D:\ACADZA_WORK\acadza-pdf-generator\data\allque030326.json")
OUTPUT_FILE = OCR_DIR / "preview_samples.html"

SAMPLES_PER_TYPE = 6


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs, flush=True)


def image_to_base64(local_path: Path) -> str:
    if not local_path or not local_path.exists():
        return ""
    suffix = local_path.suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
    }
    mime = mime_map.get(suffix, "image/png")
    data = local_path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def smiles_to_svg_data_uri(smiles_str: str) -> str:
    """Render SMILES to SVG base64 data URI using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        cleaned = smiles_str.strip()
        # Some SMILES have <br> injected from DB HTML
        cleaned = re.sub(r'<br\s*/?>',  '', cleaned)
        mol = Chem.MolFromSmiles(cleaned)
        if not mol:
            return ""
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(300, 250)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        b64 = base64.b64encode(svg.encode()).decode()
        return f"data:image/svg+xml;base64,{b64}"
    except Exception as e:
        return ""


def get_field_images(qid: str, field_name: str, qocr_map: dict, url_cache: dict) -> list:
    """Get base64 images for a specific QID + field (question or solution)."""
    images = []
    # qocr_map keys are like "qid|qtype.field"
    for map_key, map_data in qocr_map.items():
        if not map_key.startswith(qid + "|"):
            continue
        # Extract field from key: "qid|scq.question" -> "question"
        parts = map_key.split("|", 1)
        if len(parts) < 2:
            continue
        key_field = parts[1].split(".")[-1]  # "scq.question" -> "question"
        if key_field != field_name:
            continue

        for r in map_data.get("ocr_results", []):
            url = r.get("url", "")
            if not url:
                continue
            cached = url_cache.get(url, {})
            local_str = cached.get("local_path", "")
            if local_str and cached.get("status") == "downloaded":
                lp = Path(local_str)
                b64 = image_to_base64(lp)
                if b64:
                    images.append(b64)
    return images


def has_diagram_content(raw_text: str) -> bool:
    """Check if the OCR text references a diagram/figure that was in the image."""
    lower = raw_text.lower()
    return any(w in lower for w in [
        "figure", "diagram", "circuit", "shown in", "given below",
        "shown above", "shown below", "the graph", "as shown",
    ])


def build_smiles_samples(url_cache: dict) -> list:
    """Pull SMILES entries from main DB and render them."""
    if not DB_PATH.exists():
        return []

    QTYPES = ["scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"]
    samples = []

    with open(DB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if "<smiles>" not in line.lower():
                continue
            entry = json.loads(line)
            qid = entry.get("_id", {}).get("$oid", "?")

            for qtype in QTYPES:
                obj = entry.get(qtype)
                if not obj or not isinstance(obj, dict):
                    continue
                q_html = obj.get("question", "") or obj.get("ques", "") or ""
                s_html = obj.get("solution", "") or obj.get("sol", "") or ""

                # Find SMILES in question
                smiles_in_q = re.findall(r'<smiles>(.*?)</smiles>', str(q_html), re.IGNORECASE | re.DOTALL)
                if not smiles_in_q:
                    break

                # Render each SMILES to SVG
                rendered_parts = []
                for sm in smiles_in_q[:4]:
                    svg_uri = smiles_to_svg_data_uri(sm)
                    rendered_parts.append({
                        "smiles": sm.strip(),
                        "svg_uri": svg_uri,
                    })

                if any(r["svg_uri"] for r in rendered_parts):
                    samples.append({
                        "qid": qid,
                        "qtype": qtype,
                        "question_html": str(q_html),
                        "rendered_smiles": rendered_parts,
                    })
                break

            if len(samples) >= 4:
                break

    return samples


def main():
    _safe_print("Building preview HTML...")

    # Load caches
    url_cache = {}
    if IMAGE_CACHE_MAP.exists():
        with open(IMAGE_CACHE_MAP, "r", encoding="utf-8") as f:
            url_cache = json.load(f).get("url_to_local", {})
    _safe_print(f"  Image cache: {len(url_cache)} URLs")

    qocr_map = {}
    if QUESTION_OCR_MAP.exists():
        with open(QUESTION_OCR_MAP, "r", encoding="utf-8") as f:
            qocr_map = json.load(f)
    _safe_print(f"  Question OCR map: {len(qocr_map)} entries")

    # Classify converted entries
    categories = {
        "diagram": [],
        "display-math": [],
        "math-heavy": [],
        "text-mixed": [],
    }

    with open(CONVERTED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["_id"].get("$oid", "?")

            fields = {}
            total_blocks = 0
            is_diagram = False
            is_display = False

            for field, data in entry.get("ocr_fields", {}).items():
                html = data.get("html", "")
                raw = data.get("ocr_text", "") or data.get("ocr_latex", "")
                sc = data.get("latex_conversion_success", 0)
                fc = data.get("latex_conversion_fail", 0)

                # Get images for THIS specific field
                imgs = get_field_images(qid, field, qocr_map, url_cache)

                fields[field] = {
                    "html": html, "raw": raw,
                    "success": sc, "fail": fc,
                    "images": imgs,
                    "has_diagram": has_diagram_content(raw),
                }
                total_blocks += sc

                if has_diagram_content(raw):
                    is_diagram = True
                if "$$" in raw or "\\[" in raw:
                    is_display = True

            if not fields:
                continue

            sample = {"qid": qid, "fields": fields, "total_blocks": total_blocks}

            if is_diagram and len(categories["diagram"]) < SAMPLES_PER_TYPE:
                sample["type"] = "diagram"
                categories["diagram"].append(sample)
            elif is_display and len(categories["display-math"]) < SAMPLES_PER_TYPE:
                sample["type"] = "display-math"
                categories["display-math"].append(sample)
            elif total_blocks >= 4 and len(categories["math-heavy"]) < SAMPLES_PER_TYPE:
                sample["type"] = "math-heavy"
                categories["math-heavy"].append(sample)
            elif 1 <= total_blocks <= 3 and len(categories["text-mixed"]) < SAMPLES_PER_TYPE:
                sample["type"] = "text-mixed"
                categories["text-mixed"].append(sample)

            if all(len(v) >= SAMPLES_PER_TYPE for v in categories.values()):
                break

    # Build SMILES samples
    _safe_print("  Building SMILES samples...")
    smiles_samples = build_smiles_samples(url_cache)
    _safe_print(f"  SMILES samples: {len(smiles_samples)}")

    cat_list = [
        ("diagram", "Diagram / Figure References", categories["diagram"]),
        ("display-math", "Display Math (equations, arrays)", categories["display-math"]),
        ("math-heavy", "Math-Heavy (4+ LaTeX blocks)", categories["math-heavy"]),
        ("text-mixed", "Text + Math Mixed", categories["text-mixed"]),
    ]

    _safe_print(f"  OCR samples: {', '.join(f'{n}={len(s)}' for _, n, s in cat_list)}")

    # ── Build HTML ──
    parts = [HTML_HEAD]

    # Filter buttons
    parts.append('<div class="filters">')
    parts.append('<button class="filter-btn active" onclick="filterType(\'all\',this)">All</button>')
    for cid, cname, samples in cat_list:
        if samples:
            parts.append(f'<button class="filter-btn" onclick="filterType(\'{cid}\',this)">{cname} ({len(samples)})</button>')
    if smiles_samples:
        parts.append(f'<button class="filter-btn" onclick="filterType(\'smiles\',this)">SMILES / SVG ({len(smiles_samples)})</button>')
    parts.append('</div>')

    total_samples = sum(len(s) for _, _, s in cat_list) + len(smiles_samples)
    parts.append(f'<div class="stats-bar">Showing {total_samples} questions | 19,633 converted entries | 86,280 LaTeX blocks | 99.91% success | 0 SMILES in OCR output</div>')

    # Render OCR samples
    idx = 0
    for cid, cname, samples in cat_list:
        if not samples:
            continue
        parts.append(f'<h2 class="section-title" data-type="{cid}">{cname}</h2>')
        for sample in samples:
            idx += 1
            parts.append(render_ocr_sample(sample, idx))

    # Render SMILES section
    if smiles_samples:
        parts.append('<h2 class="section-title" data-type="smiles">SMILES Chemistry (SVG rendered from DB text entries)</h2>')
        parts.append('<p class="section-note">These are from the main DB text-based questions (277 entries). MathPix OCR output has zero SMILES. Rendered using RDKit.</p>')
        for i, sm in enumerate(smiles_samples):
            idx += 1
            parts.append(render_smiles_sample(sm, idx))

    parts.append(HTML_FOOT)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    _safe_print(f"\n  Output: {OUTPUT_FILE}")


def render_ocr_sample(sample: dict, idx: int) -> str:
    """Render one OCR question — each field as its own before/after card."""
    qid = sample["qid"]
    stype = sample.get("type", "")

    out = []
    out.append(f'<div class="question-block" data-type="{stype}">')
    out.append(f'<div class="qid-header">')
    out.append(f'<span class="qid-label">QID: {qid}</span>')
    out.append(f'<span class="type-badge badge-{stype}">{stype}</span>')
    out.append(f'<span class="block-count">{sample["total_blocks"]} LaTeX blocks</span>')
    out.append('</div>')

    for field_name, fdata in sample["fields"].items():
        uid = f"s{idx}_{field_name}"
        label = field_name.replace("_", " ").title()
        html = fdata["html"]
        raw = fdata["raw"]
        images = fdata["images"]
        sc, fc = fdata["success"], fdata["fail"]
        is_diagram = fdata.get("has_diagram", False)

        out.append(f'<div class="field-card">')
        out.append(f'<div class="field-header"><span class="field-label">{label}</span>')
        out.append(f'<span class="field-stats">{sc} OK, {fc} fail')
        if is_diagram:
            out.append(' | <span class="diagram-flag">has diagram</span>')
        out.append('</span></div>')

        out.append('<div class="comparison">')

        # ── BEFORE: original image(s) for THIS field ──
        out.append('<div class="before-col">')
        out.append('<div class="col-label">Before (Original Image)</div>')
        if images:
            for i, uri in enumerate(images):
                out.append(f'<img src="{uri}" class="original-img" loading="lazy" />')
        else:
            out.append('<div class="no-image">No cached image</div>')
        out.append('</div>')

        # ── AFTER: rendered HTML + diagram notice ──
        out.append('<div class="after-col">')
        out.append('<div class="col-label">After (Converted HTML)</div>')

        # Rendered content
        out.append(f'<div class="rendered-content" id="rendered_{uid}">')
        out.append(html)
        # If it references a diagram, show the original image inline (diagram stays as image)
        if is_diagram and images:
            out.append('<div class="diagram-notice">Diagram from original (not OCR-extractable, stays as image for CDN upload):</div>')
            for uri in images:
                out.append(f'<img src="{uri}" class="diagram-img" loading="lazy" />')
        out.append('</div>')

        # Toggle buttons
        out.append('<div class="toggle-bar">')
        out.append(f'<button class="toggle-btn" onclick="togglePanel(\'{uid}_latex\')">LaTeX Source</button>')
        out.append(f'<button class="toggle-btn" onclick="togglePanel(\'{uid}_html\')">MathML/HTML Source</button>')
        out.append(f'<button class="toggle-btn copy-btn" onclick="copyText(\'{uid}_html_code\')">Copy HTML</button>')
        out.append('</div>')

        # Hidden panels
        out.append(f'<div class="source-panel" id="{uid}_latex" style="display:none"><pre class="source-code">{escape_html(raw)}</pre></div>')
        out.append(f'<div class="source-panel" id="{uid}_html" style="display:none"><pre class="source-code" id="{uid}_html_code">{escape_html(html)}</pre></div>')

        out.append('</div>')  # after-col
        out.append('</div>')  # comparison
        out.append('</div>')  # field-card

    out.append('</div>')  # question-block
    return "\n".join(out)


def render_smiles_sample(sample: dict, idx: int) -> str:
    """Render one SMILES question from the DB."""
    qid = sample["qid"]
    uid = f"smiles_{idx}"

    out = []
    out.append(f'<div class="question-block" data-type="smiles">')
    out.append(f'<div class="qid-header">')
    out.append(f'<span class="qid-label">QID: {qid}</span>')
    out.append(f'<span class="type-badge badge-smiles">smiles</span>')
    out.append(f'<span class="block-count">{len(sample["rendered_smiles"])} SMILES</span>')
    out.append('</div>')

    out.append('<div class="field-card">')
    out.append('<div class="field-header"><span class="field-label">Question (from DB)</span></div>')

    out.append('<div class="comparison">')

    # Before: raw HTML with <smiles> tags
    out.append('<div class="before-col">')
    out.append('<div class="col-label">Before (raw DB HTML with &lt;smiles&gt; tags)</div>')
    out.append(f'<pre class="source-code" style="max-height:400px">{escape_html(sample["question_html"][:2000])}</pre>')
    out.append('</div>')

    # After: rendered SVGs
    out.append('<div class="after-col">')
    out.append('<div class="col-label">After (SMILES rendered to SVG via RDKit)</div>')
    out.append('<div class="rendered-content">')
    for r in sample["rendered_smiles"]:
        out.append('<div class="smiles-card">')
        out.append(f'<div class="smiles-code">{escape_html(r["smiles"])}</div>')
        if r["svg_uri"]:
            out.append(f'<img src="{r["svg_uri"]}" class="smiles-svg" />')
        else:
            out.append('<div class="no-image">RDKit failed to parse this SMILES</div>')
        out.append('</div>')
    out.append('</div>')

    # Copy button for raw HTML
    out.append('<div class="toggle-bar">')
    out.append(f'<button class="toggle-btn" onclick="togglePanel(\'{uid}_src\')">Full Source HTML</button>')
    out.append('</div>')
    out.append(f'<div class="source-panel" id="{uid}_src" style="display:none"><pre class="source-code">{escape_html(sample["question_html"])}</pre></div>')

    out.append('</div>')  # after-col
    out.append('</div>')  # comparison
    out.append('</div>')  # field-card
    out.append('</div>')  # question-block
    return "\n".join(out)


# ── HTML Template ──────────────────────────────────────────────────────

HTML_HEAD = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>OCR Conversion Preview</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f2f5;color:#1a1a1a;max-width:1400px;margin:0 auto;padding:16px}
h1{font-size:22px;margin-bottom:6px}
.stats-bar{background:#e3f2fd;border-radius:6px;padding:8px 14px;font-size:13px;color:#1565c0;margin-bottom:14px}
.section-note{font-size:13px;color:#666;margin-bottom:12px;padding-left:4px}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.filter-btn{padding:6px 14px;border:1px solid #ccc;border-radius:20px;background:#fff;cursor:pointer;font-size:13px;transition:.2s}
.filter-btn:hover{border-color:#1976d2;color:#1976d2}
.filter-btn.active{background:#1976d2;color:#fff;border-color:#1976d2}
.section-title{font-size:17px;color:#333;margin:24px 0 10px;padding-bottom:6px;border-bottom:2px solid #ddd}
.question-block{background:#fff;border-radius:10px;margin-bottom:18px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden}
.qid-header{background:#fafafa;padding:10px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #eee}
.qid-label{font-weight:600;font-size:13px;color:#333;font-family:monospace}
.type-badge{font-size:11px;padding:2px 10px;border-radius:12px;font-weight:500}
.badge-diagram{background:#fff3e0;color:#e65100}
.badge-display-math{background:#e8eaf6;color:#283593}
.badge-math-heavy{background:#e0f7fa;color:#006064}
.badge-text-mixed{background:#f3e5f5;color:#6a1b9a}
.badge-smiles{background:#fce4ec;color:#c62828}
.block-count{font-size:12px;color:#888;margin-left:auto}
.field-card{margin:12px 16px;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden}
.field-header{background:#f5f5f5;padding:8px 12px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e0e0e0}
.field-label{font-weight:600;font-size:13px;text-transform:uppercase;color:#555}
.field-stats{font-size:12px;color:#888}
.diagram-flag{color:#e65100;font-weight:600}
.comparison{display:grid;grid-template-columns:1fr 1fr;gap:0}
.before-col,.after-col{padding:12px}
.before-col{border-right:2px solid #e0e0e0;background:#fafafa}
.after-col{background:#fff}
.col-label{font-size:11px;text-transform:uppercase;color:#999;margin-bottom:8px;font-weight:600;letter-spacing:.5px}
.original-img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;margin-bottom:6px;display:block}
.no-image{color:#bbb;font-style:italic;font-size:13px;padding:20px;text-align:center;background:#f9f9f9;border:1px dashed #ddd;border-radius:4px}
.rendered-content{font-size:15px;line-height:1.7;padding:8px;min-height:40px}
.rendered-content math{font-size:1.05em}
.rendered-content math[display="block"]{display:block;margin:10px 0;text-align:center}
.diagram-notice{margin-top:12px;padding:6px 10px;background:#fff8e1;border:1px solid #ffe082;border-radius:4px;font-size:12px;color:#f57f17;font-weight:600}
.diagram-img{max-width:100%;height:auto;border:1px solid #ffe082;border-radius:4px;margin-top:6px}
.toggle-bar{display:flex;gap:6px;padding:8px 0;margin-top:8px;border-top:1px solid #eee}
.toggle-btn{padding:4px 10px;border:1px solid #ccc;border-radius:4px;background:#f9f9f9;cursor:pointer;font-size:12px}
.toggle-btn:hover{background:#e3f2fd;border-color:#90caf9}
.copy-btn{background:#e8f5e9;border-color:#a5d6a7}
.copy-btn:hover{background:#c8e6c9}
.source-panel{margin-top:6px;border-radius:4px;overflow:hidden}
.source-code{background:#263238;color:#eeffff;padding:12px;font-size:12px;line-height:1.5;overflow-x:auto;white-space:pre-wrap;word-break:break-all;max-height:300px;overflow-y:auto;border-radius:4px}
.smiles-card{display:inline-block;border:1px solid #e0e0e0;border-radius:8px;padding:8px;margin:6px;text-align:center;vertical-align:top}
.smiles-code{font-family:monospace;font-size:11px;color:#666;margin-bottom:4px;word-break:break-all}
.smiles-svg{max-width:280px;height:auto}
@media(max-width:900px){.comparison{grid-template-columns:1fr}.before-col{border-right:none;border-bottom:2px solid #e0e0e0}}
</style>
</head>
<body>
<h1>OCR Conversion Preview</h1>
<p style="color:#666;font-size:13px;margin-bottom:12px">
Each field shows <b>Before</b> (original image) and <b>After</b> (converted HTML) for the <b>same</b> question or solution.
Diagrams that can't be OCR'd appear in the After column too (they stay as images for CDN upload).
Use toggle buttons to view/copy LaTeX or MathML source.
</p>
"""

HTML_FOOT = """
<script>
function togglePanel(id){
    const el=document.getElementById(id);
    if(el)el.style.display=el.style.display==='none'?'block':'none';
}
function copyText(id){
    const el=document.getElementById(id);
    if(!el)return;
    navigator.clipboard.writeText(el.textContent).then(()=>{
        const btn=event.target;const o=btn.textContent;
        btn.textContent='Copied!';btn.style.background='#4caf50';btn.style.color='#fff';
        setTimeout(()=>{btn.textContent=o;btn.style.background='';btn.style.color=''},1500);
    });
}
function filterType(type,btn){
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.question-block').forEach(b=>{
        b.style.display=(type==='all'||b.dataset.type===type)?'':'none';
    });
    document.querySelectorAll('.section-title,.section-note').forEach(t=>{
        t.style.display=(type==='all'||t.dataset.type===type)?'':'none';
    });
}
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
