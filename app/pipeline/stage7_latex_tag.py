"""Stage 7: LaTeX + Tagging Batch --pipeline adapter.

Builds JSONL batch files for:
  1. Concept tagging (subject -> chapter -> concept -> subConcept)
  2. LaTeX-to-MathML conversion (raw LaTeX in HTML -> MathML)

Submits batches to OpenAI Batch API, polls for completion, downloads results,
and merges tags + MathML back into metadata_v2 files.

Forked from: CodeBase/Gemini_Bulk_Batch_Processor.py + CodeBase/PCMB_Bulk_Post_Processor.py
CHANGES: Adapted for session-based pipeline, OpenAI batch API, actual latex2mathml conversion.
"""

import json
import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.pipeline.base import BasePipelineStage

try:
    from latex2mathml.converter import convert as latex_to_mathml
    HAS_LATEX2MATHML = True
except ImportError:
    HAS_LATEX2MATHML = False

log = logging.getLogger(__name__)

# ============== FAISS Tagging Constants (forked from production) ==============
TOP_K = 28
SIMILARITY_CUTOFF = 0.50
MAX_CHUNKS_PER_CHAPTER = 5
MAX_TOTAL_CHUNKS = 51
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_EMBEDDING_CHARS = 8000

SUBJECT_FAISS_MAP = {
    "Physics": "physics",
    "Chemistry": "chemistry",
    "Mathematics": "math",
    "Biology": "biology",
    "Botany": "biology",
    "Zoology": "biology",
}

# Production tagger prompt (forked from Gemini_Bulk_Batch_Processor.py)
TAGGER_INSTRUCTIONS = """You are an expert JEE/NEET exam content tagger. Your task is to identify the most relevant concept tags for questions.

CRITICAL INSTRUCTION:
- You MUST return ONLY the chunk IDs provided in the reference list
- Do NOT invent, modify, or paraphrase any chunk information
- Simply select 1-3 chunk IDs that best match the question's concepts

TASK:
1. Read the question and solution carefully
2. Identify the main concepts being tested
3. Select 1-3 chunk IDs from the AVAILABLE CHUNKS that best match
4. Return ONLY the chunk IDs - the system will look up the full details

RULES:
- Return EXACTLY 1-3 chunk IDs (no more, no less)
- Use ONLY IDs from the provided list (e.g., "PHY_0001", "CHE_0234")
- If no chunks match, return an empty array
- Never mix chunks from different subjects for the same question
- Focus on concepts DIRECTLY tested in the question

OUTPUT FORMAT (JSON only):
{
  "selected_chunk_ids": ["PHY_0001", "PHY_0023"]
}

- Return ONLY the JSON object
- No explanation, no commentary
- Just the chunk IDs array
"""

TAGGING_OUTPUT_SPEC = """OUTPUT (STRICT JSON):
{
  "selected_chunk_ids": ["<ID1>", "<ID2>"]
}

Maximum 3 IDs. Use exact IDs from the list. No other keys or text."""



# Production LaTeX prompt (forked from Gemini_Bulk_Batch_Processor.py)
LATEX_SYSTEM_INSTRUCTION = (
    "You are a LaTeX delimiter gatekeeper. "
    "You ONLY ensure mathematical content has proper \\( ... \\) delimiters. "
    "You NEVER change content, meaning, spelling, or HTML structure. "
    "Output valid JSON only."
)

LATEX_GUARD_SPEC = r"""You are a STRICT LaTeX delimiter gatekeeper.

YOUR ONLY TASK: Ensure mathematical expressions have proper \( ... \) delimiters.
You are NOT a math solver, editor, or proofreader. You ONLY fix LaTeX delimiters.

CRITICAL RULES - READ CAREFULLY:

1. COPY EVERYTHING VERBATIM
   - Copy ALL text exactly as-is, character by character
   - Do NOT change spelling, punctuation, or wording
   - Do NOT "fix" perceived errors in the content
   - Do NOT add explanations or clarifications
   - If text says "H2O", keep it as "H2O" (don't change to H2O unless already formatted)

2. ONLY ADD \( ... \) DELIMITERS WHERE NEEDED
   You should ONLY wrap mathematical content that is MISSING delimiters:

   WRAP these (if not already wrapped):
   - Standalone math symbols: alpha, beta, theta, pi
   - Subscripts/superscripts: H2O -> \(H_2O\), x^2 -> \(x^2\)
   - Fractions written mathematically: 1/2 in math context
   - Equations: E=mc^2 -> \(E=mc^2\)

   DO NOT WRAP these:
   - Regular text and words
   - HTML tags and attributes
   - Already-delimited content: \(existing\)
   - Chemical formulas in plain text context

3. PRESERVE HTML EXACTLY
   - Keep ALL HTML tags: <p>, <br/>, <strong>, <img>, etc.
   - Keep ALL attributes: src="...", alt="...", style="..."
   - Do NOT escape &, <, > (they're already HTML-encoded)
   - Do NOT add or remove any HTML tags

4. FIX ONLY BROKEN DELIMITERS
   - Unpaired \( without \) -> add the missing \)
   - Unpaired \) without \( -> add the missing \(
   - $...$ -> convert to \(...\)
   - $$...$$ -> convert to \(...\) (inline only)

SYMBOLS THAT MUST BE LATEX-WRAPPED IF LONE IN TEXT:
- subscripts / superscripts ( V_B , g sin^-1 )
- fractions ( 1/2 when mathematical )
- Greek or special letters ( theta, DeltaQ, pi )
- trig / math functions ( sin, cos, tan, log, ln, lim, cot, etc. )
- integrals, summations, products
- degree sign 30 degrees -> \(30^\circ\)
- units with powers  m*s^-2 -> \(m \cdot s^{-2}\)
"""

LATEX_OUTPUT_SPEC = r"""INPUT/OUTPUT FORMAT:

INPUT (exactly this structure):
{
  "question": "<HTML string with question content>",
  "solution": "<HTML string with solution content>"
}

OUTPUT (exactly this structure):
{
  "question": "<HTML string - same content, with LaTeX properly delimited>",
  "solution": "<HTML string - same content, with LaTeX properly delimited>"
}

STRICT OUTPUT REQUIREMENTS:
- Return ONLY the JSON object, nothing else
- Same keys in same order: "question", "solution"
- No markdown code blocks
- No explanation text before or after
- No trailing commas
- Preserve ALL original content - only add/fix \( \) delimiters
"""


# ============== LaTeX Table-to-HTML Converter ==============

def latex_table_to_html(latex: str) -> str:
    """Convert a LaTeX \\begin{tabular}...\\end{tabular} block to an HTML <table>.

    Handles:
      - Column specs like {|l|l|l|}, {||l|l||l|l|}, {|l|l||c|l|}
      - \\hline -> bordered rows
      - & -> cell separator
      - \\\\ -> row separator
      - Nested \\begin{tabular}...\\end{tabular} inside cells (flattened to text)
      - $(...) inline math left as-is
    """
    # Step 1: Iteratively flatten innermost nested tabular blocks FIRST.
    text = latex.strip()
    changed = True
    while changed:
        changed = False
        pattern = re.compile(
            r'\\begin\{tabular\}\{[^}]*\}((?:(?!\\begin\{tabular\}).)*?)\\end\{tabular\}',
            re.DOTALL
        )
        matches = list(pattern.finditer(text))
        if not matches:
            break
        if len(matches) == 1:
            break
        for m in reversed(matches):
            inner = m.group(1).strip()
            inner = re.sub(r'\s*\\\\\s*', ' ', inner)
            inner = inner.replace('\\hline', '').strip()
            text = text[:m.start()] + inner + text[m.end():]
        changed = True

    # Step 2: Extract the single remaining outermost tabular
    outer = re.match(
        r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}',
        text.strip(), re.DOTALL
    )
    if not outer:
        return f"<!-- ERROR: could not parse tabular -->\n<pre>{latex}</pre>"

    col_spec = outer.group(1)
    body = outer.group(2).strip()

    # Step 3: Determine border from col_spec
    has_border = '|' in col_spec

    # Step 4: Split into rows by \\
    raw_rows = re.split(r'\\\\', body)

    rows_html = []
    for raw_row in raw_rows:
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        row_text = raw_row.replace('\\hline', '').strip()
        if not row_text:
            continue
        cells = [c.strip() for c in row_text.split('&')]
        cells_html = ''.join(f'<td>{c}</td>' for c in cells)
        rows_html.append(f'  <tr>{cells_html}</tr>')

    border_attr = ' border="1" cellpadding="4" cellspacing="0"' if has_border else ''
    table_html = f'<table{border_attr}>\n' + '\n'.join(rows_html) + '\n</table>'
    return table_html


def _convert_tabular_blocks(html: str) -> tuple:
    """Find and convert all \\begin{tabular}...\\end{tabular} blocks in HTML to HTML tables.

    Runs iteratively to handle nested tabulars (inner ones get flattened first
    by latex_table_to_html, then outer wrapper gets converted on next pass).

    Returns:
        (converted_html, count_converted)
    """
    if not html:
        return html, 0

    total_count = 0
    max_iterations = 10  # safety limit

    for _ in range(max_iterations):
        pattern = re.compile(
            r'\\begin\{tabular\}.*?\\end\{tabular\}',
            re.DOTALL
        )
        matches = list(pattern.finditer(html))
        if not matches:
            break

        converted_this_pass = 0
        for m in reversed(matches):
            table_html = latex_table_to_html(m.group(0))
            if '<!-- ERROR' not in table_html:
                html = html[:m.start()] + table_html + html[m.end():]
                converted_this_pass += 1

        if converted_this_pass == 0:
            break
        total_count += converted_this_pass

    return html, total_count


HTML_TAG_RE = re.compile(r"<[^>]+>")

def _strip_tags(s: str) -> str:
    """Strip HTML tags from a string."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", HTML_TAG_RE.sub(" ", s)).strip()


def _one_line_html(s: str) -> str:
    """Collapse HTML to single line for JSON embedding."""
    if not s:
        return "<p></p>"
    s = s.replace("\r", "").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s.startswith("<p") and not s.startswith("<h"):
        s = f"<p>{s}</p>"
    return s


# ============== ChunkCache (forked from production) ==============

class ChunkCache:
    """Cache for concept chunks with unique IDs. Eliminates hallucination."""

    def __init__(self, subject: str, chunk_data: List[Dict]):
        self.subject = subject
        self.id_to_chunk: Dict[str, Dict] = {}
        self.chunk_to_id: Dict[str, str] = {}
        self.display_lines: Dict[str, str] = {}

        for idx, chunk in enumerate(chunk_data):
            subj_code = self._get_subject_code(chunk.get("subject", ""))
            chunk_id = f"{subj_code}_{idx:04d}"
            self.id_to_chunk[chunk_id] = chunk

            subj = chunk.get("subject", "")
            chap = chunk.get("chapter", "")
            concept = chunk.get("concept", "")
            subc = chunk.get("subconcept", "")
            display = f"{subj} > {chap} > {concept} > {subc}"
            self.display_lines[chunk_id] = display
            self.chunk_to_id[display] = chunk_id

    def _get_subject_code(self, subject: str) -> str:
        s = subject.lower()
        if "physics" in s: return "PHY"
        elif "chemistry" in s: return "CHE"
        elif "math" in s: return "MAT"
        elif "botany" in s: return "BOT"
        elif "zoology" in s: return "ZOO"
        elif "biology" in s: return "BIO"
        return "UNK"

    def format_chunks_for_prompt(self, chunk_ids: List[str]) -> str:
        lines = []
        for idx, cid in enumerate(chunk_ids, 1):
            display = self.display_lines.get(cid, "UNKNOWN")
            lines.append(f"{idx}. [{cid}] {display}")
        return "\n".join(lines)

    def resolve_tags_from_ids(self, chunk_ids: List[str]) -> List[Dict]:
        tags = []
        for cid in chunk_ids:
            chunk = self.id_to_chunk.get(cid)
            if chunk:
                tags.append({
                    "subConcept": chunk.get("subconcept", ""),
                    "concept": chunk.get("concept", ""),
                    "chapter": chunk.get("chapter", ""),
                    "subject": chunk.get("subject", ""),
                    "_chunk_id": cid,
                })
        return tags


# ============== FAISS Loading + Retrieval ==============

_faiss_cache: Dict[str, Tuple] = {}
_chunk_caches: Dict[str, ChunkCache] = {}


def _load_faiss_for_subject(subject: str):
    """Load FAISS index + ChunkCache for a subject. Cached across calls."""
    if subject in _faiss_cache:
        return _faiss_cache[subject], _chunk_caches[subject]

    import faiss
    import numpy as np

    from app.config import settings
    base_dir = Path(settings.tagging_modules_dir)

    s = SUBJECT_FAISS_MAP.get(subject, subject.lower())
    module_dir = base_dir / f"{s}_tagging_module"
    index_path = module_dir / f"{s}_faiss_index.idx"
    chunks_path = module_dir / f"{s}_chunk_data.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    log.info(f"Loading FAISS index for {subject} from {module_dir}...")
    index = faiss.read_index(str(index_path))

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    cache = ChunkCache(s, chunk_data)
    _faiss_cache[subject] = (index, chunk_data)
    _chunk_caches[subject] = cache

    log.info(f"Loaded {len(chunk_data)} chunks, index vectors: {index.ntotal}")
    return (index, chunk_data), cache


def _get_embeddings(texts: List[str], openai_client) -> List:
    """Get embeddings for texts using OpenAI text-embedding-3-small."""
    import numpy as np
    truncated = [t[:MAX_EMBEDDING_CHARS] if len(t) > MAX_EMBEDDING_CHARS else (t if t.strip() else "empty")
                 for t in texts]
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=truncated)
    return [np.array(item.embedding, dtype="float32") for item in response.data]


def _retrieve_chunk_ids(query_vector, index, chunk_data, cache) -> List[str]:
    """Retrieve relevant chunk IDs via FAISS similarity search."""
    import numpy as np
    scores, indices = index.search(np.array([query_vector]), 100)

    filtered = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(chunk_data) and score >= SIMILARITY_CUTOFF:
            chunk = chunk_data[idx]
            display = f"{chunk.get('subject','')} > {chunk.get('chapter','')} > {chunk.get('concept','')} > {chunk.get('subconcept','')}"
            chunk_id = cache.chunk_to_id.get(display)
            if chunk_id:
                filtered.append((chunk_id, chunk, score))

    if not filtered:
        return []

    chapter_chunks = defaultdict(list)
    for cid, chunk, score in filtered:
        key = f"{chunk.get('subject')}::{chunk.get('chapter')}"
        chapter_chunks[key].append(cid)

    selected = []
    for key, cids in chapter_chunks.items():
        selected.extend(cids[:MAX_CHUNKS_PER_CHAPTER])
        if len(selected) >= MAX_TOTAL_CHUNKS:
            return selected[:MAX_TOTAL_CHUNKS]
    return selected


class LatexTagStage(BasePipelineStage):

    @property
    def stage_name(self) -> str:
        return "stage7_batch"

    def get_resume_state(self) -> Dict[str, Any]:
        """Determine resume point by walking the ledger backwards.

        Returns the current sub-step state with what action is needed next.
        This is the core resume logic --called on page load and Refresh.
        """
        steps = self.ledger_get_all()

        # Walk backwards from final step to find where we are
        merge = steps.get("merge", {})
        if merge.get("status") == "completed":
            return {"resume_from": "completed", "action": "advance",
                    "message": "All steps complete --ready to advance"}

        download = steps.get("batch_download", {})
        if download.get("status") == "completed":
            # Verify files actually exist
            tag_ok = self.ledger_verify_file("batch_results/tagging_results.jsonl")
            if tag_ok:
                return {"resume_from": "batch_download", "action": "merge",
                        "message": "Results downloaded --ready to merge"}
            else:
                return {"resume_from": "batch_poll", "action": "download",
                        "message": "Download claimed complete but files missing --re-download needed"}

        poll = steps.get("batch_poll", {})
        if poll.get("status") == "completed":
            return {"resume_from": "batch_poll", "action": "download",
                    "message": "Batches complete --ready to download"}

        submit = steps.get("batch_submit", {})
        if submit.get("status") == "submitted":
            batch_id = submit.get("tagging_batch_id")
            if batch_id:
                return {"resume_from": "batch_submit", "action": "poll",
                        "batch_ids": {"tagging": batch_id, "latex": submit.get("latex_batch_id")},
                        "message": f"Batches submitted --poll for progress (tagging: {batch_id[:20]}...)"}
            return {"resume_from": "batch_submit", "action": "poll",
                    "message": "Batches submitted --poll for progress"}

        build = steps.get("batch_build", {})
        if build and build.get("status") == "completed":
            return {"resume_from": "batch_build", "action": "submit",
                    "message": "Batches built --ready to submit"}

        mathml = steps.get("mathml_conversion", {})
        if mathml.get("status") == "completed":
            if not mathml.get("latex_batch_needed", True):
                return {"resume_from": "mathml_conversion", "action": "merge_direct",
                        "message": "All MathML converted locally --no batch needed, ready to merge"}
            return {"resume_from": "mathml_conversion", "action": "build",
                    "message": f"MathML done ({mathml.get('fail', 0)} failed) --ready to build batch"}

        # Also check batch_status.json as fallback (for sessions without ledger)
        batch_status = self._load_batch_status()
        if batch_status.get("tagging_batch_id"):
            return {"resume_from": "batch_submit", "action": "poll",
                    "batch_ids": {"tagging": batch_status["tagging_batch_id"],
                                  "latex": batch_status.get("latex_batch_id")},
                    "message": "Found batch_status.json with batch IDs --resuming from poll",
                    "from_legacy": True}

        return {"resume_from": "idle", "action": "scan",
                "message": "No previous progress --start from scan"}

    def _get_metadata_dir(self) -> Path:
        return self.session_dir / "metadata_v2"

    def _get_batch_dir(self) -> Path:
        d = self.session_dir / "batch_files"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_batch_results_dir(self) -> Path:
        d = self.session_dir / "batch_results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_all_metadata(self, filter_quarantine: bool = True) -> List[Dict]:
        """Load all metadata_v2 entries across all files.

        Args:
            filter_quarantine: If True, exclude quarantined QIDs (default).
        """
        meta_dir = self._get_metadata_dir()
        if not meta_dir.exists():
            return []

        all_entries = []
        for f in sorted(meta_dir.glob("*_metadata_v2.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    all_entries.extend(data)
            except Exception:
                continue

        if filter_quarantine and all_entries:
            from app.db import get_quarantined_qids
            q_qids = get_quarantined_qids(self.session_id)
            if q_qids:
                before = len(all_entries)
                all_entries = [e for e in all_entries if e.get("qid") not in q_qids]
                skipped = before - len(all_entries)
                if skipped:
                    self.log("info", f"Filtered {skipped} quarantined questions from input")

        return all_entries

    def _count_latex_blocks(self, html: str) -> int:
        """Count LaTeX blocks ($...$, $$...$$, \\(...\\), \\[...\\], \\begin{tabular}) in HTML."""
        if not html:
            return 0
        patterns = [
            r'\$\$[^$]+\$\$',
            r'\$[^$]+\$',
            r'\\\([^)]+\\\)',
            r'\\\[[^\]]+\\\]',
            r'\\begin\{tabular\}',
        ]
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, html))
        return count

    # Common hex entities from latex2mathml that CKEditor may not handle.
    # Replace with direct Unicode characters for maximum compatibility.
    _HEX_ENTITY_MAP = {
        "&#x0003D;": "=", "&#x0002B;": "+", "&#x02212;": "\u2212",
        "&#x0003C;": "<", "&#x0003E;": ">",
        "&#x00028;": "(", "&#x00029;": ")",
        "&#x0005B;": "[", "&#x0005D;": "]",
        "&#x0007B;": "{", "&#x0007D;": "}",
        "&#x0007C;": "|",
        "&#x0222B;": "\u222B", "&#x0222C;": "\u222C", "&#x0222D;": "\u222D",
        "&#x0222E;": "\u222E",
        "&#x02211;": "\u2211", "&#x0220F;": "\u220F",
        "&#x0221E;": "\u221E", "&#x02264;": "\u2264", "&#x02265;": "\u2265",
        "&#x02260;": "\u2260", "&#x02248;": "\u2248",
        "&#x003B1;": "\u03B1", "&#x003B2;": "\u03B2", "&#x003B3;": "\u03B3",
        "&#x003B4;": "\u03B4", "&#x003B5;": "\u03B5", "&#x003B8;": "\u03B8",
        "&#x003BB;": "\u03BB", "&#x003BC;": "\u03BC", "&#x003C0;": "\u03C0",
        "&#x003C3;": "\u03C3", "&#x003C6;": "\u03C6", "&#x003C9;": "\u03C9",
        "&#x02207;": "\u2207", "&#x02202;": "\u2202",
        "&#x0221A;": "\u221A",
        "&#x000B1;": "\u00B1",  # ±
        "&#x000D7;": "\u00D7",  # ×
        "&#x000F7;": "\u00F7",  # ÷
        "&#x02026;": "\u2026",  # …
        "&#x02192;": "\u2192",  # →
        "&#x02190;": "\u2190",  # ←
        "&#x021D2;": "\u21D2",  # ⇒
        "&#x02203;": "\u2203",  # ∃
        "&#x02200;": "\u2200",  # ∀
        "&#x02208;": "\u2208",  # ∈
        "&#x02282;": "\u2282",  # ⊂
        "&#x02229;": "\u2229",  # ∩
        "&#x0222A;": "\u222A",  # ∪
        "&#x000A0;": "\u00A0",
        "&#x0005E;": "^",      # hat (circumflex)
        "&#x0007E;": "~",      # tilde
        "&#x000AF;": "\u00AF", # macron (bar)
        "&#x002D9;": "\u02D9", # dot above
        "&#x02220;": "\u2220", # angle
        "&#x022C5;": "\u22C5", # dot operator
        "&#x02297;": "\u2297", # tensor product
        "&#x02295;": "\u2295", # direct sum
        "&#x02261;": "\u2261", # identical to
        "&#x02243;": "\u2243", # asymptotically equal
        "&#x021D4;": "\u21D4", # ⇔
        "&#x02205;": "\u2205", # empty set
    }

    @classmethod
    def _postprocess_mathml(cls, mathml: str, is_display: bool) -> str:
        """Fix known latex2mathml output issues for CKEditor compatibility.

        Targeted fixes (only when specific structures are detected):
        1. Hex entities → Unicode chars (CKEditor rejects some hex entities)
        2. Matrix: <mtable> inside parentheses/brackets --add spacing
        3. Integral/Sum limits: <msubsup> with ∫/∑/∏ → <munderover> in display mode
        """
        # ── Fix 0: Replace hex entities with Unicode chars for CKEditor ──
        for entity, char in cls._HEX_ENTITY_MAP.items():
            if entity in mathml:
                mathml = mathml.replace(entity, char)
        # ── Fix 1: Matrix row separation ──
        # latex2mathml merges rows when \\\\ isn't parsed properly.
        # Detect: <mtable> with multiple <mtd> but only one <mtr> --split into rows.
        # Also add columnspacing for readability.
        if '<mtable>' in mathml:
            # Detect matrix context: <mo>( or &#x00028;</mo> anywhere before <mtable>,
            # AND <mo>) or &#x00029;</mo> anywhere after </mtable>
            # This catches pmatrix, bmatrix, vmatrix etc. but NOT standalone tables/arrays
            # Convert bare <mo>DELIM</mo><mtable>...</mtable><mo>DELIM</mo> to
            # <mfenced open="DELIM" close="DELIM"><mtable>...</mtable></mfenced>
            # This matches CKEditor's native MathML format for matrices.
            _MATRIX_PATTERN = re.compile(
                r'<mo>([(\[{|])</mo>\s*(<mtable[^>]*>.*?</mtable>)\s*<mo>([)\]}|])</mo>',
                re.DOTALL
            )
            _matrix_match = _MATRIX_PATTERN.search(mathml)
            if _matrix_match:
                left_delim = _matrix_match.group(1)
                table_content = _matrix_match.group(2)
                right_delim = _matrix_match.group(3)
                # Add spacing to mtable
                if 'columnspacing=' not in table_content:
                    table_content = table_content.replace('<mtable', '<mtable columnspacing="1em" rowspacing="0.5em"', 1)
                replacement = f'<mfenced open="{left_delim}" close="{right_delim}">{table_content}</mfenced>'
                mathml = mathml[:_matrix_match.start()] + replacement + mathml[_matrix_match.end():]

        # ── Fix 2: Integral/Sum/Product limits in display mode ──
        # latex2mathml generates <msubsup><mo>∫</mo>... for \int_{a}^{b}
        # In display mode, limits should be above/below (munderover), not side (msubsup)
        if is_display:
            # Convert msubsup → munderover for large operators in display mode.
            # CKEditor quirk: ∑/∏ in munderover only work with displaystyle="false"
            # wrapper, while ∫ works with displaystyle="true". Handle separately.
            _INTEGRAL_OPS = ('\u222B', '\u222C', '\u222D', '\u222E')
            _SUMPRODUCT_OPS = ('\u2211', '\u220F')

            for op_char in _INTEGRAL_OPS:
                _pattern = (
                    r'<msubsup>\s*<mo>' + re.escape(op_char) + r'</mo>'
                    r'\s*(<mrow>.*?</mrow>)\s*(<mrow>.*?</mrow>)\s*</msubsup>'
                )
                _match = re.search(_pattern, mathml, re.DOTALL)
                if _match:
                    old = _match.group(0)
                    lower = _match.group(1)
                    upper = _match.group(2)
                    new = f'<munderover><mo>{op_char}</mo>{lower}{upper}</munderover>'
                    mathml = mathml.replace(old, new, 1)

            # For ∑/∏: same munderover conversion, limits render on top
            for op_char in _SUMPRODUCT_OPS:
                _pattern = (
                    r'<msubsup>\s*<mo>' + re.escape(op_char) + r'</mo>'
                    r'\s*(<mrow>.*?</mrow>)\s*(<mrow>.*?</mrow>)\s*</msubsup>'
                )
                _match = re.search(_pattern, mathml, re.DOTALL)
                if _match:
                    old = _match.group(0)
                    lower = _match.group(1)
                    upper = _match.group(2)
                    new = f'<munderover><mo>{op_char}</mo>{lower}{upper}</munderover>'
                    mathml = mathml.replace(old, new, 1)

        # ── Fix 3: Remove LaTeX alignment artifact <mi>&</mi> ──
        # LaTeX \begin{aligned} column separators leak into MathML as <mi>&</mi>.
        # These have no mathematical meaning -- safe to remove entirely.
        # Confirmed safe: legitimate & uses &#x00026; entity, not bare &.
        mathml = re.sub(r'<mi>\s*(?:&amp;|&)\s*</mi>', '', mathml)

        # ── Fix 4: Hat/circumflex rendering ──
        # <mo stretchy="false">^</mo> inside <mover> renders as double-dots in CKEditor.
        # Replace with proper combining circumflex &#x0302; for correct hat display.
        # Only appears inside <mover> blocks (unit vectors i-hat, j-hat, k-hat).
        mathml = mathml.replace('<mo stretchy="false">^</mo>', '<mo>&#x0302;</mo>')

        # ── Fix 5: Limit underscript ──
        # <msub><mo>lim</mo>...</msub> renders limit subscript as tiny text.
        # Correct: <munder><mo>lim</mo>...</munder> puts it underneath.
        mathml = re.sub(
            r'<msub>(\s*<mo>lim</mo>)(.*?)</msub>',
            r'<munder>\1\2</munder>',
            mathml, flags=re.DOTALL,
        )

        return mathml

    def _convert_latex_to_mathml(self, html: str) -> Tuple[str, int, int]:
        """Convert all LaTeX blocks in HTML to MathML.

        Finds all LaTeX patterns ($...$, $$...$$, \\(...\\), \\[...\\]),
        converts each to MathML using latex2mathml, and replaces in-place.
        Post-processes for matrix spacing and integral limit placement.

        Returns:
            (converted_html, success_count, fail_count)
        """
        if not html or not HAS_LATEX2MATHML:
            return html, 0, 0

        success_count = 0
        fail_count = 0

        # Define patterns in order: longer delimiters first to avoid partial matches
        # Each tuple: (regex_pattern, group_extractor_for_raw_latex)
        # We process $$...$$ before $...$ to avoid double-matching
        latex_patterns = [
            # $$...$$ (display math)
            (r'\$\$([^$]+)\$\$', True),
            # \[...\] (display math)
            (r'\\\[([^\]]+)\\\]', True),
            # \(...\) (inline math)
            (r'\\\(([^)]+)\\\)', False),
            # $...$ (inline math) --last to avoid matching $$ content
            (r'(?<!\$)\$(?!\$)([^$]+)\$(?!\$)', False),
        ]

        # Matrix environments where \\ must be replaced with \cr for latex2mathml
        _MATRIX_ENVS = ('pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix', 'matrix', 'Bmatrix',
                         'smallmatrix', 'cases', 'array')

        for pattern, is_display in latex_patterns:
            def _replace_match(match, _is_display=is_display):
                nonlocal success_count, fail_count
                raw_latex = match.group(1).strip()

                # Pre-process: replace row separators with \cr inside matrix environments.
                # latex2mathml only understands \cr as row separator.
                # Real data uses: \\ (double backslash) OR <br/> (from MathPIX OCR).
                _TWO_BS = chr(92) + chr(92)  # literal \\
                _CR_SEP = chr(92) + 'cr '    # literal \cr
                for env in _MATRIX_ENVS:
                    begin_tag = chr(92) + 'begin{' + env + '}'
                    end_tag = chr(92) + 'end{' + env + '}'
                    bi = raw_latex.find(begin_tag)
                    ei = raw_latex.find(end_tag)
                    if bi >= 0 and ei > bi:
                        body_start = bi + len(begin_tag)
                        body = raw_latex[body_start:ei]
                        # Replace \\ with \cr
                        body = body.replace(_TWO_BS, _CR_SEP)
                        # Replace <br/> and <br /> with \cr (MathPIX OCR format)
                        body = re.sub(r'<br\s*/?\s*>', lambda _: _CR_SEP, body)
                        raw_latex = raw_latex[:body_start] + body + raw_latex[ei:]

                try:
                    mathml = latex_to_mathml(raw_latex)
                    # Post-process: fix matrix spacing + integral limits
                    mathml = self._postprocess_mathml(mathml, _is_display)
                    # Ensure xmlns is present
                    if 'xmlns=' not in mathml:
                        mathml = mathml.replace(
                            '<math', '<math xmlns="http://www.w3.org/1998/Math/MathML"', 1
                        )
                    if _is_display:
                        # Display math: set display="block" on <math> tag,
                        # wrap content in <mstyle displaystyle="true"> for proper sizing.
                        # CKEditor 4 supports <mstyle> natively (verified via MathmlTest.png).
                        mathml = re.sub(r'display="[^"]*"', '', mathml, count=1)
                        mathml = mathml.replace(
                            '<math', '<math display="block"', 1
                        )
                        mathml = re.sub(
                            r'(<math[^>]*>)(.*)(</math>)',
                            r'\1<mstyle displaystyle="true">\2</mstyle>\3',
                            mathml, count=1, flags=re.DOTALL,
                        )
                    else:
                        # Inline math: set display="inline" and add displaystyle
                        # for proper fraction/sum sizing (confirmed: CKEditor renders
                        # fractions squeezed without displaystyle even in inline mode)
                        if 'display=' not in mathml:
                            mathml = mathml.replace(
                                '<math', '<math display="inline"', 1
                            )
                        if 'displaystyle' not in mathml:
                            mathml = re.sub(
                                r'(<math[^>]*>)(.*)(</math>)',
                                r'\1<mstyle displaystyle="true">\2</mstyle>\3',
                                mathml, count=1, flags=re.DOTALL,
                            )
                    success_count += 1
                    return mathml
                except Exception:
                    # If conversion fails, leave the original text intact
                    fail_count += 1
                    return match.group(0)

            html = re.sub(pattern, _replace_match, html)

        return html, success_count, fail_count

    def scan(self) -> Dict[str, Any]:
        """Scan metadata and report what batches would be built."""
        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata_v2 files found --run Stage 4 first"}

        # Count questions needing tagging
        needs_tagging = 0
        needs_latex = 0
        total_latex_blocks = 0
        by_subject = {}

        for entry in entries:
            subject = entry.get("subject", "Unknown")
            by_subject[subject] = by_subject.get(subject, 0) + 1

            # Check if needs tagging (chapter == "TBD")
            if entry.get("chapter") == "TBD" or entry.get("concept") == "TBD":
                needs_tagging += 1

            # Check if needs LaTeX conversion
            qtype = entry.get("questionType", "")
            qdata = entry.get(qtype, {})
            for field in ["question", "solution"]:
                html = qdata.get(field, "")
                latex_count = self._count_latex_blocks(html)
                if latex_count > 0:
                    needs_latex += 1
                    total_latex_blocks += latex_count
                    break  # count question once even if both fields have LaTeX

        batch_dir = self._get_batch_dir()
        existing_batches = list(batch_dir.glob("*.jsonl"))

        self.log("info", f"Scanned {len(entries)} questions: {needs_tagging} need tagging, {needs_latex} need LaTeX conversion")
        return {
            "total_questions": len(entries),
            "needs_tagging": needs_tagging,
            "needs_latex": needs_latex,
            "total_latex_blocks": total_latex_blocks,
            "by_subject": by_subject,
            "existing_batch_files": len(existing_batches),
            "batch_dir": str(batch_dir),
        }

    def test_mathml(self, count: int = 3) -> Dict[str, Any]:
        """Test MathML conversion on sample questions with LaTeX blocks.

        Finds questions containing LaTeX, converts to MathML using latex2mathml,
        and returns both the original text and the converted version with counts.
        """
        if not HAS_LATEX2MATHML:
            self.log("warning", "latex2mathml not installed --run: pip install latex2mathml")
            return {
                "error": "latex2mathml library not installed",
                "mathml_available": False,
                "samples": [],
                "message": "Install latex2mathml: pip install latex2mathml",
            }

        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata to test"}

        # Find questions with LaTeX
        samples = []
        total_success = 0
        total_fail = 0

        for entry in entries:
            if len(samples) >= count:
                break
            qtype = entry.get("questionType", "")
            qdata = entry.get(qtype, {})
            html = qdata.get("question", "")
            latex_count = self._count_latex_blocks(html)
            if latex_count > 0:
                converted_html, s_count, f_count = self._convert_latex_to_mathml(html)
                total_success += s_count
                total_fail += f_count

                samples.append({
                    "qid": entry.get("qid", "?"),
                    "latex_blocks": latex_count,
                    "success_count": s_count,
                    "fail_count": f_count,
                    "original_text": html[:300] + "..." if len(html) > 300 else html,
                    "mathml_result": converted_html,
                })

        self.log("info", f"Tested MathML on {len(samples)} samples: {total_success} converted, {total_fail} failed")
        return {
            "samples": samples,
            "mathml_available": True,
            "total_success": total_success,
            "total_fail": total_fail,
            "message": f"Converted {total_success} LaTeX blocks, {total_fail} failures across {len(samples)} samples",
        }

    def convert_all_mathml(self) -> Dict[str, Any]:
        """Convert ALL LaTeX blocks to MathML across all metadata files.

        Processes every question, applies successful conversions to metadata in-place,
        and tracks which questions still have unconverted LaTeX (failures).
        These failures are then candidates for the OpenAI LaTeX batch.

        Returns dict with counts + list of failed QIDs for batch processing.
        """
        if not HAS_LATEX2MATHML:
            self.log("warning", "latex2mathml not installed --cannot convert")
            return {"error": "latex2mathml library not installed", "failed_qids": []}

        meta_dir = self._get_metadata_dir()
        if not meta_dir.exists():
            return {"error": "No metadata_v2 files found", "failed_qids": []}

        total_questions = 0
        questions_with_latex = 0
        total_success = 0
        total_fail = 0
        fully_converted = 0
        partially_converted = 0
        failed_qids = []
        files_modified = 0

        for f in sorted(meta_dir.glob("*_metadata_v2.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, list):
                    continue
            except Exception:
                continue

            file_modified = False
            for entry in data:
                total_questions += 1
                qid = entry.get("qid", "?")
                qtype = entry.get("questionType", "")
                qdata = entry.get(qtype, {})
                if not qdata:
                    continue

                entry_success = 0
                entry_fail = 0

                # Convert both question and solution fields
                for field in ["question", "solution"]:
                    html = qdata.get(field, "")
                    if not html:
                        continue

                    # Step A: Convert LaTeX tabular blocks to HTML tables FIRST
                    # (these can't go through latex2mathml)
                    html, tables_converted = _convert_tabular_blocks(html)
                    if tables_converted > 0:
                        qdata[field] = html
                        file_modified = True
                        entry_success += tables_converted

                    # Step B: Convert remaining LaTeX to MathML
                    latex_count = self._count_latex_blocks(html)
                    if latex_count > 0:
                        converted, s, f_cnt = self._convert_latex_to_mathml(html)
                        entry_success += s
                        entry_fail += f_cnt
                        if s > 0:
                            qdata[field] = converted
                            file_modified = True

                if entry_success > 0 or entry_fail > 0:
                    questions_with_latex += 1
                    total_success += entry_success
                    total_fail += entry_fail

                    if entry_fail > 0:
                        failed_qids.append(qid)
                        if entry_success > 0:
                            partially_converted += 1
                    else:
                        fully_converted += 1

            if file_modified:
                # Backup before writing
                self.backup(f)
                with open(f, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                files_modified += 1

        self.log("success",
                 f"MathML conversion complete: {total_success} blocks converted, {total_fail} failed "
                 f"across {questions_with_latex} questions ({fully_converted} fully converted, "
                 f"{len(failed_qids)} need batch LaTeX)")

        # Save failed QIDs for batch building
        failed_path = self._get_batch_dir() / "mathml_failed_qids.json"
        with open(failed_path, "w", encoding="utf-8") as fh:
            json.dump(failed_qids, fh)

        self.ledger_set("mathml_conversion", {
            "status": "completed",
            "total_questions": total_questions,
            "with_latex": questions_with_latex,
            "success": total_success,
            "fail": total_fail,
            "failed_qids": failed_qids,
            "latex_batch_needed": len(failed_qids) > 0,
        })
        self.save_checkpoint("mathml_converted", {
            "total_success": total_success,
            "total_fail": total_fail,
            "failed_qids_count": len(failed_qids),
        })

        return {
            "total_questions": total_questions,
            "questions_with_latex": questions_with_latex,
            "total_success": total_success,
            "total_fail": total_fail,
            "fully_converted": fully_converted,
            "partially_converted": partially_converted,
            "failed_qids_count": len(failed_qids),
            "files_modified": files_modified,
            "failed_qids": failed_qids[:20],  # First 20 for display
        }

    def build_batch(self) -> Dict[str, Any]:
        """Build JSONL batch files for tagging and LaTeX conversion.

        LaTeX batch only includes questions where local MathML conversion failed
        (from convert_all_mathml). If no failed QIDs file exists, falls back to
        building for ALL questions with LaTeX blocks.
        """
        entries = self._load_all_metadata()
        if not entries:
            return {"error": "No metadata_v2 files found"}

        batch_dir = self._get_batch_dir()

        # Load failed QIDs from MathML conversion (if available)
        failed_qids_path = batch_dir / "mathml_failed_qids.json"
        failed_qids_set = set()
        if failed_qids_path.exists():
            try:
                with open(failed_qids_path, "r", encoding="utf-8") as fh:
                    failed_qids_set = set(json.load(fh))
                self.log("info", f"Loaded {len(failed_qids_set)} failed QIDs from MathML conversion --only these will get LaTeX batch")
            except Exception:
                pass

        tagging_lines = 0
        latex_lines = 0

        tagging_path = batch_dir / "tagging_batch.jsonl"
        latex_path = batch_dir / "latex_batch.jsonl"

        with open(tagging_path, "w", encoding="utf-8") as tag_f, \
             open(latex_path, "w", encoding="utf-8") as latex_f:

            # Pre-load FAISS indexes and OpenAI client for embedding retrieval
            subjects_in_batch = set()
            tagging_entries = []

            for entry in entries:
                qid = entry.get("qid", "unknown")
                subject = entry.get("subject", "Unknown")
                qtype = entry.get("questionType", "")
                qdata = entry.get(qtype, {})
                question_text = qdata.get("question", "")
                solution_text = qdata.get("solution", "")
                answer = entry.get("answer", entry.get(qtype, {}).get("answer", ""))

                if entry.get("chapter") == "TBD":
                    tagging_entries.append({
                        "qid": qid, "subject": subject,
                        "question": question_text, "solution": solution_text,
                        "answer": answer,
                    })
                    subjects_in_batch.add(subject)

            # Load FAISS indexes for all subjects needed
            faiss_loaded = {}
            for subj in subjects_in_batch:
                try:
                    (index, chunk_data), cache = _load_faiss_for_subject(subj)
                    faiss_loaded[subj] = (index, chunk_data, cache)
                    self.log("info", f"FAISS loaded for {subj}: {len(chunk_data)} chunks")
                except FileNotFoundError as e:
                    self.log("warning", f"FAISS not available for {subj}: {e}")

            # Get embeddings in batches for FAISS retrieval
            if tagging_entries and faiss_loaded:
                from app.config import settings
                from openai import OpenAI
                embed_client = OpenAI(api_key=settings.openai_api_key)

                # Build query texts for embedding
                query_texts = []
                for te in tagging_entries:
                    q_plain = _strip_tags(te["question"])[:1500]
                    s_plain = _strip_tags(te["solution"])[:1500]
                    query_texts.append(f"{q_plain}\n{s_plain}")

                self.log("info", f"Computing embeddings for {len(query_texts)} questions...")
                # Batch embeddings (OpenAI supports up to 2048 per call)
                all_embeddings = []
                batch_size = 100
                for i in range(0, len(query_texts), batch_size):
                    batch_texts = query_texts[i:i+batch_size]
                    batch_embeddings = _get_embeddings(batch_texts, embed_client)
                    all_embeddings.extend(batch_embeddings)
                self.log("info", f"Embeddings computed: {len(all_embeddings)}")

                # Build tagging requests with FAISS chunks
                chunk_id_map = {}  # qid -> list of chunk_ids (for resolution later)
                for idx, te in enumerate(tagging_entries):
                    subj = te["subject"]
                    if subj not in faiss_loaded:
                        # Fallback: no FAISS for this subject, skip tagging
                        self.log("warning", f"Skipping tag for {te['qid']} --no FAISS for {subj}")
                        continue

                    index, chunk_data, cache = faiss_loaded[subj]
                    chunk_ids = _retrieve_chunk_ids(all_embeddings[idx], index, chunk_data, cache)
                    chunk_id_map[te["qid"]] = chunk_ids

                    if chunk_ids:
                        chunks_block = f"\n\nAVAILABLE CHUNKS (select by ID only):\n{cache.format_chunks_for_prompt(chunk_ids)}"
                    else:
                        chunks_block = "\n\nAVAILABLE CHUNKS: NONE\nNo chunks available. Return empty array."

                    user_content = (
                        f"{TAGGER_INSTRUCTIONS}\n\n{TAGGING_OUTPUT_SPEC}\n\n"
                        f"ANSWER: {te['answer']}\n"
                        f"{chunks_block}\n\n"
                        f"QUESTION:\n{_strip_tags(te['question'])[:1500]}\n\n"
                        f"SOLUTION:\n{_strip_tags(te['solution'])[:1500]}\n\n"
                        "Return ONLY the JSON object with 'selected_chunk_ids'."
                    )

                    # gpt-5-mini is a reasoning model: temperature must be 1 (default),
                    # use reasoning_effort="low" for deterministic structured output
                    tag_request = {
                        "custom_id": f"tag-{te['qid']}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-5-mini",
                            "messages": [
                                {"role": "system", "content": "You are an expert JEE/NEET exam content tagger. Return ONLY valid JSON with selected_chunk_ids."},
                                {"role": "user", "content": user_content},
                            ],
                            "max_completion_tokens": 500,
                            "reasoning_effort": "low",
                            "response_format": {"type": "json_object"},
                        },
                    }
                    tag_f.write(json.dumps(tag_request) + "\n")
                    tagging_lines += 1

                # Save chunk_id_map for resolution during download
                chunk_map_path = batch_dir / "chunk_id_map.json"
                # Save full cache data for chunk ID resolution
                cache_data = {}
                for subj, (_, _, cache) in faiss_loaded.items():
                    for cid, chunk in cache.id_to_chunk.items():
                        cache_data[cid] = chunk
                with open(chunk_map_path, "w", encoding="utf-8") as f:
                    json.dump({"question_chunks": chunk_id_map, "chunk_lookup": cache_data},
                              f, ensure_ascii=False)
                self.log("info", f"Saved chunk ID map: {len(chunk_id_map)} questions, {len(cache_data)} chunks")

            elif tagging_entries and not faiss_loaded:
                self.log("error", "No FAISS indexes available --tagging cannot proceed")

            # LaTeX conversion loop --separate from tagging
            for entry in entries:
                qid = entry.get("qid", "unknown")
                qtype = entry.get("questionType", "")
                qdata = entry.get(qtype, {})
                question_text = qdata.get("question", "")

                # LaTeX conversion --ONLY for QIDs that failed local MathML conversion
                # If no failed_qids file exists, fall back to checking for LaTeX blocks
                needs_latex = False
                if failed_qids_set:
                    needs_latex = qid in failed_qids_set
                else:
                    needs_latex = self._count_latex_blocks(question_text) > 0

                if needs_latex:
                    # Send question + solution as structured JSON (production format)
                    solution_text = qdata.get("solution", "")
                    input_json = json.dumps({
                        "question": _one_line_html(question_text),
                        "solution": _one_line_html(solution_text),
                    }, ensure_ascii=False)

                    user_content = (
                        f"{LATEX_GUARD_SPEC}\n\n"
                        f"{LATEX_OUTPUT_SPEC}\n\n"
                        f"INPUT JSON:\n{input_json}\n\n"
                        "Process the above input. Return ONLY the JSON output with properly delimited LaTeX."
                    )

                    # gpt-5-mini: temperature must be 1, use reasoning_effort for control
                    latex_request = {
                        "custom_id": f"latex-{qid}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-5-mini",
                            "messages": [
                                {"role": "system", "content": LATEX_SYSTEM_INSTRUCTION},
                                {"role": "user", "content": user_content},
                            ],
                            "max_completion_tokens": 6000,
                            "reasoning_effort": "low",
                            "response_format": {"type": "json_object"},
                        },
                    }
                    latex_f.write(json.dumps(latex_request) + "\n")
                    latex_lines += 1

        self.log("success", f"Built batch files: {tagging_lines} tagging + {latex_lines} LaTeX requests")
        if failed_qids_set and latex_lines == 0:
            self.log("info", "All LaTeX was converted locally via MathML --no LaTeX batch needed!")

        self.save_checkpoint("batch_built", {
            "tagging_lines": tagging_lines,
            "latex_lines": latex_lines,
        })

        return {
            "tagging_file": str(tagging_path),
            "latex_file": str(latex_path),
            "tagging_requests": tagging_lines,
            "latex_requests": latex_lines,
            "total_requests": tagging_lines + latex_lines,
        }

    def check_credentials(self) -> Dict[str, Any]:
        """Check if OpenAI API key is configured."""
        from app.config import settings
        has_key = bool(settings.openai_api_key)
        return {
            "configured": has_key,
            "openai_api_key": "sk-***" + settings.openai_api_key[-4:] if has_key else "(not set)",
            "message": "OpenAI API key configured" if has_key else "OpenAI API key not configured. Set OPENAI_API_KEY in .env",
        }

    def _get_openai_client(self):
        """Get an OpenAI client instance."""
        from app.config import settings
        from openai import OpenAI
        return OpenAI(api_key=settings.openai_api_key)

    def _get_batch_status_path(self) -> Path:
        return self._get_batch_dir() / "batch_status.json"

    def _load_batch_status(self) -> Dict[str, Any]:
        path = self._get_batch_status_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_batch_status(self, status: Dict[str, Any]):
        with open(self._get_batch_status_path(), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)

    def _submit_openai_file(self, client, file_path: Path, batch_type: str,
                            batch_status: Dict) -> Dict[str, Any]:
        """Upload and submit a single JSONL file to OpenAI, with auto-split if >190MB."""
        chunk_paths = self.split_jsonl_by_size(file_path)
        is_split = len(chunk_paths) > 1

        if is_split:
            self.log("info", f"{batch_type} batch is {file_path.stat().st_size // (1024*1024)}MB"
                     f" -- splitting into {len(chunk_paths)} chunks")

        chunk_batches = []
        for idx, cp in enumerate(chunk_paths):
            label = f"{batch_type}" if not is_split else f"{batch_type} chunk {idx+1}/{len(chunk_paths)}"
            self.log("info", f"Uploading {label} ({cp.stat().st_size // (1024*1024)}MB)...")
            with open(cp, "rb") as f:
                file_obj = client.files.create(file=f, purpose="batch")
            self.log("info", f"File uploaded: {file_obj.id}")

            batch = client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"session_id": self.session_id, "type": batch_type,
                          "chunk": str(idx)},
            )
            chunk_batches.append({"batch_id": batch.id, "file_id": file_obj.id,
                                  "chunk_file": cp.name})
            self.log("success", f"{label} submitted: {batch.id}")

        # Save to batch_status (backward-compatible flat fields for single batch)
        primary_id = chunk_batches[0]["batch_id"]
        primary_fid = chunk_batches[0]["file_id"]
        batch_status[f"{batch_type}_batch_id"] = primary_id
        batch_status[f"{batch_type}_file_id"] = primary_fid
        if is_split:
            batch_status[f"{batch_type}_split"] = True
            batch_status[f"{batch_type}_chunks"] = chunk_batches

        return {"batch_id": primary_id, "status": "submitted",
                "split": is_split, "chunk_count": len(chunk_batches)}

    def submit_batch(self) -> Dict[str, Any]:
        """Submit batch files to OpenAI Batch API."""
        creds = self.check_credentials()
        if not creds["configured"]:
            self.log("error", "OpenAI API key not configured")
            return {
                "error": "OpenAI API key not configured",
                "message": "Set OPENAI_API_KEY in .env file",
            }

        batch_dir = self._get_batch_dir()
        tagging_path = batch_dir / "tagging_batch.jsonl"
        latex_path = batch_dir / "latex_batch.jsonl"

        if not tagging_path.exists() and not latex_path.exists():
            return {"error": "No batch files found --run Build Batch first"}

        client = self._get_openai_client()
        batch_status = self._load_batch_status()
        results = {}

        # Submit tagging batch (with split support for >190MB files)
        if tagging_path.exists() and tagging_path.stat().st_size > 0:
            if batch_status.get("tagging_batch_id"):
                self.log("info", f"Tagging batch already submitted: {batch_status['tagging_batch_id']}")
                results["tagging"] = {"batch_id": batch_status["tagging_batch_id"], "status": "already_submitted"}
            else:
                try:
                    result = self._submit_openai_file(client, tagging_path, "tagging", batch_status)
                    results["tagging"] = result
                except Exception as e:
                    self.log("error", f"Tagging batch submit failed: {e}")
                    results["tagging"] = {"error": str(e)}

        # Submit LaTeX batch (with split support)
        if latex_path.exists() and latex_path.stat().st_size > 0:
            if batch_status.get("latex_batch_id"):
                self.log("info", f"LaTeX batch already submitted: {batch_status['latex_batch_id']}")
                results["latex"] = {"batch_id": batch_status["latex_batch_id"], "status": "already_submitted"}
            else:
                try:
                    result = self._submit_openai_file(client, latex_path, "latex", batch_status)
                    results["latex"] = result
                except Exception as e:
                    self.log("error", f"LaTeX batch submit failed: {e}")
                    results["latex"] = {"error": str(e)}

        self._save_batch_status(batch_status)

        # Update ledger with submission details
        self.ledger_set("batch_submit", {
            "status": "submitted",
            "tagging_batch_id": batch_status.get("tagging_batch_id"),
            "tagging_file_id": batch_status.get("tagging_file_id"),
            "latex_batch_id": batch_status.get("latex_batch_id"),
            "latex_file_id": batch_status.get("latex_file_id"),
            "api": "openai",
        })

        return results

    def poll_status(self) -> Dict[str, Any]:
        """Poll batch job status from OpenAI API."""
        batch_status = self._load_batch_status()

        if not batch_status.get("tagging_batch_id") and not batch_status.get("latex_batch_id"):
            return {
                "tagging_batch": {"id": None, "status": "not_submitted", "progress": 0},
                "latex_batch": {"id": None, "status": "not_submitted", "progress": 0},
                "message": "No batches submitted yet --run Submit first",
            }

        client = self._get_openai_client()
        result = {}

        for batch_type in ["tagging", "latex"]:
            batch_id = batch_status.get(f"{batch_type}_batch_id")
            if not batch_id:
                result[f"{batch_type}_batch"] = {"id": None, "status": "not_submitted", "progress": 0}
                continue

            try:
                batch = client.batches.retrieve(batch_id)
                total = batch.request_counts.total if batch.request_counts else 0
                completed = batch.request_counts.completed if batch.request_counts else 0
                failed = batch.request_counts.failed if batch.request_counts else 0
                progress = int((completed / total * 100)) if total > 0 else 0

                info = {
                    "id": batch_id,
                    "status": batch.status,
                    "progress": progress,
                    "total": total,
                    "completed": completed,
                    "failed": failed,
                }

                # Save output file ID if completed
                if batch.status == "completed" and batch.output_file_id:
                    batch_status[f"{batch_type}_output_file_id"] = batch.output_file_id
                    info["output_file_id"] = batch.output_file_id

                if batch.status == "failed":
                    info["error"] = str(batch.errors) if batch.errors else "Unknown error"

                result[f"{batch_type}_batch"] = info
                self.log("info", f"{batch_type.title()} batch: {batch.status} ({progress}% --{completed}/{total})")
            except Exception as e:
                result[f"{batch_type}_batch"] = {"id": batch_id, "status": "error", "error": str(e), "progress": 0}
                self.log("error", f"{batch_type.title()} batch poll error: {e}")

        self._save_batch_status(batch_status)

        # Check if both complete
        tag_done = result.get("tagging_batch", {}).get("status") in ("completed", "not_submitted")
        latex_done = result.get("latex_batch", {}).get("status") in ("completed", "not_submitted")
        result["all_complete"] = tag_done and latex_done

        # Update ledger with poll results
        self.ledger_set("batch_poll", {
            "status": "completed" if result["all_complete"] else "polling",
            "tagging_status": result.get("tagging_batch", {}).get("status", "unknown"),
            "latex_status": result.get("latex_batch", {}).get("status", "unknown"),
            "tagging_output_file_id": batch_status.get("tagging_output_file_id"),
            "latex_output_file_id": batch_status.get("latex_output_file_id"),
        })

        return result

    def cancel_batches(self) -> Dict[str, Any]:
        """Cancel active OpenAI batches for this session."""
        batch_status = self._load_batch_status()
        cancelled = []
        errors = []

        if not batch_status.get("tagging_batch_id") and not batch_status.get("latex_batch_id"):
            return {"status": "nothing_to_cancel", "message": "No active batches found"}

        client = self._get_openai_client()

        for batch_type in ["tagging", "latex"]:
            batch_id = batch_status.get(f"{batch_type}_batch_id")
            if not batch_id:
                continue
            try:
                client.batches.cancel(batch_id)
                cancelled.append(batch_type)
                self.log("info", f"Cancelled {batch_type} batch: {batch_id}")
            except Exception as e:
                err_str = str(e)
                # Already completed/cancelled batches can't be cancelled --that's fine
                if "Cannot cancel" in err_str or "already" in err_str.lower():
                    self.log("info", f"{batch_type} batch already finished: {batch_id}")
                else:
                    errors.append(f"{batch_type}: {err_str}")
                    self.log("error", f"Failed to cancel {batch_type} batch: {e}")

        batch_status["status"] = "cancelled"
        self._save_batch_status(batch_status)

        return {
            "status": "cancelled" if cancelled else "stop_requested",
            "cancelled": cancelled,
            "errors": errors,
        }

    def download_results(self) -> Dict[str, Any]:
        """Download batch results from OpenAI and merge into metadata_v2."""
        batch_status = self._load_batch_status()
        client = self._get_openai_client()
        results_dir = self._get_batch_results_dir()

        tags_merged = 0
        latex_merged = 0
        mathml_converted = 0
        errors = []
        empty_tag_qids = []      # QIDs where tagging returned empty/TBD
        still_failed_qids = []   # QIDs where MathML still fails after batch

        # Download and parse results for each batch type
        for batch_type in ["tagging", "latex"]:
            output_file_id = batch_status.get(f"{batch_type}_output_file_id")
            if not output_file_id:
                self.log("info", f"No output file for {batch_type} batch --skipping")
                continue

            try:
                self.log("info", f"Downloading {batch_type} results from OpenAI...")
                content = client.files.content(output_file_id)
                result_path = results_dir / f"{batch_type}_results.jsonl"
                result_path.write_bytes(content.content)
                self.log("success", f"Downloaded {batch_type} results: {result_path.name}")

                # Parse results
                results = []
                with open(result_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                results.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

                self.log("info", f"Parsed {len(results)} {batch_type} results")

                # Build lookup by custom_id
                result_lookup = {}
                for r in results:
                    custom_id = r.get("custom_id", "")
                    response = r.get("response", {})
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if choices:
                        content_text = choices[0].get("message", {}).get("content", "")
                        # Extract QID from custom_id (e.g., "tag-PHY_TEST01_Q1" -> "PHY_TEST01_Q1")
                        prefix = "tag-" if batch_type == "tagging" else "latex-"
                        qid = custom_id[len(prefix):] if custom_id.startswith(prefix) else custom_id
                        result_lookup[qid] = content_text

                # Merge into metadata_v2 files
                meta_dir = self._get_metadata_dir()

                # Load chunk lookup ONCE before iterating files (used for tag resolution)
                chunk_lookup = {}
                if batch_type == "tagging":
                    chunk_map_path = self._get_batch_dir() / "chunk_id_map.json"
                    if chunk_map_path.exists():
                        try:
                            with open(chunk_map_path, "r", encoding="utf-8") as cl:
                                cm = json.load(cl)
                            chunk_lookup = cm.get("chunk_lookup", {})
                            self.log("info", f"Loaded chunk lookup: {len(chunk_lookup)} chunks")
                        except Exception as e:
                            self.log("warning", f"Failed to load chunk_id_map.json: {e}")

                for f in sorted(meta_dir.glob("*_metadata_v2.json")):
                    try:
                        with open(f, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        if not isinstance(data, list):
                            continue

                        modified = False
                        for entry in data:
                            qid = entry.get("qid", "")
                            if qid not in result_lookup:
                                continue

                            content_text = result_lookup[qid]

                            if batch_type == "tagging":
                                try:
                                    tag_data = json.loads(content_text)
                                    # New format: resolve chunk IDs to tags
                                    selected_ids = tag_data.get("selected_chunk_ids", [])
                                    if selected_ids and chunk_lookup:
                                        # Use first chunk ID for primary tag
                                        first_chunk = chunk_lookup.get(selected_ids[0], {})
                                        if first_chunk:
                                            entry["chapter"] = first_chunk.get("chapter", "TBD")
                                            entry["concept"] = first_chunk.get("concept", "TBD")
                                            entry["subConcept"] = first_chunk.get("subconcept", "TBD")
                                            # Update subject from chunk (Biology -> Botany/Zoology)
                                            chunk_subject = first_chunk.get("subject", "")
                                            if chunk_subject:
                                                entry["subject"] = chunk_subject

                                            # Build tagSubConcept array from ALL resolved chunks
                                            tag_sub_concepts = []
                                            for cid in selected_ids[:3]:
                                                chunk = chunk_lookup.get(cid, {})
                                                if chunk:
                                                    tag_sub_concepts.append({
                                                        "subject": chunk.get("subject", ""),
                                                        "chapter": chunk.get("chapter", ""),
                                                        "concept": chunk.get("concept", ""),
                                                        "subConcept": chunk.get("subconcept", ""),
                                                    })
                                            entry["tagSubConcept"] = tag_sub_concepts

                                            tags_merged += 1
                                            modified = True
                                        else:
                                            empty_tag_qids.append(qid)
                                    elif tag_data.get("chapter"):
                                        # Fallback: old format with direct chapter/concept/subConcept
                                        entry["chapter"] = tag_data["chapter"]
                                        if tag_data.get("concept"):
                                            entry["concept"] = tag_data["concept"]
                                        if tag_data.get("subConcept"):
                                            entry["subConcept"] = tag_data["subConcept"]
                                        tags_merged += 1
                                        modified = True
                                        if tag_data["chapter"] == "TBD":
                                            empty_tag_qids.append(qid)
                                    else:
                                        empty_tag_qids.append(qid)
                                except json.JSONDecodeError:
                                    errors.append(f"Invalid JSON for tag-{qid}")
                                    empty_tag_qids.append(qid)
                            else:  # latex
                                qtype = entry.get("questionType", "")
                                qdata = entry.setdefault(qtype, {})
                                if content_text and len(content_text) > 10:
                                    # New format: JSON with question/solution keys
                                    q_part = ""
                                    s_part = ""
                                    try:
                                        latex_json = json.loads(content_text)
                                        q_part = latex_json.get("question", "")
                                        s_part = latex_json.get("solution", "")
                                    except json.JSONDecodeError:
                                        # Fallback: old QUESTION:/SOLUTION: split format
                                        q_part = content_text
                                        if "QUESTION:" in content_text and "SOLUTION:" in content_text:
                                            parts = content_text.split("SOLUTION:", 1)
                                            q_part = parts[0].replace("QUESTION:", "", 1).strip()
                                            s_part = parts[1].strip()

                                    if q_part:
                                        qdata["question"] = q_part
                                    latex_merged += 1
                                    modified = True

                                    # Also do MathML conversion locally on question
                                    converted_q, sq, fq = self._convert_latex_to_mathml(q_part)
                                    if sq > 0:
                                        qdata["question"] = converted_q
                                        mathml_converted += sq
                                    if fq > 0:
                                        still_failed_qids.append(qid)

                                    # Process solution if we got one back
                                    if s_part:
                                        qdata["solution"] = s_part
                                        converted_s, ss, fs = self._convert_latex_to_mathml(s_part)
                                        if ss > 0:
                                            qdata["solution"] = converted_s
                                            mathml_converted += ss
                                        if fs > 0 and qid not in still_failed_qids:
                                            still_failed_qids.append(qid)

                        if modified:
                            with open(f, "w", encoding="utf-8") as fh:
                                json.dump(data, fh, indent=2, ensure_ascii=False)

                    except Exception as e:
                        errors.append(f"Error processing {f.name}: {e}")

            except Exception as e:
                self.log("error", f"Download {batch_type} results failed: {e}")
                errors.append(str(e))

        self.log("success", f"Merged: {tags_merged} tags + {latex_merged} LaTeX + {mathml_converted} MathML conversions")
        if errors:
            self.log("warning", f"{len(errors)} errors during merge")

        # Quarantine MathML failures and empty tags
        from app.db import quarantine_question
        quarantined_count = 0

        if still_failed_qids:
            self.log("warning", f"{len(still_failed_qids)} questions still have MathML failures after batch --quarantining")
            for qid in still_failed_qids:
                quarantine_question(
                    self.session_id, qid, self.stage_name,
                    reason="mathml_failure",
                    data={"source": "stage7_mathml_still_failed_after_batch"},
                )
                quarantined_count += 1

        if empty_tag_qids:
            self.log("warning", f"{len(empty_tag_qids)} questions have empty/TBD tags after batch --quarantining")
            for qid in empty_tag_qids:
                quarantine_question(
                    self.session_id, qid, self.stage_name,
                    reason="empty_tags",
                    data={"source": "stage7_tagging_empty_chapter"},
                )
                quarantined_count += 1

        # Save quarantine info for downstream stages
        quarantine_path = self._get_batch_dir() / "stage7_quarantine.json"
        quarantine_data = {
            "mathml_failures": still_failed_qids,
            "empty_tags": empty_tag_qids,
            "total_quarantined": quarantined_count,
        }
        with open(quarantine_path, "w", encoding="utf-8") as qf:
            json.dump(quarantine_data, qf, indent=2)

        # GUARD: Warn if 0 results were merged --don't mark complete blindly
        if tags_merged == 0 and latex_merged == 0:
            self.log("error", "Download produced 0 merged results --NOT marking complete")
            return {
                "error": "Batch download returned 0 results to merge. "
                         "Check batch status or retry download.",
                "tags_merged": 0, "latex_merged": 0,
            }

        # Update ledger: download + merge complete
        results_dir = self._get_batch_results_dir()
        tagging_file = str(results_dir / "tagging_results.jsonl")
        latex_file = str(results_dir / "latex_results.jsonl")
        self.ledger_set("batch_download", {
            "status": "completed",
            "tagging_file": tagging_file,
            "latex_file": latex_file,
            "tagging_verified": self.ledger_verify_file("batch_results/tagging_results.jsonl"),
            "latex_verified": self.ledger_verify_file("batch_results/latex_results.jsonl"),
        })
        self.ledger_set("merge", {
            "status": "completed",
            "tags_merged": tags_merged,
            "latex_merged": latex_merged,
            "mathml_converted": mathml_converted,
            "quarantined": quarantined_count,
        })
        self.save_checkpoint("completed", {
            "tags_merged": tags_merged,
            "latex_merged": latex_merged,
            "mathml_converted": mathml_converted,
            "quarantined": quarantined_count,
        })

        return {
            "tags_merged": tags_merged,
            "latex_merged": latex_merged,
            "mathml_converted": mathml_converted,
            "errors": errors[:10] if errors else [],
            "quarantined": quarantined_count,
            "mathml_still_failed": len(still_failed_qids),
            "empty_tags": len(empty_tag_qids),
        }

    def dry_run(self, **kwargs) -> Dict[str, Any]:
        """Preview what batches would be built."""
        scan = self.scan()
        creds = self.check_credentials()
        return {**scan, "credentials": creds}

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the full batch pipeline: build -> submit -> poll -> download."""
        # In real implementation, this would orchestrate the full flow
        # For now, just build the batch files
        return self.build_batch()
