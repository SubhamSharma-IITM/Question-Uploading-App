# Question Uploading App - Claude Code Instructions

## Project Overview
10-stage wizard web application for uploading exam questions (JEE/NEET) to the Acadza database. Processes PDFs through OCR, LLM extraction, cleaning, metadata conversion, watermark removal (CV2 inpainting + adaptive threshold + CLAHE), SMILES chemistry rendering, image CDN upload, LaTeX↔MathML bidirectional conversion, FAISS semantic deduplication (157K vector DB), 3-model QA verification (Vertex AI + OpenAI batch APIs), and final database upload with FAISS absorption.

## How to Run
```bash
cd "Question Uploading App"
python -m uvicorn app.main:app --reload --port 8000
```
Then open http://localhost:8000 in a browser.

## Project Structure
```
app/
  main.py           - FastAPI entry point, mounts all routers
  config.py         - Settings via pydantic-settings + .env
  db.py             - SQLite session/checkpoint/quarantine DB
  sse.py            - Server-Sent Events manager
  models.py         - Pydantic response models
  pipeline/         - Stage processing logic (stage1 through stage10)
    _mathml_to_latex.py - MathML→LaTeX converter (used by Stage 8 embeddings)
  routes/           - FastAPI route handlers per stage
  templates/        - Jinja2 HTML templates (wizard UI)
  static/           - CSS/JS assets
data/
  allque030326.json       - 162K question JSONL database (Git LFS)
  db_index_cache/         - Global FAISS index cache (Git LFS, 965MB)
    db_reference/
      faiss.index         - 157K vectors, text-embedding-3-small, MathML→LaTeX
      metadata.json       - OID map + text snippets for preview
  sessions/               - Per-session runtime data (not in git)
```

## Key Rules
- **Gemini model**: ALWAYS use `gemini-3-flash-preview` via `google.genai` SDK (NOT deprecated `google.generativeai`)
- **OpenAI models**: `gpt-5-mini` (Pass 2), `gpt-5.1` (Pass 3)
- **Vertex AI**: PRIMARY for Gemini batches (paid quota). AI Studio is fallback only.
- **Never change model IDs** without explicit user permission
- **Windows cp1252**: Use safe print with ascii fallback for console output
- **Error codes**: error=1 (3/3 approved), error=2 (2/3 approved), error=3/4 (rejected/quarantined)
- **Protected duplicate flags**: Only `isPrevious`, `positivePick`, `negativePick`, `isSolvedExample` lock the quarantine checkbox. Other flags (isBoard, isNCERT, etc.) are informational only.
- **MathML in embeddings**: Use `_mathml_to_latex.py` to convert MathML→LaTeX before generating embeddings. NEVER replace MathML with generic `MATH` token.
- **QA prompt error types**: `malformed_mathml` = broken MathML TAGS only. `math_logic_error` = wrong math content/logic. Keep these separate.
- **FAISS index**: Cached globally at `data/db_index_cache/db_reference/`. Rebuilt when `_strip_html_keep_images` logic changes. ~157K vectors, ~923MB.

## Environment Variables (.env)
```
# Required
GEMINI_API_KEY=...
OPENAI_API_KEY=...
MATHPIX_APP_ID=...
MATHPIX_APP_KEY=...

# Vertex AI (PRIMARY for Stage 9 QA Pass 1)
# WARNING: VERTEX_AI_CREDENTIALS must be an ABSOLUTE path to the service account JSON.
# This path is MACHINE-SPECIFIC — update it when deploying to a new machine.
VERTEX_AI_CREDENTIALS=/path/to/service-account.json
VERTEX_AI_PROJECT=your-project-id
VERTEX_AI_LOCATION=global
VERTEX_AI_BUCKET=your-gcs-bucket

# AWS S3 (Stage 6 CDN upload)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=...

# Optional
ANTHROPIC_API_KEY=...
```

## Setup for New System
1. Install Python 3.11+
2. `pip install -r requirements.txt`
3. Install Playwright: `playwright install chromium` (if using PDF generator)
4. Create `.env` file with API keys (see .env.example)
5. `python -m uvicorn app.main:app --reload --port 8000`
6. FAISS index ships with repo via Git LFS — no rebuild needed on fresh clone

## Stage Pipeline
```
Stage 1:  PDF Upload → MathPIX OCR → .mmd files
Stage 2:  LLM Extraction (Gemini) → structured JSON per question
Stage 3:  Text Cleaning → normalize HTML, strip artifacts
Stage 4:  Metadata V2 → option normalization (a)→(A), image refs, type detection
Stage 5:  Watermark Removal (CV2 inpainting) + SMILES→SVG rendering
Stage 6:  Image CDN Upload (S3) + SVG→PNG conversion
Stage 7:  LaTeX→MathML batch conversion + concept tagging (OpenAI batch)
Stage 8:  Duplicate Check — FAISS semantic search (self + cross-DB 157K)
Stage 9:  QA Verification — 3-model voting (Gemini/GPT-5-mini/GPT-5.1 batch)
Stage 10: Upload to Acadza DB API + local DB absorption + FAISS update
```

## Critical Implementation Details

### Vertex AI Credentials
`_ensure_vertex_credentials()` MUST be called before ANY `gcs_storage.Client()` or `genai.Client(vertexai=True)` creation. The method sets `GOOGLE_APPLICATION_CREDENTIALS` env var.

### Vertex AI Model Format
- **Vertex AI batch**: `model=model` (bare model ID, e.g. `gemini-3-flash-preview`)
- **Gemini AI Studio batch**: `model=f"models/{model}"` (with `models/` prefix)
- Do NOT mix these — Vertex AI rejects `models/` prefix, AI Studio requires it.

### Stage 7 Metadata Merge
Use `entry.setdefault(qtype, {})` NOT `entry.get(qtype, {})` when modifying nested question data. `.get()` returns a detached dict — changes are lost.

### Stage 8 Embeddings
- MathML→LaTeX via `app/pipeline/_mathml_to_latex.py` (forked from PROOF-READING_v2.0)
- Session embeddings cached to disk (`session_embeddings/vectors.npy`) — reused between self-check and cross-check
- DB OIDs (24-digit hex from `_id.$oid`) used as identifiers, not `qid`

### Stage 8 PYQ Auto-Skip
When `session_config.json` has `is_pyq: true` (set in Stage 2), Stage 8 auto-skips duplicate checking on page load. All PYQs go to DB regardless of similarity.

### Stage 9 Skip Path
Skip QA sets `error=2` on ALL metadata entries. No QA verification performed. Questions uploaded with error=Yes for manual review.

### CSV Naming Convention
All downloadable CSVs are prefixed with the sanitized session name:
`SessionName_upload_log.csv`, `SessionName_qa_3of3.csv`, `SessionName_pipeline_stats.csv`

### CSV View Link
Stage 10 CSVs use the uploaded MongoDB OID (24-digit hex) for View Link after successful upload. FAILED/QUARANTINED rows keep meta QID. Stage 9 CSVs use meta QID (pre-upload).

### Session Persistence & Ledger System
All state survives server restart -- SQLite checkpoints, disk files, batch IDs. Polling resumes automatically on page load via `DOMContentLoaded` handler.

**Ledger sub-steps** are stored in the SQLite checkpoint `data` JSON column under `sub_steps`. Each stage writes ledger entries AFTER operations succeed. On resume, the code reads the ledger, verifies local files exist, polls external APIs if needed, and continues from the last confirmed state.

### CRITICAL: Windows cp1252 Encoding Rules
1. **ALL file opens** for JSON/JSONL with potential Unicode content MUST use `encoding="utf-8"`. Omitting this on Windows defaults to cp1252, which silently crashes on Unicode chars (arrows, em-dashes, math symbols). The `except Exception: pass` pattern then hides the failure.
2. **ALL `self.log()` messages** MUST use ASCII-only characters. No em-dashes (`--`), no arrows (`->`), no checkmarks. Use `--` instead of `--`, `->` instead of arrows. The log chain (insert_log -> SSE -> console) can crash on cp1252 if non-ASCII chars are present.
3. **After `git pull`**: Always clear `__pycache__` before restarting the server. Stale `.pyc` bytecode files will ignore source changes. Run: `find app -name "__pycache__" -exec rm -rf {} +` then restart uvicorn.

### Deployment on New Machine (After git pull)
```bash
# MANDATORY: Clear bytecode cache before first run
find app -name "__pycache__" -exec rm -rf {} +
# Then start server
python -m uvicorn app.main:app --reload --port 8000
```
The `--reload` flag watches source files, but stale `.pyc` from a previous run can override source changes. Always clear cache after pull.

### Stage 9 QA Pass Validation
- Ledger files (`quality_check/ledgers/qa_pass*_.json`) are the source of truth for pass completion
- Validation checks: file exists AND size > 10 bytes AND content is a non-empty JSON array
- `poll_passes()` reads ledger with `encoding="utf-8"` first; falls back to status file + API polling only if ledger is invalid
- Empty completion guard: download functions reject 0-result batches and do NOT mark as completed
- For Vertex AI: GCS files are cleaned after download, so re-download is impossible -- must re-submit if data is lost
- For OpenAI: `output_file_id` is preserved in status file, allowing re-download anytime

### OpenAI Batch Size Limit
OpenAI Batch API rejects files > 200MB. For large sessions (300+ questions with images), batch JSONL files can exceed this. Batch splitting with proper merge-on-download is required (see Stage 7 and Stage 9 implementations).

## Future Work (Pending)
- **Image-based question embeddings**: ~6K DB questions are image-only (OCR text via MathPIX exists but not embedded). Need to generate text embeddings from OCR output and add to FAISS index for cross-duplicate matching against image-based DB entries.
- **UI overhaul**: Auto dark/light mode via `prefers-color-scheme`, progress time estimates, session dashboard cards, cost tracker per session.
- **Celery/Redis job queue**: Replace `asyncio.to_thread` for horizontal scaling beyond 5 concurrent sessions.
- **Token cost dashboard**: Track API spend per session across Gemini/OpenAI/Vertex AI.
- **Model cascading**: Use cheaper models for obvious-pass questions, expensive models for edge cases.
