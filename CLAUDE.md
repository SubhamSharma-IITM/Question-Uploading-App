# Question Uploading App - Claude Code Instructions

## Project Overview
This is a 10-stage wizard web application for uploading exam questions (JEE/NEET) to the Acadza database. It processes PDFs through OCR, LLM extraction, cleaning, metadata conversion, watermarking, image CDN upload, LaTeX/tagging batch processing, duplicate checking, quality verification, and final database upload.

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
  routes/           - FastAPI route handlers per stage
  templates/        - Jinja2 HTML templates (wizard UI)
  static/           - CSS/JS assets
data/               - Runtime data (sessions, SQLite DB) - NOT in git
tests/              - Integration tests
```

## Key Rules
- **Gemini model**: ALWAYS use `gemini-3-flash-preview` via `google.genai` SDK (NOT deprecated `google.generativeai`)
- **OpenAI models**: `gpt-5-mini` (Pass 2), `gpt-5.1` (Pass 3)
- **Never change model IDs** without explicit user permission
- **Windows cp1252**: Use safe print with ascii fallback for console output
- **Error codes**: error=1 (3/3 approved), error=2 (2/3 approved), error=3/4 (rejected/quarantined)

## Environment Variables (.env)
```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
MATHPIX_APP_ID=...
MATHPIX_APP_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=...
```

## Setup for New System
1. Install Python 3.11+
2. `pip install -r requirements.txt`
3. Create `.env` file with API keys (see above)
4. `python -m uvicorn app.main:app --port 8000`
