# Acadza Question Uploading App

A 10-stage wizard web application for processing and uploading exam questions (JEE Mains, JEE Advanced, NEET) to the Acadza database.

---

## Prerequisites Setup (First-Time Only)

### Step 1: Install Git

**Download Git** from: https://git-scm.com/downloads

- During installation, keep all defaults. Click "Next" through every screen.
- After installation, **close and reopen** your terminal (Command Prompt or PowerShell).

**Verify Git is installed:**
```cmd
git --version
```
You should see something like `git version 2.47.1.windows.1`. If you see an error, restart your computer and try again.

### Step 2: Install Python

**Download Python 3.11+** from: https://www.python.org/downloads/

- **IMPORTANT**: During installation, check the box that says **"Add Python to PATH"**
- Click "Install Now"

**Verify Python and pip are installed:**
```cmd
python --version
pip --version
```
You should see `Python 3.11.x` (or newer) and `pip 24.x.x`. If `python` is not found:
- Open **Settings > System > Advanced System Settings > Environment Variables**
- Under "Path" in User variables, add: `C:\Users\<YourName>\AppData\Local\Programs\Python\Python311\` and `C:\Users\<YourName>\AppData\Local\Programs\Python\Python311\Scripts\`
- Close and reopen the terminal

---

## Installation

### Step 3: Clone the Repository

Open **Command Prompt** (search "cmd" in Start menu) and run:

```cmd
cd Desktop
git clone https://github.com/SubhamSharma-IITM/Question-Uploading-App.git
cd Question-Uploading-App
```

### Step 4: Install Dependencies

```cmd
pip install -r requirements.txt
```

This may take 2-3 minutes. Wait until it finishes.

**Verify installation:**
```cmd
python -c "import fastapi; import openai; import google.genai; print('All packages OK')"
```

### Step 5: Set Up Environment File

Create a file called `.env` in the project root folder (`Question-Uploading-App\.env`).

**Where to create it:** Open the `Question-Uploading-App` folder in File Explorer, right-click > New > Text Document, name it `.env` (remove the `.txt` extension). If Windows doesn't let you remove the extension, open Notepad, paste the contents below, then File > Save As > type `.env` (with the dot) > Save as type: "All Files".

**Paste these contents into the `.env` file** (replace the placeholder values with your actual API keys):

```env
# Required: At least one LLM provider
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Required for Stage 1 (PDF OCR)
MATHPIX_APP_ID=your_mathpix_app_id
MATHPIX_APP_KEY=your_mathpix_app_key

# Optional: For Stage 6 (Image CDN Upload)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_S3_BUCKET=your_bucket_name

# Optional: For Stage 3 (Anthropic cleaner)
ANTHROPIC_API_KEY=your_anthropic_key
```

**Note:** The `.env` file will NOT be uploaded to GitHub. You will receive the API keys separately.

---

## Running the Server

### Step 6: Start the Application

Open Command Prompt, navigate to the project folder, and run:

```cmd
cd Desktop\Question-Uploading-App
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 7: Open in Browser

Open your web browser and go to:

```
http://localhost:8000
```

The dashboard will appear. Create a new session and follow the 10-stage wizard.

### To Stop the Server

Press `Ctrl + C` in the Command Prompt window.

---

## Troubleshooting

### "python is not recognized"
Python is not in your PATH. See Step 2 above for adding it.

### "pip is not recognized"
Try `python -m pip install -r requirements.txt` instead.

### "Module not found" errors
Run `pip install -r requirements.txt` again.

### Port 8000 already in use
Use a different port: `python -m uvicorn app.main:app --port 8001`

### Checking if everything is installed correctly
```cmd
python --version
pip --version
git --version
python -c "import fastapi; print('FastAPI OK')"
python -c "import openai; print('OpenAI OK')"
python -c "from google import genai; print('Google GenAI OK')"
python -c "import uvicorn; print('Uvicorn OK')"
```

---

## Claude Code Setup (for developers)

If you have [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed, it can set up and run the project automatically:

```cmd
npm install -g @anthropic-ai/claude-code
cd Question-Uploading-App
claude
```

Claude Code will read the `CLAUDE.md` file and understand the project structure. You can ask it to:
- "Set up the project and install dependencies"
- "Start the server"
- "Run the tests"

To install Claude Code, you need Node.js first:
1. Download Node.js from https://nodejs.org/ (LTS version)
2. Install with defaults
3. Open a new terminal and run: `npm install -g @anthropic-ai/claude-code`

---

## Project Structure

```
Question-Uploading-App/
  app/
    main.py              # FastAPI entry point
    config.py            # Settings + .env loader
    db.py                # SQLite database
    pipeline/            # Processing logic per stage
    routes/              # API endpoints per stage
    templates/           # HTML templates (wizard UI)
    static/              # CSS/JS
  tests/                 # Integration tests
  requirements.txt       # Python dependencies
  CLAUDE.md             # Claude Code instructions
  .env                  # API keys (YOU CREATE THIS - not in repo)
  data/                 # Created at runtime (sessions, DB)
```

## Stages Overview

| Stage | Name | What It Does |
|-------|------|-------------|
| 1 | PDF Upload + OCR | Upload PDF, process via MathPix OCR |
| 2 | LLM Extraction | Extract questions using Gemini/GPT |
| 3 | Cleaning | Clean HTML, fix formatting, remove artifacts |
| 4 | Metadata Conversion | Convert to metadata_v2 format |
| 5 | Watermark | Add/remove watermarks from images |
| 6 | Image CDN | Upload images to AWS S3 CDN |
| 7 | LaTeX + Tagging | Batch process LaTeX conversion + concept tagging |
| 8 | Duplicate Check | Self + cross-DB duplicate detection |
| 9 | Quality Verification | 3-model voting QA (Gemini + GPT-5-mini + GPT-5.1) |
| 10 | Upload to DB | Upload approved questions to Acadza API |
