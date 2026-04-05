"""Teacher Video Recording — upload to YouTube, stream via YouTube embed.

Flow:
1. Teacher records in browser → POST /video/api/upload
2. Server saves to temp disk → uploads to YouTube (unlisted)
3. Client polls GET /video/api/status/{id} until YouTube processing completes
4. Player page embeds YouTube iframe player

Migration note: YouTube credentials (OAuth token + client secret) stored in
{data_dir}/youtube_token.json and youtube_client_secret.json. Token auto-refreshes.
To switch to S3: replace _upload_to_youtube() with S3 put_object, remove polling.
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import settings

router = APIRouter()

# ── Config ──
VIDEOS_DIR = Path(settings.data_dir) / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB
ALLOWED_EXTENSIONS = {".webm", ".mp4", ".mkv"}
EXPIRY_DAYS = 7
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# YouTube credentials
YT_CLIENT_SECRET = Path(settings.data_dir) / "youtube_client_secret.json"
YT_TOKEN_FILE = Path(settings.data_dir) / "youtube_token.json"
YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
              "https://www.googleapis.com/auth/youtube.readonly"]

# In-memory upload status tracker
# {request_id: {status, progress, youtube_url, error, ...}}
_upload_status = {}
_status_lock = threading.Lock()


def _get_youtube_client():
    """Build authenticated YouTube API client from saved OAuth token."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    if not YT_TOKEN_FILE.exists():
        raise RuntimeError("YouTube token not found. Run test_youtube_upload.py first.")

    creds = Credentials.from_authorized_user_file(str(YT_TOKEN_FILE), YT_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        YT_TOKEN_FILE.write_text(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def _upload_to_youtube(request_id: str, video_path: Path, title: str, uploader: str):
    """Background thread: upload video to YouTube + poll until processed."""
    from googleapiclient.http import MediaFileUpload

    def set_status(**kwargs):
        with _status_lock:
            _upload_status[request_id].update(kwargs)

    try:
        youtube = _get_youtube_client()

        body = {
            "snippet": {
                "title": title,
                "description": f"Recorded by {uploader} via Acadza Video",
                "tags": ["acadza", "education"],
                "categoryId": "27",  # Education
            },
            "status": {
                "privacyStatus": "unlisted",
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(str(video_path), chunksize=5 * 1024 * 1024, resumable=True)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

        set_status(status="uploading", phase="Uploading to YouTube...")

        # Upload with progress tracking
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                set_status(progress=pct, phase=f"Uploading... {pct}%")

        video_id = response["id"]
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        set_status(status="processing", progress=100, youtube_id=video_id,
                   youtube_url=youtube_url, phase="YouTube is processing...")

        # Poll YouTube for processing completion
        for i in range(60):  # up to 10 minutes
            time.sleep(5)
            result = youtube.videos().list(
                part="processingDetails,status", id=video_id
            ).execute()

            if result["items"]:
                item = result["items"][0]
                proc = item.get("processingDetails", {})
                upload_status = item.get("status", {}).get("uploadStatus", "")
                proc_status = proc.get("processingStatus", "")

                elapsed = i * 5
                set_status(phase=f"Processing... ({elapsed}s)")

                if proc_status == "succeeded" or upload_status == "processed":
                    set_status(status="ready", phase="Video ready!",
                               youtube_url=youtube_url, youtube_id=video_id)
                    # Clean up temp file
                    try:
                        video_path.unlink()
                    except Exception:
                        pass
                    return

                if upload_status == "failed":
                    reason = item.get("status", {}).get("failureReason", "unknown")
                    set_status(status="failed", error=f"YouTube rejected: {reason}")
                    return

        set_status(status="ready", phase="Video uploaded (processing may continue)",
                   youtube_url=youtube_url)

    except Exception as e:
        import traceback
        traceback.print_exc()
        set_status(status="failed", error=str(e), phase="Upload failed")


def cleanup_expired_videos():
    """Delete temp videos older than EXPIRY_DAYS."""
    cutoff = time.time() - (EXPIRY_DAYS * 86400)
    cleaned = 0
    for f in VIDEOS_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink()
            cleaned += 1
    return cleaned


# ── Pages ──

@router.get("/recorder", response_class=HTMLResponse)
async def recorder_page():
    return (TEMPLATES_DIR / "video_recorder.html").read_text(encoding="utf-8")


@router.get("/player", response_class=HTMLResponse)
async def player_page():
    return (TEMPLATES_DIR / "video_player.html").read_text(encoding="utf-8")


# ── Upload (starts YouTube upload in background) ──

@router.post("/api/upload")
async def upload_video(request: Request, file: UploadFile = File(...), duration: float = 0):
    user = getattr(request.state, "user", {})
    phone = user.get("phone", "")
    name = user.get("name", "")
    if not phone:
        return JSONResponse(status_code=401, content={"success": False, "error": "Not authenticated"})

    ext = Path(file.filename or "video.webm").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(status_code=400, content={"success": False, "error": f"Invalid format: {ext}"})

    # Stream-read with size check
    chunks = []
    total_size = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={
                "success": False, "error": f"File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)}MB"
            })
        chunks.append(chunk)

    if total_size == 0:
        return JSONResponse(status_code=400, content={"success": False, "error": "Empty file"})

    # Save to temp disk
    request_id = uuid.uuid4().hex[:12]
    video_path = VIDEOS_DIR / f"{request_id}{ext}"
    with open(video_path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)

    # Init status tracker
    with _status_lock:
        _upload_status[request_id] = {
            "status": "queued",
            "progress": 0,
            "phase": "Preparing upload...",
            "youtube_url": None,
            "youtube_id": None,
            "error": None,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "uploader": name,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    # Start YouTube upload in background thread
    title = f"Acadza - {name} - {datetime.now().strftime('%d %b %Y %H:%M')}"
    thread = threading.Thread(
        target=_upload_to_youtube,
        args=(request_id, video_path, title, name),
        daemon=True,
    )
    thread.start()

    return {"success": True, "request_id": request_id, "size_mb": round(total_size / (1024 * 1024), 2)}


# ── Poll upload status ──

@router.get("/api/status/{request_id}")
async def upload_status(request_id: str):
    with _status_lock:
        status = _upload_status.get(request_id)
    if not status:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return {"success": True, "data": status}
