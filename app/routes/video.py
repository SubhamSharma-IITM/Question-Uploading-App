"""Teacher Video Recording — upload to Google Drive, embed via iframe.

Flow:
1. Teacher records in browser -> POST /video/api/upload
2. Server saves to temp disk -> uploads to Google Drive (public link)
3. Server returns Drive file ID instantly (no processing delay)
4. Player page embeds Google Drive preview iframe
5. Temp file deleted from server after Drive upload succeeds

Credentials: OAuth token + client secret in {data_dir}/youtube_token.json
and youtube_client_secret.json (same creds, Drive scope added).
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
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB (Drive handles storage, temp file deleted after upload)
ALLOWED_EXTENSIONS = {".webm", ".mp4", ".mkv"}
EXPIRY_DAYS = 7
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# OAuth credentials (shared with YouTube — same project)
OAUTH_CLIENT_SECRET = Path(settings.data_dir) / "youtube_client_secret.json"
OAUTH_TOKEN_FILE = Path(settings.data_dir) / "youtube_token.json"
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

# Google Drive folder for videos (created on first upload)
_drive_folder_id = None

# In-memory upload status tracker
_upload_status = {}
_status_lock = threading.Lock()


def _get_drive_client():
    """Build authenticated Google Drive API client from saved OAuth token."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    if not OAUTH_TOKEN_FILE.exists():
        raise RuntimeError("OAuth token not found. Run auth flow first.")

    creds = Credentials.from_authorized_user_file(str(OAUTH_TOKEN_FILE), OAUTH_SCOPES)

    # Check if token has Drive scope — if not, need re-auth
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        OAUTH_TOKEN_FILE.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def _get_or_create_folder(drive):
    """Get or create 'Acadza Videos' folder in Drive root."""
    global _drive_folder_id
    if _drive_folder_id:
        return _drive_folder_id

    # Search for existing folder
    results = drive.files().list(
        q="name='Acadza Videos' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        spaces="drive", fields="files(id)"
    ).execute()

    if results.get("files"):
        _drive_folder_id = results["files"][0]["id"]
    else:
        # Create folder
        meta = {"name": "Acadza Videos", "mimeType": "application/vnd.google-apps.folder"}
        folder = drive.files().create(body=meta, fields="id").execute()
        _drive_folder_id = folder["id"]

    return _drive_folder_id


def _upload_to_drive(request_id: str, video_path: Path, title: str, uploader: str):
    """Background thread: upload video to Google Drive + set public sharing."""
    from googleapiclient.http import MediaFileUpload

    def set_status(**kwargs):
        with _status_lock:
            _upload_status[request_id].update(kwargs)

    try:
        # Transcode WebM -> MP4 for Drive compatibility (Drive can't preview WebM)
        upload_path = video_path
        if video_path.suffix == ".webm":
            import shutil, subprocess
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                set_status(status="uploading", phase="Converting video...")
                mp4_path = video_path.with_suffix(".mp4")
                result = subprocess.run(
                    [ffmpeg, "-i", str(video_path),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                     "-c:a", "aac", "-b:a", "64k",
                     "-movflags", "+faststart", "-y", str(mp4_path)],
                    capture_output=True, timeout=30,
                )
                if mp4_path.exists() and mp4_path.stat().st_size > 0:
                    upload_path = mp4_path
                    video_path.unlink()  # delete original WebM
                    # Fix title extension
                    if title.endswith(".webm"):
                        title = title[:-5] + ".mp4"

        set_status(status="uploading", phase="Connecting to Google Drive...")
        drive = _get_drive_client()
        folder_id = _get_or_create_folder(drive)

        set_status(phase="Uploading to Google Drive...")

        # Upload file
        file_meta = {
            "name": title,
            "parents": [folder_id],
            "description": f"Recorded by {uploader} via Acadza Video",
        }

        mime = "video/mp4" if upload_path.suffix == ".mp4" else "video/webm"
        media = MediaFileUpload(str(upload_path), mimetype=mime, resumable=True,
                                chunksize=5 * 1024 * 1024)

        request = drive.files().create(body=file_meta, media_body=media, fields="id")

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                set_status(progress=pct, phase=f"Uploading... {pct}%")

        file_id = response["id"]
        set_status(progress=100, phase="Setting permissions...")

        # Set public sharing (anyone with link can view)
        drive.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

        # Build URLs
        embed_url = f"https://drive.google.com/file/d/{file_id}/preview"
        view_url = f"https://drive.google.com/file/d/{file_id}/view"

        set_status(
            status="ready",
            phase="Video ready!",
            file_id=file_id,
            embed_url=embed_url,
            view_url=view_url,
        )

        # Persist to SQLite so video list survives restarts
        try:
            size_mb = round(upload_path.stat().st_size / (1024 * 1024), 2) if upload_path.exists() else 0
            _save_video_record(file_id, request_id, uploader, size_mb)
        except Exception:
            pass

        # Delete temp file
        try:
            upload_path.unlink()
        except Exception:
            pass

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


def _save_video_record(file_id: str, request_id: str, uploader: str, size_mb: float):
    """Persist video metadata to SQLite after Drive upload succeeds."""
    from app.db import _get_conn, _lock
    # Get uploader phone from in-memory status
    phone = ""
    with _status_lock:
        st = _upload_status.get(request_id, {})
        phone = st.get("uploader_phone", "")
        if not phone:
            # Try to get from started_at context
            phone = st.get("phone", "")
    duration = 0
    with _status_lock:
        duration = _upload_status.get(request_id, {}).get("duration", 0)

    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO videos (file_id, uploader_phone, uploader_name, size_mb, duration, created_at, play_count) VALUES (?, ?, ?, ?, ?, ?, 0)",
            (file_id, phone, uploader, size_mb, duration,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def _increment_play_count(file_id: str):
    """Increment play count for a video."""
    from app.db import _get_conn, _lock
    with _lock:
        conn = _get_conn()
        conn.execute("UPDATE videos SET play_count = play_count + 1 WHERE file_id = ?", (file_id,))
        conn.commit()


# ── Check if Drive has finished processing a video ──

@router.get("/api/check/{file_id}")
async def check_video_ready(file_id: str):
    """Check if Google Drive has finished processing a video.

    Drive populates videoMediaMetadata only after transcoding completes.
    Returns {ready: true/false} — used by player to decide iframe vs proxy.
    """
    import asyncio

    def _check():
        drive = _get_drive_client()
        meta = drive.files().get(
            fileId=file_id, fields="videoMediaMetadata,mimeType"
        ).execute()
        vmm = meta.get("videoMediaMetadata")
        return vmm is not None

    try:
        ready = await asyncio.to_thread(_check)
        return {"ready": ready}
    except Exception:
        return {"ready": False}


# ── Stream: return Drive direct download URL for client-side fetch ──

@router.get("/api/stream/{file_id}")
async def get_stream_url(file_id: str):
    """Return the direct download URL for a Drive file.

    The client fetches the video directly from Google's CDN using this URL.
    Our server is NOT in the data path — zero RAM usage, zero bandwidth.
    """
    import asyncio

    def _get_url():
        drive = _get_drive_client()
        meta = drive.files().get(
            fileId=file_id, fields="webContentLink,size,mimeType"
        ).execute()
        return {
            "url": meta.get("webContentLink", ""),
            "size": int(meta.get("size", 0)),
            "mimeType": meta.get("mimeType", "video/mp4"),
        }

    try:
        result = await asyncio.to_thread(_get_url)
        if not result["url"]:
            return JSONResponse(status_code=404, content={"error": "No download URL"})
        return result
    except Exception as e:
        return JSONResponse(status_code=404, content={"error": f"Video not found: {e}"})


# ── Proxy fallback: server downloads from Drive, serves to client ──

@router.get("/api/proxy/{file_id}")
async def proxy_from_drive(file_id: str):
    """Fallback: download video from Drive server-side and serve to browser.

    Used when CORS blocks direct client-side fetch from Google Drive.
    Loads full file into memory — only for small videos (<15MB).
    """
    import asyncio
    import io
    from fastapi.responses import Response

    def _download():
        drive = _get_drive_client()
        meta = drive.files().get(fileId=file_id, fields="size,mimeType").execute()
        content_type = meta.get("mimeType", "video/mp4")
        from googleapiclient.http import MediaIoBaseDownload
        fh = io.BytesIO()
        dl = MediaIoBaseDownload(fh, drive.files().get_media(fileId=file_id))
        done = False
        while not done:
            _, done = dl.next_chunk()
        fh.seek(0)
        return fh.read(), content_type

    try:
        video_bytes, content_type = await asyncio.to_thread(_download)
        return Response(
            content=video_bytes,
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(video_bytes)),
                "Cache-Control": "public, max-age=3600",
            },
        )
    except Exception as e:
        return JSONResponse(status_code=404, content={"error": str(e)})


# ── Playback logging ──

PLAYBACK_LOG = VIDEOS_DIR / "playback_log.jsonl"


@router.post("/api/log")
async def log_playback(request: Request):
    """Log playback events (iframe vs proxy, timing, fallback triggers).

    Appended to data/videos/playback_log.jsonl for local analysis.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "file_id": str(body.get("file_id", ""))[:64],
        "method": str(body.get("method", ""))[:10],
        "elapsed_ms": int(body.get("elapsed_ms", 0)),
        "success": bool(body.get("success")),
        "iframe_loaded": bool(body.get("iframe_loaded")),
        "user_agent": str(body.get("user_agent", ""))[:200],
        "events": body.get("events", [])[:30],  # cap at 30 entries
    }

    try:
        with open(PLAYBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # Increment play count in SQLite
    file_id = entry.get("file_id", "")
    if file_id:
        try:
            _increment_play_count(file_id)
        except Exception:
            pass

    return {"ok": True}


# ── List all recorded videos (from SQLite) ──

@router.get("/api/videos")
async def list_videos():
    """List all recorded videos with metadata and play counts.

    Reads from SQLite — persists across restarts, works on Render.
    """
    from app.db import _get_conn, _lock
    try:
        with _lock:
            conn = _get_conn()
            rows = conn.execute(
                "SELECT file_id, uploader_name, size_mb, duration, created_at, play_count FROM videos ORDER BY created_at DESC"
            ).fetchall()
        videos = []
        for r in rows:
            fid = r["file_id"]
            videos.append({
                "file_id": fid,
                "uploader": r["uploader_name"] or "",
                "size_mb": r["size_mb"] or 0,
                "duration": r["duration"] or 0,
                "created_at": r["created_at"] or "",
                "play_count": r["play_count"] or 0,
                "player_url": f"/video/player?v={fid}",
            })
        return {"success": True, "count": len(videos), "videos": videos}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Pages ──

@router.get("/recorder", response_class=HTMLResponse)
async def recorder_page():
    return (TEMPLATES_DIR / "video_recorder.html").read_text(encoding="utf-8")


@router.get("/player", response_class=HTMLResponse)
async def player_page():
    return (TEMPLATES_DIR / "video_player.html").read_text(encoding="utf-8")


# ── Upload (starts Drive upload in background) ──

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
            "file_id": None,
            "embed_url": None,
            "view_url": None,
            "error": None,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "uploader": name,
            "uploader_phone": phone,
            "duration": duration,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    # Start Drive upload in background thread
    title = f"Acadza - {name} - {datetime.now().strftime('%d %b %Y %H:%M')}{ext}"
    thread = threading.Thread(
        target=_upload_to_drive,
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
