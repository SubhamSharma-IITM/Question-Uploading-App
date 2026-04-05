"""Teacher Video Recording — upload, stream, and manage 1-min videos.

Mounted at /video/* in the main app. Reuses existing cookie auth.
Videos stored on persistent disk at {data_dir}/videos/ with 7-day expiry.

Migration note: when moving to production, swap VIDEOS_DIR to S3 and
replace stream_video() with a pre-signed URL redirect. The API contract
(upload returns player_url, stream returns video bytes) stays the same.
"""

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from app.config import settings

router = APIRouter()


def _remux_and_probe(file_path: Path) -> float:
    """Remux video to fix container metadata (duration), then probe exact duration.

    Chrome's MediaRecorder writes WebM in live-streaming mode without duration.
    A quick ffmpeg remux (-c copy, no re-encoding) rewrites the container with
    correct duration metadata. Takes <0.1s for a 10MB file.

    Also converts WebM to MP4 (H.264) for Safari/iOS compatibility if needed.

    Returns duration in seconds, or 0.0 if ffmpeg/ffprobe unavailable.
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        return 0.0

    # Remux: rewrite container with proper metadata (no re-encoding)
    tmp_path = file_path.with_suffix(".tmp" + file_path.suffix)
    try:
        subprocess.run(
            [ffmpeg, "-i", str(file_path), "-c", "copy", "-y", str(tmp_path)],
            capture_output=True, timeout=15,
        )
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.replace(file_path)  # atomic replace
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()

    # Probe duration from the fixed file
    try:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(file_path)],
            capture_output=True, text=True, timeout=5,
        )
        dur = float(result.stdout.strip())
        return round(dur, 2) if dur > 0 else 0.0
    except Exception:
        return 0.0


# ── Config ──
VIDEOS_DIR = Path(settings.data_dir) / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB
ALLOWED_EXTENSIONS = {".webm", ".mp4", ".mkv"}
EXPIRY_DAYS = 7
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def cleanup_expired_videos():
    """Delete videos older than EXPIRY_DAYS."""
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


# ── Upload ──

@router.post("/api/upload")
async def upload_video(request: Request, file: UploadFile = File(...), duration: float = 0):
    # Auth: reuse existing cookie session
    user = getattr(request.state, "user", {})
    phone = user.get("phone", "")
    name = user.get("name", "")
    if not phone:
        return JSONResponse(status_code=401, content={"success": False, "error": "Not authenticated"})

    # Validate extension
    ext = Path(file.filename or "video.webm").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(status_code=400, content={
            "success": False, "error": f"Invalid format: {ext}"
        })

    # Stream-read with size check (never load full file into memory)
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

    # Save
    video_id = uuid.uuid4().hex[:12]
    video_path = VIDEOS_DIR / f"{video_id}{ext}"
    with open(video_path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)

    # Post-upload: remux to fix container + transcode WebM->MP4 for Safari/iOS
    duration_seconds = _remux_and_probe(video_path)

    # Safari/iOS can't play WebM — transcode to MP4 (H.264+AAC)
    # Keep both: WebM for Chrome/Firefox, MP4 for Safari/iOS
    if ext == ".webm":
        import shutil, subprocess
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            mp4_path = VIDEOS_DIR / f"{video_id}.mp4"
            try:
                subprocess.run(
                    [ffmpeg, "-i", str(video_path),
                     "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                     "-c:a", "aac", "-b:a", "64k",
                     "-movflags", "+faststart",  # moov atom at start for instant playback
                     "-y", str(mp4_path)],
                    capture_output=True, timeout=30,
                )
                if mp4_path.exists() and mp4_path.stat().st_size > 0:
                    total_size = mp4_path.stat().st_size  # update size to MP4
            except Exception:
                pass  # WebM still available as fallback

    if duration_seconds <= 0 and duration > 0:
        duration_seconds = round(float(duration), 2)
    if duration_seconds <= 0:
        duration_seconds = round(total_size * 8 / 1_000_000, 1)
    if duration_seconds > 65:
        duration_seconds = 60.0

    # Sidecar metadata
    expires_at = datetime.now(timezone.utc) + timedelta(days=EXPIRY_DAYS)
    metadata = {
        "video_id": video_id,
        "filename": f"{video_id}{ext}",
        "size_bytes": total_size,
        "size_mb": round(total_size / (1024 * 1024), 2),
        "duration_seconds": duration_seconds,
        "uploader_phone": phone,
        "uploader_name": name,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expires_at.isoformat(),
        "content_type": file.content_type or "video/webm",
    }
    with open(VIDEOS_DIR / f"{video_id}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Build URLs relative to host
    base = str(request.base_url).rstrip("/")
    return {
        "success": True,
        "video_id": video_id,
        "stream_url": f"{base}/video/api/stream/{video_id}",
        "player_url": f"{base}/video/player?v={video_id}",
        "size_bytes": total_size,
        "size_mb": metadata["size_mb"],
        "expires_at": expires_at.isoformat(),
    }


# ── Stream (HTTP 206 range requests) ──

@router.get("/api/stream/{video_id}")
async def stream_video(video_id: str, request: Request):
    # Find video file — prefer MP4 (Safari/iOS compatible) over WebM
    video_file = None
    for ext in (".mp4", ".webm", ".mkv"):
        candidate = VIDEOS_DIR / f"{video_id}{ext}"
        if candidate.exists():
            video_file = candidate
            break

    if not video_file:
        return JSONResponse(status_code=404, content={"error": "Video not found or expired"})

    file_size = video_file.stat().st_size
    content_type = "video/webm" if video_file.suffix == ".webm" else "video/mp4"

    range_header = request.headers.get("range")
    if range_header:
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        def iter_range():
            with open(video_file, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk_size = min(64 * 1024, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Cache-Control": "public, max-age=3600",
            },
        )

    # No range header: serve full file (browser discovers duration from full content)
    return FileResponse(
        video_file,
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": "public, max-age=3600",
        },
    )


# ── Video info ──

@router.get("/api/info/{video_id}")
async def video_info(video_id: str):
    meta_path = VIDEOS_DIR / f"{video_id}.json"
    if not meta_path.exists():
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    with open(meta_path, "r") as f:
        return {"success": True, "data": json.load(f)}
