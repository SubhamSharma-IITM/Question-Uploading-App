#!/usr/bin/env python3
"""Batch image replacement: upload cleaned images + replace S3 URLs.

For uploaded sessions: re-upload cleaned PNGs to S3, then API-update question HTML.
For non-uploaded sessions: re-upload cleaned PNGs to S3, then update metadata_v2 JSON.

Usage:
    python tools/batch_image_replace.py --session 35e7d78e --dry-run
    python tools/batch_image_replace.py --session all
    python tools/batch_image_replace.py --session all --skip-upload  (if images already uploaded)
"""

import argparse
import csv
import json
import glob
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests as http_requests

IMAGE_UPLOAD_API = "https://api.acadza.in/question/upload/androidbulk"
API_BASE = "https://api.acadza.in"
DETAILS_PATH = "/question/details"
UPDATE_PATH = "/question/update/field"

SESSIONS = {
    "35e7d78e": {"uploaded": True, "year": 2026},
    "56589a45": {"uploaded": True, "year": 2026},
    "09477d7d": {"uploaded": True, "year": 2025},
    "73d46401": {"uploaded": True, "year": 2026},
    "95e55c16": {"uploaded": False, "year": 0},  # not PYQ, skip year
}


def _guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "webp": "image/webp", "svg": "image/svg+xml"
            }.get(ext.lstrip("."), "application/octet-stream")


def upload_image(local_path: str, retries: int = 3) -> Optional[str]:
    """Upload one image to Acadza CDN. Returns CDN URL or None."""
    for attempt in range(retries):
        try:
            mime = _guess_mime(local_path)
            with open(local_path, "rb") as f:
                resp = http_requests.post(
                    IMAGE_UPLOAD_API,
                    files={"image": (os.path.basename(local_path), f, mime)},
                    timeout=60,
                )
            resp.raise_for_status()
            data = resp.json()
            url = data.get("url") or data.get("URL")
            if url:
                return url
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    [ERR] Upload failed after {retries} attempts: {e}")
    return None


def build_url_replacement_map(session_id: str, skip_upload: bool = False, dry_run: bool = False) -> Dict[str, str]:
    """Upload cleaned images and build old_url -> new_url mapping.

    Returns dict: {old_cdn_url: new_cdn_url}
    """
    data_dir = Path(f"data/sessions/{session_id}")
    cdn_map_path = data_dir / "images_cdn_map.json"
    cleaned_dir = data_dir / "images_cleaned"

    if not cdn_map_path.exists():
        print(f"  [WARN] No CDN map for {session_id}")
        return {}

    with open(cdn_map_path, "r", encoding="utf-8") as f:
        cdn_map = json.load(f)

    if not cleaned_dir.exists():
        print(f"  [WARN] No images_cleaned/ for {session_id}")
        return {}

    # Check if local_path points to images_original (needs replacement)
    # or images_cleaned (already correct)
    needs_replacement = {}
    for key, val in cdn_map.items():
        if not isinstance(val, dict):
            continue
        local_path = val.get("local_path", "")
        cdn_url = val.get("cdn_url", "")
        if not cdn_url:
            continue

        # Check if this was uploaded from original (watermarked)
        if "images_original" in local_path:
            # Find corresponding cleaned file
            basename = os.path.basename(local_path)
            stem = os.path.splitext(basename)[0]
            # Cleaned images are PNG
            cleaned_path = cleaned_dir / f"{stem}.png"
            if not cleaned_path.exists():
                # Try original extension
                cleaned_path = cleaned_dir / basename
            if cleaned_path.exists():
                needs_replacement[cdn_url] = str(cleaned_path)

    if not needs_replacement:
        print(f"  [INFO] No watermarked images to replace for {session_id}")
        return {}

    print(f"  [INFO] {len(needs_replacement)} images need replacement")

    # Load existing new URL map if we've done a partial run before
    new_map_path = data_dir / "images_cleaned_cdn_map.json"
    url_map = {}
    if new_map_path.exists():
        with open(new_map_path, "r", encoding="utf-8") as f:
            url_map = json.load(f)

    if skip_upload:
        print(f"  [INFO] Skipping upload, using existing map ({len(url_map)} entries)")
        return url_map

    # Upload each cleaned image
    uploaded = 0
    for old_url, cleaned_path in needs_replacement.items():
        if old_url in url_map:
            continue  # already uploaded

        if dry_run:
            url_map[old_url] = f"DRY_RUN_{os.path.basename(cleaned_path)}"
            uploaded += 1
            continue

        new_url = upload_image(cleaned_path)
        if new_url:
            url_map[old_url] = new_url
            uploaded += 1
            if uploaded % 10 == 0 or uploaded <= 3:
                print(f"    Uploaded {uploaded}/{len(needs_replacement)}: {os.path.basename(cleaned_path)}")
        else:
            print(f"    [ERR] Failed: {os.path.basename(cleaned_path)}")

        time.sleep(0.1)

    # Save map for resume
    with open(new_map_path, "w", encoding="utf-8") as f:
        json.dump(url_map, f, indent=2)

    print(f"  [INFO] Uploaded {uploaded} cleaned images, total map: {len(url_map)}")
    return url_map


def replace_urls_in_text(text: str, url_map: Dict[str, str]) -> Tuple[str, int]:
    """Replace old CDN URLs with new ones in HTML text. Returns (new_text, count)."""
    count = 0
    for old_url, new_url in url_map.items():
        if old_url in text:
            text = text.replace(old_url, new_url)
            count += 1
    return text, count


def patch_uploaded_session(session_id: str, url_map: Dict[str, str], dry_run: bool = False):
    """For uploaded sessions: API-update each question's HTML with new URLs."""
    data_dir = Path(f"data/sessions/{session_id}")

    oid_map = {}
    for cf in glob.glob(str(data_dir / "upload_logs" / "*_upload_log.csv")):
        with open(cf, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = row.get("UploadedQuestionID", "")
                if uid and len(uid) == 24:
                    oid_map[row["Question META ID"]] = uid

    if not oid_map:
        print(f"  [ERR] No upload OIDs for {session_id}")
        return

    session = http_requests.Session()
    hdrs = {"Accept": "application/json", "Content-Type": "application/json"}

    updated = 0
    skipped = 0
    total = len(oid_map)

    for i, (qid, oid) in enumerate(oid_map.items(), 1):
        r = session.post(f"{API_BASE}{DETAILS_PATH}",
                          headers={**hdrs, "questionid": oid}, json={})
        if r.status_code != 200:
            continue

        data = r.json().get("question", r.json())
        payload = dict(data)
        for k in ("_id", "createdAt", "updatedAt", "__v"):
            payload.pop(k, None)

        qtype = payload.get("questionType", "scq")
        qdata = payload.get(qtype, {})
        if not isinstance(qdata, dict):
            skipped += 1
            continue

        total_replaced = 0
        for field in ("question", "solution"):
            text = qdata.get(field, "")
            if text:
                new_text, count = replace_urls_in_text(text, url_map)
                if count > 0:
                    qdata[field] = new_text
                    total_replaced += count

        if total_replaced == 0:
            skipped += 1
            continue

        if dry_run:
            if i <= 5 or i % 50 == 0:
                print(f"    [{i}/{total}] {oid} DRY-RUN: {total_replaced} URLs replaced")
            updated += 1
        else:
            r2 = session.put(f"{API_BASE}{UPDATE_PATH}",
                              headers={**hdrs, "questionid": oid}, json=payload)
            if r2.status_code == 200:
                updated += 1
                if i <= 3 or i % 25 == 0:
                    print(f"    [{i}/{total}] {oid} OK: {total_replaced} URLs")
            else:
                print(f"    [{i}/{total}] {oid} FAILED: {r2.status_code}")

        time.sleep(0.08)

    print(f"  [DONE] API update: {updated} updated, {skipped} skipped")


def patch_local_session(session_id: str, url_map: Dict[str, str], dry_run: bool = False):
    """For non-uploaded sessions: update metadata_v2 JSON files directly."""
    data_dir = Path(f"data/sessions/{session_id}")

    total_replaced = 0
    for mf in glob.glob(str(data_dir / "metadata_v2" / "*_metadata_v2.json")):
        with open(mf, "r", encoding="utf-8") as f:
            entries = json.load(f)

        modified = False
        for entry in entries:
            qtype = entry.get("questionType", "scq")
            qdata = entry.get(qtype, {})
            if not isinstance(qdata, dict):
                continue

            for field in ("question", "solution"):
                text = qdata.get(field, "")
                if text:
                    new_text, count = replace_urls_in_text(text, url_map)
                    if count > 0:
                        qdata[field] = new_text
                        total_replaced += count
                        modified = True

        if modified and not dry_run:
            with open(mf, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"  [DONE] Local metadata: {total_replaced} URLs replaced")


def process_session(session_id: str, dry_run: bool = False, skip_upload: bool = False):
    """Process one session: upload cleaned images + replace URLs."""
    info = SESSIONS.get(session_id)
    if not info:
        print(f"[ERR] Unknown session: {session_id}")
        return

    print(f"\n{'='*60}")
    print(f"Session {session_id} ({'UPLOADED' if info['uploaded'] else 'LOCAL'})")
    print(f"{'='*60}")

    # Step 1: Upload cleaned images and build URL map
    url_map = build_url_replacement_map(session_id, skip_upload=skip_upload, dry_run=dry_run)

    if not url_map:
        print(f"  [INFO] No URL replacements needed")
        return

    # Step 2: Replace URLs
    if info["uploaded"]:
        patch_uploaded_session(session_id, url_map, dry_run=dry_run)
    else:
        patch_local_session(session_id, url_map, dry_run=dry_run)


def main():
    ap = argparse.ArgumentParser(description="Batch image replacement for watermarked images")
    ap.add_argument("--session", required=True, help="Session ID or 'all'")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-upload", action="store_true", help="Skip S3 upload, use existing map")
    args = ap.parse_args()

    if args.session == "all":
        for sid in SESSIONS:
            process_session(sid, dry_run=args.dry_run, skip_upload=args.skip_upload)
    else:
        process_session(args.session, dry_run=args.dry_run, skip_upload=args.skip_upload)


if __name__ == "__main__":
    main()
