"""Download and cache ALL images from the production DB (streaming).

Streams JSONL line-by-line, collects URLs in batches, fires async
downloads immediately. Never loads entire DB into memory.

Usage:
    python scripts/download_db_images.py

Resume-safe: skips already-downloaded URLs on restart.
"""

import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

# ── Paths ────────────────────────────────────────────────────────────────
APP_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(r"D:\ACADZA_WORK\acadza-pdf-generator\data\allque030326.json")
CACHE_DIR = APP_ROOT / "data" / "image_cache"
IMAGES_DIR = CACHE_DIR / "images"
CACHE_MAP_FILE = CACHE_DIR / "image_cache_map.json"
QUESTION_MAP_FILE = CACHE_DIR / "question_image_map.json"

# ── Download settings ────────────────────────────────────────────────────
MAX_CONCURRENT = 256
LIMIT_PER_HOST = 64
CONNECT_TIMEOUT = 8
READ_TIMEOUT = 30
BATCH_QUESTIONS = 200       # process N questions worth of URLs per batch
SAVE_EVERY_BATCHES = 5      # save cache map every N batches
MAX_RETRIES = 2

IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
DATA_URI_RE = re.compile(r'^data:')
LOCAL_RE = re.compile(r'^(file://|\.\/|\.\.\/|[A-Za-z]:\\)')

QUESTION_TYPES = ["scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"]


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs, flush=True)


# ── Field extraction ─────────────────────────────────────────────────────

def get_question_id(entry: Dict) -> str:
    _id = entry.get("_id", {})
    if isinstance(_id, dict):
        return _id.get("$oid", str(_id))
    return str(_id)


def extract_html_fields(entry: Dict) -> Dict[str, str]:
    """Extract all HTML fields from nested DB entry."""
    fields: Dict[str, str] = {}

    for qtype in QUESTION_TYPES:
        obj = entry.get(qtype)
        if not obj or not isinstance(obj, dict):
            continue

        # Question + Solution
        q = obj.get("question") or obj.get("ques") or ""
        if q: fields["question"] = str(q)
        s = obj.get("solution") or obj.get("sol") or ""
        if s: fields["solution"] = str(s)

        if qtype == "passageQuestion":
            passage = obj.get("passage") or obj.get("stem") or ""
            if passage: fields["passage"] = str(passage)
            psol = obj.get("solution") or obj.get("passageSolution") or ""
            if psol: fields["passage_solution"] = str(psol)
            sub_qs = obj.get("questions", [])
            if isinstance(sub_qs, list):
                for i, sq in enumerate(sub_qs):
                    if not isinstance(sq, dict): continue
                    for sub_type in ["scq", "mcq", "integerQuestion", "matchQuestion"]:
                        sub_obj = sq.get(sub_type)
                        if sub_obj and isinstance(sub_obj, dict):
                            sq_q = sub_obj.get("question") or sub_obj.get("ques") or ""
                            if sq_q: fields[f"sub_q{i}_question"] = str(sq_q)
                            sq_s = sub_obj.get("solution") or sub_obj.get("sol") or ""
                            if sq_s: fields[f"sub_q{i}_solution"] = str(sq_s)
                            break
                    else:
                        sq_q = sq.get("question") or sq.get("ques") or ""
                        if sq_q: fields[f"sub_q{i}_question"] = str(sq_q)
                        sq_s = sq.get("solution") or sq.get("sol") or ""
                        if sq_s: fields[f"sub_q{i}_solution"] = str(sq_s)
        break

    return fields


def extract_urls_from_entry(entry: Dict) -> Tuple[str, Dict[str, List[str]]]:
    """Extract question ID and {field: [urls]} from a DB entry."""
    qid = get_question_id(entry)
    html_fields = extract_html_fields(entry)
    field_map: Dict[str, List[str]] = {}

    for field_name, html in html_fields.items():
        urls = IMG_SRC_RE.findall(html)
        downloadable = []
        for url in urls:
            if DATA_URI_RE.match(url) or LOCAL_RE.match(url):
                continue
            if url.startswith(("http://", "https://")):
                downloadable.append(url)
        if downloadable:
            field_map[field_name] = downloadable

    return qid, field_map


# ── URL → local path ────────────────────────────────────────────────────

def url_to_local_path(url: str) -> Path:
    parsed = urlparse(url)
    host = parsed.hostname or "unknown"
    url_path = parsed.path.lstrip("/")
    if not url_path:
        import hashlib
        url_path = hashlib.md5(url.encode()).hexdigest() + ".img"
    parts = url_path.split("/")
    parts = [p[:80] for p in parts]
    return IMAGES_DIR / host / "/".join(parts)


# ── Cache map (lightweight — just url_to_local) ─────────────────────────

def load_url_cache() -> Dict[str, Dict]:
    """Load just the URL→local mapping (not the huge question map)."""
    if CACHE_MAP_FILE.exists():
        try:
            with open(CACHE_MAP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("url_to_local", {})
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_url_cache(url_cache: Dict[str, Dict], metadata: Dict):
    """Save URL cache to disk (atomic write)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    obj = {"metadata": metadata, "url_to_local": url_cache}
    tmp = CACHE_MAP_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(CACHE_MAP_FILE)


def save_question_map(question_map: Dict[str, Dict[str, List[str]]]):
    """Save question→images mapping separately (can be large)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = QUESTION_MAP_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(question_map, f, ensure_ascii=False)
    tmp.replace(QUESTION_MAP_FILE)


# ── Async download engine ────────────────────────────────────────────────

async def download_batch_async(urls: List[str], url_cache: Dict[str, Dict],
                                stats: Dict) -> None:
    """Download a batch of URLs using aiohttp with 256 concurrency."""
    import aiohttp

    # Filter out already-downloaded
    pending = []
    for url in urls:
        existing = url_cache.get(url)
        if existing and existing.get("status") == "downloaded":
            lp = Path(existing["local_path"])
            if lp.exists():
                stats["skipped"] += 1
                continue
        pending.append(url)

    if not pending:
        return

    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT,
        limit_per_host=LIMIT_PER_HOST,
        ttl_dns_cache=600,
        keepalive_timeout=120,
        enable_cleanup_closed=True,
        force_close=False,
    )
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _fetch_one(session, url):
        async with sem:
            local_path = url_to_local_path(url)
            for attempt in range(MAX_RETRIES + 1):
                try:
                    timeout = aiohttp.ClientTimeout(
                        connect=CONNECT_TIMEOUT, total=CONNECT_TIMEOUT + READ_TIMEOUT
                    )
                    async with session.get(url, timeout=timeout) as resp:
                        if resp.status == 404:
                            url_cache[url] = {"local_path": str(local_path), "status": "not_found"}
                            stats["failed"] += 1
                            return
                        if resp.status != 200:
                            if attempt < MAX_RETRIES:
                                await asyncio.sleep(0.5)
                                continue
                            url_cache[url] = {"local_path": str(local_path), "status": "failed",
                                              "http_status": resp.status}
                            stats["failed"] += 1
                            return

                        content = await resp.read()
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_bytes(content)

                        ct = resp.headers.get("Content-Type", "")
                        mime = ct.split(";")[0].strip() if ct else None

                        url_cache[url] = {
                            "local_path": str(local_path),
                            "status": "downloaded",
                            "size_bytes": len(content),
                            "mime": mime,
                        }
                        stats["downloaded"] += 1
                        return

                except asyncio.TimeoutError:
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(0.5)
                        continue
                    url_cache[url] = {"local_path": str(local_path), "status": "timeout"}
                    stats["failed"] += 1
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(0.3)
                        continue
                    url_cache[url] = {"local_path": str(local_path), "status": "error",
                                      "error": str(e)[:100]}
                    stats["failed"] += 1

    async with aiohttp.ClientSession(
        connector=connector,
        headers={"User-Agent": "acadza-image-cache/1.0"}
    ) as session:
        tasks = [_fetch_one(session, url) for url in pending]
        await asyncio.gather(*tasks)


# ── Main: streaming pipeline ─────────────────────────────────────────────

def main():
    _safe_print("=" * 60)
    _safe_print("DB Image Download — Streaming Pipeline")
    _safe_print("=" * 60)

    # Load existing cache (just URL map, not the full DB)
    _safe_print("Loading existing cache...")
    url_cache = load_url_cache()
    _safe_print(f"  Existing cached URLs: {len(url_cache)}")

    # Stats
    stats = {"downloaded": 0, "failed": 0, "skipped": 0}
    question_map: Dict[str, Dict[str, List[str]]] = {}
    total_entries = 0
    entries_with_images = 0
    batch_urls: List[str] = []
    batch_num = 0
    questions_in_batch = 0
    start_time = time.time()

    _safe_print(f"\nStreaming DB from {DB_PATH} ...")

    with open(DB_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_entries += 1

            if entry.get("error") != 1:
                continue

            # Extract URLs from this entry
            qid, field_map = extract_urls_from_entry(entry)

            if not field_map:
                continue

            entries_with_images += 1
            question_map[qid] = field_map

            # Collect URLs for this batch
            for urls in field_map.values():
                for url in urls:
                    if url not in url_cache or url_cache[url].get("status") != "downloaded":
                        batch_urls.append(url)

            questions_in_batch += 1

            # Fire download when we hit batch size
            if questions_in_batch >= BATCH_QUESTIONS:
                batch_num += 1
                # Deduplicate within batch
                unique_batch = list(dict.fromkeys(batch_urls))

                if unique_batch:
                    asyncio.run(download_batch_async(unique_batch, url_cache, stats))

                done = stats["downloaded"] + stats["failed"] + stats["skipped"]
                elapsed = time.time() - start_time
                rate = stats["downloaded"] / elapsed if elapsed > 0 else 0
                _safe_print(
                    f"  Batch {batch_num} | Q#{entries_with_images} | "
                    f"DL={stats['downloaded']} FAIL={stats['failed']} SKIP={stats['skipped']} | "
                    f"{rate:.0f} img/s | {elapsed:.0f}s | "
                    f"line {line_num}/{162014}"
                )

                # Save periodically
                if batch_num % SAVE_EVERY_BATCHES == 0:
                    metadata = {
                        "db_path": str(DB_PATH),
                        "total_entries": total_entries,
                        "entries_with_images": entries_with_images,
                        "downloaded": stats["downloaded"],
                        "failed": stats["failed"],
                        "last_save": datetime.now(timezone.utc).isoformat(),
                        "last_line": line_num,
                    }
                    save_url_cache(url_cache, metadata)

                batch_urls = []
                questions_in_batch = 0

    # Final batch
    if batch_urls:
        unique_batch = list(dict.fromkeys(batch_urls))
        if unique_batch:
            asyncio.run(download_batch_async(unique_batch, url_cache, stats))

    # Final save
    elapsed = time.time() - start_time
    metadata = {
        "db_path": str(DB_PATH),
        "total_entries": total_entries,
        "entries_with_images": entries_with_images,
        "total_unique_urls": len(url_cache),
        "downloaded": stats["downloaded"],
        "failed": stats["failed"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
    }
    save_url_cache(url_cache, metadata)
    save_question_map(question_map)

    _safe_print(f"\n{'=' * 60}")
    _safe_print(f"DONE in {elapsed:.0f}s")
    _safe_print(f"  Entries scanned: {total_entries}")
    _safe_print(f"  Entries with images: {entries_with_images}")
    _safe_print(f"  Unique URLs in cache: {len(url_cache)}")
    _safe_print(f"  Downloaded: {stats['downloaded']}")
    _safe_print(f"  Failed: {stats['failed']}")
    _safe_print(f"  Skipped (cached): {stats['skipped']}")
    _safe_print(f"  Cache map: {CACHE_MAP_FILE}")
    _safe_print(f"  Question map: {QUESTION_MAP_FILE}")
    _safe_print(f"  Images dir: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
