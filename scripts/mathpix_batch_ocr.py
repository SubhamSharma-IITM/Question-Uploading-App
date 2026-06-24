"""MathPix OCR for image-only DB questions.

Scans DB for image-only questions/solutions (text < 10 chars with images),
submits to MathPix v3/text endpoint (better accuracy than v3/batch which
uses v3/latex codepath and ignores plain text). Concurrent async requests
with rate limiting.

Output: LaTeX with $ $ and $$ $$ delimiters → downstream latex2mathml converter.

Excludes matchQuestion. Includes scq, mcq, integerQuestion, passageQuestion.

Usage:
    # Dry run (count only, no API calls):
    python scripts/mathpix_batch_ocr.py --dry-run

    # OCR questions only (cheapest first):
    python scripts/mathpix_batch_ocr.py --questions-only

    # OCR everything (questions + solutions):
    python scripts/mathpix_batch_ocr.py

    # Limit to N images (for testing):
    python scripts/mathpix_batch_ocr.py --limit 10

Resume-safe: always loads existing cache and skips already-OCR'd URLs.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    print("ERROR: pip install aiohttp")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────
APP_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(r"D:\ACADZA_WORK\acadza-pdf-generator\data\allque030326.json")
OCR_DIR = APP_ROOT / "data" / "ocr_cache"
OCR_MAP_FILE = OCR_DIR / "ocr_results_map.json"
OCR_QUEUE_FILE = OCR_DIR / "ocr_queue.json"

# ── MathPix API ──────────────────────────────────────────────────────────
MATHPIX_TEXT_URL = "https://api.mathpix.com/v3/text"

# v3/text with text+latex_styled → downstream latex2mathml handles conversion
MATHPIX_OCR_OPTIONS = {
    "formats": ["text", "latex_styled"],
    "math_inline_delimiters": ["$", "$"],
    "math_display_delimiters": ["$$", "$$"],
    "confidence_rate_threshold": 0.05,  # accept most results, filter ourselves
}

# ── Settings ─────────────────────────────────────────────────────────────
# v3/text: individual calls, rate-limited at 50 req/min by MathPix.
# Best accuracy (text+math codepath). ~17 hrs for 50K images. Run overnight.
RATE_LIMIT_PER_MIN = 250          # MathPix approved quota (upgraded from 50)
RATE_LIMIT_DELAY = 60.0 / RATE_LIMIT_PER_MIN  # 0.24s between requests
MAX_CONCURRENT = 20               # parallel connections (stay within rate limit)
REQUEST_TIMEOUT = 60              # seconds per request
SAVE_EVERY = 50                   # save cache every N images
MAX_RETRIES = 2                   # retries per image on failure

QUESTION_TYPES = ["scq", "mcq", "integerQuestion", "passageQuestion"]  # NO matchQuestion

IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs, flush=True)


def strip_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-z]+;|&#\d+;|&#x[0-9a-fA-F]+;', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def get_question_id(entry: dict) -> str:
    _id = entry.get("_id", {})
    if isinstance(_id, dict):
        return _id.get("$oid", str(_id))
    return str(_id)


def extract_image_urls(html: str) -> List[str]:
    if not html:
        return []
    urls = IMG_SRC_RE.findall(str(html))
    return [u for u in urls if u.startswith(("http://", "https://"))]


# ── Credentials ──────────────────────────────────────────────────────────

def get_credentials() -> Tuple[str, str]:
    app_id = os.environ.get("MATHPIX_APP_ID", "")
    app_key = os.environ.get("MATHPIX_APP_KEY", "")
    if not app_id or not app_key:
        env_path = APP_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("MATHPIX_APP_ID="):
                    app_id = line.split("=", 1)[1].strip()
                elif line.startswith("MATHPIX_APP_KEY="):
                    app_key = line.split("=", 1)[1].strip()
    return app_id, app_key


# ── OCR Cache ────────────────────────────────────────────────────────────

def load_ocr_cache() -> Dict[str, Dict]:
    """Load OCR results cache: {image_url: {text, latex_styled, confidence, ...}}"""
    if OCR_MAP_FILE.exists():
        try:
            with open(OCR_MAP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("url_to_ocr", {})
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_ocr_cache(ocr_cache: Dict[str, Dict], metadata: Dict):
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    obj = {"metadata": metadata, "url_to_ocr": ocr_cache}
    tmp = OCR_MAP_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(OCR_MAP_FILE)


# ── DB Scanning ──────────────────────────────────────────────────────────

def scan_image_only_entries(questions_only: bool = False) -> List[Dict]:
    """Scan DB for image-only entries.

    Returns list of tasks: {qid, error, qtype, field, field_path, image_urls}
    """
    tasks = []

    with open(DB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            err = entry.get("error")
            if err not in (1, 2):
                continue

            qid = get_question_id(entry)

            for qtype in QUESTION_TYPES:
                obj = entry.get(qtype)
                if not obj or not isinstance(obj, dict):
                    continue

                # Main question
                q = str(obj.get("question") or obj.get("ques") or "")
                q_field = "question" if obj.get("question") else "ques"
                if q and len(strip_html(q)) < 10:
                    urls = extract_image_urls(q)
                    if urls:
                        tasks.append({
                            "qid": qid, "error": err, "qtype": qtype,
                            "field": "question",
                            "field_path": f"{qtype}.{q_field}",
                            "image_urls": urls,
                        })

                # Solution
                if not questions_only:
                    sol = str(obj.get("solution") or obj.get("sol") or "")
                    sol_field = "solution" if obj.get("solution") else "sol"
                    if sol and len(strip_html(sol)) < 10:
                        urls = extract_image_urls(sol)
                        if urls:
                            tasks.append({
                                "qid": qid, "error": err, "qtype": qtype,
                                "field": "solution",
                                "field_path": f"{qtype}.{sol_field}",
                                "image_urls": urls,
                            })

                # passageQuestion: passage + sub-questions
                if qtype == "passageQuestion":
                    passage = str(obj.get("passage") or obj.get("stem") or "")
                    p_field = "passage" if obj.get("passage") else "stem"
                    if passage and len(strip_html(passage)) < 10:
                        urls = extract_image_urls(passage)
                        if urls:
                            tasks.append({
                                "qid": qid, "error": err, "qtype": qtype,
                                "field": "passage",
                                "field_path": f"{qtype}.{p_field}",
                                "image_urls": urls,
                            })

                    sub_qs = obj.get("questions", [])
                    if isinstance(sub_qs, list):
                        for i, sq in enumerate(sub_qs):
                            if not isinstance(sq, dict):
                                continue
                            for sub_type in ["scq", "mcq", "integerQuestion"]:
                                sub_obj = sq.get(sub_type)
                                if not sub_obj or not isinstance(sub_obj, dict):
                                    continue

                                sq_q = str(sub_obj.get("question") or sub_obj.get("ques") or "")
                                sq_qf = "question" if sub_obj.get("question") else "ques"
                                if sq_q and len(strip_html(sq_q)) < 10:
                                    urls = extract_image_urls(sq_q)
                                    if urls:
                                        tasks.append({
                                            "qid": qid, "error": err, "qtype": qtype,
                                            "field": f"sub_q{i}_question",
                                            "field_path": f"{qtype}.questions.{i}.{sub_type}.{sq_qf}",
                                            "image_urls": urls,
                                        })

                                if not questions_only:
                                    sq_s = str(sub_obj.get("solution") or sub_obj.get("sol") or "")
                                    sq_sf = "solution" if sub_obj.get("solution") else "sol"
                                    if sq_s and len(strip_html(sq_s)) < 10:
                                        urls = extract_image_urls(sq_s)
                                        if urls:
                                            tasks.append({
                                                "qid": qid, "error": err, "qtype": qtype,
                                                "field": f"sub_q{i}_solution",
                                                "field_path": f"{qtype}.questions.{i}.{sub_type}.{sq_sf}",
                                                "image_urls": urls,
                                            })
                                break
                break

    return tasks


# ── Async v3/text OCR ────────────────────────────────────────────────────

async def ocr_single_image(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    rate_lock: asyncio.Lock,
    url: str,
    app_id: str,
    app_key: str,
    stats: Dict,
) -> Tuple[str, Dict]:
    """OCR a single image via v3/text. Single attempt, no inline retries.

    Returns (url, result_dict) with status: success | empty | dead | retriable.
    - dead: image is broken/placeholder, never retry (Content not found, etc.)
    - retriable: transient error (timeout, 429, server error), retry in pass 2
    """
    async with sem:
        # Rate limiting: ensure minimum gap between requests
        async with rate_lock:
            now = asyncio.get_event_loop().time()
            last = stats.get("_last_request_time", 0)
            wait = RATE_LIMIT_DELAY - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
            stats["_last_request_time"] = asyncio.get_event_loop().time()

        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with session.post(
                MATHPIX_TEXT_URL,
                headers={
                    "app_id": app_id,
                    "app_key": app_key,
                    "Content-Type": "application/json",
                },
                json={"src": url, **MATHPIX_OCR_OPTIONS},
                timeout=timeout,
            ) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    return url, {
                        "status": "retriable",
                        "error_msg": f"429 rate limited (retry-after={retry_after}s)",
                        "ocr_at": datetime.now(timezone.utc).isoformat(),
                    }

                if resp.status >= 500:
                    return url, {
                        "status": "retriable",
                        "error_msg": f"HTTP {resp.status} server error",
                        "ocr_at": datetime.now(timezone.utc).isoformat(),
                    }

                if resp.status != 200:
                    return url, {
                        "status": "dead",
                        "error_msg": f"HTTP {resp.status}",
                        "ocr_at": datetime.now(timezone.utc).isoformat(),
                    }

                data = await resp.json()

                result = {
                    "text": data.get("text", ""),
                    "latex_styled": data.get("latex_styled", ""),
                    "confidence": data.get("confidence", 0),
                    "confidence_rate": data.get("confidence_rate", 0),
                    "is_printed": data.get("is_printed"),
                    "is_handwritten": data.get("is_handwritten"),
                    "image_width": data.get("image_width", 0),
                    "image_height": data.get("image_height", 0),
                    "ocr_at": datetime.now(timezone.utc).isoformat(),
                }

                if data.get("error"):
                    err_msg = data["error"][:200]
                    # Classify: dead (permanent) vs retriable (transient)
                    err_lower = err_msg.lower()
                    is_dead = (
                        "not found" in err_lower
                        or "invalid" in err_lower
                        or err_msg.startswith("http")  # MathPix returns URL = can't fetch source
                    )
                    result["status"] = "dead" if is_dead else "retriable"
                    result["error_msg"] = err_msg
                elif result["text"].strip() and len(result["text"].strip()) > 2:
                    result["status"] = "success"
                else:
                    result["status"] = "empty"  # placeholder/blank image = dead
                    result["error_msg"] = "empty content"

                return url, result

        except asyncio.TimeoutError:
            return url, {
                "status": "retriable", "error_msg": "timeout",
                "ocr_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return url, {
                "status": "retriable", "error_msg": str(e)[:200],
                "ocr_at": datetime.now(timezone.utc).isoformat(),
            }


async def process_urls_async(
    urls: List[str],
    app_id: str,
    app_key: str,
    ocr_cache: Dict[str, Dict],
    metadata_base: Dict,
    pass_label: str = "Pass 1",
) -> Tuple[int, int, int, List[str]]:
    """Process all URLs with rate-limited concurrent v3/text calls.

    Returns (success_count, dead_count, retriable_count, retriable_urls).
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    rate_lock = asyncio.Lock()
    stats = {"_last_request_time": 0}

    total_success = 0
    total_dead = 0
    total_retriable = 0
    retriable_urls = []
    processed = 0
    start_time = time.time()

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, keepalive_timeout=120)
    async with aiohttp.ClientSession(connector=connector) as session:
        for chunk_start in range(0, len(urls), SAVE_EVERY):
            chunk = urls[chunk_start:chunk_start + SAVE_EVERY]

            tasks = [
                ocr_single_image(session, sem, rate_lock, url, app_id, app_key, stats)
                for url in chunk
            ]
            results = await asyncio.gather(*tasks)

            chunk_success = 0
            chunk_dead = 0
            chunk_retry = 0
            for url, result in results:
                ocr_cache[url] = result
                status = result.get("status")
                if status == "success":
                    chunk_success += 1
                    total_success += 1
                elif status in ("dead", "empty"):
                    chunk_dead += 1
                    total_dead += 1
                elif status == "retriable":
                    chunk_retry += 1
                    total_retriable += 1
                    retriable_urls.append(url)
                else:
                    chunk_dead += 1
                    total_dead += 1

            processed += len(chunk)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta_min = (len(urls) - processed) / rate / 60 if rate > 0 else 0

            _safe_print(
                f"  {pass_label} [{processed}/{len(urls)}] "
                f"+{chunk_success} OK, +{chunk_dead} dead, +{chunk_retry} retry | "
                f"Total: {total_success} OK, {total_dead} dead, {total_retriable} retry | "
                f"{rate:.1f} img/s | {elapsed:.0f}s | "
                f"ETA: {eta_min:.0f}min | "
                f"Cost: ${processed * 0.002:.2f}"
            )

            # Save cache after each chunk
            metadata = {**metadata_base,
                        "total_ocr_cached": len(ocr_cache),
                        "total_success": total_success,
                        "total_dead": total_dead,
                        "total_retriable": total_retriable,
                        "last_save": datetime.now(timezone.utc).isoformat(),
                        "processed": processed,
                        "total_to_process": len(urls),
                        "pass": pass_label}
            save_ocr_cache(ocr_cache, metadata)

    return total_success, total_dead, total_retriable, retriable_urls


# ── Question OCR Map Builder ─────────────────────────────────────────────

def build_question_ocr_map(tasks: List[Dict], ocr_cache: Dict[str, Dict]) -> Dict:
    """Build qid+field_path → OCR results mapping for pipeline use."""
    question_ocr_map = {}
    for task in tasks:
        qid = task["qid"]
        fp = task["field_path"]
        key = f"{qid}|{fp}"
        ocr_texts = []
        for url in task["image_urls"]:
            cached = ocr_cache.get(url)
            if cached and cached.get("status") == "success":
                ocr_texts.append({
                    "url": url,
                    "text": cached.get("text", ""),
                    "latex": cached.get("latex_styled", ""),
                    "confidence": cached.get("confidence_rate", 0),
                })
        if ocr_texts:
            question_ocr_map[key] = {
                "qid": qid,
                "error": task["error"],
                "qtype": task["qtype"],
                "field": task["field"],
                "field_path": fp,
                "ocr_results": ocr_texts,
            }
    return question_ocr_map


def _build_and_save_question_map(tasks: List[Dict], ocr_cache: Dict[str, Dict]):
    """Build and save the question OCR map."""
    question_ocr_map = build_question_ocr_map(tasks, ocr_cache)
    q_ocr_map_file = OCR_DIR / "question_ocr_map.json"
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    with open(q_ocr_map_file, "w", encoding="utf-8") as f:
        json.dump(question_ocr_map, f, ensure_ascii=False, indent=2)
    _safe_print(f"  Question OCR map: {len(question_ocr_map)} entries -> {q_ocr_map_file}")


def _count_cache_statuses(ocr_cache: Dict[str, Dict]) -> Dict[str, int]:
    """Count entries by status in the cache."""
    from collections import Counter
    return dict(Counter(r.get("status", "unknown") for r in ocr_cache.values()))


def _build_failure_report(tasks: List[Dict], ocr_cache: Dict[str, Dict]):
    """Build a categorized failure report: dead/empty URLs mapped to question IDs.

    Output: data/ocr_cache/failure_report.json with:
    - by_domain: {domain: [{url, qid, qtype, field, error_msg, status}]}
    - by_category: {category: [qids]}  (for bulk DB operations)
    - summary: counts per domain and category
    """
    from collections import defaultdict
    from urllib.parse import urlparse

    # Build reverse map: url → list of (qid, qtype, field, field_path, error_level)
    url_to_questions = defaultdict(list)
    for task in tasks:
        for url in task["image_urls"]:
            url_to_questions[url].append({
                "qid": task["qid"],
                "error": task["error"],
                "qtype": task["qtype"],
                "field": task["field"],
                "field_path": task["field_path"],
            })

    # Collect failures
    by_domain = defaultdict(list)
    affected_qids = defaultdict(set)  # category → set of qids

    for url, cached in ocr_cache.items():
        status = cached.get("status", "")
        if status not in ("dead", "empty", "retriable"):
            continue

        err_msg = cached.get("error_msg", "")

        # Parse domain
        try:
            domain = urlparse(url).hostname or "unknown"
        except Exception:
            domain = "unknown"

        # Categorize
        if "googleusercontent.com" in domain or "google.com" in domain:
            category = "google_dead_links"
        elif "sarthaks.com" in domain:
            category = "sarthaks_dead_links"
        elif "acadza" in domain and status == "empty":
            category = "acadza_placeholder_images"
        elif "acadza" in domain and status == "dead":
            category = "acadza_dead_images"
        elif status == "empty":
            category = "empty_placeholder"
        else:
            category = f"other_{status}"

        # Get affected questions
        questions = url_to_questions.get(url, [])
        for q in questions:
            affected_qids[category].add(q["qid"])

        by_domain[domain].append({
            "url": url,
            "status": status,
            "error_msg": err_msg[:150],
            "category": category,
            "questions": questions,
        })

    # Build summary
    summary = {
        "total_failed_urls": sum(len(v) for v in by_domain.values()),
        "total_affected_qids": len(set().union(*affected_qids.values())) if affected_qids else 0,
        "by_domain": {d: len(entries) for d, entries in sorted(by_domain.items(), key=lambda x: -len(x[1]))},
        "by_category": {cat: len(qids) for cat, qids in sorted(affected_qids.items(), key=lambda x: -len(x[1]))},
    }

    # Convert sets to sorted lists for JSON
    category_qids = {cat: sorted(qids) for cat, qids in affected_qids.items()}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "category_qids": category_qids,
        "by_domain": {d: entries for d, entries in sorted(by_domain.items(), key=lambda x: -len(x[1]))},
    }

    report_file = OCR_DIR / "failure_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _safe_print(f"\n  Failure report: {report_file}")
    _safe_print(f"    Total failed URLs: {summary['total_failed_urls']}")
    _safe_print(f"    Total affected question IDs: {summary['total_affected_qids']}")
    _safe_print(f"    By domain:")
    for d, cnt in summary["by_domain"].items():
        _safe_print(f"      {cnt:5d}x  {d}")
    _safe_print(f"    By category (unique question IDs):")
    for cat, cnt in summary["by_category"].items():
        _safe_print(f"      {cnt:5d}x  {cat}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MathPix OCR for image-only DB questions")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no API calls")
    parser.add_argument("--questions-only", action="store_true", help="OCR questions only (skip solutions)")
    parser.add_argument("--limit", type=int, default=0, help="Limit total images to OCR (0=unlimited)")
    args = parser.parse_args()

    _safe_print("=" * 60)
    _safe_print("MathPix OCR — Image-Only Questions (v3/text)")
    _safe_print("=" * 60)

    app_id, app_key = get_credentials()
    if not args.dry_run and (not app_id or not app_key):
        _safe_print("ERROR: MathPix credentials not found in .env")
        sys.exit(1)

    # Scan DB
    _safe_print("\nScanning DB for image-only entries...")
    mode = "questions only" if args.questions_only else "questions + solutions"
    _safe_print(f"  Mode: {mode}")
    _safe_print(f"  Question types: {', '.join(QUESTION_TYPES)}")

    tasks = scan_image_only_entries(questions_only=args.questions_only)

    all_urls = set()
    for task in tasks:
        all_urls.update(task["image_urls"])

    q_tasks = [t for t in tasks if "question" in t["field"] or t["field"] == "passage"]
    s_tasks = [t for t in tasks if "solution" in t["field"]]

    _safe_print(f"\n  Total tasks: {len(tasks)}")
    _safe_print(f"    Question/passage tasks: {len(q_tasks)}")
    _safe_print(f"    Solution tasks: {len(s_tasks)}")
    _safe_print(f"  Unique image URLs: {len(all_urls)}")
    _safe_print(f"  error=1: {sum(1 for t in tasks if t['error']==1)} | error=2: {sum(1 for t in tasks if t['error']==2)}")
    _safe_print(f"  Est. cost @ $0.002/img: ${len(all_urls) * 0.002:.2f}")
    _safe_print(f"  Rate limit: {RATE_LIMIT_PER_MIN} req/min, {MAX_CONCURRENT} concurrent")
    est_time_min = len(all_urls) * RATE_LIMIT_DELAY / 60
    _safe_print(f"  Est. time: ~{est_time_min:.0f} min ({est_time_min/60:.1f} hrs)")

    if args.dry_run:
        _safe_print("\n[DRY RUN] No API calls made.")
        OCR_DIR.mkdir(parents=True, exist_ok=True)
        with open(OCR_QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "total_tasks": len(tasks),
                "unique_urls": len(all_urls),
                "est_cost_usd": round(len(all_urls) * 0.002, 2),
                "tasks": tasks,
            }, f, ensure_ascii=False, indent=2)
        _safe_print(f"  Queue saved to: {OCR_QUEUE_FILE}")
        return

    # Load existing cache (always resume)
    ocr_cache = load_ocr_cache()
    _safe_print(f"\n  Existing OCR cache: {len(ocr_cache)} URLs")

    # Reclassify old "error" entries: dead vs retriable
    reclassified = 0
    for url, entry in ocr_cache.items():
        if entry.get("status") == "error":
            err = entry.get("error_msg") or ""
            err_lower = err.lower()
            if "not found" in err_lower or "invalid" in err_lower or err.startswith("http"):
                entry["status"] = "dead"
                reclassified += 1
    if reclassified:
        _safe_print(f"  Reclassified {reclassified} old errors as dead")

    # Filter already-done URLs (skip success + empty/dead)
    urls_to_ocr = []
    seen = set()
    for task in tasks:
        for url in task["image_urls"]:
            if url in seen:
                continue
            seen.add(url)
            existing = ocr_cache.get(url)
            if existing and existing.get("status") in ("success", "empty", "dead"):
                continue
            urls_to_ocr.append(url)

    _safe_print(f"  URLs to OCR (after cache filter): {len(urls_to_ocr)}")

    if args.limit > 0:
        urls_to_ocr = urls_to_ocr[:args.limit]
        _safe_print(f"  Limited to: {len(urls_to_ocr)}")

    if not urls_to_ocr:
        _safe_print("\n  Nothing to OCR! All URLs already cached.")
        _build_and_save_question_map(tasks, ocr_cache)
        _build_failure_report(tasks, ocr_cache)
        return

    cost = len(urls_to_ocr) * 0.002
    est_min = len(urls_to_ocr) * RATE_LIMIT_DELAY / 60
    _safe_print(f"\n  Will OCR {len(urls_to_ocr)} images")
    _safe_print(f"  Estimated cost: ${cost:.2f}")
    _safe_print(f"  Estimated time: ~{est_min:.0f} min ({est_min/60:.1f} hrs)")

    # ── Pass 1: Process all images ──────────────────────────────────────
    metadata_base = {"db_path": str(DB_PATH), "mode": mode}
    _safe_print(f"\n{'=' * 60}")
    _safe_print("PASS 1: Processing all images...")
    _safe_print(f"{'=' * 60}")

    p1_success, p1_dead, p1_retriable, retriable_urls = asyncio.run(
        process_urls_async(urls_to_ocr, app_id, app_key, ocr_cache, metadata_base, "P1")
    )

    _safe_print(f"\n  Pass 1 complete:")
    _safe_print(f"    Success: {p1_success}")
    _safe_print(f"    Dead (won't retry): {p1_dead}")
    _safe_print(f"    Retriable: {p1_retriable}")
    _safe_print(f"    Cost: ~${len(urls_to_ocr) * 0.002:.2f}")

    # ── Pass 2: Retry only retriable failures ───────────────────────────
    total_success = p1_success
    total_dead = p1_dead

    if retriable_urls:
        _safe_print(f"\n{'=' * 60}")
        _safe_print(f"PASS 2: Retrying {len(retriable_urls)} retriable failures...")
        _safe_print(f"{'=' * 60}")

        # Clear retriable status so they get re-processed
        for url in retriable_urls:
            if url in ocr_cache:
                del ocr_cache[url]

        p2_success, p2_dead, p2_retriable, still_failing = asyncio.run(
            process_urls_async(retriable_urls, app_id, app_key, ocr_cache, metadata_base, "P2")
        )

        total_success += p2_success
        total_dead += p2_dead

        _safe_print(f"\n  Pass 2 complete:")
        _safe_print(f"    Recovered: {p2_success}")
        _safe_print(f"    Still dead: {p2_dead}")
        _safe_print(f"    Still failing: {p2_retriable}")

        if still_failing:
            _safe_print(f"\n  {len(still_failing)} URLs failed both passes (logged as dead):")
            for url in still_failing[:10]:
                err = ocr_cache.get(url, {}).get("error_msg", "unknown")
                _safe_print(f"    {url[-50:]} -> {err[:60]}")
            if len(still_failing) > 10:
                _safe_print(f"    ... and {len(still_failing) - 10} more")
            # Mark remaining as dead so future runs skip them
            for url in still_failing:
                if url in ocr_cache:
                    ocr_cache[url]["status"] = "dead"
                    ocr_cache[url]["error_msg"] = f"failed 2 passes: {ocr_cache[url].get('error_msg', '')}"

    # ── Final save ──────────────────────────────────────────────────────
    total_processed = len(urls_to_ocr) + len(retriable_urls)
    metadata = {
        **metadata_base,
        "total_ocr_cached": len(ocr_cache),
        "total_success": total_success,
        "total_dead": total_dead,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_ocr_cache(ocr_cache, metadata)

    # Build question OCR map + failure report
    _build_and_save_question_map(tasks, ocr_cache)
    _build_failure_report(tasks, ocr_cache)

    # Summary
    cache_stats = _count_cache_statuses(ocr_cache)
    _safe_print(f"\n{'=' * 60}")
    _safe_print(f"DONE — ALL PASSES COMPLETE")
    _safe_print(f"{'=' * 60}")
    _safe_print(f"  Pass 1 processed: {len(urls_to_ocr)}")
    _safe_print(f"  Pass 2 retried: {len(retriable_urls)}")
    _safe_print(f"  Total success: {total_success}")
    _safe_print(f"  Total dead: {total_dead}")
    _safe_print(f"  Total cached: {len(ocr_cache)}")
    _safe_print(f"  Cache breakdown: {cache_stats}")
    _safe_print(f"  Est. cost: ~${total_processed * 0.002:.2f}")
    _safe_print(f"\n  OCR cache: {OCR_MAP_FILE}")
    _safe_print(f"  Question map: {OCR_DIR / 'question_ocr_map.json'}")


if __name__ == "__main__":
    main()
