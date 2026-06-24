"""Metadata-driven stats utility -- single source of truth for active sessions.

Reads live metadata_v2/*.json files and computes per-subject counts.
Used by: sidebar stats, per-stage cards, dashboard in-process counts.

All patterns (image detection, SMILES detection, field traversal, question types)
are derived from production code in stage5_watermark.py and stage6_images.py.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- Production patterns (derived from stage5/stage6 code) ----

# Question type keys used across all stages (stage5_watermark.py:1001, stage6_images.py:218)
QUESTION_TYPES = ("scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion", "subjective")

# Fields scanned for images/SMILES in each question type dict
# (stage5_watermark.py:1017, stage5_watermark.py:1103)
SCAN_FIELDS = ("question", "ques", "solution", "sol", "passage")

# Sub-question types inside passageQuestion.questions[]
# (stage5_watermark.py:1026, stage6_images.py:231)
PASSAGE_SUB_TYPES = ("scq", "mcq", "integerQuestion")

# Sub-question fields (stage5_watermark.py:1029, stage6_images.py:234)
PASSAGE_SUB_FIELDS = ("question", "ques", "solution", "sol")

# Image src pattern -- matches all <img> tags (stage6_images.py:144)
IMG_SRC_RE = re.compile(r'<img[^>]*\bsrc="([^"]+)"', re.IGNORECASE)

# SMILES tag pattern (stage5_watermark.py:488)
SMILES_TAG_RE = re.compile(r'<smiles>(.*?)</smiles>', re.IGNORECASE | re.DOTALL)


def iter_metadata_files(session_dir: Path) -> List[Path]:
    """Find all *_metadata_v2.json files in a session directory.

    Handles both flat layout (metadata_v2/*.json) and subdirectory layout
    (metadata_v2/Physics/*.json). Prefers metadata_v2_output/ if it exists
    (post-Stage-9 with error codes applied).

    Same lookup logic as stage5_watermark.py:1044 and stage6_images.py:422.
    """
    meta_dir = None
    for dir_name in ("metadata_v2_output", "metadata_v2"):
        candidate = session_dir / dir_name
        if candidate.exists():
            meta_dir = candidate
            break

    if meta_dir is None:
        return []

    files = []
    # Flat files first (Stage 4 web app layout)
    for f in sorted(meta_dir.glob("*_metadata_v2.json")):
        if "backup" not in f.name:
            files.append(f)
    # Then subdirectories (production pipeline layout)
    for sub_dir in sorted(meta_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        for f in sorted(sub_dir.glob("*_metadata_v2.json")):
            if "backup" not in f.name:
                files.append(f)
    return files


def load_metadata_entries(session_dir: Path) -> List[Tuple[Path, List[Dict]]]:
    """Load all metadata_v2 entries grouped by file.

    Returns: [(filepath, [entry, ...]), ...]
    """
    groups = []
    for f in iter_metadata_files(session_dir):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list) and data:
                groups.append((f, data))
        except Exception:
            continue
    return groups


def _count_images_in_entry(entry: Dict) -> int:
    """Count total image references in a metadata entry.

    Traverses the same fields and patterns as stage6_images.py:_extract_img_srcs_from_question()
    but counts ALL image src (not just local ./images/ ones) since we want total image count.
    """
    count = 0
    for qt in QUESTION_TYPES:
        obj = entry.get(qt)
        if not isinstance(obj, dict):
            continue

        # Direct fields
        for field in SCAN_FIELDS:
            val = obj.get(field, "") or ""
            if val:
                count += len(IMG_SRC_RE.findall(val))

        # Passage sub-questions (stage6_images.py:229-236)
        if qt == "passageQuestion":
            for sub_q in obj.get("questions", []):
                if not isinstance(sub_q, dict):
                    continue
                for sub_key in PASSAGE_SUB_TYPES:
                    sub_obj = sub_q.get(sub_key)
                    if isinstance(sub_obj, dict):
                        for field in PASSAGE_SUB_FIELDS:
                            val = sub_obj.get(field, "") or ""
                            if val:
                                count += len(IMG_SRC_RE.findall(val))

    return count


def _count_smiles_in_entry(entry: Dict) -> int:
    """Count SMILES tags in a metadata entry.

    Same traversal as stage5_watermark.py:scan_smiles() (lines 1096-1122).
    """
    total = 0
    for qt in QUESTION_TYPES:
        obj = entry.get(qt)
        if not isinstance(obj, dict):
            continue

        for field in SCAN_FIELDS:
            val = str(obj.get(field, "") or "")
            tags = SMILES_TAG_RE.findall(val)
            if tags:
                total += len(tags)

        # Passage sub-questions
        if qt == "passageQuestion":
            for sub_q in obj.get("questions", []):
                if not isinstance(sub_q, dict):
                    continue
                for sub_key in PASSAGE_SUB_TYPES:
                    sub_obj = sub_q.get(sub_key)
                    if isinstance(sub_obj, dict):
                        for field in PASSAGE_SUB_FIELDS:
                            val = str(sub_obj.get(field, "") or "")
                            tags = SMILES_TAG_RE.findall(val)
                            if tags:
                                total += len(tags)

    return total


def _extract_subject(entry: Dict, filepath: Path) -> str:
    """Extract subject from entry or filename.

    Primary: entry["subject"] field (set by Stage 4).
    Fallback: derive from filename (Physics_metadata_v2.json -> Physics).
    """
    subj = entry.get("subject", "")
    if subj:
        return subj
    name = filepath.stem
    name = name.replace("_metadata_v2", "").replace("_cleaned", "").replace("_mapped", "")
    return name or "Unknown"


def compute_live_stats(session_dir: Path,
                       quarantined_qids: Optional[set] = None) -> Dict:
    """Compute comprehensive stats from live metadata files.

    This is THE single source of truth for active session stats.
    Every stage sidebar and card should derive counts from this.

    Returns:
    {
        "total": int,               # Total triplets across all subjects
        "quarantined": int,         # Questions in quarantine set
        "live": int,                # total - quarantined
        "with_images": int,         # Questions that have image references
        "total_images": int,        # Total image count across all questions
        "with_smiles": int,         # Questions with SMILES tags
        "total_smiles_tags": int,   # Total SMILES tag count
        "err1": int,                # error=1 count (live only)
        "err2": int,                # error=2 count (live only)
        "err_other": int,           # error>=3 count (live only)
        "by_subject": {
            "Physics": {"total", "live", "quarantined",
                        "with_images", "total_images", "err1", "err2"},
            ...
        },
        "subjects": ["Physics", "Chemistry", ...],
    }
    """
    if quarantined_qids is None:
        quarantined_qids = set()

    result = {
        "total": 0,
        "quarantined": 0,
        "live": 0,
        "with_images": 0,
        "total_images": 0,
        "with_smiles": 0,
        "total_smiles_tags": 0,
        "err1": 0,
        "err2": 0,
        "err_other": 0,
        "by_subject": {},
        "subjects": [],
    }

    file_groups = load_metadata_entries(session_dir)
    if not file_groups:
        return result

    subject_order = []

    for filepath, entries in file_groups:
        for entry in entries:
            qid = entry.get("qid", "")
            is_quarantined = qid in quarantined_qids
            subject = _extract_subject(entry, filepath)

            # Init subject bucket
            if subject not in result["by_subject"]:
                result["by_subject"][subject] = {
                    "total": 0, "live": 0, "quarantined": 0,
                    "with_images": 0, "total_images": 0,
                    "err1": 0, "err2": 0,
                }
                subject_order.append(subject)

            subj = result["by_subject"][subject]
            result["total"] += 1
            subj["total"] += 1

            if is_quarantined:
                result["quarantined"] += 1
                subj["quarantined"] += 1
                continue  # Skip quarantined from live stats

            result["live"] += 1
            subj["live"] += 1

            # Image stats
            img_count = _count_images_in_entry(entry)
            if img_count > 0:
                result["with_images"] += 1
                subj["with_images"] += 1
            result["total_images"] += img_count
            subj["total_images"] += img_count

            # SMILES stats
            smiles_count = _count_smiles_in_entry(entry)
            if smiles_count > 0:
                result["with_smiles"] += 1
                result["total_smiles_tags"] += smiles_count

            # Error level
            error = entry.get("error")
            if error == 1:
                result["err1"] += 1
                subj["err1"] += 1
            elif error == 2:
                result["err2"] += 1
                subj["err2"] += 1
            elif error is not None and error >= 3:
                result["err_other"] += 1

    # Account for quarantined qids that were physically REMOVED from
    # metadata files by Stage 5 / Stage 8 quarantine. Without this, after
    # Stage 8 removes 34 of 45 entries, the walk above sees only 11 and
    # reports total=11, quarantined=0. The truth: total stayed 45, only
    # 11 are live, 34 are quarantined-and-removed.
    seen_qids = {
        entry.get("qid", "")
        for _, entries in file_groups
        for entry in entries
    }
    removed_quarantined = {
        q for q in (quarantined_qids or set())
        if q and q not in seen_qids
    }
    if removed_quarantined:
        n = len(removed_quarantined)
        result["total"] += n
        result["quarantined"] += n
        # Per-subject buckets stay accurate for entries that remain in
        # metadata. Removed entries have no recoverable subject -- they
        # contribute only to top-level total/quarantined counts.

    result["subjects"] = subject_order
    return result
