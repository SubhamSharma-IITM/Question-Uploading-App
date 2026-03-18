#!/usr/bin/env python3
"""Experiment: Compare full MMD vs per-question MMD context for QA.

Picks 5 diverse questions from session 09477d7d, builds both:
  A) Current system prompt (full MMD ~82K tokens)
  B) Optimized prompt (per-question MMD ~8-10K tokens)

Compares token counts and optionally runs both through a model to check if
error detection quality differs.

Usage:
    python tools/experiment_qa_token_optimization.py --mode compare
    python tools/experiment_qa_token_optimization.py --mode run --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_mmd_lines(session_dir: Path) -> List[str]:
    """Load all lines from the MMD file."""
    mmd_dir = session_dir / "mathpix_output"
    lines = []
    for f in sorted(mmd_dir.iterdir()):
        if f.suffix == ".mmd":
            lines = f.read_text(encoding="utf-8").splitlines()
            break
    return lines


def load_mapper_data(session_dir: Path) -> List[Dict]:
    """Load Stage 2 mapper output."""
    mapped_dir = session_dir / "mapped_output"
    for f in sorted(mapped_dir.glob("*_mapped.json")):
        with open(f, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []


def load_metadata(session_dir: Path) -> Dict[str, Dict]:
    """Load metadata_v2 indexed by qid."""
    result = {}
    for f in (session_dir / "metadata_v2").glob("*_metadata_v2.json"):
        with open(f, "r", encoding="utf-8") as fh:
            for entry in json.load(fh):
                result[entry.get("qid", "")] = entry
    return result


def extract_per_question_mmd(mmd_lines: List[str], mapper_entry: Dict,
                              padding: int = 10) -> str:
    """Extract relevant MMD lines for a single question.

    Uses question line_start/line_end and sol_line_start/sol_line_end
    from Stage 2 mapper output.
    """
    q_start = mapper_entry.get("line_start", 0)
    q_end = mapper_entry.get("line_end", 0)
    s_start = mapper_entry.get("sol_line_start")
    s_end = mapper_entry.get("sol_line_end")

    total_lines = len(mmd_lines)
    parts = []

    # Question block with padding
    qs = max(0, q_start - padding)
    qe = min(total_lines, q_end + padding + 1)
    q_block = mmd_lines[qs:qe]
    parts.append(f"--- QUESTION CONTEXT (lines {qs+1}-{qe}) ---")
    for i, line in enumerate(q_block, qs + 1):
        parts.append(f"L{i}: {line}")

    # Solution block (if available and separate from question)
    if s_start is not None and s_end is not None:
        s_start = int(s_start)
        s_end = int(s_end)
        if s_start > q_end + padding:
            # Solution is separate from question block
            ss = max(0, s_start - padding)
            se = min(total_lines, s_end + padding + 1)
            s_block = mmd_lines[ss:se]
            parts.append(f"\n--- SOLUTION CONTEXT (lines {ss+1}-{se}) ---")
            for i, line in enumerate(s_block, ss + 1):
                parts.append(f"L{i}: {line}")
        # else: solution is within the question padding, already captured
    elif s_start is None:
        # No solution line refs -- try to find solution near question
        # Look in the gap between this question's end and next question's start
        # Use a heuristic: check 30 lines after question end
        ss = q_end + 1
        se = min(total_lines, q_end + 30)
        gap = mmd_lines[ss:se]
        # Check if there's a solution marker
        sol_found = False
        for i, line in enumerate(gap):
            if re.match(r'^Sol\.?\s|^Solution', line, re.IGNORECASE):
                sol_found = True
                break
        if sol_found:
            parts.append(f"\n--- SOLUTION CONTEXT (lines {ss+1}-{se}) ---")
            for i, line in enumerate(gap, ss + 1):
                parts.append(f"L{i}: {line}")

    return "\n".join(parts)


def get_chapter_filtered_tree(concept_tree: Dict, chapter: str) -> Dict:
    """Filter concept tree to only the specified chapter."""
    filtered = {}
    for exam, subjects in concept_tree.items():
        if not isinstance(subjects, dict):
            continue
        for subj, chapters in subjects.items():
            if not isinstance(chapters, dict):
                continue
            for ch, concepts in chapters.items():
                if ch.lower() == chapter.lower():
                    if exam not in filtered:
                        filtered[exam] = {}
                    if subj not in filtered[exam]:
                        filtered[exam][subj] = {}
                    filtered[exam][subj][ch] = concepts
    return filtered


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4


def compare_mode(session_id: str = "09477d7d"):
    """Compare token counts between full MMD and per-question MMD."""
    session_dir = Path(f"data/sessions/{session_id}")
    mmd_lines = load_mmd_lines(session_dir)
    mapper_data = load_mapper_data(session_dir)
    metadata = load_metadata(session_dir)

    if not mmd_lines or not mapper_data:
        print("[ERR] Missing MMD or mapper data")
        return

    # Build full MMD (current approach)
    full_mmd = "\n".join(mmd_lines)
    full_mmd_tokens = estimate_tokens(full_mmd)

    # Load system prompt base
    from app.pipeline.stage9_quality import INSPECTOR_SYSTEM_PROMPT
    base_tokens = estimate_tokens(INSPECTOR_SYSTEM_PROMPT)

    print(f"Session: {session_id}")
    print(f"MMD lines: {len(mmd_lines)}")
    print(f"Questions: {len(mapper_data)}")
    print(f"Base prompt: ~{base_tokens} tokens")
    print(f"Full MMD: ~{full_mmd_tokens} tokens")
    print(f"Current total per request: ~{base_tokens + full_mmd_tokens} tokens")
    print()

    # Pick 5 diverse questions
    indices = [0, len(mapper_data)//4, len(mapper_data)//2,
               3*len(mapper_data)//4, len(mapper_data)-1]
    samples = [mapper_data[i] for i in indices]

    print("=== Per-Question MMD Extraction ===\n")
    per_q_tokens = []

    for entry in samples:
        qid = entry.get("qid", "?")
        q_mmd = extract_per_question_mmd(mmd_lines, entry)
        tokens = estimate_tokens(q_mmd)
        per_q_tokens.append(tokens)
        lines_used = q_mmd.count("\n") + 1

        # Get chapter from metadata for concept tree filtering
        meta = metadata.get(qid, {})
        tags = meta.get("tagSubConcept", [{}])
        chapter = tags[0].get("chapter", "?") if tags else "?"

        has_sol_refs = entry.get("sol_line_start") is not None
        print(f"  {qid}: ~{tokens} tokens ({lines_used} lines), chapter={chapter}, sol_refs={has_sol_refs}")

    avg_per_q = sum(per_q_tokens) // len(per_q_tokens) if per_q_tokens else 0
    optimized_total = base_tokens + avg_per_q

    print(f"\n=== Token Comparison ===")
    print(f"  Current (full MMD):     ~{base_tokens + full_mmd_tokens:,} tokens/request")
    print(f"  Optimized (per-Q MMD):  ~{optimized_total:,} tokens/request")
    print(f"  Savings per request:    ~{base_tokens + full_mmd_tokens - optimized_total:,} tokens ({100 * (1 - optimized_total / (base_tokens + full_mmd_tokens)):.0f}%)")
    print()
    total_qs = len(mapper_data)
    print(f"  Current per session (3 passes):   ~{(base_tokens + full_mmd_tokens) * total_qs * 3:,} tokens")
    print(f"  Optimized per session (3 passes):  ~{optimized_total * total_qs * 3:,} tokens")
    print(f"  Total savings per session:         ~{((base_tokens + full_mmd_tokens) - optimized_total) * total_qs * 3:,} tokens")

    # Show a sample extraction
    print(f"\n=== Sample: {samples[2].get('qid')} ===")
    sample_mmd = extract_per_question_mmd(mmd_lines, samples[2])
    print(sample_mmd[:500])
    if len(sample_mmd) > 500:
        print(f"  ... ({len(sample_mmd)} chars total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="compare", choices=["compare"])
    ap.add_argument("--session", default="09477d7d")
    args = ap.parse_args()

    if args.mode == "compare":
        compare_mode(args.session)


if __name__ == "__main__":
    main()
