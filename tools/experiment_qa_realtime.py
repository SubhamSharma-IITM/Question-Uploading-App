#!/usr/bin/env python3
"""Real-time experiment: Build optimized QA payloads and test with actual models.

Picks 7 diverse questions from session 09477d7d (has backfilled sol line refs).
Builds optimized payloads: system prompt (base rules only) + user message (question +
per-Q MMD with padding + chapter-scoped concept tree).
Sends to GPT-5-mini in real-time to compare response quality.

Usage:
    python tools/experiment_qa_realtime.py --mode build     # Build payloads, show structure
    python tools/experiment_qa_realtime.py --mode test      # Send to model, show responses
"""
import argparse, json, sys, re, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

SESSION_ID = "09477d7d"
SESSION_DIR = Path(f"data/sessions/{SESSION_ID}")


def load_data():
    """Load all needed data for the experiment."""
    # Mapper (with backfilled sol refs)
    with open(SESSION_DIR / "mapped_output/MATH2025_Mathematics_mapped.json", "r", encoding="utf-8") as f:
        mapper = json.load(f)

    # Metadata
    with open(SESSION_DIR / "metadata_v2/MATH2025_Mathematics_metadata_v2.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # MMD lines
    mmd_path = list((SESSION_DIR / "mathpix_output").glob("*.mmd"))[0]
    mmd_lines = mmd_path.read_text(encoding="utf-8").splitlines()

    # Concept tree
    from app.config import settings
    with open(Path(settings.concept_tree_path), "r", encoding="utf-8") as f:
        concept_tree = json.load(f)

    # System prompt (base only, no MMD/tree injection)
    from app.pipeline.stage9_quality import INSPECTOR_SYSTEM_PROMPT
    base_prompt = INSPECTOR_SYSTEM_PROMPT

    return mapper, metadata, mmd_lines, concept_tree, base_prompt


def extract_per_q_mmd(mmd_lines, mapper_entry, padding=7):
    """Extract relevant MMD lines for one question with labeled padding."""
    q_start = mapper_entry.get("line_start", 0)
    q_end = mapper_entry.get("line_end", 0)
    s_start = mapper_entry.get("sol_line_start")
    s_end = mapper_entry.get("sol_line_end")

    total = len(mmd_lines)
    parts = []

    # Question block with padding
    qs = max(0, q_start - padding)
    qe = min(total, q_end + padding + 1)
    parts.append(f"--- SOURCE DOCUMENT: Question region (lines {qs+1}-{qe}) ---")
    parts.append(f"--- Note: Actual question is lines {q_start+1}-{q_end+1}. "
                 f"Extra lines are padding for context verification. ---")
    for i in range(qs, qe):
        marker = " <<" if q_start <= i <= q_end else ""
        parts.append(f"L{i+1}: {mmd_lines[i]}{marker}")

    # Solution block
    if s_start is not None and s_end is not None:
        s_start, s_end = int(s_start), int(s_end)
        if s_start > q_end + padding:
            ss = max(0, s_start - padding)
            se = min(total, s_end + padding + 1)
            parts.append(f"\n--- SOURCE DOCUMENT: Solution region (lines {ss+1}-{se}) ---")
            parts.append(f"--- Note: Actual solution is lines {s_start+1}-{s_end+1}. ---")
            for i in range(ss, se):
                marker = " <<" if s_start <= i <= s_end else ""
                parts.append(f"L{i+1}: {mmd_lines[i]}{marker}")

    return "\n".join(parts)


def extract_chapter_tree(concept_tree, tag_sub_concepts):
    """Extract chapter-scoped concept tree from tags."""
    chapters = set()
    subjects = set()
    for t in tag_sub_concepts:
        ch = t.get("chapter", "")
        subj = t.get("subject", "")
        if ch:
            chapters.add(ch)
        if subj:
            subjects.add(subj)

    filtered = {}
    for exam, exam_subjects in concept_tree.items():
        if not isinstance(exam_subjects, dict):
            continue
        for subj, subj_chapters in exam_subjects.items():
            if not isinstance(subj_chapters, dict):
                continue
            if subj not in subjects:
                continue
            for ch, concepts in subj_chapters.items():
                if ch in chapters:
                    if exam not in filtered:
                        filtered[exam] = {}
                    if subj not in filtered[exam]:
                        filtered[exam][subj] = {}
                    filtered[exam][subj][ch] = concepts

    return filtered


def build_optimized_payload(entry, mapper_entry, mmd_lines, concept_tree, base_prompt):
    """Build the full optimized payload for one question."""
    qid = entry.get("qid", "?")
    qtype = entry.get("questionType", "scq")
    qdata = entry.get(qtype, {})
    subject = entry.get("subject", "")

    # User message parts
    q_html = qdata.get("question", "")
    s_html = qdata.get("solution", "")
    answer = qdata.get("answer", entry.get("answer", ""))

    # Image handling (same as production)
    from app.pipeline.stage9_quality import _collect_images_with_placeholders
    q_text, q_imgs = _collect_images_with_placeholders(q_html, "ImageQ")
    s_text, s_imgs = _collect_images_with_placeholders(s_html, "ImageS")

    # Tags
    tags = entry.get("tagSubConcept", [])
    chapter = tags[0].get("chapter", "") if tags else ""
    concept = tags[0].get("concept", "") if tags else ""
    sub_concept = tags[0].get("subConcept", "") if tags else ""

    # Per-Q MMD context
    mmd_context = extract_per_q_mmd(mmd_lines, mapper_entry)

    # Chapter-scoped concept tree
    ch_tree = extract_chapter_tree(concept_tree, tags)
    ch_tree_str = json.dumps(ch_tree, indent=1, ensure_ascii=False) if ch_tree else "No concept tree available"

    # Build user message
    user_text = f"""================================================================================
TARGET QUESTION FOR AUDIT
================================================================================

QID: {qid}
Subject: {subject}
Question Type: {qtype}

TAGS (verify against concept tree below):
  Subject:    {subject}
  Chapter:    {chapter}
  Concept:    {concept}
  SubConcept: {sub_concept}

QUESTION HTML:
{q_text}

ANSWER: {answer}

SOLUTION HTML:
{s_text}

================================================================================
CONCEPT TREE (chapter-scoped for tag validation):
{ch_tree_str}

================================================================================
SOURCE DOCUMENT CONTEXT (for content verification):
{mmd_context}

================================================================================
EXECUTE AUDIT: Run Sections 0-11. Return ONLY valid JSON.
================================================================================"""

    user_parts = [{"type": "text", "text": user_text}]
    for img in q_imgs + s_imgs:
        user_parts.append(img)

    return {
        "system": base_prompt,
        "user_parts": user_parts,
        "user_text_only": user_text,
        "images": len(q_imgs) + len(s_imgs),
        "qid": qid,
    }


def pick_diverse_samples(mapper, metadata):
    """Pick 7 diverse questions."""
    meta_by_qid = {e["qid"]: e for e in metadata}
    mapper_by_qid = {e["qid"]: e for e in mapper}

    # Uploadable only
    uploadable = [e for e in metadata if e.get("error", 2) < 3 and e["qid"] in mapper_by_qid]

    # Pick diverse: text-only, with images, with limit, with fractions, different chapters
    indices = [0, len(uploadable)//6, len(uploadable)//3, len(uploadable)//2,
               2*len(uploadable)//3, 5*len(uploadable)//6, len(uploadable)-1]
    samples = []
    for i in indices:
        e = uploadable[i]
        m = mapper_by_qid[e["qid"]]
        samples.append((e, m))

    return samples


def build_mode():
    """Build and display optimized payloads."""
    mapper, metadata, mmd_lines, concept_tree, base_prompt = load_data()
    samples = pick_diverse_samples(mapper, metadata)

    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        count_tok = lambda t: len(enc.encode(t))
        print("[OK] tiktoken available")
    except ImportError:
        count_tok = lambda t: len(t) // 4
        print("[WARN] using char/4 estimate")

    base_tokens = count_tok(base_prompt)
    print(f"Base system prompt: {base_tokens} tokens (CACHED after first request)")
    print(f"Samples: {len(samples)} questions\n")

    for i, (entry, mapper_entry) in enumerate(samples, 1):
        payload = build_optimized_payload(entry, mapper_entry, mmd_lines, concept_tree, base_prompt)
        user_tokens = count_tok(payload["user_text_only"])
        total = base_tokens + user_tokens + payload["images"] * 255  # high detail estimate

        tags = entry.get("tagSubConcept", [])
        chapter = tags[0].get("chapter", "") if tags else "?"

        print(f"Q{i}: {payload['qid']} ({entry.get('questionType','?')}, ch={chapter})")
        print(f"  System: {base_tokens} tok (cached) | User: {user_tokens} tok | "
              f"Images: {payload['images']} | Total: ~{total} tok")
        print(f"  Has sol_line_refs: {mapper_entry.get('sol_line_start') is not None}")
        print()

    # Total batch estimate
    avg_user = sum(count_tok(build_optimized_payload(e, m, mmd_lines, concept_tree, base_prompt)["user_text_only"])
                   for e, m in samples) // len(samples)
    print(f"Avg user message: {avg_user} tokens")
    print(f"Avg total per request: {base_tokens + avg_user} tokens")
    print(f"With caching (system cached at 25%): {base_tokens * 0.25 + avg_user:.0f} effective tokens")


def test_mode():
    """Send optimized payloads to GPT-5-mini and show responses."""
    mapper, metadata, mmd_lines, concept_tree, base_prompt = load_data()
    samples = pick_diverse_samples(mapper, metadata)

    from app.config import settings
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)

    from app.pipeline.stage9_quality import OPENAI_QA_RESPONSE_SCHEMA

    print(f"Testing {len(samples)} questions with GPT-5-mini (real-time)...\n")

    for i, (entry, mapper_entry) in enumerate(samples, 1):
        payload = build_optimized_payload(entry, mapper_entry, mmd_lines, concept_tree, base_prompt)

        print(f"Q{i}: {payload['qid']} ...", end=" ", flush=True)

        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": payload["system"]},
                    {"role": "user", "content": payload["user_parts"]},
                ],
                response_format=OPENAI_QA_RESPONSE_SCHEMA,
                max_completion_tokens=4096,
                reasoning_effort="low",
            )

            usage = response.usage
            result = json.loads(response.choices[0].message.content)

            print(f"OK | in={usage.prompt_tokens:,} out={usage.completion_tokens:,} "
                  f"cached={usage.prompt_tokens_details.cached_tokens:,}")
            print(f"  has_error={result.get('has_error')}", end="")
            if result.get("errors"):
                for err in result["errors"][:2]:
                    print(f" | [{err.get('type')}] {err.get('detail','')[:60]}", end="")
            print()

        except Exception as e:
            print(f"ERROR: {e}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="build", choices=["build", "test"])
    args = ap.parse_args()

    if args.mode == "build":
        build_mode()
    elif args.mode == "test":
        test_mode()


if __name__ == "__main__":
    main()
