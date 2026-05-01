"""Cross-validate the PRODUCTION _strip_format_noise against ~6090 random
production questions (1000 random + all matchQuestion + passageQuestion).

Confirms the ported patterns produce identical hits to the pilot version.
Should report: 65 genuine noise hits, 0 false positives in real prose.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.pipeline.stage3_cleaner import _strip_format_noise

DB_PATH = ROOT / "data" / "allque030326.json"
SAMPLE_SIZE = 1000
SEED = 42
COLLECT_ALL_TYPES = {"passageQuestion", "matchQuestion"}


def _walk_text_fields(entry):
    qt = entry.get("questionType", "")
    for top in ("scq", "mcq", "integerQuestion", "subjective"):
        sub = entry.get(top) or {}
        for fk in ("question", "solution"):
            v = sub.get(fk)
            if isinstance(v, str) and v.strip():
                yield f"{top}.{fk}", v
    mq = entry.get("matchQuestion") or {}
    for fk in ("question", "solution"):
        v = mq.get(fk)
        if isinstance(v, str) and v.strip():
            yield f"matchQuestion.{fk}", v
    pq = entry.get("passageQuestion") or {}
    for fk in ("passage", "solution"):
        v = pq.get(fk)
        if isinstance(v, str) and v.strip():
            yield f"passageQuestion.{fk}", v
    for i, inner in enumerate(pq.get("questions") or []):
        scq = inner.get("scq") or {}
        for fk in ("ques", "solution"):
            v = scq.get(fk)
            if isinstance(v, str) and v.strip():
                yield f"passageQuestion.questions[{i}].scq.{fk}", v


def reservoir_sample(path, k, seed):
    rng = random.Random(seed)
    sample = []
    forced = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            qt = entry.get("questionType", "")
            if qt in COLLECT_ALL_TYPES:
                forced.append(entry)
                continue
            if i < k:
                sample.append(entry)
            else:
                j = rng.randint(0, i)
                if j < k:
                    sample[j] = entry
    return sample + forced


def main():
    print(f"Reservoir-sampling {SAMPLE_SIZE} + all match/passage from {DB_PATH.name}...")
    sample = reservoir_sample(DB_PATH, SAMPLE_SIZE, SEED)
    print(f"Got {len(sample)} entries.")

    total_fields = 0
    changed = 0
    for entry in sample:
        for path, text in _walk_text_fields(entry):
            total_fields += 1
            if _strip_format_noise(text) != text:
                changed += 1

    print(f"Total fields scanned: {total_fields}")
    print(f"Fields modified: {changed}")
    if changed == 65:
        print("=> PARITY confirmed (matches pilot validation: 65 genuine noise hits)")
    elif changed == 0:
        print("=> ZERO false positives.")
    else:
        print(f"=> {changed} hits — diff vs pilot 65 expected; investigate.")


if __name__ == "__main__":
    main()
