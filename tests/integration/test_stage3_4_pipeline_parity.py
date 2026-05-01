"""End-to-end pipeline parity: run the 22 v2 pilot entries through PRODUCTION
stage3+4 and confirm output matches the pilot rebuild.

This is the critical correctness gate per user direction:
"compilation is not equals to confirmation. If test cases that passed through
the actual pilot phase properly also passed through production, then we are
done."
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PILOT_DIR = ROOT / "tests" / "_pilot" / "_pilot_v2_session"


def main():
    # Use PRODUCTION stage3 + stage4
    from app.pipeline.stage3_cleaner import _clean_question_entry
    from app.pipeline.stage4_metadata import _build_metadata_v2

    selected = json.loads((PILOT_DIR / "02b_selected.json").read_text(encoding="utf-8"))
    expected = json.loads((PILOT_DIR / "rebuilt" / "04_metadata_v2.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(selected)} entries from 02b_selected.json")
    print(f"Loaded {len(expected)} expected entries from pilot rebuilt 04_metadata_v2.json")

    # Stage 3 cleaning
    cleaned = [_clean_question_entry(e) for e in selected]

    # Stage 4 metadata build
    actual = []
    for e in cleaned:
        actual.append(_build_metadata_v2(e, "pilot_match_passage_v2", "PILOT_V2", tagged_by="AITAI"))

    if len(actual) != len(expected):
        raise AssertionError(f"Length mismatch: actual={len(actual)} expected={len(expected)}")

    # Compare critical fields per entry
    ok = 0
    diffs = []
    for i, (a, e) in enumerate(zip(actual, expected)):
        a_qt = a.get("questionType")
        e_qt = e.get("questionType")
        if a_qt != e_qt:
            diffs.append(f"[{i}] questionType: {a_qt} vs {e_qt}")
            continue
        # Compare the active container
        a_c = a.get(a_qt, {})
        e_c = e.get(e_qt, {})
        # For passage: passage HTML, sub-question HTMLs
        # For match: question HTML, solution HTML, rows
        # For scq/etc: question HTML, solution HTML
        for key in sorted(set(a_c.keys()) | set(e_c.keys())):
            if a_c.get(key) != e_c.get(key):
                diffs.append(f"[{i}] {a_qt}.{key} differs (lengths {len(str(a_c.get(key)))} vs {len(str(e_c.get(key)))})")
        if a_c == e_c:
            ok += 1

    print(f"\n=> Container parity: {ok}/{len(actual)}")
    if diffs:
        print(f"\nDiff summary ({len(diffs)} differences):")
        for d in diffs[:30]:
            print(f"  {d}")
        if len(diffs) > 30:
            print(f"  ... and {len(diffs) - 30} more")
        raise AssertionError(f"{len(diffs)} container differences vs pilot rebuild")
    print("ALL ENTRIES PRODUCE IDENTICAL CONTAINERS — production stage3+4 matches pilot rebuild")


if __name__ == "__main__":
    main()
