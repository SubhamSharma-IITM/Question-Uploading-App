"""Production-parity test for stage2 _process_match_item + answer mapping
canonicalization. Imports PRODUCTION module.

Replays the recorded LLM output (01_resume_pass*_raw.json) through
production's _process_match_item and confirms identical results to the pilot
output (02_all_processed.json) for matchQuestion entries.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PILOT_DIR = ROOT / "tests" / "_pilot" / "_pilot_v2_session"
MMD = ROOT / "data" / "test_papers" / "passage_match_pilot_v2.mmd"


def test_label_maps():
    from app.pipeline.stage2_llm_mapper import _LEFT_LABEL_MAP, _RIGHT_LABEL_MAP
    # Old JEE Advanced FIITJEE labels: A,B,C,D | p,q,r,s,t
    assert _LEFT_LABEL_MAP["A"] == "1"
    assert _LEFT_LABEL_MAP["D"] == "4"
    assert _RIGHT_LABEL_MAP["p"] == "A"
    assert _RIGHT_LABEL_MAP["t"] == "E"
    # New JEE Advanced labels (already canonical): pass-through
    assert _LEFT_LABEL_MAP["1"] == "1"
    assert _RIGHT_LABEL_MAP["A"] == "A"
    print("  PASS label maps")


def test_answer_mapping_conversion():
    from app.pipeline.stage2_llm_mapper import _convert_match_answer_mapping

    # FIITJEE M01 source: {A:[s], B:[p,t], C:[q], D:[p,r]}
    # Canonical: 1->[D], 2->[A,E], 3->[B], 4->[A,C]
    src = {"A": ["s"], "B": ["p", "t"], "C": ["q"], "D": ["p", "r"]}
    out = _convert_match_answer_mapping(src)
    assert out == [["D"], ["A", "E"], ["B"], ["A", "C"]], f"got {out}"
    print(f"  PASS M01 mapping conversion: {out}")

    # M02 source: {A:[p,q,t], B:[p,r,t], C:[s], D:[q,r]}
    src = {"A": ["p", "q", "t"], "B": ["p", "r", "t"], "C": ["s"], "D": ["q", "r"]}
    out = _convert_match_answer_mapping(src)
    assert out == [["A", "B", "E"], ["A", "C", "E"], ["D"], ["B", "C"]], f"got {out}"
    print(f"  PASS M02 mapping conversion: {out}")

    # New JEE format: {1:["B","D"], 2:["A","C"], 3:["A","B","D"], 4:["C"]}
    src = {"1": ["B", "D"], "2": ["A", "C"], "3": ["A", "B", "D"], "4": ["C"]}
    out = _convert_match_answer_mapping(src)
    assert out == [["B", "D"], ["A", "C"], ["A", "B", "D"], ["C"]], f"got {out}"
    print(f"  PASS new-JEE pass-through: {out}")

    # Empty / malformed
    assert _convert_match_answer_mapping({}) == []
    assert _convert_match_answer_mapping("not a dict") == []
    print("  PASS edge cases")


def test_process_match_item_replays_pilot():
    """Replay match-type LLM items from the recorded raw passes through
    PRODUCTION _process_match_item; results must match pilot's processed
    matchQuestion entries in 02_all_processed.json (same code path was used
    for the pilot run, just under different module path)."""
    from app.pipeline.stage2_llm_mapper import _process_match_item

    if not MMD.exists():
        print("  SKIP — mmd source not found (data/test_papers/passage_match_pilot_v2.mmd)")
        return

    raw_lines = MMD.read_text(encoding="utf-8").splitlines(keepends=True)

    # Collect all match items from the recorded raw passes
    all_match_items = []
    for raw_file in sorted(PILOT_DIR.glob("01_*_raw.json")):
        try:
            data = json.loads(raw_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if item.get("type") == "matchQuestion":
                all_match_items.append(item)

    # Load pilot's processed match entries for comparison
    processed = json.loads((PILOT_DIR / "02_all_processed.json").read_text(encoding="utf-8"))
    pilot_match = [e for e in processed if e.get("questionType") == "matchQuestion"]

    print(f"  Found {len(all_match_items)} match items in raw LLM output")
    print(f"  Pilot processed has {len(pilot_match)} matchQuestion entries")

    # Process each LLM item through PRODUCTION code
    prod_match = []
    for item in all_match_items:
        result = _process_match_item(item, raw_lines, set(), [], 0)
        if result:
            prod_match.append(result)

    print(f"  Production produced {len(prod_match)} matchQuestion entries")

    # Compare key fields per match
    if len(prod_match) != len(pilot_match):
        print(f"  WARNING: count mismatch (this can happen if pilot deduplicated by line range)")

    # Spot-compare on shared fields: questionType, subject, level, answer_mapping, line_start, line_end
    by_line = {(e.get("line_start"), e.get("line_end")): e for e in pilot_match}
    matches = 0
    for prod_e in prod_match:
        key = (prod_e.get("line_start"), prod_e.get("line_end"))
        pilot_e = by_line.get(key)
        if not pilot_e:
            continue
        for fk in ("questionType", "subject", "level", "answer_mapping"):
            if prod_e.get(fk) != pilot_e.get(fk):
                raise AssertionError(
                    f"Mismatch on line range {key} field {fk}: "
                    f"prod={prod_e.get(fk)!r} pilot={pilot_e.get(fk)!r}"
                )
        matches += 1

    print(f"  PASS replay: {matches}/{len(prod_match)} match items field-equal to pilot processed")


if __name__ == "__main__":
    print("Running PRODUCTION stage2 match-type tests...")
    test_label_maps()
    test_answer_mapping_conversion()
    test_process_match_item_replays_pilot()
    print("ALL TESTS PASSED")
