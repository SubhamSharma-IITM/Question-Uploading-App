"""Real-data regression: 8 match-type questions extracted by Stage 2 from
Vector Algebra .mmd. Pre-fix, 5 of 8 had all-empty canonical rows due to digit
RHS labels not being mapped. Post-fix, all 8 must produce non-empty rows AND
pass through Stage 4 to a valid match container.

Fixture: tests/integration/fixtures/replay/vector_algebra_match_8q.json (8 entries)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pipeline.stage2_llm_mapper import _convert_match_answer_mapping
from app.pipeline.stage4_metadata import _build_metadata_v2

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "vector_algebra_match_8q.json"


def test_all_8_match_qs_canonicalize_post_fix():
    match_qs = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(match_qs) == 8, f"fixture should have 8 match Q's, got {len(match_qs)}"

    broken = []
    for q in match_qs:
        canon = _convert_match_answer_mapping(q.get("answer_mapping_source", {}))
        if not canon or all(not r for r in canon):
            broken.append((q.get("qid"), q.get("answer_mapping_source")))
    assert not broken, (
        f"All 8 match Q's must produce non-empty canonical rows. Broken: {broken}"
    )


def test_stage4_match_container_built_for_all_8():
    match_qs = json.loads(FIXTURE.read_text(encoding="utf-8"))
    failures = []
    for q in match_qs:
        # Re-canonicalize with current code (fixture's stored answer_mapping was pre-fix)
        q = dict(q)
        q["answer_mapping"] = _convert_match_answer_mapping(q.get("answer_mapping_source", {}))
        q.setdefault("subject", "Mathematics")
        try:
            meta = _build_metadata_v2(
                entry=q,
                source_name="EVAM_REPLAY",
                test_code="EVAM_REPLAY",
                tagged_by="AITAI",
                is_pyq=False,
            )
        except Exception as exc:
            failures.append((q["qid"], f"{type(exc).__name__}: {exc}"))
            continue

        mc = meta.get("matchQuestion", {})
        rows = mc.get("rows")
        q_html = mc.get("question", "") or ""

        if not rows or all(not r for r in rows):
            failures.append((q["qid"], f"empty rows {rows}"))
        if not q_html or "<table" not in q_html.lower():
            failures.append((q["qid"], "missing/non-table question HTML"))

    assert not failures, f"Stage 4 failures on real match data: {failures}"


def test_specific_digit_rhs_cases_now_correct():
    """Lock the exact pre-fix-broken QIDs and their expected post-fix canonical."""
    match_qs = json.loads(FIXTURE.read_text(encoding="utf-8"))
    by_qid = {q["qid"]: q for q in match_qs}

    expected = {
        "EVAM_REPLAY_MATH_Q59":  [["C"], ["D"], ["A"], ["B"]],
        "EVAM_REPLAY_MATH_Q62":  [["D"], ["C"], ["B"], ["A"]],
        "EVAM_REPLAY_MATH_Q78":  [["E"], ["D"], ["C"], ["A"]],
        "EVAM_REPLAY_MATH_Q120": [["C"], ["B"], ["B"], ["D"]],
        "EVAM_REPLAY_MATH_Q121": [["B"], ["A", "C"], ["C"], ["D"]],
    }
    for qid, exp_canon in expected.items():
        q = by_qid[qid]
        canon = _convert_match_answer_mapping(q["answer_mapping_source"])
        assert canon == exp_canon, f"{qid}: expected {exp_canon}, got {canon}"


if __name__ == "__main__":
    test_all_8_match_qs_canonicalize_post_fix()
    test_stage4_match_container_built_for_all_8()
    test_specific_digit_rhs_cases_now_correct()
    print("OK")
