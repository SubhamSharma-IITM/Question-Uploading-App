"""Regression: Stage 4 orchestration `_meta_is_triplet` must NOT quarantine
match + passage questions whose canonical containers are valid.

Bug: pre-fix orchestration treated matchQuestion as non_triplet because match
container has answer="" (the answer lives in `rows`); and passages because
sub-Q field name is `ques` (canonical) while triplet check looked for `question`.
Result: ALL 19 match + passage Q's across 6 production sessions were quarantined
as non_triplet, never reaching Stage 9.

Fix:
- matchQuestion branch: require `question` HTML + non-empty `rows` (skip answer)
- passage sub-Q field: accept both `ques` (canonical) and `question` (legacy)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.pipeline.stage2_llm_mapper import _convert_match_answer_mapping
from app.pipeline.stage4_metadata import _build_metadata_v2


def _meta_is_triplet(m):
    """Mirror of stage4_metadata.py orchestration triplet check (POST-fix)."""
    qtype = m["questionType"]
    if qtype == "passageQuestion":
        pq = m.get(qtype, {})
        has_p = bool(pq.get("passage", "").replace("<h3></h3>", ""))
        subs = pq.get("questions", [])
        if not has_p or not subs:
            return False
        for sq in subs:
            if not sq:
                return False
            sq_type = list(sq.keys())[0]
            sq_data = sq[sq_type]
            sq_q = sq_data.get("ques", sq_data.get("question", ""))
            if not (sq_q.replace("<h3></h3>", "") and sq_data.get("answer", "")):
                return False
        return True
    if qtype == "matchQuestion":
        mq = m.get(qtype, {})
        has_q = bool((mq.get("question", "") or "").replace("<h3></h3>", ""))
        rows = mq.get("rows", [])
        has_rows = isinstance(rows, list) and any(isinstance(r, list) and r for r in rows)
        return has_q and has_rows
    qdata = m.get(qtype, {})
    return (bool(qdata.get("question", "").replace("<h3></h3>", ""))
            and bool(qdata.get("answer", ""))
            and bool(qdata.get("solution", "").replace("<h3></h3>", "")))


MATCH_FIXTURE = ROOT / "tests" / "integration" / "fixtures" / "replay" / "vector_algebra_match_8q.json"
PASSAGE_FIXTURE = ROOT / "tests" / "integration" / "fixtures" / "replay" / "vector_algebra_passage_2q.json"


def test_all_match_qs_pass_triplet():
    match_qs = json.loads(MATCH_FIXTURE.read_text(encoding="utf-8"))
    failures = []
    for q in match_qs:
        # Re-canonicalize (fixture's stored answer_mapping was pre-fix)
        q = dict(q)
        q["answer_mapping"] = _convert_match_answer_mapping(q.get("answer_mapping_source", {}))
        q.setdefault("subject", "Mathematics")
        meta = _build_metadata_v2(q, source_name="EVAM", test_code="EVAM", tagged_by="AITAI", is_pyq=False)
        if not _meta_is_triplet(meta):
            failures.append(q["qid"])
    assert not failures, f"match Q's wrongly quarantined: {failures}"


def test_all_passage_qs_pass_triplet():
    passage_qs = json.loads(PASSAGE_FIXTURE.read_text(encoding="utf-8"))
    failures = []
    for q in passage_qs:
        q = dict(q)
        q.setdefault("subject", "Mathematics")
        meta = _build_metadata_v2(q, source_name="EVAM", test_code="EVAM", tagged_by="AITAI", is_pyq=False)
        if not _meta_is_triplet(meta):
            failures.append(q["qid"])
    assert not failures, f"passage Q's wrongly quarantined: {failures}"


def test_match_with_all_empty_rows_correctly_fails_triplet():
    """If rows is all-empty, match is genuinely broken — quarantine is correct."""
    meta = {
        "questionType": "matchQuestion",
        "matchQuestion": {
            "rows": [[], [], [], []],
            "question": "<table>...</table>",
            "solution": "",
            "quesImages": None,
            "solutionImages": None,
        }
    }
    assert _meta_is_triplet(meta) is False


def test_match_with_no_question_html_correctly_fails_triplet():
    meta = {
        "questionType": "matchQuestion",
        "matchQuestion": {
            "rows": [["A"], ["B"]],
            "question": "",
            "solution": "",
            "quesImages": None,
            "solutionImages": None,
        }
    }
    assert _meta_is_triplet(meta) is False


def test_passage_with_canonical_ques_field_passes():
    """Sub-Q canonical schema uses 'ques' field — must be accepted."""
    meta = {
        "questionType": "passageQuestion",
        "passageQuestion": {
            "passage": "<h3>passage body</h3>",
            "solution": "",
            "questions": [
                {"scq": {"ques": "<h3>Q1</h3>", "solution": "<h3>S1</h3>",
                         "answer": "C", "passage": "P11"}},
                {"scq": {"ques": "<h3>Q2</h3>", "solution": "<h3>S2</h3>",
                         "answer": "B", "passage": "P11"}},
            ],
            "passageImages": [],
            "passageSolutionImages": [],
        }
    }
    assert _meta_is_triplet(meta) is True


def test_passage_with_legacy_question_field_still_passes():
    """Legacy schema with 'question' field must still be accepted (back-compat)."""
    meta = {
        "questionType": "passageQuestion",
        "passageQuestion": {
            "passage": "<h3>passage body</h3>",
            "solution": "",
            "questions": [
                {"integerQuestion": {"question": "<h3>Q1</h3>", "solution": "<h3>S1</h3>",
                                     "answer": "42", "passage": "P11"}},
            ],
        }
    }
    assert _meta_is_triplet(meta) is True


if __name__ == "__main__":
    test_all_match_qs_pass_triplet()
    test_all_passage_qs_pass_triplet()
    test_match_with_all_empty_rows_correctly_fails_triplet()
    test_match_with_no_question_html_correctly_fails_triplet()
    test_passage_with_canonical_ques_field_passes()
    test_passage_with_legacy_question_field_still_passes()
    print("OK")
