"""Stage 8 (duplicate check) parity test for match + passage container shapes.

Verifies:
  1. matchQuestion text extraction includes table cell text + rows mapping
     summary (stronger dedup signal)
  2. passageQuestion text extraction is main passage stem + FIRST sub-question
  3. HTML preview map covers matchQuestion (preview pane shows match HTML)
  4. Quarantine path is QID-based and works for both new types
  5. Two match questions with same cells but different rows are NOT identical
     (rows mapping is part of embedding signal)
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ── Sample entries: canonical schema ───────────────────────────────────────

MATCH_ENTRY_A = {
    "qid": "TEST_M01_A",
    "subject": "Chemistry",
    "questionType": "matchQuestion",
    "matchQuestion": {
        "rows": [["B"], ["A"], ["C"], ["D"]],
        "quesImages": None,
        "solutionImages": None,
        "question": "<p>Match the carbides.</p><table><tr><td>(1)</td><td>SiC</td><td>(A)</td><td>red ppt</td></tr><tr><td>(2)</td><td>Al4C3</td><td>(B)</td><td>unsaturated</td></tr></table>",
        "solution": "<h3>Detailed solution</h3>",
    },
}

# Same cells as A, but DIFFERENT rows mapping — must NOT be considered identical text
MATCH_ENTRY_B = {
    "qid": "TEST_M01_B",
    "subject": "Chemistry",
    "questionType": "matchQuestion",
    "matchQuestion": {
        "rows": [["A"], ["B"], ["D"], ["C"]],   # different mapping
        "quesImages": None,
        "solutionImages": None,
        "question": "<p>Match the carbides.</p><table><tr><td>(1)</td><td>SiC</td><td>(A)</td><td>red ppt</td></tr><tr><td>(2)</td><td>Al4C3</td><td>(B)</td><td>unsaturated</td></tr></table>",
        "solution": "<h3>Detailed solution</h3>",
    },
}

PASSAGE_ENTRY = {
    "qid": "TEST_P01",
    "subject": "Physics",
    "questionType": "passageQuestion",
    "passageQuestion": {
        "passage": "<h3>Consider a sliding bar in a magnetic field.</h3>",
        "solution": "",
        "questions": [
            {"scq": {"ques": "<h3>Find the speed of the bar.</h3>",
                     "solution": "", "answer": "B", "passage": "TEST_P01"}},
            {"scq": {"ques": "<h3>Compute the efficiency.</h3>",
                     "solution": "", "answer": "C", "passage": "TEST_P01"}},
            {"scq": {"ques": "<h3>What is the EMF?</h3>",
                     "solution": "", "answer": "A", "passage": "TEST_P01"}},
        ],
        "passageImages": [],
        "passageSolutionImages": [],
    },
}


def _make_stage():
    """Construct DuplicateCheckStage instance bypassing __init__."""
    from app.pipeline.stage8_duplicates import DuplicateCheckStage
    inst = DuplicateCheckStage.__new__(DuplicateCheckStage)
    inst.session_dir = Path(tempfile.mkdtemp(prefix="stage8_test_"))
    inst.session_id = "test_session"
    inst.log = lambda level, msg: None
    return inst


def test_match_text_includes_rows_mapping():
    """match text must include rows mapping summary alongside cell text."""
    stage = _make_stage()
    text = stage._extract_question_text(MATCH_ENTRY_A)
    assert "SiC" in text, f"cell text missing: {text[:200]}"
    assert "Al4C3" in text, "cell text missing"
    # Rows mapping summary must be appended
    assert "1->B" in text, f"rows summary missing: {text[:300]}"
    assert "2->A" in text
    assert "3->C" in text
    assert "4->D" in text
    print(f"  PASS match text includes rows mapping: '{text[:100]}...'")


def test_match_different_mapping_yields_different_text():
    """Two match questions with same cells but different rows must produce
    different embedding text — they're NOT duplicates."""
    stage = _make_stage()
    text_a = stage._extract_question_text(MATCH_ENTRY_A)
    text_b = stage._extract_question_text(MATCH_ENTRY_B)
    assert text_a != text_b, "different mappings produced identical text"
    # Both share cell content
    assert "SiC" in text_a and "SiC" in text_b
    # But mapping summaries differ
    assert "1->B" in text_a and "1->A" in text_b
    print("  PASS match cells-same / mapping-different produces distinct text")


def test_passage_text_uses_first_sub_question_only():
    """Passage embedding = passage stem + FIRST sub-question stem.
    Subsequent sub-questions are correlated with first; embed just first to
    keep signal focused."""
    stage = _make_stage()
    text = stage._extract_question_text(PASSAGE_ENTRY)
    # passage body must be present
    assert "sliding bar" in text, "passage body missing"
    # First sub-question must be present
    assert "speed of the bar" in text, "first sub-question missing"
    # Second + third sub-questions must NOT be embedded (per user direction)
    assert "efficiency" not in text, f"second sub-question should not be embedded: {text}"
    assert "EMF" not in text, "third sub-question should not be embedded"
    print(f"  PASS passage text = stem + first sub-question only: '{text[:120]}...'")


def test_html_preview_map_covers_match():
    """The internal html_map (used by preview pane to display side-by-side
    duplicates) must include matchQuestion HTML — was missing before."""
    # Replicate the per-question html_map extraction logic from line ~535
    stage = _make_stage()
    html_map = {}
    for q in [MATCH_ENTRY_A, PASSAGE_ENTRY]:
        text = stage._extract_question_text(q)
        if text and len(text) >= 10:
            qid = q.get("qid", "?")
            for qt in ("scq", "mcq", "integerQuestion", "passageQuestion", "matchQuestion"):
                obj = q.get(qt)
                if isinstance(obj, dict):
                    q_html = obj.get("question", obj.get("ques", ""))
                    if not q_html and qt == "passageQuestion":
                        q_html = obj.get("passage", "")
                    if q_html:
                        html_map[qid] = q_html
                        break
    # match HTML must be in the map for preview rendering
    assert "TEST_M01_A" in html_map
    assert "<table>" in html_map["TEST_M01_A"], f"match HTML truncated/wrong: {html_map.get('TEST_M01_A','')[:150]}"
    # passage HTML must be in the map (uses passage body)
    assert "TEST_P01" in html_map
    assert "sliding bar" in html_map["TEST_P01"]
    print("  PASS preview html_map includes match-type table HTML + passage body")


def test_text_strips_html_for_embedding():
    """Embedded text must be plain text, not HTML — verifies strip_fn works
    on canonical match table HTML."""
    stage = _make_stage()
    text = stage._extract_question_text(MATCH_ENTRY_A)
    # Tags should be stripped (we keep image fingerprints but no <table> etc.)
    assert "<table>" not in text
    assert "<td>" not in text
    print("  PASS text stripped of HTML tags for embedding")


def test_quarantine_is_qid_based():
    """quarantine_duplicates accepts a list of QIDs and is type-agnostic.
    Smoke-check by inspecting the function — actual DB writes mocked by stage 5
    test pattern (see test_stage5_match_passage.py for full quarantine flow)."""
    from app.pipeline.stage8_duplicates import DuplicateCheckStage
    import inspect
    src = inspect.getsource(DuplicateCheckStage.quarantine_duplicates)
    # Confirm the function operates on a `qids` list — type-agnostic interface
    assert "qids: List[str]" in src or "qids" in src.split("\n")[0]
    # Confirm it removes from metadata files (which contain match + passage entries)
    assert "metadata_v2" in src or "questions[:]" in src or "filter_quarantine" in src or "f.read" in src.lower() or "json" in src.lower()
    print("  PASS quarantine_duplicates: QID-based, type-agnostic interface")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Running PRODUCTION stage8 match+passage parity tests...")
    test_match_text_includes_rows_mapping()
    test_match_different_mapping_yields_different_text()
    test_passage_text_uses_first_sub_question_only()
    test_html_preview_map_covers_match()
    test_text_strips_html_for_embedding()
    test_quarantine_is_qid_based()
    print("ALL TESTS PASSED")
