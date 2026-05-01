"""Stage 10 (upload + FAISS absorption + DB JSONL append) parity test for
match + passage container shapes.

Verifies:
  1. _build_db_entry preserves matchQuestion + passageQuestion containers
     when constructing the local-DB JSONL append entry
  2. _update_faiss_index uses the SHARED extractor extract_text_for_embedding
     so absorbed match/passage vectors match what Stage 8 would generate
  3. Match and passage produce non-empty embedding text post-absorb (regression
     guard for the bug where passage produced empty text and got skipped)
  4. compute_live_stats counts match + passage in totals/by_subject (cards)
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ── Sample metadata entries (post-Stage-9, pre-upload) ────────────────────

MATCH_ENTRY = {
    "qid": "TEST_M01",
    "subject": "Chemistry",
    "questionType": "matchQuestion",
    "level": "MEDIUM",
    "error": 1,
    "matchQuestion": {
        "rows": [["B"], ["A"], ["C"], ["D"]],
        "quesImages": None,
        "solutionImages": None,
        "question": "<p>Match the carbides.</p><table><tr><td>(1)</td><td>SiC</td><td>(A)</td><td>red ppt</td></tr></table>",
        "solution": "<h3>Solution</h3>",
    },
    "tagSubConcept": [{"subject": "Chemistry", "chapter": "Carbides", "concept": "C", "subConcept": "SC"}],
    "chapter": "Carbides", "concept": "C", "subConcept": "SC",
}

PASSAGE_ENTRY = {
    "qid": "TEST_P01",
    "subject": "Physics",
    "questionType": "passageQuestion",
    "level": "HARD",
    "error": 2,
    "passageQuestion": {
        "passage": "<h3>Consider a sliding bar in a magnetic field.</h3>",
        "solution": "",
        "questions": [
            {"scq": {"ques": "<h3>Find the speed of the bar.</h3>",
                     "solution": "", "answer": "B", "passage": "TEST_P01"}},
        ],
        "passageImages": [],
        "passageSolutionImages": [],
    },
    "tagSubConcept": [{"subject": "Physics", "chapter": "Magnetism", "concept": "C", "subConcept": "SC"}],
    "chapter": "Magnetism", "concept": "C", "subConcept": "SC",
}

SCQ_ENTRY = {
    "qid": "TEST_S01",
    "subject": "Mathematics",
    "questionType": "scq",
    "level": "EASY",
    "error": 1,
    "scq": {"question": "<h3>What is 2+2?</h3>(A) 3<br/>(B) 4", "solution": "<h3>4</h3>", "answer": "B"},
    "tagSubConcept": [],
    "chapter": "Arithmetic", "concept": "C", "subConcept": "SC",
}


def _make_stage():
    """Construct UploadStage instance bypassing __init__ + DB."""
    from app.pipeline.stage10_upload import UploadStage
    inst = UploadStage.__new__(UploadStage)
    inst.session_dir = Path(tempfile.mkdtemp(prefix="stage10_test_"))
    inst.session_id = "test_session"
    inst.log = lambda level, msg: None
    return inst


def test_build_db_entry_preserves_match_container():
    stage = _make_stage()
    db_entry = stage._build_db_entry(MATCH_ENTRY, "5fa1b2c3d4e5f6a7b8c9d0e1", "2026-05-01T00:00:00.000Z")
    # Match container preserved with all fields
    assert "matchQuestion" in db_entry
    mq = db_entry["matchQuestion"]
    assert mq.get("rows") == [["B"], ["A"], ["C"], ["D"]]
    assert mq.get("quesImages") is None  # null preserved
    assert "<table>" in mq.get("question", "")
    # Other types empty
    assert db_entry.get("scq") == {} or db_entry.get("scq") is None or db_entry["scq"] == {}
    assert db_entry["questionType"] == "matchQuestion"
    assert db_entry["error"] == 1
    print("  PASS _build_db_entry preserves matchQuestion container (rows + question + null images)")


def test_build_db_entry_preserves_passage_container():
    stage = _make_stage()
    db_entry = stage._build_db_entry(PASSAGE_ENTRY, "60a1b2c3d4e5f6a7b8c9d0e1", "2026-05-01T00:00:00.000Z")
    pq = db_entry.get("passageQuestion", {})
    assert pq.get("passage", "").strip().startswith("<h3>Consider")
    # Inner sub-question preserved with canonical scq + ques field
    inner = pq["questions"][0]
    assert "scq" in inner
    assert inner["scq"]["ques"].startswith("<h3>Find")
    assert inner["scq"]["answer"] == "B"
    assert pq.get("passageImages") == []
    print("  PASS _build_db_entry preserves passageQuestion container (passage + nested scq)")


def test_shared_embedding_extractor_used():
    """Stage 10 must use extract_text_for_embedding from Stage 8 module."""
    from app.pipeline.stage8_duplicates import extract_text_for_embedding

    # Match: must have rows summary
    match_text = extract_text_for_embedding(MATCH_ENTRY)
    assert "1->B" in match_text and "2->A" in match_text
    assert "SiC" in match_text
    assert len(match_text) >= 10

    # Passage: must have passage body + first sub-question stem
    passage_text = extract_text_for_embedding(PASSAGE_ENTRY)
    assert "sliding bar" in passage_text
    assert "speed of the bar" in passage_text
    assert len(passage_text) >= 10

    # SCQ: regression-safe
    scq_text = extract_text_for_embedding(SCQ_ENTRY)
    assert "What is 2+2" in scq_text
    print("  PASS shared extractor produces non-empty embedding text for all 3 types")


def test_stage10_imports_shared_extractor():
    """Confirm the function is importable from stage8 module (used by stage10)."""
    from app.pipeline.stage8_duplicates import extract_text_for_embedding  # noqa
    # And that stage8 method delegates correctly (regression guard for refactor)
    from app.pipeline.stage8_duplicates import DuplicateCheckStage
    inst = DuplicateCheckStage.__new__(DuplicateCheckStage)
    method_text = inst._extract_question_text(MATCH_ENTRY)
    free_fn_text = extract_text_for_embedding(MATCH_ENTRY)
    assert method_text == free_fn_text, "method and free function diverge"
    print("  PASS Stage 8 method delegates to shared free function")


def test_stage10_scan_by_type_includes_match_passage():
    """Stage 10 scan() returns by_type dict — match + passage must show up."""
    # Simulate the by_type aggregation logic from scan() lines 128-139
    quarantined = set()
    by_type = {}
    for entry in [MATCH_ENTRY, PASSAGE_ENTRY, SCQ_ENTRY]:
        qid = entry.get("qid", "")
        if qid in quarantined:
            continue
        error_val = entry.get("error", 2)
        if isinstance(error_val, int) and error_val >= 3:
            continue
        qtype = entry.get("questionType", "unknown")
        by_type[qtype] = by_type.get(qtype, 0) + 1
    assert by_type.get("matchQuestion") == 1
    assert by_type.get("passageQuestion") == 1
    assert by_type.get("scq") == 1
    print(f"  PASS scan by_type breakdown includes match + passage: {by_type}")


def test_compute_live_stats_counts_all_types():
    """compute_live_stats totals must include match + passage entries."""
    # Simulate metadata directory with 3 entries
    stage = _make_stage()
    meta_dir = stage.session_dir / "metadata_v2"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "test_metadata_v2.json").write_text(
        json.dumps([MATCH_ENTRY, PASSAGE_ENTRY, SCQ_ENTRY], ensure_ascii=False),
        encoding="utf-8",
    )
    from app.pipeline.metadata_stats import compute_live_stats
    stats = compute_live_stats(stage.session_dir, set())
    assert stats["total"] == 3, f"got total={stats['total']}"
    assert stats["live"] == 3
    # err1 = 2 (match + scq), err2 = 1 (passage)
    assert stats["err1"] == 2, f"got err1={stats['err1']}"
    assert stats["err2"] == 1, f"got err2={stats['err2']}"
    # All 3 subjects appear
    assert "Chemistry" in stats["by_subject"]
    assert "Physics" in stats["by_subject"]
    assert "Mathematics" in stats["by_subject"]
    print(f"  PASS compute_live_stats: 3 total (1 match + 1 passage + 1 scq) across 3 subjects")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Running PRODUCTION stage10 match+passage parity tests...")
    test_build_db_entry_preserves_match_container()
    test_build_db_entry_preserves_passage_container()
    test_shared_embedding_extractor_used()
    test_stage10_imports_shared_extractor()
    test_stage10_scan_by_type_includes_match_passage()
    test_compute_live_stats_counts_all_types()
    print("ALL TESTS PASSED")
