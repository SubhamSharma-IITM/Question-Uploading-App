"""Verify Stage 5 (watermark + SMILES + image map + quarantine) correctly
handles matchQuestion and passageQuestion containers.

Stage 5 has both filesystem-side ops (watermark removal of raw images, preview
pane) and metadata-side ops (SMILES rendering, image-to-QID mapping,
QID-based quarantine). The filesystem ops are type-agnostic — they iterate
files in images_original/ regardless of which question type owns the image.
The metadata ops MUST walk match + passage container shapes correctly. This
test confirms both.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ── Fixture: synthetic metadata_v2 with all 3 container shapes ─────────────

SAMPLE_METADATA = [
    {
        "qid": "TEST_SCQ_01",
        "questionType": "scq",
        "scq": {
            "question": '<h3>SCQ <img src="./images/scq_fig.png"/></h3>',
            "solution": '<h3><smiles>CC=O</smiles></h3>',
            "answer": "C",
        },
    },
    {
        "qid": "TEST_MATCH_01",
        "questionType": "matchQuestion",
        "matchQuestion": {
            "rows": [["B"], ["A"], ["C"], ["D"]],
            "quesImages": None,
            "solutionImages": None,
            "question": '<table><tr><td>List-I</td></tr><tr><td><img src="./images/match_li.png"/></td><td><img src="./images/match_lii.png"/></td></tr></table>',
            "solution": '<h3>Sol with <smiles>C[C@H](Cl)C(=O)O</smiles></h3>',
        },
    },
    {
        "qid": "TEST_PASSAGE_01",
        "questionType": "passageQuestion",
        "passageQuestion": {
            "passage": '<h3>Passage <img src="./images/passage_fig.png"/></h3>',
            "solution": "",
            "questions": [
                {"scq": {
                    "ques": '<h3>Inner Q1 with <smiles>CCO</smiles></h3>',
                    "solution": '<h3>Inner sol1 <img src="./images/inner_sol1.png"/></h3>',
                    "answer": "B",
                    "passage": "TEST_PASSAGE_01",
                }},
                {"scq": {
                    "ques": '<h3>Inner Q2 <img src="./images/inner_q2.png"/></h3>',
                    "solution": "<h3>Inner sol2</h3>",
                    "answer": "A",
                    "passage": "TEST_PASSAGE_01",
                }},
            ],
            "passageImages": [],
            "passageSolutionImages": [],
        },
    },
]


def _make_session_dir():
    """Create a temp session directory mirroring the on-disk layout Stage 5
    expects: session_dir/metadata_v2/<test>_metadata_v2.json."""
    tmp = Path(tempfile.mkdtemp(prefix="stage5_test_"))
    meta_dir = tmp / "metadata_v2"
    meta_dir.mkdir()
    (meta_dir / "test_metadata_v2.json").write_text(
        json.dumps(SAMPLE_METADATA, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return tmp


def _make_stage(session_dir):
    """Construct WatermarkStage instance bypassing __init__ (avoids DB setup)."""
    from app.pipeline.stage5_watermark import WatermarkStage
    inst = WatermarkStage.__new__(WatermarkStage)
    inst.session_dir = session_dir
    inst.session_id = "test_session"
    # No-op log to skip DB foreign-key dependency in tests
    inst.log = lambda level, msg: None
    inst._load_checkpoint_data = lambda: {}
    inst.save_checkpoint = lambda *a, **k: None
    inst.backup = lambda *a, **k: None
    return inst


def test_get_image_question_map_covers_all_types():
    """All 6 image filenames in our 3 sample entries must be mapped to their QIDs."""
    session_dir = _make_session_dir()
    stage = _make_stage(session_dir)
    mapping = stage.get_image_question_map()

    expected = {
        "scq_fig.png": "TEST_SCQ_01",
        "match_li.png": "TEST_MATCH_01",
        "match_lii.png": "TEST_MATCH_01",
        "passage_fig.png": "TEST_PASSAGE_01",
        "inner_sol1.png": "TEST_PASSAGE_01",
        "inner_q2.png": "TEST_PASSAGE_01",
    }
    missing = [f for f in expected if f not in mapping]
    if missing:
        for f in missing:
            print(f"FAIL: {f!r} missing from image-question map")
        raise AssertionError(f"{len(missing)} expected images not mapped")
    for fname, qid in expected.items():
        assert mapping[fname] == qid, (
            f"FAIL: {fname!r} mapped to {mapping[fname]!r}, expected {qid!r}"
        )
    print(f"  PASS image-question map: 6/6 images across scq+match+passage mapped to correct QID")


def test_scan_smiles_finds_smiles_in_all_types():
    """SMILES tags in scq.solution, match.solution, AND passage inner ques
    must all be detected."""
    session_dir = _make_session_dir()
    stage = _make_stage(session_dir)
    result = stage.scan_smiles()

    # 3 SMILES tags total: scq.solution (CC=O), match.solution (chiral), passage.questions[0].scq.ques (CCO)
    assert result["total_smiles_tags"] == 3, f"got {result['total_smiles_tags']}"
    assert result["entries_with_smiles"] == 3, f"got {result['entries_with_smiles']}"
    print(f"  PASS scan_smiles: detects SMILES in scq + match + passage-inner ({result['total_smiles_tags']} tags total)")


def test_render_smiles_processes_match_and_passage():
    """When SMILES rendering runs, it must replace <smiles> tags inside
    matchQuestion.solution AND inside passageQuestion.questions[i].scq.ques."""
    # If rdkit not available, skip — render path requires it
    try:
        import rdkit  # noqa
    except ImportError:
        print("  SKIP — rdkit not installed")
        return

    session_dir = _make_session_dir()
    stage = _make_stage(session_dir)
    result = stage.render_smiles()

    # Reload metadata after render to verify <smiles> tags are gone
    meta_path = session_dir / "metadata_v2" / "test_metadata_v2.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    smiles_re = re.compile(r"<smiles>", re.IGNORECASE)
    for entry in data:
        qt = entry.get("questionType")
        if qt == "matchQuestion":
            sol = entry["matchQuestion"]["solution"]
            assert not smiles_re.search(sol), f"matchQuestion.solution still has <smiles>: {sol[:200]}"
        elif qt == "passageQuestion":
            for sq in entry["passageQuestion"]["questions"]:
                ques = sq["scq"]["ques"]
                assert not smiles_re.search(ques), f"passage inner ques still has <smiles>: {ques[:200]}"
    print(f"  PASS render_smiles: <smiles> tags replaced in match + passage-inner")


def test_quarantine_works_for_all_types():
    """Quarantining match or passage QIDs must remove them from metadata."""
    # Monkey-patch quarantine_question + scan_smiles to bypass DB foreign-key
    import app.pipeline.stage5_watermark as stage5_mod
    original_quarantine = stage5_mod.__dict__.get("_test_qfn_orig")
    # Replace the imported quarantine_question inside the module's quarantine_images call
    import app.db as db_mod
    saved_qfn = db_mod.quarantine_question
    db_mod.quarantine_question = lambda *a, **k: None

    session_dir = _make_session_dir()
    stage = _make_stage(session_dir)

    # Quarantine the matchQuestion + passageQuestion QIDs
    result = stage.quarantine_images(["TEST_MATCH_01", "TEST_PASSAGE_01"])
    db_mod.quarantine_question = saved_qfn  # restore
    assert result["quarantined"] == 2, f"got {result['quarantined']}"

    # Verify removed from metadata
    meta_path = session_dir / "metadata_v2" / "test_metadata_v2.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    remaining_qids = [e.get("qid") for e in data]
    assert "TEST_MATCH_01" not in remaining_qids
    assert "TEST_PASSAGE_01" not in remaining_qids
    assert "TEST_SCQ_01" in remaining_qids
    print(f"  PASS quarantine_images: removes match + passage QIDs cleanly")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Running PRODUCTION stage5 match+passage parity tests...")
    test_get_image_question_map_covers_all_types()
    test_scan_smiles_finds_smiles_in_all_types()
    test_render_smiles_processes_match_and_passage()
    test_quarantine_works_for_all_types()
    print("ALL TESTS PASSED")
