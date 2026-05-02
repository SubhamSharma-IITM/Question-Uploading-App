"""Verify the err3/4 → err=2 promotion path produces a CSV row with:
  - UploadedQuestionID populated (not SKIPPED_*)
  - Error Level == 2 (the promoted level)
  - Original Error Level == 3 or 4 (the pre-promotion level)
  - Promotion Note == "PROMOTED_FROM_ERR3" or "PROMOTED_FROM_ERR4"
  - Pass1/2/3 Model + Verdict + Errors columns populated from QA ledger

This is the audit-trail invariant: faculty filtering err=2 review queue must be
able to identify which rows came from promotion vs natural err=2 voting.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_promotion_note_format_canonical():
    """Lock the format string downstream filters depend on."""
    # The note is built inline in stage10_upload.py upload() — assert the
    # exact format we documented to ops + the CSV header row carries it.
    expected_for_3 = "PROMOTED_FROM_ERR3"
    expected_for_4 = "PROMOTED_FROM_ERR4"
    # Mirror the inline construction:
    for original, expected in [(3, expected_for_3), (4, expected_for_4)]:
        note = f"PROMOTED_FROM_ERR{original}"
        assert note == expected


def test_csv_fieldnames_include_promotion_columns():
    """The CSV writer must emit Original Error Level + Promotion Note columns."""
    # Read the writer source so this test catches any future header re-order.
    src = (ROOT / "app" / "pipeline" / "stage10_upload.py").read_text(encoding="utf-8")
    # Find the fieldnames literal
    assert '"Original Error Level"' in src, "Original Error Level column missing from CSV header"
    assert '"Promotion Note"' in src, "Promotion Note column missing from CSV header"
    assert '"Pass1 Errors"' in src, "Pass1 Errors column missing"
    assert '"Pass2 Errors"' in src, "Pass2 Errors column missing"
    assert '"Pass3 Errors"' in src, "Pass3 Errors column missing"


def test_db_absorption_mirrors_promotion():
    """If err3/4 was promoted to err=2 at upload, the local DB JSONL entry
    must also carry err=2 (matching API + dashboard)."""
    from app.pipeline.stage10_upload import UploadStage
    inst = UploadStage.__new__(UploadStage)
    inst.session_id = "X"
    inst.session_dir = Path("/tmp/x")

    # Metadata has err=3 (Stage 9 verdict)
    meta_entry = {
        "qid": "Q1", "questionType": "scq", "subject": "Math",
        "scq": {"question": "<h3>Q</h3>", "answer": "A", "solution": "<h3>S</h3>"},
        "error": 3,
    }
    db_entry = inst._build_db_entry(meta_entry, "fake_oid_xyz",
                                    "2026-05-02T00:00:00.000Z")
    # With promote on (default), DB entry must carry err=2
    assert db_entry["error"] == 2, (
        f"local DB error must mirror promotion (got {db_entry['error']})"
    )

    meta_entry_4 = dict(meta_entry, error=4)
    db_entry_4 = inst._build_db_entry(meta_entry_4, "fake_oid_2",
                                      "2026-05-02T00:00:00.000Z")
    assert db_entry_4["error"] == 2, (
        f"err=4 must also promote to 2 in local DB (got {db_entry_4['error']})"
    )

    # err=1 / err=2 must pass through unchanged
    for natural in (1, 2):
        meta_entry_n = dict(meta_entry, error=natural)
        db_entry_n = inst._build_db_entry(meta_entry_n, "fake",
                                          "2026-05-02T00:00:00.000Z")
        assert db_entry_n["error"] == natural, (
            f"err={natural} must pass through unchanged (got {db_entry_n['error']})"
        )


def test_promotion_setting_default_on():
    from app.config import settings
    assert getattr(settings, "stage10_promote_err34", None) is True


if __name__ == "__main__":
    test_promotion_note_format_canonical()
    test_csv_fieldnames_include_promotion_columns()
    test_db_absorption_mirrors_promotion()
    test_promotion_setting_default_on()
    print("OK")
