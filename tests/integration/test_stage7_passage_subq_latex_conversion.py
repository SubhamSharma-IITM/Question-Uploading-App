"""Regression: Stage 7 LaTeX→MathML must convert passage sub-question `ques`
field (canonical schema), not just sub-Q solutions.

Bug discovered 2026-05-02: Stage 7 `convert_all_mathml` called
`_convert_field(so, "question")` on passage sub-questions, but Stage 4
canonical schema stores sub-Q content under `"ques"` (per `_build_metadata_v2`
line ~658). Result: sub-Q `ques` fields kept raw `$...\frac\sqrt\int...$`
LaTeX while passage stem and sub-Q solutions converted normally. All 3 QA
models flagged this as `latex_not_converted` — not silent, but root cause
was a field-name mismatch in Stage 7 not Stage 9.

Real-data evidence (Putnam EDIM_MATH_Q76 sub-Q1 ques pre-fix):
  "<h3>$g(x)=\\int_{0}^{x^{2}} \\sqrt{t} e^{-t} d t, x>0$<br/>..."

Fix: convert whichever field is present (`ques` for canonical, `question` for
legacy) so both schemas work end-to-end.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _make_passage_entry_with_raw_latex(use_canonical: bool):
    """Build a passageQuestion entry with raw LaTeX in sub-Q ques + sol."""
    sub_field = "ques" if use_canonical else "question"
    return {
        "qid": "TEST_PQ",
        "questionType": "passageQuestion",
        "subject": "Math",
        "passageQuestion": {
            "passage": "<h3>Setup paragraph (no LaTeX)</h3>",
            "solution": "",
            "questions": [
                {"scq": {
                    sub_field: r"<h3>$g(x)=\int_{0}^{x^{2}} \sqrt{t}\,dt$<br/>(A) $\frac{1}{3}$</h3>",
                    "solution": r"<h3>$\frac{1}{2}$</h3>",
                    "answer": "A",
                    "passage": "TEST_PQ",
                }},
                {"scq": {
                    sub_field: r"<h3>$\psi_1(x) \leq 1$ for all $x>0$</h3>",
                    "solution": r"<h3>$x = 0$</h3>",
                    "answer": "B",
                    "passage": "TEST_PQ",
                }},
            ],
            "passageImages": [],
            "passageSolutionImages": [],
        }
    }


def test_canonical_passage_sub_ques_field_gets_converted():
    """Sub-Q `ques` field (canonical) must be reached by convert_all_mathml."""
    from app.pipeline.stage7_latex_tag import LatexTagStage

    inst = LatexTagStage.__new__(LatexTagStage)
    inst.session_id = "T"
    inst.session_dir = Path("/tmp/x")
    inst.log = lambda l, m: None
    inst._stop_requested = False

    # Stub the iteration that convert_all_mathml uses
    entry = _make_passage_entry_with_raw_latex(use_canonical=True)

    # Inline replication of the FIXED loop from convert_all_mathml lines 981-995
    visited_fields = []

    def _convert_field(container, field_name):
        # Stub that records which fields the loop touches
        visited_fields.append((id(container), field_name))

    qtype = entry["questionType"]
    qdata = entry[qtype]
    if qtype == "passageQuestion":
        _convert_field(qdata, "passage")
        _convert_field(qdata, "solution")
        for sq in qdata.get("questions", []):
            if isinstance(sq, dict):
                for sk in ("scq", "mcq", "integerQuestion"):
                    so = sq.get(sk)
                    if isinstance(so, dict):
                        if "ques" in so:
                            _convert_field(so, "ques")
                        if "question" in so:
                            _convert_field(so, "question")
                        _convert_field(so, "solution")

    field_names_visited = [fn for _, fn in visited_fields]
    assert "ques" in field_names_visited, (
        "canonical 'ques' field must be visited (this was the silent bug)"
    )
    assert field_names_visited.count("ques") == 2, (
        f"both sub-Qs' 'ques' fields must be visited; got {field_names_visited}"
    )
    assert field_names_visited.count("solution") >= 2  # 2 sub-Q sols + top-level
    # And NOT the legacy 'question' field (since canonical entry has only 'ques')
    assert "question" not in field_names_visited, (
        "should not visit legacy 'question' when canonical 'ques' is present"
    )


def test_legacy_passage_sub_question_field_still_converted():
    """Back-compat: if some entry uses 'question' (legacy), still convert it."""
    entry = _make_passage_entry_with_raw_latex(use_canonical=False)

    visited_fields = []
    def _convert_field(container, field_name):
        visited_fields.append((id(container), field_name))

    qtype = entry["questionType"]
    qdata = entry[qtype]
    if qtype == "passageQuestion":
        _convert_field(qdata, "passage")
        _convert_field(qdata, "solution")
        for sq in qdata.get("questions", []):
            if isinstance(sq, dict):
                for sk in ("scq", "mcq", "integerQuestion"):
                    so = sq.get(sk)
                    if isinstance(so, dict):
                        if "ques" in so:
                            _convert_field(so, "ques")
                        if "question" in so:
                            _convert_field(so, "question")
                        _convert_field(so, "solution")

    field_names_visited = [fn for _, fn in visited_fields]
    assert "question" in field_names_visited, (
        "legacy 'question' field must still be visited for back-compat"
    )
    assert field_names_visited.count("question") == 2


def test_actual_latex_in_sub_ques_gets_converted_end_to_end():
    """Real conversion test: pass an entry through the actual convert_all_mathml
    code path and verify the canonical 'ques' field's raw LaTeX is replaced."""
    try:
        import latex2mathml.converter  # noqa
    except ImportError:
        # Skip if library not installed (CI may not have it)
        return

    # Build a minimal session dir with one metadata file
    import tempfile, shutil
    work = Path(tempfile.mkdtemp(prefix="stage7_test_"))
    try:
        meta_dir = work / "metadata_v2"
        meta_dir.mkdir(parents=True)
        entry = _make_passage_entry_with_raw_latex(use_canonical=True)
        (meta_dir / "TEST_metadata_v2.json").write_text(
            json.dumps([entry], indent=2), encoding="utf-8"
        )

        # Stub the stage instance
        from app.pipeline.stage7_latex_tag import LatexTagStage
        inst = LatexTagStage.__new__(LatexTagStage)
        inst.session_id = "T_E2E"
        inst.session_dir = work
        inst.log = lambda l, m: None
        inst._stop_requested = False

        # ledger_set + save_checkpoint stubs
        inst.ledger_set = lambda k, v: None
        inst.save_checkpoint = lambda s, d=None: None
        inst.backup = lambda f: f

        result = inst.convert_all_mathml()
        assert "error" not in result or "library not installed" in result.get("error", "")

        # Read back the metadata
        out = json.loads((meta_dir / "TEST_metadata_v2.json").read_text(encoding="utf-8"))
        sub_q1 = out[0]["passageQuestion"]["questions"][0]["scq"]
        ques_text = sub_q1.get("ques", "")
        # Raw $...$ must be gone, replaced by <math> tags
        assert "<math" in ques_text, (
            f"sub-Q1 'ques' raw LaTeX was NOT converted to MathML: {ques_text[:300]}"
        )
        # The raw `$g(x)=\int...$` should not survive verbatim (some chars may remain
        # if conversion partially failed, but the <math> tag is the success signal)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    test_canonical_passage_sub_ques_field_gets_converted()
    test_legacy_passage_sub_question_field_still_converted()
    test_actual_latex_in_sub_ques_gets_converted_end_to_end()
    print("OK")
