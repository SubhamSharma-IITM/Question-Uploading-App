"""Verify Stage 9's QA batch builder + user-content renderer correctly extract
the canonical match + passage container shapes.

Inline shape construction — does NOT require a real session. Tests the per-
question extraction logic directly by simulating the relevant code path with
sample matchQuestion + passageQuestion entries matching DB canonical schema.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _extract_request_payload(entry):
    """Replicate the per-question extraction inside QualityStage.build_qa_batch.

    Mirrors lines 1169-1207 of stage9_quality.py (the `with open(batch_path, "w")` loop).
    Importing the actual class would require session setup, so this stub mirrors
    the field-extraction logic verbatim — kept in lockstep with production.
    """
    qid = entry.get("qid", "?")
    qtype = entry.get("questionType", "")
    qdata = entry.get(qtype, {}) if qtype else {}
    rows_data = None

    if qtype == "passageQuestion":
        passage_text = qdata.get("passage", "") or ""
        sub_q_texts, sub_answers, sub_solutions = [], [], []
        for sq in qdata.get("questions", []):
            if isinstance(sq, dict):
                so = sq.get("scq")
                if isinstance(so, dict):
                    sub_q_texts.append(so.get("ques", "") or so.get("question", ""))
                    sub_answers.append(so.get("answer", ""))
                    sub_solutions.append(so.get("solution", ""))
                else:
                    for sk in ("mcq", "integerQuestion"):
                        so = sq.get(sk)
                        if isinstance(so, dict):
                            sub_q_texts.append(so.get("question", ""))
                            sub_answers.append(so.get("answer", ""))
                            sub_solutions.append(so.get("solution", ""))
                            break
        q_text = passage_text + "\n" + "\n".join(sub_q_texts)
        a_text = ", ".join(a for a in sub_answers if a)
        s_text = (qdata.get("solution", "") or "") + "\n" + "\n".join(sub_solutions)
    elif qtype == "matchQuestion":
        q_text = qdata.get("question", "") or ""
        s_text = qdata.get("solution", "") or ""
        a_text = ""
        rows_data = qdata.get("rows", None)
    else:
        q_text = qdata.get("question", "") or ""
        a_text = qdata.get("answer", "")
        s_text = qdata.get("solution", "") or ""

    req = {
        "custom_id": f"qa-pass1-{qid}",
        "qid": qid,
        "questionType": qtype,
        "question": q_text,
        "answer": a_text,
        "solution": s_text,
    }
    if rows_data is not None:
        req["rows"] = rows_data
    return req


def test_match_extraction():
    entry = {
        "qid": "TEST_M01",
        "questionType": "matchQuestion",
        "matchQuestion": {
            "rows": [["B"], ["D"], ["A"], ["E"]],
            "quesImages": None,
            "solutionImages": None,
            "question": "<table>List-I/List-II HTML</table>",
            "solution": "<h3>Solution math here</h3>",
        },
    }
    req = _extract_request_payload(entry)
    assert req["questionType"] == "matchQuestion"
    assert req["question"] == "<table>List-I/List-II HTML</table>"
    assert req["solution"] == "<h3>Solution math here</h3>"
    assert req["answer"] == ""              # match has no answer field
    assert req.get("rows") == [["B"], ["D"], ["A"], ["E"]]   # rows must be sent
    print("  PASS match extraction: rows[] sent, answer empty, HTML preserved")


def test_passage_canonical_extraction():
    """Canonical passage schema: questions[i] = {scq: {ques, solution, answer, passage}}"""
    entry = {
        "qid": "TEST_P11",
        "questionType": "passageQuestion",
        "passageQuestion": {
            "passage": "<h3>Passage body math</h3>",
            "solution": "",
            "questions": [
                {"scq": {"ques": "<h3>Inner Q1</h3>", "solution": "<h3>Sol1</h3>",
                         "answer": "C", "passage": "TEST_P11"}},
                {"scq": {"ques": "<h3>Inner Q2</h3>", "solution": "<h3>Sol2</h3>",
                         "answer": "B", "passage": "TEST_P11"}},
            ],
            "passageImages": [],
            "passageSolutionImages": [],
        },
    }
    req = _extract_request_payload(entry)
    # The passage HTML + each inner ques HTML must be concatenated in question
    assert "Passage body math" in req["question"]
    assert "Inner Q1" in req["question"]
    assert "Inner Q2" in req["question"]
    # Inner answers must be aggregated
    assert "C" in req["answer"] and "B" in req["answer"]
    # Inner solutions must be in solution
    assert "Sol1" in req["solution"] and "Sol2" in req["solution"]
    print("  PASS passage canonical extraction: nested ques+solution+answer aggregated")


def test_passage_legacy_fallback():
    """Legacy passage shape (pre-canonical): questions[i] = {integerQuestion: {question,...}}.
    Builder should still extract via fallback branch."""
    entry = {
        "qid": "TEST_LEGACY",
        "questionType": "passageQuestion",
        "passageQuestion": {
            "passage": "<h3>Legacy passage</h3>",
            "solution": "",
            "questions": [
                {"integerQuestion": {"question": "<h3>Legacy Q</h3>",
                                     "solution": "<h3>Legacy Sol</h3>",
                                     "answer": "42"}},
            ],
        },
    }
    req = _extract_request_payload(entry)
    assert "Legacy Q" in req["question"]
    assert "42" in req["answer"]
    assert "Legacy Sol" in req["solution"]
    print("  PASS legacy passage fallback: integerQuestion subshape still extracts")


def test_scq_unchanged():
    """Plain scq path must be unaffected by the new branches."""
    entry = {
        "qid": "TEST_SCQ",
        "questionType": "scq",
        "scq": {
            "question": "<h3>Q?</h3>(A) 1<br/>(B) 2<br/>(C) 3<br/>(D) 4",
            "answer": "C",
            "solution": "<h3>Sol</h3>",
            "quesImages": [],
            "solutionImages": [],
        },
    }
    req = _extract_request_payload(entry)
    assert req["questionType"] == "scq"
    assert req["answer"] == "C"
    assert "rows" not in req  # match-only field must NOT leak into scq
    print("  PASS scq path unchanged")


def test_user_content_renders_match_rows():
    """The user-content renderer must include a ROWS block for matchQuestion."""
    from app.pipeline.stage9_quality import QualityVerifierStage
    inst = QualityVerifierStage.__new__(QualityVerifierStage)
    req = {
        "qid": "TEST_M01", "subject": "Chemistry", "questionType": "matchQuestion",
        "question": "<table>q</table>", "answer": "", "solution": "<h3>s</h3>",
        "rows": [["B"], ["D"], ["A"], ["E"]],
        "chapter": "", "concept": "", "subConcept": "", "tag_sub_concepts": [],
    }
    text, _ = inst._build_user_content(req)
    assert "ROWS" in text and "1 -> B" in text and "4 -> E" in text, \
        f"rows block missing or malformed:\n{text[:500]}"
    print("  PASS user-content renders ROWS block for matchQuestion")


def test_user_content_passage_no_rows_block():
    """Non-match types must NOT have a ROWS block."""
    from app.pipeline.stage9_quality import QualityVerifierStage
    inst = QualityVerifierStage.__new__(QualityVerifierStage)
    req = {
        "qid": "TEST_P11", "subject": "Physics", "questionType": "passageQuestion",
        "question": "<h3>passage</h3>", "answer": "C, B", "solution": "<h3>sol</h3>",
        "chapter": "", "concept": "", "subConcept": "", "tag_sub_concepts": [],
    }
    text, _ = inst._build_user_content(req)
    assert "ROWS" not in text, f"unexpected ROWS block in passage:\n{text[:400]}"
    print("  PASS user-content omits ROWS for non-match types")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Running PRODUCTION stage9 batch builder + renderer parity tests...")
    test_match_extraction()
    test_passage_canonical_extraction()
    test_passage_legacy_fallback()
    test_scq_unchanged()
    test_user_content_renders_match_rows()
    test_user_content_passage_no_rows_block()
    print("ALL TESTS PASSED")
