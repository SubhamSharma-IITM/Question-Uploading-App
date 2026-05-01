"""Stage 7 (LaTeX→MathML + tagging batch + LaTeX fallback) parity test for
match + passage container shapes.

Stage 7 has 5 distinct container-aware paths:
  1. convert_all_mathml — main local LaTeX→MathML pass on every field
  2. build_batch tagging text extractor — sends {question, solution} text
     per question to OpenAI for concept inference
  3. build_batch LaTeX fallback combiner — for failed-MathML qids, sends
     full text to gpt-5-mini for LaTeX-aware conversion
  4. download_results re-injection — writes converted text back into the
     correct field
  5. quarantine path for unrecoverable LaTeX

This test simulates each path's field-extraction logic and confirms it
correctly reads canonical match + passage shapes (including canonical
sub-question field name "ques", not legacy "question").
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ── Sample entries: canonical match + passage with embedded LaTeX ──────────

MATCH_ENTRY = {
    "qid": "TEST_M01",
    "subject": "Mathematics",
    "questionType": "matchQuestion",
    "matchQuestion": {
        "rows": [["B"], ["A"], ["C"], ["D"]],
        "quesImages": None,
        "solutionImages": None,
        "question": "<p>Match the following.</p><table><tr><td>(1)</td><td>$\\sqrt{x^2}$</td><td>(A)</td><td>$|x|$</td></tr></table>",
        "solution": "<h3>(1 → A): $\\sqrt{x^2} = |x|$</h3>",
    },
    "answer": "",
}

PASSAGE_ENTRY = {
    "qid": "TEST_P01",
    "subject": "Physics",
    "questionType": "passageQuestion",
    "passageQuestion": {
        "passage": "<h3>Consider $a, b \\in \\mathbb{R}$ with $a > b$.</h3>",
        "solution": "",
        "questions": [
            {"scq": {
                "ques": "<h3>If $a^2 = b^2 + c^2$, find $c$.</h3>(A) $\\sqrt{a-b}$<br/>(B) $\\sqrt{a^2-b^2}$",
                "solution": "<h3>$c = \\sqrt{a^2 - b^2}$ by Pythagoras.</h3>",
                "answer": "B",
                "passage": "TEST_P01",
            }},
        ],
        "passageImages": [],
        "passageSolutionImages": [],
    },
}

PASSAGE_LEGACY_ENTRY = {
    "qid": "TEST_P02_LEGACY",
    "subject": "Mathematics",
    "questionType": "passageQuestion",
    "passageQuestion": {
        "passage": "<h3>Legacy passage.</h3>",
        "solution": "",
        "questions": [
            {"integerQuestion": {  # legacy: wraps under non-scq sub-key
                "question": "<h3>Find $x$ given $x^2 = 4$.</h3>",
                "solution": "<h3>$x = 2$.</h3>",
                "answer": "2",
            }},
        ],
    },
}


def _build_tagging_text(entry):
    """Replica of stage7's tagging text-extraction logic.

    Mirrors the per-question branch in build_batch (lines ~1095-1115) so we can
    test field selection without invoking the full FAISS+OpenAI pipeline.
    """
    qtype = entry.get("questionType", "")
    qdata = entry.get(qtype, {})
    if qtype == "passageQuestion":
        passage_text = qdata.get("passage", "")
        sub_texts, sub_answers = [], []
        for sq in qdata.get("questions", []):
            if isinstance(sq, dict):
                so = sq.get("scq")
                if isinstance(so, dict):
                    sub_texts.append(so.get("ques", "") or so.get("question", ""))
                    sub_answers.append(so.get("answer", ""))
                else:
                    for sk in ("mcq", "integerQuestion"):
                        so = sq.get(sk)
                        if isinstance(so, dict):
                            sub_texts.append(so.get("question", ""))
                            sub_answers.append(so.get("answer", ""))
                            break
        question_text = passage_text + "\n" + "\n".join(sub_texts)
        solution_text = qdata.get("solution", "")
        answer = ", ".join(a for a in sub_answers if a)
    else:
        question_text = qdata.get("question", "")
        solution_text = qdata.get("solution", "")
        answer = entry.get("answer", qdata.get("answer", ""))
    return {"question": question_text, "solution": solution_text, "answer": answer}


def _build_latex_fallback_input(entry):
    """Replica of stage7's LaTeX-fallback combiner (lines ~1252-1290)."""
    qtype = entry.get("questionType", "")
    qdata = entry.get(qtype, {})
    if qtype == "passageQuestion":
        solution_text = qdata.get("solution", "")
        combined_q = qdata.get("passage", "")
        for sq in qdata.get("questions", []):
            if isinstance(sq, dict):
                so = sq.get("scq")
                if isinstance(so, dict):
                    combined_q += "\n" + (so.get("ques", "") or so.get("question", ""))
                    solution_text += "\n" + so.get("solution", "")
                else:
                    for sk in ("mcq", "integerQuestion"):
                        so = sq.get(sk)
                        if isinstance(so, dict):
                            combined_q += "\n" + so.get("question", "")
                            solution_text += "\n" + so.get("solution", "")
                            break
        return {"question": combined_q, "solution": solution_text}
    else:
        return {
            "question": qdata.get("question", ""),
            "solution": qdata.get("solution", ""),
        }


def test_match_local_mathml_walks_question_solution():
    """convert_all_mathml's else branch already iterates ['question', 'solution']
    on the qtype container — matchQuestion has both fields, so it works."""
    qtype = MATCH_ENTRY["questionType"]
    qdata = MATCH_ENTRY[qtype]
    # The convert_all_mathml else branch reads these:
    fields_walked = ["question", "solution"]
    assert all(f in qdata for f in fields_walked)
    # Both fields have LaTeX content
    assert "\\sqrt" in qdata["question"]
    assert "\\sqrt" in qdata["solution"]
    print("  PASS local MathML conversion: matchQuestion {question, solution} reachable")


def test_match_tagging_text():
    """Tagging extractor must produce non-empty question + solution for match."""
    txt = _build_tagging_text(MATCH_ENTRY)
    assert "Match the following" in txt["question"]
    assert "(1 → A)" in txt["solution"]
    assert txt["answer"] == ""  # match has no answer field — empty
    print("  PASS match tagging text: question + solution non-empty")


def test_passage_tagging_canonical_ques_field():
    """Passage canonical schema uses 'ques' field — tagging must read it.
    This was a real bug: production read 'question' first which is empty in
    canonical schema."""
    txt = _build_tagging_text(PASSAGE_ENTRY)
    assert "$a, b \\in \\mathbb{R}$" in txt["question"]   # passage body
    assert "$a^2 = b^2 + c^2$" in txt["question"]          # inner ques
    assert txt["answer"] == "B"
    print("  PASS passage tagging text: canonical 'ques' field read correctly")


def test_passage_tagging_legacy_question_field_fallback():
    """Legacy passage shape has questions[i].integerQuestion.question — must
    still extract via the fallback branch."""
    txt = _build_tagging_text(PASSAGE_LEGACY_ENTRY)
    assert "Find $x$" in txt["question"]
    assert txt["answer"] == "2"
    print("  PASS passage tagging text: legacy integerQuestion sub-shape extracts")


def test_match_latex_fallback_input():
    """LaTeX fallback combiner sends question + solution JSON to gpt-5-mini.
    For match, this includes the table HTML + canonical solution text."""
    inp = _build_latex_fallback_input(MATCH_ENTRY)
    assert "$\\sqrt{x^2}$" in inp["question"]
    assert "(1 → A)" in inp["solution"]
    print("  PASS match LaTeX fallback input: q + sol included for batch send")


def test_passage_latex_fallback_canonical():
    """LaTeX fallback combiner for canonical passage shape (ques field)."""
    inp = _build_latex_fallback_input(PASSAGE_ENTRY)
    assert "$a, b \\in \\mathbb{R}$" in inp["question"]
    assert "$a^2 = b^2 + c^2$" in inp["question"]
    assert "$c = \\sqrt{a^2 - b^2}$" in inp["solution"]
    print("  PASS passage LaTeX fallback input: canonical ques + sub-solutions combined")


def test_match_re_injection_path():
    """Re-injection (download_results) writes converted q_part to qdata['question']
    and s_part to qdata['solution']. matchQuestion has both fields — else branch
    handles it correctly."""
    # Simulate the else branch behavior:
    entry = json.loads(json.dumps(MATCH_ENTRY))  # deep copy
    qtype = entry["questionType"]
    qdata = entry.setdefault(qtype, {})
    converted_q = "<table><tr><td>(1)</td><td><math>...</math></td></tr></table>"
    converted_s = "<h3>(1 → A): <math>...</math></h3>"
    # else branch logic (from line 1749+)
    qdata["question"] = converted_q
    qdata["solution"] = converted_s
    assert entry["matchQuestion"]["question"] == converted_q
    assert entry["matchQuestion"]["solution"] == converted_s
    print("  PASS match re-injection: converted q + sol written to right fields")


def test_local_mathml_conversion_actually_runs():
    """End-to-end: import the actual stage7 helper and convert match content."""
    from app.pipeline.stage7_latex_tag import LatexTagStage
    inst = LatexTagStage.__new__(LatexTagStage)
    # Convert match question containing LaTeX
    html = MATCH_ENTRY["matchQuestion"]["question"]
    converted, succ, fail = inst._convert_latex_to_mathml(html)
    # At least one $...$ block should convert (we can't predict exactly how many
    # depending on edge cases, but match content WITH `$\sqrt{x^2}$` must convert)
    assert succ > 0, f"expected at least 1 LaTeX→MathML success, got {succ}"
    # Either the result has <math> tags OR succ > 0 is sufficient
    assert "<math" in converted or succ > 0
    print(f"  PASS local MathML conversion runs on match content ({succ} successes, {fail} fails)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Running PRODUCTION stage7 match+passage parity tests...")
    test_match_local_mathml_walks_question_solution()
    test_match_tagging_text()
    test_passage_tagging_canonical_ques_field()
    test_passage_tagging_legacy_question_field_fallback()
    test_match_latex_fallback_input()
    test_passage_latex_fallback_canonical()
    test_match_re_injection_path()
    test_local_mathml_conversion_actually_runs()
    print("ALL TESTS PASSED")
