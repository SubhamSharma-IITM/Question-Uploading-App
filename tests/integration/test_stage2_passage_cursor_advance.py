"""Regression: passageQuestion line_end must NOT include sub-question solution end.

Bug: _process_passage_item pulled ss_end (solution end, ~1793) into overall_end,
causing the chunker to advance cursor past whole question exercises (lines 220-1062
of FIITJEE-style PDFs), losing 80%+ of questions.

Fix verified by: passage at lines 164-187 with solutions at line 1793 must produce
line_end <= 187 (max question end), not 1793.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pipeline.stage2_llm_mapper import _process_passage_item


def _make_lines(n: int) -> list:
    """Build n synthetic numbered lines so 0-indexed slicing succeeds."""
    return [f"line {i+1}\n" for i in range(n)]


def test_passage_line_end_does_not_include_solution_end():
    raw_lines = _make_lines(2000)
    item = {
        "type": "passageQuestion",
        "subject": "Mathematics",
        "passage": {"start": 164, "end": 174},
        "sub_questions": [
            {
                "type": "scq",
                "answer_key": "A",
                "level": "MEDIUM",
                "question": {"start": 175, "end": 181},
                "solution": {"start": 1750, "end": 1793},  # solution far below
            },
            {
                "type": "scq",
                "answer_key": "B",
                "level": "MEDIUM",
                "question": {"start": 182, "end": 187},
                "solution": {"start": 1794, "end": 1820},
            },
        ],
    }
    result = _process_passage_item(item, raw_lines, set(), [], 100)
    assert result is not None, "passage processor returned None"
    # The bug: line_end was 1820 (last sub-Q solution end). Fix: must be <= 187.
    assert result["line_end"] <= 187, (
        f"line_end leaked into solution range: got {result['line_end']}, expected <= 187"
    )
    assert result["line_start"] == 164
    assert len(result["sub_questions"]) == 2
    # Solution ranges still preserved on each sub-question (we didn't drop the data)
    assert result["sub_questions"][0]["sol_line_end"] == 1793
    assert result["sub_questions"][1]["sol_line_end"] == 1820


def test_passage_line_end_uses_max_question_end_among_subquestions():
    raw_lines = _make_lines(500)
    item = {
        "type": "passageQuestion",
        "subject": "Physics",
        "passage": {"start": 100, "end": 110},
        "sub_questions": [
            {"type": "scq", "answer_key": "A", "question": {"start": 111, "end": 115}},
            {"type": "integerQuestion", "answer_key": "9",
             "question": {"start": 116, "end": 120}},
            {"type": "scq", "answer_key": "C", "question": {"start": 121, "end": 130}},
        ],
    }
    result = _process_passage_item(item, raw_lines, set(), [], 90)
    assert result is not None
    assert result["line_end"] == 130, (
        f"expected line_end=130 (max sub-q end), got {result['line_end']}"
    )


if __name__ == "__main__":
    test_passage_line_end_does_not_include_solution_end()
    test_passage_line_end_uses_max_question_end_among_subquestions()
    print("OK")
