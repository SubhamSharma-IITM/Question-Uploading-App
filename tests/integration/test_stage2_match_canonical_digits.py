"""Regression: matchQuestion canonical mapping must handle digit RHS labels.

Bug: _RIGHT_LABEL_MAP was missing digit→letter entries, so when source emitted
{P:[3], Q:[4], R:[1], S:[2]} (left=letters, right=digits — New JEE Adv style),
canonical came out as [[],[],[],[]] (all empty rows).

Fix: added "1"→"A", "2"→"B", "3"→"C", "4"→"D", "5"→"E" to _RIGHT_LABEL_MAP.

This test locks in handling for:
- letter LHS, letter RHS (FIITJEE old: A-D vs p-t / P-T) — already worked
- letter LHS, digit RHS (NEW JEE Adv: P-S vs 1-4) — was broken, fixed
- letter LHS, mixed sources
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pipeline.stage2_llm_mapper import _convert_match_answer_mapping


def test_letter_lhs_lowercase_letter_rhs_works():
    """FIITJEE old format — letters both sides."""
    src = {"A": ["r"], "B": ["p", "s"], "C": ["p", "q", "r", "s"], "D": ["p"]}
    canon = _convert_match_answer_mapping(src)
    assert canon == [["C"], ["A", "D"], ["A", "B", "C", "D"], ["A"]], canon


def test_letter_lhs_uppercase_letter_rhs_works():
    """FIITJEE uppercase variant."""
    src = {"A": ["R"], "B": ["Q", "S"], "C": ["Q", "R"], "D": ["P", "Q", "R", "S"]}
    canon = _convert_match_answer_mapping(src)
    assert canon == [["C"], ["B", "D"], ["B", "C"], ["A", "B", "C", "D"]], canon


def test_letter_lhs_digit_rhs_now_works():
    """NEW JEE Adv: P,Q,R,S on left, 1,2,3,4 on right.

    Was broken before the _RIGHT_LABEL_MAP digit fix.
    """
    src = {"P": ["3"], "Q": ["4"], "R": ["1"], "S": ["2"]}
    canon = _convert_match_answer_mapping(src)
    assert canon == [["C"], ["D"], ["A"], ["B"]], canon


def test_digit_lhs_letter_rhs_works():
    """Other NEW JEE Adv variant: 1,2,3,4 on left, A,B,C,D on right."""
    src = {"1": ["B"], "2": ["A", "D"], "3": ["C"], "4": ["A"]}
    canon = _convert_match_answer_mapping(src)
    assert canon == [["B"], ["A", "D"], ["C"], ["A"]], canon


def test_multi_match_with_digits():
    """Multi-match with digit RHS."""
    src = {"P": ["2"], "Q": ["1", "3"], "R": ["3"], "S": ["4"]}
    canon = _convert_match_answer_mapping(src)
    assert canon == [["B"], ["A", "C"], ["C"], ["D"]], canon


def test_unknown_label_yields_empty_for_that_value():
    """Unknown label still drops the value — that row may end up empty."""
    src = {"A": ["X"], "B": ["Y"]}
    canon = _convert_match_answer_mapping(src)
    assert canon == [[], []], canon


def test_empty_input_returns_empty_list():
    assert _convert_match_answer_mapping({}) == []
    assert _convert_match_answer_mapping(None) == []


if __name__ == "__main__":
    test_letter_lhs_lowercase_letter_rhs_works()
    test_letter_lhs_uppercase_letter_rhs_works()
    test_letter_lhs_digit_rhs_now_works()
    test_digit_lhs_letter_rhs_works()
    test_multi_match_with_digits()
    test_unknown_label_yields_empty_for_that_value()
    test_empty_input_returns_empty_list()
    print("OK")
