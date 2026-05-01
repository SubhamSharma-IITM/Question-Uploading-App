"""Production-parity test for stage4 match-type + bold + canonicalization.

Mirrors tests/_pilot/test_stage4_bold_conversion.py and
tests/_pilot/test_match_solution_canonicalize.py but imports from PRODUCTION
(app.pipeline.stage4_metadata). Confirms ported behavior.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.pipeline.stage4_metadata import (
    _convert_markdown_bold,
    _fill_match_label_gaps,
    _canonicalize_match_labels,
)


def test_bold_conversion():
    cases = [
        ("**Definition:** A function...", "<strong>Definition:</strong> A function..."),
        ("The answer is **42**.", "The answer is <strong>42</strong>."),
        ("**A** and **B**", "<strong>A</strong> and <strong>B</strong>"),
        ("Plain text", "Plain text"),
        ("", ""),
        ("**Line 1\nstill**", "**Line 1\nstill**"),
        ("text * not ** bold", "text * not ** bold"),
        ("** **", "<strong> </strong>"),
        ("<math><mtext>a</mtext></math>", "<math><mtext>a</mtext></math>"),
        ("\n\nA metal wire PQ slides on parallel rails.", "\n\nA metal wire PQ slides on parallel rails."),
    ]
    failures = []
    for inp, expected in cases:
        got = _convert_markdown_bold(inp)
        if got != expected:
            failures.append((inp, expected, got))
    if failures:
        for inp, exp, got in failures:
            print(f"FAIL: {inp!r} expected {exp!r} got {got!r}")
        raise AssertionError(f"{len(failures)} cases failed")
    print(f"  PASS bold conversion: {len(cases)} cases")


def test_fill_match_label_gaps():
    # M02-style: List-II A,B,D,E (missing C)
    items = [{"label": L, "content": f"c{L}"} for L in "ABDE"]
    out = _fill_match_label_gaps(items, ["A", "B", "C", "D", "E"])
    assert len(out) == 5 and [x["label"] for x in out] == list("ABCDE")
    assert out[2]["content"] == "&nbsp;"
    print(f"  PASS M02-style label gap")

    # 5 entries A-E
    items = [{"label": L, "content": f"c{L}"} for L in "ABCDE"]
    out = _fill_match_label_gaps(items, ["A", "B", "C", "D", "E"])
    assert len(out) == 5
    print(f"  PASS no-gap")

    # 4 entries A-D (no extension to E)
    items = [{"label": L, "content": f"c{L}"} for L in "ABCD"]
    out = _fill_match_label_gaps(items, ["A", "B", "C", "D", "E"])
    assert len(out) == 4
    print(f"  PASS no extension beyond max")

    # M10-style: missing A, B at top
    items = [{"label": L, "content": f"c{L}"} for L in "CDE"]
    out = _fill_match_label_gaps(items, ["A", "B", "C", "D", "E"])
    assert len(out) == 5 and [x["label"] for x in out] == list("ABCDE")
    assert out[0]["content"] == "&nbsp;" and out[1]["content"] == "&nbsp;"
    print(f"  PASS M10-style top-gap")

    # Empty
    assert _fill_match_label_gaps([], ["A", "B", "C"]) == []
    print(f"  PASS empty")


def test_canonicalize_match_labels():
    positive = [
        ("(A → p,r,t)", "(1 → A,C,E)"),
        ("(B → q)", "(2 → B)"),
        ("(C → q,s)", "(3 → B,D)"),
        ("(D → r,s,t)", "(4 → C,D,E)"),
        ("(A → p,r,t)(B → q)(C → q,s)(D → r,s,t)",
         "(1 → A,C,E)(2 → B)(3 → B,D)(4 → C,D,E)"),
        (r"(A \rightarrow p, r)", "(1 → A,C)"),
        (r"(B \to q)", "(2 → B)"),
        (r"(\mathrm{A} \rightarrow \mathrm{s})", "(1 → D)"),
        (r"(\mathrm{B} \rightarrow \mathrm{p}, \mathrm{t})", "(2 → A,E)"),
        (r"(A \rightarrow p,q,t)(B \rightarrow p,r,t)(C \rightarrow s)(D \rightarrow q,r)",
         "(1 → A,B,E)(2 → A,C,E)(3 → D)(4 → B,C)"),
        (r"(A \rightarrow p,q,s),(B \rightarrow q),(C \rightarrow r,t)(D \rightarrow q).",
         "(1 → A,B,D),(2 → B),(3 → C,E)(4 → B)."),
    ]
    negative = [
        "The correct option is (A).",
        "Compare options (A) and (B).",
        "From (i) we obtain x = 5.",
        "f(x) = 2x + 1",
        "(2,3) is a point on the curve.",
        r"\mathrm{HCl} + \mathrm{Na} \rightarrow \mathrm{NaCl} + \mathrm{H}",
        r"\mathrm{cm}/\mathrm{s}",
        r"\mathrm{kg}",
        "The reaction proceeds A → B at high temperature.",
        "Prove that f(x) → g(x) as x → ∞.",
        "A → p means option A maps to p.",
        "",
        "No arrow here, just labels (A) and (B).",
    ]
    failures_p = []
    for inp, expected in positive:
        got = _canonicalize_match_labels(inp)
        if got != expected:
            failures_p.append((inp, expected, got))
    if failures_p:
        for inp, exp, got in failures_p:
            print(f"FAIL POS: {inp!r} expected {exp!r} got {got!r}")
        raise AssertionError(f"{len(failures_p)} positive cases failed")
    print(f"  PASS canonicalize positive: {len(positive)} cases")

    failures_n = []
    for inp in negative:
        got = _canonicalize_match_labels(inp)
        if got != inp:
            failures_n.append((inp, got))
    if failures_n:
        for inp, got in failures_n:
            print(f"FAIL NEG: {inp!r} got {got!r}")
        raise AssertionError(f"{len(failures_n)} negative cases failed")
    print(f"  PASS canonicalize negative: {len(negative)} cases")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Running PRODUCTION stage4 match-type + bold + canonicalization tests...")
    test_bold_conversion()
    test_fill_match_label_gaps()
    test_canonicalize_match_labels()
    print("ALL TESTS PASSED")
