"""Production-parity test for stage3 _strip_format_noise.

Mirror of tests/_pilot/test_stage3_noise_strip.py but imports from the
PRODUCTION module (app.pipeline.stage3_cleaner). Confirms the pilot fixes
ported into production behave identically. compilation != correctness;
this test is the correctness gate.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.pipeline.stage3_cleaner import _strip_format_noise


POSITIVE_CASES = [
    # Paragraph header — bolded
    ("**Paragraph for Question Nos. 9 to 10**\n\nA metal wire PQ slides...",
     "\n\nA metal wire PQ slides..."),
    ("**Paragraph for question Nos. 11 to 12**\nThe binary compounds...",
     "\nThe binary compounds..."),
    ("**Paragraph for Questions 12 & 13**", ""),
    ("**Paragraph for Question Nos. 13 to 14**", ""),
    # Without bold wrapper
    ("Paragraph for Question Nos. 14 to 16", ""),
    # LaTeX `\section*{Paragraph for ...}` form
    (r"\section*{Paragraph for question Nos. 11 to 12}" + "\n\nThe binary compounds...",
     "\n\nThe binary compounds..."),
    (r"\section*{Paragraph for Question Nos. 50 and 51:}", ""),
    # LaTeX-escaped ampersand `\&`
    (r"\section*{Paragraph for Questions 12 \& 13}" + "\nA disc...", "\nA disc..."),
    (r"\section*{Paragraph for Questions 14 \& 16}", ""),
    # LaTeX-wrapped Read forms
    (r"\section*{Read the following write up carefully and answer the following questions:}" + "\n\nIf P(...)",
     "\n\nIf P(...)"),
    (r"\section*{Read the above paragraph carefully and answer the questions given below:}", ""),
    # Bare Read forms
    ("**Read the above paragraph carefully and answer the questions given below:**", ""),
    ("**Read the following write up carefully and answer the following questions:**", ""),
    # "Then find the answer..."
    ("Then find the answer of following questions based on above paragraph.\nQuestion content here.",
     "\nQuestion content here."),
    # Printer artifact
    ("Some text. Space for Rough work", "Some text. "),
    ("(B) C6T6 > C6D6 > C6H6\n\nSpace for Rough work\n\nOptions: C",
     "(B) C6T6 > C6D6 > C6H6\n\n\n\nOptions: C"),
    # Line-range prefix at start
    ("14-16. iBℓ=0.5", "iBℓ=0.5"),
    ("11-12. ⇒i=0.5/Bℓ", "⇒i=0.5/Bℓ"),
    # Combined
    ("**Paragraph for Question Nos. 13 to 14**\n\n**Read the above paragraph carefully and answer the questions given below:**\n\nA metal is regarded as...",
     "\n\n\n\nA metal is regarded as..."),
]


NEGATIVE_CASES = [
    "A particle of mass 2 kg is moving on straight line under the action of force F=(8-2x)N.",
    "There is a 2Ω resistor and 10 V battery as shown in figure.",
    "What is the answer for x?",
    "Find the area under the parabola y=x^2.",
    "The paragraph mentions three types of compounds.",
    "The ratio is 14-16 cm in length.",
    "Equations 14-16 from chapter 5 give the answer.",
    "**Definition:** A function is continuous if...",
    "The answer is **42**.",
    "Match the following.",
    "Match the solution given in column I with the properties given in column II.",
    "1. Match the following",
    "2. Match the following:",
    "1. Energy conservation holds.",
    "Step 1. Calculate the velocity.",
]


def test_positive():
    failures = []
    for inp, expected in POSITIVE_CASES:
        got = _strip_format_noise(inp)
        if got != expected:
            failures.append((inp, expected, got))
    if failures:
        for inp, exp, got in failures:
            print(f"FAIL POS:")
            print(f"  inp:      {inp!r}")
            print(f"  expected: {exp!r}")
            print(f"  got:      {got!r}")
        raise AssertionError(f"{len(failures)} positive cases failed")
    print(f"  PASS positive: {len(POSITIVE_CASES)} cases")


def test_negative():
    failures = []
    for inp in NEGATIVE_CASES:
        got = _strip_format_noise(inp)
        if got != inp:
            failures.append((inp, got))
    if failures:
        for inp, got in failures:
            print(f"FAIL NEG (real prose modified):")
            print(f"  inp: {inp!r}")
            print(f"  got: {got!r}")
        raise AssertionError(f"{len(failures)} negative cases failed")
    print(f"  PASS negative: {len(NEGATIVE_CASES)} cases")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Running PRODUCTION stage3 noise-strip parity tests...")
    test_positive()
    test_negative()
    print("ALL TESTS PASSED")
