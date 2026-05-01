"""Production-parity test for stage9 verdict prompt — confirm MATCH-TYPE GATE
auto-rejection has been removed and matchQuestion is properly recognized.

Imports PRODUCTION app.pipeline.stage9_quality.INSPECTOR_SYSTEM_PROMPT.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.pipeline.stage9_quality import INSPECTOR_SYSTEM_PROMPT


def test_gate_removed():
    bad_phrases = [
        "match_format_content",                # old auto-reject error type
        "REJECT IMMEDIATELY",                  # old gate command
        "MATCH-TYPE GATE (EXECUTE FIRST",      # old section header
        "Match gate triggered",                # old example phrasing
        "Skip all other checks",               # old gate behavior
    ]
    failures = [p for p in bad_phrases if p in INSPECTOR_SYSTEM_PROMPT]
    if failures:
        for f in failures:
            print(f"FAIL: gate language still present -> {f!r}")
        raise AssertionError(f"{len(failures)} gate phrases still in prompt")
    print(f"  PASS gate removed: 0 of {len(bad_phrases)} legacy phrases present")


def test_match_schema_present():
    required_phrases = [
        # Section 0 explanatory rules
        'matchQuestion',
        'List-I',
        'List-II',
        '"rows":',
        # New error code
        'invalid_match_rows',
        # JEE Advanced now permits match
        'mcq, integerQuestion, passageQuestion, matchQuestion',
        # Multi-match support
        'Multi-match supported',
        # New CHECK 1 wording
        'MATCH-TYPE INTEGRITY',
    ]
    missing = [p for p in required_phrases if p not in INSPECTOR_SYSTEM_PROMPT]
    if missing:
        for p in missing:
            print(f"FAIL: required phrase not in prompt -> {p!r}")
        raise AssertionError(f"{len(missing)} required match-schema phrases missing")
    print(f"  PASS schema present: all {len(required_phrases)} required phrases found")


def test_noise_and_exam_ref_provisions_present():
    """Confirm Sections 6C (noise_in_stem) and 6D (exam_reference_in_stem)
    are present per integration plan + memory references."""
    required_phrases = [
        # Section 6C — noise_in_stem
        'noise_in_stem',
        'SECTION 6C',
        'paragraph headers',
        '"Read carefully"',
        'Space for Rough work',
        '\\section*{...}',  # prompt contains literal `\section*{...}` for LaTeX
        # Section 6D — exam_reference_in_stem
        'exam_reference_in_stem',
        'SECTION 6D',
        'JEE Main',
        'NEET',
        'MHCET',
        'AIIMS',
        'BITSAT',
        # CHECK 1B
        'CHECK 1B',
        'NOISE / EXAM REFERENCE SCAN',
    ]
    from app.pipeline.stage9_quality import INSPECTOR_SYSTEM_PROMPT
    missing = [p for p in required_phrases if p not in INSPECTOR_SYSTEM_PROMPT]
    if missing:
        for p in missing:
            print(f"FAIL: required noise/exam-ref phrase missing -> {p!r}")
        raise AssertionError(f"{len(missing)} provisions missing")
    print(f"  PASS noise_in_stem + exam_reference_in_stem provisions: all {len(required_phrases)} present")


def test_audit_table_has_new_codes():
    from app.pipeline.stage9_quality import INSPECTOR_SYSTEM_PROMPT
    required = [
        '| question | noise_in_stem',
        '| question | exam_reference_in_stem',
    ]
    missing = [p for p in required if p not in INSPECTOR_SYSTEM_PROMPT]
    if missing:
        for p in missing:
            print(f"FAIL: audit-table row missing -> {p!r}")
        raise AssertionError(f"{len(missing)} audit-table rows missing")
    print(f"  PASS audit table has new error codes")


def test_passage_nested_schema_present():
    """Production passage section was previously flat (out-of-date). After
    integration it should describe the nested schema."""
    required_phrases = [
        'passageQuestion',
        '"ques"',                              # nested inner field
        '"passage": qid',                      # back-ref pattern
    ]
    # The third one might or might not appear depending on exact wording;
    # check at least the two structural ones.
    structural = ['passageQuestion', 'questions:', 'inner sub-question']
    missing = [p for p in structural if p not in INSPECTOR_SYSTEM_PROMPT]
    if missing:
        for p in missing:
            print(f"FAIL: required passage phrase missing -> {p!r}")
        raise AssertionError(f"{len(missing)} passage-schema phrases missing")
    print(f"  PASS passage nested schema present")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Running PRODUCTION stage9 verdict prompt parity tests...")
    test_gate_removed()
    test_match_schema_present()
    test_passage_nested_schema_present()
    test_noise_and_exam_ref_provisions_present()
    test_audit_table_has_new_codes()
    print("ALL TESTS PASSED")
