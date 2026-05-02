"""Real-data + targeted regression: Stage 3 _strip_format_noise must remove
match-type section markers and JEE/NEET year-reference citations without
touching real question content.

Cross-validated against the .mmd source files for 6 audited sessions
(see tmp_render_audit/) — these patterns appear hundreds of times as pure noise.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.pipeline.stage3_cleaner import _strip_format_noise


def test_match_section_markers_stripped():
    cases = [
        # Match-type section headers (these were leaking into match question HTML)
        (r'\section*{MATCHING LIST TYPE QUESTION}', ''),
        (r'\section*{MATCHING LIST TYPE QUESTIONS}', ''),
        (r'\section*{MATRIX MATCH TYPE QUESTION}', ''),
        (r'\section*{MATRIX MATCH TYPE QUESTIONS}', ''),
        # Combined slash form (real case from EDM_MATH_Q25 — Determinants session)
        (r'\section*{MATRIX MATCH/MATCHING LIST TYPE QUESTION}', ''),
        (r'\section*{MATRIX MATCH / MATCHING LIST TYPE QUESTIONS}', ''),
        (r'\section*{MATCHING LIST/MATRIX MATCH TYPE QUESTION}', ''),
        (r'\section*{List-I}', ''),
        (r'\section*{List-II}', ''),
        (r'\section*{List - I}', ''),
        (r'\section*{List I}', ''),
        (r'\section*{Column-I}', ''),
        (r'\section*{Column-II}', ''),
        (r'\section*{Column - II}', ''),
    ]
    for inp, expected in cases:
        out = _strip_format_noise(inp).strip()
        assert out == expected, f"input {inp!r} expected {expected!r} got {out!r}"


def test_match_section_with_surrounding_content_preserved():
    """Body content + (I)/(P) labels must remain — only \section*{} wrapper goes."""
    inp = (r'\section*{MATCHING LIST TYPE QUESTION}' + '\n34.\n\n'
           r'\section*{List-I}' + '\n(I) tan20.tan40\n'
           r'\section*{List-II}' + '\n(P) Odd natural\n')
    out = _strip_format_noise(inp)
    assert '(I) tan20.tan40' in out
    assert '(P) Odd natural' in out
    assert 'MATCHING LIST' not in out
    assert 'List-I' not in out
    assert 'List-II' not in out


def test_year_reference_brackets_stripped():
    cases = [
        ('Q stem [JEE (Advanced) 2017] more text', 'Q stem  more text'),
        ('text [JEE Advanced 2018] x', 'text  x'),
        ('y [JEE (Main) 2020] z', 'y  z'),
        ('a [JEE Main 2021] b', 'a  b'),
        ('p [NEET 2019] q', 'p  q'),
        ('r [AIEEE 2010] s', 'r  s'),
    ]
    for inp, expected in cases:
        out = _strip_format_noise(inp)
        assert out == expected, f"input {inp!r} expected {expected!r} got {out!r}"


def test_year_required_for_jee_bracket_strip():
    """Plain `[JEE Advanced]` without a year is NOT noise — leave alone."""
    inp = 'See [JEE Advanced] reference book.'
    out = _strip_format_noise(inp)
    assert '[JEE Advanced]' in out, f"plain bracket without year was wrongly stripped: {out!r}"


def test_match_label_inside_text_not_stripped():
    """`(I)` `(II)` `(P)` etc. inside content must NEVER be stripped — they're
    match-type LABELS."""
    inp = '(I) tan20° tan40° (II) sin45 (P) Odd Number (Q) Even Number'
    out = _strip_format_noise(inp)
    assert out == inp, f"label content wrongly modified: {out!r}"


def test_existing_paragraph_pattern_still_works():
    """The pre-existing paragraph-for-question pattern must still strip."""
    inp = '**Paragraph for Question Nos. 9 to 10**\nbody'
    out = _strip_format_noise(inp).strip()
    assert 'body' in out
    assert 'Paragraph for' not in out


def test_real_match_data_from_replay():
    """Use the actual question_html from a Vector Algebra match Q."""
    raw = (r'<p>\section*{MATCHING LIST TYPE QUESTION}' + '\n34.\n\n'
           r'\section*{List-I}' + '\n(I) The value of <math>...</math>\n\n'
           r'\section*{List-II}' + '\n(II) The value of <math>...</math>\n')
    out = _strip_format_noise(raw)
    # All section markers gone
    assert 'MATCHING LIST' not in out, f"didn't strip MATCHING LIST: {out[:200]}"
    assert r'\section*{List-I}' not in out, f"didn't strip List-I: {out[:200]}"
    assert r'\section*{List-II}' not in out, f"didn't strip List-II: {out[:200]}"
    # But labels and content preserved
    assert '(I) The value of' in out
    assert '(II) The value of' in out


if __name__ == "__main__":
    test_match_section_markers_stripped()
    test_match_section_with_surrounding_content_preserved()
    test_year_reference_brackets_stripped()
    test_year_required_for_jee_bracket_strip()
    test_match_label_inside_text_not_stripped()
    test_existing_paragraph_pattern_still_works()
    test_real_match_data_from_replay()
    print("OK")
