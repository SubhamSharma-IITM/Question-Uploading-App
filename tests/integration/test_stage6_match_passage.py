"""Stage 6 (image CDN) parity test for match + passage container shapes.

Confirms:
  1. _extract_img_srcs_from_question covers matchQuestion.{question, solution}
  2. _extract_img_srcs_from_question covers passageQuestion.{passage, solution}
     and each questions[i].scq.{ques, question, solution}
  3. _replace_in_question substitutes URLs in ALL container/field paths above
  4. Base64 SMILES SVG image scan walks matchQuestion + passageQuestion fields
  5. Counts in scan() reflect match/passage images correctly
  6. Failure handling: when CDN cache has no entry for a src, the original
     <img> tag is preserved untouched (no broken HTML)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.pipeline.stage6_images import (
    _extract_img_srcs_from_question,
    _replace_in_question,
    _extract_base64_imgs_from_text,
)


# ── Sample entries: SCQ + match + passage (canonical schema) ───────────────

SCQ_ENTRY = {
    "qid": "TEST_SCQ_01",
    "questionType": "scq",
    "scq": {
        "question": '<h3>SCQ <img src="./images/scq_q.png"/></h3>',
        "solution": '<h3>SCQ sol <img src="./images/scq_s.png"/></h3>',
        "answer": "C",
    },
}

MATCH_ENTRY = {
    "qid": "TEST_MATCH_01",
    "questionType": "matchQuestion",
    "matchQuestion": {
        "rows": [["B"], ["A"], ["C"], ["D"]],
        "quesImages": None,
        "solutionImages": None,
        "question": '<table><tr><td><img src="./images/match_q1.png"/></td><td>List-I item 1</td><td><img src="./images/match_q2.png"/></td><td>List-II item</td></tr></table>',
        "solution": '<h3>Sol <img src="./images/match_sol.png"/> derivation</h3>',
    },
}

PASSAGE_ENTRY = {
    "qid": "TEST_PASSAGE_01",
    "questionType": "passageQuestion",
    "passageQuestion": {
        "passage": '<h3>Passage with <img src="./images/passage_fig.png"/></h3>',
        "solution": '<h3>Top sol <img src="./images/passage_sol.png"/></h3>',
        "questions": [
            {"scq": {
                "ques": '<h3>Inner Q1 <img src="./images/inner_q1.png"/></h3>',
                "solution": '<h3>Inner sol1 <img src="./images/inner_sol1.png"/></h3>',
                "answer": "B",
                "passage": "TEST_PASSAGE_01",
            }},
        ],
        "passageImages": [],
        "passageSolutionImages": [],
    },
}


def test_extract_match_image_srcs():
    srcs = _extract_img_srcs_from_question(MATCH_ENTRY)
    expected = {
        "./images/match_q1.png",
        "./images/match_q2.png",
        "./images/match_sol.png",
    }
    got = set(srcs)
    if got != expected:
        print(f"FAIL: expected {expected}, got {got}")
        raise AssertionError("match-type image srcs not extracted")
    print(f"  PASS extract from matchQuestion: 3/3 image srcs (q×2 + sol×1)")


def test_extract_passage_image_srcs():
    srcs = _extract_img_srcs_from_question(PASSAGE_ENTRY)
    expected = {
        "./images/passage_fig.png",
        "./images/passage_sol.png",
        "./images/inner_q1.png",
        "./images/inner_sol1.png",
    }
    got = set(srcs)
    if got != expected:
        print(f"FAIL: expected {expected}, got {got}")
        raise AssertionError("passage-type image srcs not extracted")
    print(f"  PASS extract from passageQuestion: 4/4 srcs (passage + top-sol + inner-q + inner-sol)")


def test_extract_scq_unchanged():
    srcs = _extract_img_srcs_from_question(SCQ_ENTRY)
    assert set(srcs) == {"./images/scq_q.png", "./images/scq_s.png"}
    print("  PASS extract from scq: regression-safe (unchanged behavior)")


def test_replace_in_match():
    cdn_cache = {
        "./images/match_q1.png": {"cdn_url": "https://cdn.example.com/m_q1.png", "display_width": 200, "display_height": 100},
        "./images/match_q2.png": {"cdn_url": "https://cdn.example.com/m_q2.png", "display_width": 220, "display_height": 110},
        "./images/match_sol.png": {"cdn_url": "https://cdn.example.com/m_sol.png", "display_width": 300, "display_height": 150},
    }
    entry = {"qid": "X", "questionType": "matchQuestion", "matchQuestion": dict(MATCH_ENTRY["matchQuestion"])}
    n = _replace_in_question(entry, cdn_cache)
    assert n == 3, f"expected 3 replacements, got {n}"
    q_html = entry["matchQuestion"]["question"]
    assert "https://cdn.example.com/m_q1.png" in q_html
    assert "https://cdn.example.com/m_q2.png" in q_html
    assert "./images/" not in q_html, f"raw ./images/ still present: {q_html}"
    s_html = entry["matchQuestion"]["solution"]
    assert "https://cdn.example.com/m_sol.png" in s_html
    assert "./images/" not in s_html
    print("  PASS replace in matchQuestion: 3 URLs substituted in q + sol, no ./images/ left")


def test_replace_in_passage():
    cdn_cache = {
        "./images/passage_fig.png": {"cdn_url": "https://cdn.example.com/p_fig.png", "display_width": 350, "display_height": 200},
        "./images/passage_sol.png": {"cdn_url": "https://cdn.example.com/p_sol.png", "display_width": 350, "display_height": 200},
        "./images/inner_q1.png": {"cdn_url": "https://cdn.example.com/iq1.png", "display_width": 350, "display_height": 200},
        "./images/inner_sol1.png": {"cdn_url": "https://cdn.example.com/is1.png", "display_width": 350, "display_height": 200},
    }
    import copy
    entry = copy.deepcopy(PASSAGE_ENTRY)
    n = _replace_in_question(entry, cdn_cache)
    assert n == 4, f"expected 4 replacements, got {n}"
    pq = entry["passageQuestion"]
    assert "https://cdn.example.com/p_fig.png" in pq["passage"]
    assert "https://cdn.example.com/p_sol.png" in pq["solution"]
    inner = pq["questions"][0]["scq"]
    assert "https://cdn.example.com/iq1.png" in inner["ques"]
    assert "https://cdn.example.com/is1.png" in inner["solution"]
    # No raw ./images/ refs anywhere
    import json as _json
    blob = _json.dumps(entry)
    assert "./images/" not in blob, f"raw ./images/ still present somewhere"
    print("  PASS replace in passageQuestion: 4 URLs substituted in body + top-sol + inner ques + inner sol")


def test_replace_failure_preserves_original():
    """If CDN cache has no entry for a src, the <img> tag is preserved untouched.
    Failure path must NOT corrupt HTML."""
    import copy
    entry = copy.deepcopy(MATCH_ENTRY)
    cdn_cache = {}  # empty — every lookup misses
    n = _replace_in_question(entry, cdn_cache)
    assert n == 0, f"expected 0 replacements with empty cache, got {n}"
    # Original ./images/ refs preserved exactly
    assert './images/match_q1.png' in entry["matchQuestion"]["question"]
    assert './images/match_q2.png' in entry["matchQuestion"]["question"]
    assert './images/match_sol.png' in entry["matchQuestion"]["solution"]
    print("  PASS failure path: empty cdn_cache leaves original <img> tags untouched")


def test_base64_smiles_scan_covers_match():
    """SMILES SVG base64 inside matchQuestion.{question, solution} must be detected."""
    text = '<table><tr><td><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==" alt="" /></td></tr></table>'
    base64_imgs = _extract_base64_imgs_from_text(text)
    assert len(base64_imgs) == 1, f"expected 1 base64 img, got {len(base64_imgs)}"
    assert base64_imgs[0]["data_uri"].startswith("data:image/svg+xml;base64,")
    print("  PASS base64 scan: SVG data-URI in match HTML detected (Stage 5 SMILES output)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Running PRODUCTION stage6 match+passage parity tests...")
    test_extract_match_image_srcs()
    test_extract_passage_image_srcs()
    test_extract_scq_unchanged()
    test_replace_in_match()
    test_replace_in_passage()
    test_replace_failure_preserves_original()
    test_base64_smiles_scan_covers_match()
    print("ALL TESTS PASSED")
