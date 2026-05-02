"""Regression: Stage 10 must auto-promote err=3/4 entries to err=2 in the
upload payload when settings.stage10_promote_err34 is True (default).

Lock the post-fix behavior:
- _post_question with error_override=2 sends payload error=2 even if entry.error=3
- _post_question without override falls back to entry.error or default 2 as before

We don't actually hit the API — we monkeypatch authenticated_request to capture
the payload that would have been sent.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.pipeline.stage10_upload import UploadStage


def _make_stage_stub():
    inst = UploadStage.__new__(UploadStage)
    inst.session_id = "TEST_PROMOTE"
    inst.session_dir = Path("/tmp/test_promote")
    inst.log = lambda level, msg: None
    return inst


def test_post_question_error_override_promotes_to_2():
    """When error_override=2, payload['error']==2 regardless of entry['error']=3."""
    stage = _make_stage_stub()
    entry = {
        "qid": "TEST_Q1",
        "subject": "Math",
        "questionType": "scq",
        "scq": {"question": "<h3>Q</h3>", "answer": "A", "solution": "<h3>S</h3>"},
        "error": 3,
        "tagSubConcept": [{"subject": "Math", "chapter": "X"}],
    }
    captured = {}
    def fake_request(method, url, **kwargs):
        captured["json"] = kwargs.get("json")
        resp = MagicMock()
        resp.json.return_value = {"isSuccessful": True, "question": {"_id": "fake_oid"}}
        resp.status_code = 200
        resp.raise_for_status = lambda: None
        return resp

    with patch("app.pipeline.auth.authenticated_request", side_effect=fake_request):
        ok, oid = stage._post_question(req_lib=None, entry=entry, error_override=2)

    assert ok is True
    assert oid == "fake_oid"
    assert captured["json"]["error"] == 2, (
        f"expected promoted error=2 in payload, got {captured['json'].get('error')!r}"
    )
    # Original entry untouched
    assert entry["error"] == 3, "entry mutated — must use copy in _post_question"


def test_post_question_no_override_keeps_entry_error():
    """When error_override is None, payload uses entry's error."""
    stage = _make_stage_stub()
    entry = {
        "qid": "TEST_Q2",
        "subject": "Math",
        "questionType": "scq",
        "scq": {"question": "<h3>Q</h3>", "answer": "B", "solution": "<h3>S</h3>"},
        "error": 1,
    }
    captured = {}
    def fake_request(method, url, **kwargs):
        captured["json"] = kwargs.get("json")
        resp = MagicMock()
        resp.json.return_value = {"isSuccessful": True, "question": {"_id": "fake_oid_2"}}
        resp.status_code = 200
        resp.raise_for_status = lambda: None
        return resp

    with patch("app.pipeline.auth.authenticated_request", side_effect=fake_request):
        ok, oid = stage._post_question(req_lib=None, entry=entry)

    assert ok
    assert captured["json"]["error"] == 1, (
        f"expected entry's error=1 to flow through, got {captured['json'].get('error')!r}"
    )


def test_post_question_no_override_no_error_field_defaults_to_2():
    """Entry without 'error' field defaults to 2 (legacy behavior preserved)."""
    stage = _make_stage_stub()
    entry = {
        "qid": "TEST_Q3",
        "subject": "Math",
        "questionType": "scq",
        "scq": {"question": "<h3>Q</h3>", "answer": "C", "solution": "<h3>S</h3>"},
        # no "error" key
    }
    captured = {}
    def fake_request(method, url, **kwargs):
        captured["json"] = kwargs.get("json")
        resp = MagicMock()
        resp.json.return_value = {"isSuccessful": True, "question": {"_id": "fake_oid_3"}}
        resp.status_code = 200
        resp.raise_for_status = lambda: None
        return resp

    with patch("app.pipeline.auth.authenticated_request", side_effect=fake_request):
        ok, oid = stage._post_question(req_lib=None, entry=entry)

    assert ok
    assert captured["json"]["error"] == 2


def test_settings_default_is_promote_on():
    """settings.stage10_promote_err34 must default to True so production benefits."""
    from app.config import settings
    assert getattr(settings, "stage10_promote_err34", None) is True


if __name__ == "__main__":
    test_post_question_error_override_promotes_to_2()
    test_post_question_no_override_keeps_entry_error()
    test_post_question_no_override_no_error_field_defaults_to_2()
    test_settings_default_is_promote_on()
    print("OK")
