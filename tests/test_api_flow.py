"""API flow test: Exercise stages 1-4 via HTTP endpoints.

Simulates what a browser user would do: create session, load data,
run each stage via API calls, verify results.

Usage:
    1. Start the app: python -m uvicorn app.main:app --port 8000
    2. Run this test: python -m tests.test_api_flow
"""

import json
import shutil
import sys
import time
import requests
from pathlib import Path

BASE = "http://localhost:8000"
TEST_DATA = Path(__file__).parent / "test_data"

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(text):
    print(f"  {GREEN}[OK]{RESET} {text}")


def fail(text):
    print(f"  {RED}[FAIL]{RESET} {text}")
    sys.exit(1)


def section(text):
    print(f"\n{BOLD}{CYAN}--- {text} ---{RESET}")


def main():
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f"  API Flow Test: 5 Questions Through HTTP Endpoints")
    print(f"{'='*60}{RESET}")

    # ── Health check ──
    section("Health check")
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200
        ok(f"Server is up: {r.json()}")
    except Exception as e:
        fail(f"Server not running at {BASE}: {e}")

    # ── Create session ──
    section("Create session")
    r = requests.post(f"{BASE}/api/sessions", json={"name": "API Test: AIATS-05 Q1-5"})
    assert r.status_code == 200
    data = r.json()
    assert data["success"]
    session_id = data["data"]["session_id"]
    ok(f"Session created: {session_id}")

    # ── Stage 1: Load test MMD files into session ──
    section("Stage 1: Load test data")
    # Copy test files directly into session mathpix_output (simulating MathPIX)
    from app.config import settings
    session_dir = Path(settings.sessions_dir) / session_id
    mmd_dir = session_dir / "mathpix_output"
    mmd_dir.mkdir(parents=True, exist_ok=True)

    for src_name in ["AIATS05_questions.mmd", "AIATS05_solutions.mmd", "AIATS05_answer_key.txt"]:
        src = TEST_DATA / src_name
        shutil.copy2(src, mmd_dir / src_name)
        ok(f"Loaded {src_name}")

    # Advance to stage 1
    r = requests.post(f"{BASE}/api/sessions/{session_id}/advance")
    assert r.json()["success"]
    ok("Advanced to stage1_upload")

    # ── Stage 2: Run LLM mapper via API ──
    section("Stage 2: LLM Mapper")
    r = requests.post(f"{BASE}/api/sessions/{session_id}/stage2/start?max_questions=5", timeout=30)
    data = r.json()
    if not data["success"]:
        fail(f"Stage 2 failed: {json.dumps(data, indent=2)}")
    ok(f"Extracted {data['data']['questions']} questions, {data['data']['triplets']} triplets")

    # Advance
    r = requests.post(f"{BASE}/api/sessions/{session_id}/advance")
    assert r.json()["success"]

    # ── Stage 3: Cleaner ──
    section("Stage 3: Cleaner")
    # Dry run
    r = requests.post(f"{BASE}/api/sessions/{session_id}/stage3/dry-run")
    data = r.json()
    assert data["success"]
    ok(f"Dry run: {data['data']['questions_modified']} would change")

    # Apply
    r = requests.post(f"{BASE}/api/sessions/{session_id}/stage3/apply")
    data = r.json()
    assert data["success"]
    ok(f"Cleaned {data['data']['questions_cleaned']} questions")

    # Advance
    r = requests.post(f"{BASE}/api/sessions/{session_id}/advance")
    assert r.json()["success"]

    # ── Stage 4: Metadata V2 ──
    section("Stage 4: Metadata V2")
    r = requests.post(
        f"{BASE}/api/sessions/{session_id}/stage4/convert",
        params={"source": "AIATS-05 Aakash", "test_code": "AIATS05"},
    )
    data = r.json()
    assert data["success"]
    ok(f"Created {data['data']['total']} metadata entries, {data['data']['triplets']} triplets")

    # ── Verify final session state ──
    section("Verify final state")
    r = requests.get(f"{BASE}/api/sessions/{session_id}")
    data = r.json()["data"]

    ok(f"Session stage: {data['current_stage']}")
    ok(f"Questions tracked: {data['stats']['total']}")
    ok(f"With answer: {data['stats']['with_answer']}")
    ok(f"With solution: {data['stats']['with_solution']}")
    ok(f"By subject: {data['stats']['by_subject']}")

    # Verify checkpoints
    for cp in data["checkpoints"]:
        status = cp["status"]
        color = GREEN if status == "completed" else RED
        print(f"    {cp['stage']:<20} {color}{status}{RESET}")

    # ── Verify metadata file content ──
    section("Verify metadata content")
    meta_file = session_dir / "metadata_v2" / "AIATS05_Physics_metadata_v2.json"
    assert meta_file.exists(), "Metadata file not found"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    assert len(meta_data) == 5, f"Expected 5, got {len(meta_data)}"

    expected_answers = {"PHY_AIATS05_Q1": "D", "PHY_AIATS05_Q2": "C",
                        "PHY_AIATS05_Q3": "D", "PHY_AIATS05_Q4": "B",
                        "PHY_AIATS05_Q5": "B"}

    all_match = True
    for meta in meta_data:
        qid = meta["qid"]
        actual = meta["scq"]["answer"]
        expected = expected_answers.get(qid, "?")
        match = actual == expected
        if not match:
            all_match = False
        status = f"{GREEN}MATCH{RESET}" if match else f"{RED}MISMATCH{RESET}"
        print(f"    {qid}: answer={actual} expected={expected} {status}")

    assert all_match, "Answer mismatch detected!"
    ok("All 5 answers verified correct")

    print(f"\n{BOLD}{GREEN}{'='*60}")
    print(f"  ALL API TESTS PASSED!")
    print(f"  5 questions: MMD -> Mapper -> Cleaner -> Metadata V2")
    print(f"{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()
