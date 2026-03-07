"""End-to-end pipeline test: 5 questions through Stages 1-4.

Verifies that 5 real questions from AIATS-05 (Aakash) travel correctly
from raw MMD → mapped JSON → cleaned JSON → metadata_v2 JSON.

Usage:
    cd "D:\ACADZA_WORK\Question Uploading App"
    python -m tests.test_pipeline_e2e
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.db import init_db, create_session, get_session, get_session_stats, get_all_checkpoints, get_logs
from app.sse import sse_manager
from app.utils import now_iso, short_uuid, _safe_print

# Test data paths
TEST_DATA = Path(__file__).parent / "test_data"
MMD_QUESTIONS = TEST_DATA / "AIATS05_questions.mmd"
MMD_SOLUTIONS = TEST_DATA / "AIATS05_solutions.mmd"
ANSWER_KEY = TEST_DATA / "AIATS05_answer_key.txt"

# Terminal colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(text):
    _safe_print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    _safe_print(f"{BOLD}{CYAN}  {text}{RESET}")
    _safe_print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


def section(text):
    _safe_print(f"\n{BOLD}{YELLOW}--- {text} ---{RESET}")


def ok(text):
    _safe_print(f"  {GREEN}[OK]{RESET} {text}")


def fail(text):
    _safe_print(f"  {RED}[FAIL]{RESET} {text}")


def info(text):
    _safe_print(f"  {CYAN}[..]{RESET} {text}")


def run_test():
    banner("ACADZA Pipeline E2E Test: 5 Questions Through Stages 1-4")

    # ── Check test data exists ──
    section("Checking test data files")
    for f, label in [(MMD_QUESTIONS, "Questions MMD"), (MMD_SOLUTIONS, "Solutions MMD"), (ANSWER_KEY, "Answer Key")]:
        if f.exists():
            ok(f"{label}: {f.name} ({f.stat().st_size} bytes)")
        else:
            fail(f"{label}: {f.name} NOT FOUND")
            return False

    # ── Init DB ──
    section("Initializing database")
    init_db()
    ok("Database initialized")

    # ── Create session ──
    section("Creating test session")
    session_id = short_uuid()
    session_name = "E2E Test: AIATS-05 Physics Q1-5"
    create_session(session_id, session_name, now_iso())

    session_dir = Path(settings.sessions_dir) / session_id
    for subdir in ["input", "mathpix_output", "mapped_output", "cleaned_output",
                    "metadata_v2", "backups"]:
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)

    ok(f"Session: {session_id} ({session_name})")
    ok(f"Directory: {session_dir}")

    # ═══════════════════════════════════════════════════
    # STAGE 1: MathPIX (simulated — load test MMD files)
    # ═══════════════════════════════════════════════════
    banner("STAGE 1: PDF Upload / MathPIX OCR (Simulated)")

    from app.pipeline.stage1_mathpix import MathPIXStage
    stage1 = MathPIXStage(session_id, session_dir)

    result1 = stage1.execute(
        mmd_questions=str(MMD_QUESTIONS),
        mmd_solutions=str(MMD_SOLUTIONS),
        answer_key=str(ANSWER_KEY),
    )

    ok(f"Files loaded: {result1['count']}")
    for fname in result1["files"]:
        info(f"  {fname}")

    # Verify files exist
    mmd_dir = session_dir / "mathpix_output"
    assert len(list(mmd_dir.iterdir())) == 3, "Expected 3 files in mathpix_output"
    ok("Stage 1 PASSED")

    # ═══════════════════════════════════════════════════
    # STAGE 2: LLM Mapper (regex extraction)
    # ═══════════════════════════════════════════════════
    banner("STAGE 2: LLM Mapper (Regex Extraction)")

    from app.pipeline.stage2_llm_mapper import LLMMapperStage
    stage2 = LLMMapperStage(session_id, session_dir)

    result2 = stage2.execute(max_questions=5)

    ok(f"Questions extracted: {result2['questions']}")
    ok(f"Q+A+S triplets: {result2['triplets']}")

    # Verify mapped output
    mapped_file = Path(result2["output_file"])
    assert mapped_file.exists(), f"Mapped output not found: {mapped_file}"
    with open(mapped_file, "r", encoding="utf-8") as f:
        mapped_data = json.load(f)

    assert len(mapped_data) == 5, f"Expected 5 questions, got {len(mapped_data)}"

    _safe_print("")
    _safe_print(f"  {BOLD}{'Q#':<5} {'Answer':<8} {'Has Solution':<14} {'Question Preview'}{RESET}")
    _safe_print(f"  {'─'*70}")
    for entry in mapped_data:
        qn = entry["question_number"]
        ans = entry.get("answer", "—")
        has_sol = "Yes" if entry.get("solution") else "No"
        preview = entry["question"][:50] + "..."
        _safe_print(f"  Q{qn:<4} {ans:<8} {has_sol:<14} {preview}")

    ok("Stage 2 PASSED")

    # ═══════════════════════════════════════════════════
    # STAGE 3: Cleaner
    # ═══════════════════════════════════════════════════
    banner("STAGE 3: Advanced Cleaner")

    from app.pipeline.stage3_cleaner import CleanerStage
    stage3 = CleanerStage(session_id, session_dir)

    # Dry run first
    dry_result = stage3.dry_run()
    info(f"Dry run: {dry_result['questions_modified']} questions would be modified")

    # Apply
    result3 = stage3.execute()
    ok(f"Questions cleaned: {result3['questions_cleaned']}")
    ok(f"Fields modified: {result3['fields_modified']}")

    # Verify cleaned output
    cleaned_dir = session_dir / "cleaned_output"
    cleaned_files = list(cleaned_dir.glob("*_cleaned.json"))
    assert len(cleaned_files) > 0, "No cleaned files produced"

    with open(cleaned_files[0], "r", encoding="utf-8") as f:
        cleaned_data = json.load(f)
    assert len(cleaned_data) == 5, f"Expected 5 cleaned questions, got {len(cleaned_data)}"

    ok("Stage 3 PASSED")

    # ═══════════════════════════════════════════════════
    # STAGE 4: Metadata V2 Mapper
    # ═══════════════════════════════════════════════════
    banner("STAGE 4: Metadata V2 Mapper")

    from app.pipeline.stage4_metadata import MetadataMapperStage
    stage4 = MetadataMapperStage(session_id, session_dir)

    result4 = stage4.execute(
        source="AIATS-05 Aakash Preparatory Test-2",
        test_code="AIATS05",
    )

    ok(f"Metadata entries: {result4['total']}")
    ok(f"Q+A+S triplets: {result4['triplets']}")

    # Verify metadata_v2 output
    meta_dir = session_dir / "metadata_v2"
    meta_files = list(meta_dir.glob("*_metadata_v2.json"))
    assert len(meta_files) > 0, "No metadata_v2 files produced"

    with open(meta_files[0], "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert len(meta_data) == 5, f"Expected 5 metadata entries, got {len(meta_data)}"

    _safe_print("")
    _safe_print(f"  {BOLD}{'QID':<20} {'Type':<8} {'Answer':<8} {'Q':<3} {'A':<3} {'S':<3} {'Level'}{RESET}")
    _safe_print(f"  {'─'*60}")
    for meta in meta_data:
        qid = meta["qid"]
        qtype = meta["questionType"]
        qdata = meta[qtype]
        ans = qdata.get("answer", "—")
        has_q = "Y" if qdata.get("question", "").replace("<h3></h3>", "") else "N"
        has_a = "Y" if qdata.get("answer") else "N"
        has_s = "Y" if qdata.get("solution", "").replace("<h3></h3>", "") else "N"
        level = meta.get("level", "?")
        _safe_print(f"  {qid:<20} {qtype:<8} {ans:<8} {has_q:<3} {has_a:<3} {has_s:<3} {level}")

    # Validate schema fields
    for meta in meta_data:
        assert "qid" in meta, "Missing qid"
        assert "questionType" in meta, "Missing questionType"
        assert "subject" in meta, "Missing subject"
        assert meta["questionType"] in meta, f"Missing {meta['questionType']} data"
        assert meta["addedByAi"] == True, "addedByAi should be True"
        assert meta["class"] == 12, "class should be 12"
        assert meta["error"] == 2, "error should be 2"

    ok("Schema validation PASSED")
    ok("Stage 4 PASSED")

    # ═══════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════
    banner("TEST SUMMARY")

    # Get session stats from DB
    stats = get_session_stats(session_id)
    checkpoints = get_all_checkpoints(session_id)

    _safe_print(f"  Session: {session_id}")
    _safe_print(f"  Name: {session_name}")
    _safe_print(f"  Questions tracked: {stats['total']}")
    _safe_print(f"  With answer: {stats['with_answer']}")
    _safe_print(f"  With solution: {stats['with_solution']}")
    _safe_print(f"  By subject: {stats['by_subject']}")
    _safe_print("")

    _safe_print(f"  {BOLD}Stage Checkpoints:{RESET}")
    for cp in checkpoints:
        status_color = GREEN if cp["status"] == "completed" else RED
        _safe_print(f"    {cp['stage']:<20} {status_color}{cp['status']}{RESET}")

    _safe_print("")

    # Show file sizes
    _safe_print(f"  {BOLD}Output Files:{RESET}")
    for stage_dir, label in [
        (mmd_dir, "Stage 1 (MathPIX)"),
        (session_dir / "mapped_output", "Stage 2 (Mapped)"),
        (session_dir / "cleaned_output", "Stage 3 (Cleaned)"),
        (session_dir / "metadata_v2", "Stage 4 (Metadata V2)"),
    ]:
        files = list(stage_dir.iterdir()) if stage_dir.exists() else []
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        _safe_print(f"    {label:<25} {len(files)} files, {total_size:,} bytes")

    # Show logs
    logs = get_logs(session_id, limit=500)
    _safe_print(f"\n  Total log entries: {len(logs)}")

    _safe_print(f"\n{BOLD}{GREEN}{'='*60}")
    _safe_print(f"  ALL 4 STAGES PASSED — 5 questions successfully processed!")
    _safe_print(f"  Questions traveled: MMD -> Mapped -> Cleaned -> Metadata V2")
    _safe_print(f"{'='*60}{RESET}")

    return True


if __name__ == "__main__":
    try:
        success = run_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        _safe_print(f"\n{RED}[FATAL] {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
