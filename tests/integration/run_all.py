"""Consolidated integration validation gate.

Runs every production-parity test in tests/integration/ in sequence and
reports a single pass/fail summary. Exits nonzero if any test fails.

This is the gate per user direction: "compilation is not equals to confirmation.
If test cases that passed through the actual pilot phase properly also passed
through production, then we are done."
"""
import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Order: smaller → bigger; run pipeline parity LAST as the end-to-end gate
SUITES = [
    "test_stage2_match_processing",
    "test_stage2_passage_cursor_advance",
    "test_stage2_match_canonical_digits",
    "test_stage2_match_real_data",
    "test_stage3_noise_strip",
    "test_stage3_match_passage_noise",
    "test_stage4_match_passage",
    "test_stage4_triplet_match_passage",
    "test_stage5_match_passage",
    "test_stage6_match_passage",
    "test_stage7_match_passage",
    "test_stage8_match_passage",
    "test_stage9_match_gate",
    "test_stage9_batch_builder",
    "test_stage10_match_passage",
    "test_stage10_promote_err34",
    "test_stage10_promotion_csv_columns",
    "test_stage3_4_pipeline_parity",  # end-to-end gate, last
]


def run_module(name):
    """Import a test module and call its top-level test_* functions."""
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        print(f"  IMPORT FAIL: {e}")
        traceback.print_exc()
        return False, 0

    test_fns = [f for f in dir(mod) if f.startswith("test_") and callable(getattr(mod, f))]
    if not test_fns:
        # Module's __main__ block runs all tests inline — invoke main if present
        main_fn = getattr(mod, "main", None)
        if callable(main_fn):
            try:
                main_fn()
                return True, 1
            except Exception as e:
                print(f"  FAIL: {e}")
                return False, 0
        # Some files use only the if __name__ == '__main__' block — re-execute as script
        import subprocess
        try:
            r = subprocess.run([sys.executable, str(Path(__file__).parent / f"{name}.py")],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               cwd=str(ROOT))
            if r.returncode == 0:
                # Print last 20 lines for context
                tail = "\n".join(r.stdout.strip().split("\n")[-20:])
                print(tail)
                return True, 1
            else:
                print(r.stdout[-2000:])
                print(r.stderr[-1000:])
                return False, 0
        except Exception as e:
            print(f"  EXEC FAIL: {e}")
            return False, 0

    passes = 0
    fails = 0
    for fn_name in test_fns:
        try:
            getattr(mod, fn_name)()
            passes += 1
        except Exception as e:
            fails += 1
            print(f"  FAIL {fn_name}: {e}")
            traceback.print_exc()
    return fails == 0, passes


def main():
    t0 = time.time()
    results = []
    total_passes = 0
    for name in SUITES:
        ok, n = run_module(name)
        results.append((name, ok, n))
        total_passes += n

    print(f"\n{'='*72}")
    print("CONSOLIDATED VALIDATION GATE — SUMMARY")
    print(f"{'='*72}")
    failed = [r[0] for r in results if not r[1]]
    for name, ok, n in results:
        marker = "✅" if ok else "❌"
        print(f"  {marker}  {name}  ({n} tests)")
    elapsed = time.time() - t0
    print()
    print(f"Total suites: {len(SUITES)}, passed: {len(SUITES) - len(failed)}, failed: {len(failed)}")
    print(f"Total individual tests passed: {total_passes}")
    print(f"Elapsed: {elapsed:.1f}s")
    if failed:
        print(f"\nFAILED SUITES: {', '.join(failed)}")
        sys.exit(1)
    print("\n🟢 ALL INTEGRATION GATES PASSED — production is consistent with pilot")


if __name__ == "__main__":
    main()
