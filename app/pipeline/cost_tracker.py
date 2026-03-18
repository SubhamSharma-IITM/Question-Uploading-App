"""Cost tracker module -- reads actual token usage from batch results and calculates exact costs.

Reads from:
  - Stage 7: batch_results/tagging_results.jsonl, batch_results/latex_results.jsonl
  - Stage 9 Pass 1: quality_check/pass1_batch_results.jsonl (Gemini Vertex)
  - Stage 9 Pass 2: quality_check/pass2_batch_results.jsonl (OpenAI)
  - Stage 9 Pass 3: quality_check/pass3_batch_results.jsonl (OpenAI)
  - Stage 1: page count from session_config.json
  - Stage 2: estimated (mapper doesn't persist usage yet)

Output: cost_report.json saved to session directory.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional


# Verified batch pricing (March 2026)
# Batch = 50% of standard. Cached = additional 50% off batch (25% of standard).
PRICING = {
    "gemini_flash_batch": {
        "input_uncached": 0.25,     # $/1M tokens
        "input_cached": 0.125,      # $/1M tokens
        "output": 1.50,             # $/1M tokens
    },
    "gpt5mini_batch": {
        "input_uncached": 0.125,
        "input_cached": 0.0625,
        "output": 1.00,
    },
    "gpt51_batch": {
        "input_uncached": 0.625,
        "input_cached": 0.3125,
        "output": 5.00,
    },
    "gpt5mini_standard": {          # Stage 2 Gemini uses standard (real-time)
        "input_uncached": 0.25,
        "input_cached": 0.125,
        "output": 2.00,
    },
    "gemini_flash_standard": {
        "input_uncached": 0.50,
        "input_cached": 0.25,
        "output": 3.00,
    },
    "mathpix": {
        "per_page": 0.004,          # $/page
        "questions_per_page": 4,    # estimated
    },
}

INR_PER_USD = 85


def _cost_usd(pricing_key: str, uncached: int, cached: int, output: int) -> float:
    """Calculate cost in USD from token counts."""
    p = PRICING.get(pricing_key, {})
    return (uncached / 1e6 * p.get("input_uncached", 0) +
            cached / 1e6 * p.get("input_cached", 0) +
            output / 1e6 * p.get("output", 0))


def _parse_openai_batch_results(filepath: str) -> Dict:
    """Parse OpenAI batch results for token usage."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 10:
        return {"n": 0, "prompt": 0, "cached": 0, "completion": 0, "reasoning": 0}

    total = {"n": 0, "prompt": 0, "cached": 0, "completion": 0, "reasoning": 0}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                u = r.get("response", {}).get("body", {}).get("usage", {})
                if u:
                    total["n"] += 1
                    total["prompt"] += u.get("prompt_tokens", 0)
                    total["cached"] += u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    total["completion"] += u.get("completion_tokens", 0)
                    total["reasoning"] += u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            except (json.JSONDecodeError, KeyError):
                continue

    total["uncached"] = total["prompt"] - total["cached"]
    return total


def _parse_gemini_batch_results(filepath: str) -> Dict:
    """Parse Gemini Vertex batch results for token usage."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 10:
        return {"n": 0, "prompt": 0, "cached": 0, "completion": 0}

    total = {"n": 0, "prompt": 0, "cached": 0, "completion": 0}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                um = r.get("response", {}).get("usageMetadata", {})
                if um:
                    total["n"] += 1
                    total["prompt"] += um.get("promptTokenCount", 0)
                    total["cached"] += um.get("cachedContentTokenCount", 0)
                    total["completion"] += um.get("candidatesTokenCount", 0)
            except (json.JSONDecodeError, KeyError):
                continue

    total["uncached"] = total["prompt"] - total["cached"]
    return total


def generate_cost_report(session_dir: str) -> Dict:
    """Generate comprehensive cost report for a session.

    Reads actual token usage from batch results files.
    Returns detailed cost breakdown per stage and per question.
    """
    session_dir = Path(session_dir)
    qa_dir = session_dir / "quality_check"
    batch_dir = session_dir / "batch_results"

    report = {
        "session_id": session_dir.name,
        "stages": {},
        "totals": {},
    }

    # ── Stage 1: MathPIX OCR ──
    try:
        with open(session_dir / "session_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        # Estimate pages from total questions
        total_qs_config = config.get("total_questions", 0)
        pages_est = max(1, total_qs_config // PRICING["mathpix"]["questions_per_page"])
        mathpix_cost = pages_est * PRICING["mathpix"]["per_page"]
        report["stages"]["stage1_mathpix"] = {
            "pages_estimated": pages_est,
            "cost_usd": mathpix_cost,
            "cost_inr": mathpix_cost * INR_PER_USD,
            "source": "estimated_from_question_count",
        }
    except Exception:
        report["stages"]["stage1_mathpix"] = {"cost_usd": 0, "cost_inr": 0, "source": "unavailable"}

    # ── Stage 2: Gemini Extraction ──
    # Mapper doesn't persist usage yet -- estimate from question count
    try:
        meta_files = list((session_dir / "metadata_v2").glob("*_metadata_v2.json"))
        total_qs = 0
        for mf in meta_files:
            with open(mf, "r", encoding="utf-8") as f:
                total_qs += len(json.load(f))

        # Estimate: ~2000 input tokens, ~500 output per question (standard Gemini rate)
        est_input = total_qs * 2000
        est_output = total_qs * 500
        s2_cost = _cost_usd("gemini_flash_standard", est_input, 0, est_output)
        report["stages"]["stage2_gemini"] = {
            "questions": total_qs,
            "est_input_tokens": est_input,
            "est_output_tokens": est_output,
            "cost_usd": s2_cost,
            "cost_inr": s2_cost * INR_PER_USD,
            "source": "estimated_2000in_500out_per_q",
        }
    except Exception:
        report["stages"]["stage2_gemini"] = {"cost_usd": 0, "cost_inr": 0, "source": "unavailable"}

    # ── Stage 7: Tagging + LaTeX (OpenAI batch) ──
    for batch_type in ["tagging", "latex"]:
        filepath = batch_dir / f"{batch_type}_results.jsonl"
        usage = _parse_openai_batch_results(str(filepath))
        if usage["n"] > 0:
            cost = _cost_usd("gpt5mini_batch", usage["uncached"], usage["cached"], usage["completion"])
            report["stages"][f"stage7_{batch_type}"] = {
                "questions": usage["n"],
                "prompt_tokens": usage["prompt"],
                "cached_tokens": usage["cached"],
                "uncached_tokens": usage["uncached"],
                "completion_tokens": usage["completion"],
                "avg_prompt_per_q": usage["prompt"] // usage["n"],
                "avg_completion_per_q": usage["completion"] // usage["n"],
                "cache_rate": f"{usage['cached']/usage['prompt']*100:.1f}%" if usage["prompt"] else "0%",
                "cost_usd": cost,
                "cost_inr": cost * INR_PER_USD,
                "source": "actual_batch_results",
            }

    # ── Stage 9: QA Passes ──
    pass_configs = {
        "pass1": {"file": "pass1_batch_results.jsonl", "parser": "gemini", "pricing": "gemini_flash_batch", "model": "Gemini Flash"},
        "pass2": {"file": "pass2_batch_results.jsonl", "parser": "openai", "pricing": "gpt5mini_batch", "model": "GPT-5-mini"},
        "pass3": {"file": "pass3_batch_results.jsonl", "parser": "openai", "pricing": "gpt51_batch", "model": "GPT-5.1"},
    }

    for pass_key, cfg in pass_configs.items():
        filepath = qa_dir / cfg["file"]
        if cfg["parser"] == "openai":
            usage = _parse_openai_batch_results(str(filepath))
        else:
            usage = _parse_gemini_batch_results(str(filepath))

        if usage["n"] > 0:
            cost = _cost_usd(cfg["pricing"], usage["uncached"], usage["cached"], usage["completion"])
            report["stages"][f"stage9_{pass_key}"] = {
                "model": cfg["model"],
                "questions": usage["n"],
                "prompt_tokens": usage["prompt"],
                "cached_tokens": usage["cached"],
                "uncached_tokens": usage["uncached"],
                "completion_tokens": usage["completion"],
                "avg_prompt_per_q": usage["prompt"] // usage["n"],
                "avg_completion_per_q": usage["completion"] // usage["n"],
                "cache_rate": f"{usage['cached']/usage['prompt']*100:.1f}%" if usage["prompt"] else "0%",
                "cost_usd": cost,
                "cost_inr": cost * INR_PER_USD,
                "source": "actual_batch_results",
            }

    # ── Totals ──
    total_cost_usd = sum(s.get("cost_usd", 0) for s in report["stages"].values())
    total_cost_inr = total_cost_usd * INR_PER_USD

    # Count uploaded questions
    uploaded = 0
    error1_count = 0
    error2_count = 0
    for cf in glob.glob(str(session_dir / "upload_logs" / "*_upload_log.csv")):
        import csv
        with open(cf, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = row.get("UploadedQuestionID", "")
                if uid and len(uid) == 24:
                    uploaded += 1
                    el = row.get("Error Level", "")
                    if el == "1":
                        error1_count += 1
                    elif el == "2":
                        error2_count += 1

    report["totals"] = {
        "total_cost_usd": total_cost_usd,
        "total_cost_inr": total_cost_inr,
        "total_questions_processed": total_qs if 'total_qs' in dir() else 0,
        "total_uploaded": uploaded,
        "error1_uploaded": error1_count,
        "error2_uploaded": error2_count,
        "cost_per_uploaded_q_inr": total_cost_inr / uploaded if uploaded else 0,
        "cost_per_error1_q_inr": total_cost_inr / error1_count if error1_count else 0,
        "cost_per_all_uploaded_inr": total_cost_inr / (error1_count + error2_count) if (error1_count + error2_count) else 0,
    }

    # Save report
    report_path = session_dir / "cost_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def print_cost_report(report: Dict):
    """Pretty-print cost report."""
    print(f"\n{'='*70}")
    print(f"COST REPORT: Session {report.get('session_id', '?')}")
    print(f"{'='*70}")

    for stage_key, stage in report.get("stages", {}).items():
        cost_inr = stage.get("cost_inr", 0)
        source = stage.get("source", "?")
        n = stage.get("questions", 0)
        print(f"\n  {stage_key}:")
        print(f"    Questions: {n}")
        if "prompt_tokens" in stage:
            print(f"    Tokens: {stage['prompt_tokens']:,} prompt "
                  f"({stage.get('cache_rate', '?')} cached) "
                  f"+ {stage['completion_tokens']:,} completion")
            print(f"    Avg/q: {stage.get('avg_prompt_per_q', 0):,} in + "
                  f"{stage.get('avg_completion_per_q', 0):,} out")
        print(f"    Cost: Rs {cost_inr:.2f} (${stage.get('cost_usd', 0):.4f})")
        print(f"    Source: {source}")

    t = report.get("totals", {})
    print(f"\n  {'='*50}")
    print(f"  TOTALS:")
    print(f"    Total cost: Rs {t.get('total_cost_inr', 0):.2f}")
    print(f"    Uploaded: {t.get('total_uploaded', 0)} (E1: {t.get('error1_uploaded', 0)}, E2: {t.get('error2_uploaded', 0)})")
    print(f"    Cost/uploaded question: Rs {t.get('cost_per_uploaded_q_inr', 0):.2f}")
    print(f"    Cost/error1 question:   Rs {t.get('cost_per_error1_q_inr', 0):.2f}")
    print(f"    Cost/all uploaded (E1+E2): Rs {t.get('cost_per_all_uploaded_inr', 0):.2f}")
