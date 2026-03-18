#!/usr/bin/env python3
"""Test optimized payloads across all 3 models (GPT-5-mini, GPT-5.1, Gemini)."""
import json, sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.experiment_qa_realtime import load_data, build_optimized_payload
from app.pipeline.stage9_quality import OPENAI_QA_RESPONSE_SCHEMA
from app.config import settings

mapper, metadata, mmd_lines, concept_tree, base_prompt = load_data()
meta_by_qid = {e['qid']: e for e in metadata}
mapper_by_qid = {e['qid']: e for e in mapper}
uploadable = [e for e in metadata if e.get('error', 2) < 3 and e['qid'] in mapper_by_qid]

# 11 diverse samples
step = max(1, len(uploadable) // 11)
samples = []
for i in range(11):
    idx = i * step
    if idx < len(uploadable):
        e = uploadable[idx]
        m = mapper_by_qid[e['qid']]
        samples.append((e, m))

payloads = [build_optimized_payload(e, m, mmd_lines, concept_tree, base_prompt) for e, m in samples]
print(f"Built {len(payloads)} payloads")

# --- OpenAI models ---
from openai import OpenAI
client = OpenAI(api_key=settings.openai_api_key)

for model_name, model_id, extra in [
    ('GPT-5-mini', 'gpt-5-mini', {'reasoning_effort': 'low'}),
    ('GPT-5.1', 'gpt-5.1', {}),
]:
    print(f"\n{'='*70}")
    print(f"{model_name} ({model_id})")
    print(f"{'='*70}")

    total_in = total_out = total_cached = errors_found = 0

    for i, p in enumerate(payloads, 1):
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=[
                    {'role': 'system', 'content': p['system']},
                    {'role': 'user', 'content': p['user_parts']},
                ],
                response_format=OPENAI_QA_RESPONSE_SCHEMA,
                max_completion_tokens=4096,
                **extra,
            )
            u = r.usage
            cached = u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0
            total_in += u.prompt_tokens
            total_out += u.completion_tokens
            total_cached += cached

            result = json.loads(r.choices[0].message.content)
            has_err = result.get('has_error', False)
            if has_err: errors_found += 1
            err_types = [e.get('type', '?') for e in result.get('errors', [])]

            print(f"  Q{i:2d} {p['qid']:20s}: in={u.prompt_tokens:>6,} cached={cached:>6,} "
                  f"out={u.completion_tokens:>4,} | {'ERR' if has_err else 'OK ':3s} "
                  f"{' '.join(err_types[:2])}")
        except Exception as e:
            print(f"  Q{i:2d} {p['qid']:20s}: ERROR {str(e)[:60]}")

        time.sleep(0.5)

    cache_pct = total_cached / total_in * 100 if total_in else 0
    print(f"\n  TOTAL: in={total_in:,} cached={total_cached:,} ({cache_pct:.0f}%) out={total_out:,}")
    print(f"  AVG/q: in={total_in//len(payloads):,} out={total_out//len(payloads):,}")
    print(f"  Errors found: {errors_found}/{len(payloads)}")

# --- Gemini Flash via google.genai ---
print(f"\n{'='*70}")
print(f"Gemini 3 Flash (gemini-3-flash-preview)")
print(f"{'='*70}")

try:
    from google import genai
    from google.genai import types

    gemini_client = genai.Client()

    total_in = total_out = errors_found = 0

    for i, p in enumerate(payloads, 1):
        try:
            # Build user text (Gemini doesn't support separate system message in same way)
            # Combine system + user as single prompt
            user_text = ""
            for part in p['user_parts']:
                if part.get('type') == 'text':
                    user_text = part['text']
                    break

            r = gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=f"{p['system']}\n\n{user_text}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=4096,
                ),
            )

            # Token usage
            um = r.usage_metadata
            p_tok = um.prompt_token_count if um else 0
            c_tok = um.candidates_token_count if um else 0
            total_in += p_tok
            total_out += c_tok

            result = json.loads(r.text)
            has_err = result.get('has_error', False)
            if has_err: errors_found += 1
            err_types = [e.get('type', '?') for e in result.get('errors', [])]

            print(f"  Q{i:2d} {p['qid']:20s}: in={p_tok:>6,} out={c_tok:>4,} | "
                  f"{'ERR' if has_err else 'OK ':3s} {' '.join(err_types[:2])}")
        except Exception as e:
            print(f"  Q{i:2d} {p['qid']:20s}: ERROR {str(e)[:80]}")

        time.sleep(1.0)  # Gemini rate limit

    print(f"\n  TOTAL: in={total_in:,} out={total_out:,}")
    print(f"  AVG/q: in={total_in//max(len(payloads),1):,} out={total_out//max(len(payloads),1):,}")
    print(f"  Errors found: {errors_found}/{len(payloads)}")

except ImportError:
    print("  google.genai not available -- skipping Gemini test")
except Exception as e:
    print(f"  Gemini test failed: {e}")

print(f"\n{'='*70}")
print("DONE")
