#!/usr/bin/env python3
"""Real cost v2: Uses actual token data with cached token distinction."""
import json, sys, glob, os
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

INR = 85

# VERIFIED PRICING (March 2026)
# Batch = 50% of standard
# Cached prompt tokens = additional 50% off (i.e., 25% of standard)
# GPT-5-mini: standard $0.25/$2.00, batch $0.125/$1.00, cached batch $0.0625
# GPT-5.1:    standard $1.25/$10.00, batch $0.625/$5.00, cached batch $0.3125
# Gemini Flash Vertex batch: $0.25/$1.50 (no cached token discount documented)

PRICING = {
    'gpt5mini_batch': {
        'input_uncached': 0.125,   # per 1M
        'input_cached': 0.0625,    # 50% of batch = 25% of standard
        'output': 1.00,
    },
    'gpt51_batch': {
        'input_uncached': 0.625,
        'input_cached': 0.3125,    # 50% of batch
        'output': 5.00,
    },
    'gemini_batch': {
        'input_uncached': 0.25,
        'input_cached': 0.125,     # estimate
        'output': 1.50,
    },
}

sessions_data = {}

for sid in ['35e7d78e', '56589a45', '09477d7d', '73d46401', '95e55c16']:
    qa_dir = f'data/sessions/{sid}/quality_check'
    batch_dir = f'data/sessions/{sid}/batch_results'
    sessions_data[sid] = {}

    # QA passes
    for pf in sorted(glob.glob(f'{qa_dir}/*_batch_results.jsonl')):
        fname = os.path.basename(pf)
        if os.path.getsize(pf) < 10:
            continue

        total_prompt = 0
        total_cached = 0
        total_completion = 0
        total_reasoning = 0
        count = 0

        with open(pf, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    u = r.get('response', {}).get('body', {}).get('usage', {})
                    if u:
                        total_prompt += u.get('prompt_tokens', 0)
                        cached = u.get('prompt_tokens_details', {}).get('cached_tokens', 0)
                        total_cached += cached
                        total_completion += u.get('completion_tokens', 0)
                        reasoning = u.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
                        total_reasoning += reasoning
                        count += 1
                except:
                    continue

        if count > 0:
            pass_key = fname.split('_')[0]  # pass1, pass2, pass3
            sessions_data[sid][pass_key] = {
                'n': count,
                'prompt': total_prompt,
                'cached': total_cached,
                'uncached': total_prompt - total_cached,
                'completion': total_completion,
                'reasoning': total_reasoning,
            }

    # Tagging
    for rf in glob.glob(f'{batch_dir}/*tagging_results.jsonl'):
        total_prompt = 0
        total_cached = 0
        total_completion = 0
        count = 0
        with open(rf, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    u = r.get('response', {}).get('body', {}).get('usage', {})
                    if u:
                        total_prompt += u.get('prompt_tokens', 0)
                        total_cached += u.get('prompt_tokens_details', {}).get('cached_tokens', 0)
                        total_completion += u.get('completion_tokens', 0)
                        count += 1
                except:
                    continue
        if count > 0:
            sessions_data[sid]['tagging'] = {
                'n': count, 'prompt': total_prompt, 'cached': total_cached,
                'uncached': total_prompt - total_cached, 'completion': total_completion,
            }

# Display
print("=" * 80)
print("REAL TOKEN DATA WITH CACHE BREAKDOWN")
print("=" * 80)

grand_cost = 0
grand_qs = 0

for sid, passes in sessions_data.items():
    if 'pass2' not in passes:
        continue
    n = passes['pass2']['n']
    grand_qs += n

    print(f"\n{sid} ({n} questions):")

    session_cost = 0

    for pk in ['pass2', 'pass3', 'tagging']:
        if pk not in passes:
            continue
        d = passes[pk]

        if pk == 'pass2' or pk == 'tagging':
            p = PRICING['gpt5mini_batch']
        elif pk == 'pass3':
            p = PRICING['gpt51_batch']

        cost_usd = (d['uncached']/1e6 * p['input_uncached'] +
                    d['cached']/1e6 * p['input_cached'] +
                    d['completion']/1e6 * p['output'])
        cost_inr = cost_usd * INR
        session_cost += cost_inr

        cache_pct = d['cached']/d['prompt']*100 if d['prompt'] > 0 else 0

        model = 'GPT-5-mini' if pk in ('pass2', 'tagging') else 'GPT-5.1'
        print(f"  {pk:8s} ({model:10s}): {d['n']:>4d} Qs | "
              f"prompt {d['prompt']:>12,} (cached {cache_pct:.0f}%) | "
              f"completion {d['completion']:>8,} | "
              f"Rs {cost_inr:>8.2f} (Rs {cost_inr/d['n']:.2f}/q)")

    # Estimate Pass 1 (Gemini) - same prompt volume as pass2
    if 'pass2' in passes:
        d2 = passes['pass2']
        p_g = PRICING['gemini_batch']
        # Gemini may not cache same way, use uncached rate for all
        p1_cost = (d2['prompt']/1e6 * p_g['input_uncached'] +
                   d2['completion']/1e6 * p_g['output']) * INR
        session_cost += p1_cost
        print(f"  {'pass1':8s} ({'Gemini':10s}): {n:>4d} Qs | "
              f"(estimated from pass2 volume) | "
              f"Rs {p1_cost:>8.2f} (Rs {p1_cost/n:.2f}/q)")

    grand_cost += session_cost
    print(f"  {'TOTAL':8s}: Rs {session_cost:>8.2f} (Rs {session_cost/n:.2f}/q)")

print(f"\n{'='*80}")
print(f"GRAND TOTAL: Rs {grand_cost:.2f} for {grand_qs} questions")
print(f"AVG PER Q:   Rs {grand_cost/grand_qs:.2f}")

# Cost breakdown
print(f"\nBreakdown by pass:")
for pk_name, pk_key, model in [('Pass 2', 'pass2', 'gpt5mini_batch'),
                                 ('Pass 3', 'pass3', 'gpt51_batch'),
                                 ('Tagging', 'tagging', 'gpt5mini_batch')]:
    total = 0
    for sid, passes in sessions_data.items():
        if pk_key not in passes:
            continue
        d = passes[pk_key]
        p = PRICING[model]
        cost = (d['uncached']/1e6 * p['input_uncached'] +
                d['cached']/1e6 * p['input_cached'] +
                d['completion']/1e6 * p['output']) * INR
        total += cost
    print(f"  {pk_name}: Rs {total:.2f}")

# Gemini estimate
gemini_total = 0
for sid, passes in sessions_data.items():
    if 'pass2' in passes:
        d2 = passes['pass2']
        p_g = PRICING['gemini_batch']
        gemini_total += (d2['prompt']/1e6 * p_g['input_uncached'] +
                         d2['completion']/1e6 * p_g['output']) * INR
print(f"  Pass 1 (Gemini est): Rs {gemini_total:.2f}")

# E2E
mathpix = grand_qs * 0.004/4 * INR
gemini_s2 = grand_qs * (2000/1e6*0.50 + 500/1e6*3.00) * INR

print(f"\n{'='*80}")
print(f"END-TO-END")
print(f"{'='*80}")
print(f"  MathPIX:       Rs {mathpix:.2f} (Rs {mathpix/grand_qs:.4f}/q)")
print(f"  Gemini S2:     Rs {gemini_s2:.2f} (Rs {gemini_s2/grand_qs:.4f}/q)")
print(f"  Stage 7+9:     Rs {grand_cost:.2f} (Rs {grand_cost/grand_qs:.2f}/q)")
print(f"  E2E TOTAL:     Rs {mathpix + gemini_s2 + grand_cost:.2f}")
print(f"  E2E PER Q:     Rs {(mathpix + gemini_s2 + grand_cost)/grand_qs:.2f}")
