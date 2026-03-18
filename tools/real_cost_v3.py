#!/usr/bin/env python3
"""Real cost v3: Actual tokens from batch results + optimized batch simulation.

Step 1: Read REAL token usage from all batch results (with cache breakdown)
Step 2: Build OPTIMIZED user messages using per-Q MMD + chapter-scoped tags
Step 3: Estimate optimized token counts using tiktoken
Step 4: Calculate costs for both methods with verified pricing
"""
import json, sys, glob, os, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

INR = 85

# Try tiktoken for accurate token counting
try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")  # closest available
    def count_tokens(text):
        return len(enc.encode(text))
    print("[OK] Using tiktoken for token counting")
except ImportError:
    def count_tokens(text):
        return len(text) // 4  # rough estimate
    print("[WARN] tiktoken not available, using char/4 estimate")

# Verified batch pricing (March 2026, 50% of standard)
# Cached prompt tokens: additional 50% off batch price
PRICING = {
    'gpt5mini': {'uncached': 0.125, 'cached': 0.0625, 'output': 1.00},
    'gpt51':    {'uncached': 0.625, 'cached': 0.3125, 'output': 5.00},
    'gemini':   {'uncached': 0.25,  'cached': 0.125,  'output': 1.50},
}

def cost_usd(pricing_key, uncached_tok, cached_tok, output_tok):
    p = PRICING[pricing_key]
    return (uncached_tok/1e6 * p['uncached'] +
            cached_tok/1e6 * p['cached'] +
            output_tok/1e6 * p['output'])

# ============================================================
# STEP 1: Read REAL token usage from batch results
# ============================================================
print("\n" + "=" * 80)
print("STEP 1: REAL TOKEN USAGE FROM BATCH RESULTS")
print("=" * 80)

real_data = {}
for sid in ['35e7d78e', '56589a45', '09477d7d', '73d46401', '95e55c16']:
    qa_dir = f'data/sessions/{sid}/quality_check'
    batch_dir = f'data/sessions/{sid}/batch_results'
    real_data[sid] = {}

    for pf in sorted(glob.glob(f'{qa_dir}/*_batch_results.jsonl')):
        fname = os.path.basename(pf)
        if os.path.getsize(pf) < 10:
            continue
        pk = fname.split('_')[0]
        tot_prompt = tot_cached = tot_comp = tot_reason = 0
        n = 0
        with open(pf, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line)
                    u = r.get('response', {}).get('body', {}).get('usage', {})
                    if u:
                        tot_prompt += u.get('prompt_tokens', 0)
                        tot_cached += u.get('prompt_tokens_details', {}).get('cached_tokens', 0)
                        tot_comp += u.get('completion_tokens', 0)
                        tot_reason += u.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
                        n += 1
                except: continue
        if n:
            real_data[sid][pk] = {
                'n': n, 'prompt': tot_prompt, 'cached': tot_cached,
                'uncached': tot_prompt - tot_cached, 'comp': tot_comp, 'reason': tot_reason,
            }

    for rf in glob.glob(f'{batch_dir}/*tagging_results.jsonl'):
        tot_prompt = tot_cached = tot_comp = 0
        n = 0
        with open(rf, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line)
                    u = r.get('response', {}).get('body', {}).get('usage', {})
                    if u:
                        tot_prompt += u.get('prompt_tokens', 0)
                        tot_cached += u.get('prompt_tokens_details', {}).get('cached_tokens', 0)
                        tot_comp += u.get('completion_tokens', 0)
                        n += 1
                except: continue
        if n:
            real_data[sid]['tagging'] = {
                'n': n, 'prompt': tot_prompt, 'cached': tot_cached,
                'uncached': tot_prompt - tot_cached, 'comp': tot_comp,
            }

# Print real data summary
grand_real_cost = 0
grand_qs = 0

for sid, passes in real_data.items():
    if 'pass2' not in passes:
        continue
    n = passes['pass2']['n']
    grand_qs += n
    s_cost = 0

    print(f"\n  {sid} ({n} Qs):")
    for pk, pricing_key, label in [
        ('pass2', 'gpt5mini', 'GPT-5-mini'),
        ('pass3', 'gpt51', 'GPT-5.1'),
        ('tagging', 'gpt5mini', 'Tagging'),
    ]:
        if pk not in passes: continue
        d = passes[pk]
        c = cost_usd(pricing_key, d['uncached'], d['cached'], d['comp']) * INR
        s_cost += c
        cache_pct = d['cached']/d['prompt']*100 if d['prompt'] else 0
        print(f"    {label:12s}: prompt {d['prompt']:>12,} ({cache_pct:.0f}% cached) | "
              f"comp {d['comp']:>8,} | Rs {c:>7.2f} (Rs {c/d['n']:.2f}/q)")

    # Pass 1 Gemini estimate (same prompt as pass2, Gemini pricing)
    if 'pass2' in passes:
        d2 = passes['pass2']
        p1 = cost_usd('gemini', d2['uncached'], d2['cached'], d2['comp']) * INR
        s_cost += p1
        print(f"    {'Gemini(est)':12s}: (estimated from pass2 volume) | Rs {p1:>7.2f} (Rs {p1/n:.2f}/q)")

    grand_real_cost += s_cost
    print(f"    {'SESSION':12s}: Rs {s_cost:>7.2f} (Rs {s_cost/n:.2f}/q)")

print(f"\n  REAL TOTAL: Rs {grand_real_cost:.2f} for {grand_qs} Qs (Rs {grand_real_cost/grand_qs:.2f}/q)")

# ============================================================
# STEP 2: Build OPTIMIZED prompts and estimate tokens
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: OPTIMIZED TOKEN ESTIMATION")
print("=" * 80)

from app.pipeline.stage9_quality import INSPECTOR_SYSTEM_PROMPT
from app.config import settings

# Load concept tree
with open(Path(settings.concept_tree_path), 'r', encoding='utf-8') as f:
    full_tree = json.load(f)

base_system_tokens = count_tokens(INSPECTOR_SYSTEM_PROMPT)
print(f"\n  Base system prompt: {base_system_tokens} tokens")

# For each session, simulate optimized batches
opt_data = {}

for sid in ['35e7d78e', '56589a45', '09477d7d', '73d46401', '95e55c16']:
    session_dir = Path(f'data/sessions/{sid}')
    mapper_files = list(session_dir.glob('mapped_output/*_mapped.json'))
    meta_files = list(session_dir.glob('metadata_v2/*_metadata_v2.json'))
    mmd_files = list(session_dir.glob('mathpix_output/*.mmd'))

    if not mapper_files or not meta_files or not mmd_files:
        continue

    with open(mapper_files[0], 'r', encoding='utf-8') as f:
        mapper = json.load(f)
    with open(meta_files[0], 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    mmd_lines = mmd_files[0].read_text(encoding='utf-8').splitlines()

    meta_by_qid = {e['qid']: e for e in metadata}
    mapper_by_qid = {e['qid']: e for e in mapper}

    # Only uploadable questions (error < 3)
    uploadable = [e for e in metadata if e.get('error', 2) < 3]

    per_q_tokens = []
    for entry in uploadable:
        qid = entry.get('qid', '')
        qtype = entry.get('questionType', 'scq')
        qdata = entry.get(qtype, {})

        # User message: question + solution + answer
        user_content = f"QID: {qid}\nType: {qtype}\n"
        user_content += f"Question: {qdata.get('question', '')[:3000]}\n"
        user_content += f"Solution: {qdata.get('solution', '')[:3000]}\n"
        user_content += f"Answer: {qdata.get('answer', entry.get('answer', ''))}\n"

        # Per-Q MMD context
        m = mapper_by_qid.get(qid)
        if m:
            q_s, q_e = m.get('line_start', 0), m.get('line_end', 0)
            s_s, s_e = m.get('sol_line_start'), m.get('sol_line_end')
            pad = 10
            mmd_ctx = '\n'.join(mmd_lines[max(0,q_s-pad):min(len(mmd_lines),q_e+pad)])
            if s_s is not None and s_e is not None and int(s_s) > q_e + pad:
                mmd_ctx += '\n' + '\n'.join(mmd_lines[max(0,int(s_s)-pad):min(len(mmd_lines),int(s_e)+pad)])
        else:
            mmd_ctx = ''

        # Chapter-scoped concept tree
        tags = entry.get('tagSubConcept', [])
        chapters = set(t.get('chapter', '') for t in tags if t.get('chapter'))
        ch_tree = {}
        for exam, subjects in full_tree.items():
            for subj, chs in subjects.items():
                if isinstance(chs, dict):
                    for ch, concepts in chs.items():
                        if ch in chapters:
                            if exam not in ch_tree: ch_tree[exam] = {}
                            if subj not in ch_tree[exam]: ch_tree[exam][subj] = {}
                            ch_tree[exam][subj][ch] = concepts

        ch_tree_str = json.dumps(ch_tree, indent=1) if ch_tree else ''

        # Total per-Q input = base system + chapter tree + MMD + user message
        total_input = (base_system_tokens +
                       count_tokens(ch_tree_str) +
                       count_tokens(mmd_ctx) +
                       count_tokens(user_content))
        per_q_tokens.append(total_input)

    avg_tok = sum(per_q_tokens) // len(per_q_tokens) if per_q_tokens else 0
    min_tok = min(per_q_tokens) if per_q_tokens else 0
    max_tok = max(per_q_tokens) if per_q_tokens else 0

    opt_data[sid] = {
        'n': len(uploadable),
        'avg_input': avg_tok,
        'min_input': min_tok,
        'max_input': max_tok,
        'total_input': sum(per_q_tokens),
        'est_output': 500 * len(uploadable),  # avg 500 output tokens
    }

    print(f"\n  {sid} ({len(uploadable)} Qs):")
    print(f"    Input: avg {avg_tok:,}, min {min_tok:,}, max {max_tok:,}")
    print(f"    Total input: {sum(per_q_tokens):,} tokens")

# ============================================================
# STEP 3: Cost comparison
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: COST COMPARISON")
print("=" * 80)

grand_opt_cost = 0

for sid, od in opt_data.items():
    n = od['n']
    total_in = od['total_input']
    total_out = od['est_output']

    # Optimized: system prompt is constant -> will be cached after first request
    # Estimate ~80% cache rate for system prompt (same as current)
    cached_in = int(total_in * 0.80)
    uncached_in = total_in - cached_in

    # 3 passes: Gemini + GPT-mini + GPT-5.1
    c_gemini = cost_usd('gemini', uncached_in, cached_in, total_out) * INR
    c_mini = cost_usd('gpt5mini', uncached_in, cached_in, total_out) * INR
    c_51 = cost_usd('gpt51', uncached_in, cached_in, total_out) * INR
    c_tag = 0
    if sid in real_data and 'tagging' in real_data[sid]:
        d = real_data[sid]['tagging']
        c_tag = cost_usd('gpt5mini', d['uncached'], d['cached'], d['comp']) * INR

    s_total = c_gemini + c_mini + c_51 + c_tag
    grand_opt_cost += s_total

    # Get current cost for comparison
    curr_cost = 0
    if sid in real_data:
        for pk, pricing_key in [('pass2', 'gpt5mini'), ('pass3', 'gpt51'), ('tagging', 'gpt5mini')]:
            if pk in real_data[sid]:
                d = real_data[sid][pk]
                curr_cost += cost_usd(pricing_key, d['uncached'], d['cached'], d['comp']) * INR
        if 'pass2' in real_data[sid]:
            d2 = real_data[sid]['pass2']
            curr_cost += cost_usd('gemini', d2['uncached'], d2['cached'], d2['comp']) * INR

    pct = (1 - s_total/curr_cost) * 100 if curr_cost > 0 else 0

    print(f"\n  {sid} ({n} Qs):")
    print(f"    Current:   Rs {curr_cost:>8.2f} (Rs {curr_cost/n:.2f}/q)")
    print(f"    Optimized: Rs {s_total:>8.2f} (Rs {s_total/n:.2f}/q)")
    print(f"      Gemini:  Rs {c_gemini:.2f} | Mini: Rs {c_mini:.2f} | 5.1: Rs {c_51:.2f} | Tag: Rs {c_tag:.2f}")
    print(f"    Savings:   {pct:.0f}%")

print(f"\n  {'='*60}")
print(f"  CURRENT TOTAL:   Rs {grand_real_cost:.2f} (Rs {grand_real_cost/grand_qs:.2f}/q)")
print(f"  OPTIMIZED TOTAL: Rs {grand_opt_cost:.2f} (Rs {grand_opt_cost/sum(od['n'] for od in opt_data.values()):.2f}/q)")
print(f"  SAVINGS:         Rs {grand_real_cost - grand_opt_cost:.2f}")

# E2E
print(f"\n  {'='*60}")
print(f"  END-TO-END PER QUESTION:")
mathpix = 0.004/4 * INR
gemini_s2 = (2000/1e6*0.50 + 500/1e6*3.00) * INR
total_opt_qs = sum(od['n'] for od in opt_data.values())
print(f"    MathPIX:       Rs {mathpix:.4f}")
print(f"    Gemini S2:     Rs {gemini_s2:.4f}")
print(f"    S7+S9 current: Rs {grand_real_cost/grand_qs:.2f}")
print(f"    S7+S9 optimzd: Rs {grand_opt_cost/total_opt_qs:.2f}")
e2e_c = mathpix + gemini_s2 + grand_real_cost/grand_qs
e2e_o = mathpix + gemini_s2 + grand_opt_cost/total_opt_qs
print(f"    E2E CURRENT:   Rs {e2e_c:.2f}/q")
print(f"    E2E OPTIMIZED: Rs {e2e_o:.2f}/q")
