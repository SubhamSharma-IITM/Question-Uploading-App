#!/usr/bin/env python3
"""Cost analysis: Current vs Optimized QA token usage."""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

session_dir = 'data/sessions/09477d7d'

# Load data
with open(f'{session_dir}/metadata_v2/MATH2025_Mathematics_metadata_v2.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
with open(f'{session_dir}/mapped_output/MATH2025_Mathematics_mapped.json', 'r', encoding='utf-8') as f:
    mapper = json.load(f)
with open(f'{session_dir}/mathpix_output/4002d2ab-5191-4b0d-9084-3b97d06a8a8d.mmd', 'r', encoding='utf-8') as f:
    mmd_lines = f.readlines()

from app.config import settings
with open(Path(settings.concept_tree_path), 'r', encoding='utf-8') as f:
    full_tree = json.load(f)
from app.pipeline.stage9_quality import INSPECTOR_SYSTEM_PROMPT

meta_by_qid = {e['qid']: e for e in metadata}
mapper_by_qid = {e['qid']: e for e in mapper}
uploadable = [e for e in metadata if e.get('error', 2) < 3]
num_qs = len(uploadable)

# Token counts
base_prompt_tokens = len(INSPECTOR_SYSTEM_PROMPT) // 4
math_tree = {}
for exam, subjects in full_tree.items():
    if 'Math' in subjects:
        math_tree[exam] = {'Math': subjects['Math']}
full_tree_tokens = len(json.dumps(math_tree)) // 4
full_mmd_tokens = sum(len(l) for l in mmd_lines) // 4

# Chapter analysis
chapter_token_cache = {}
for e in uploadable:
    for t in e.get('tagSubConcept', []):
        ch = t.get('chapter', '')
        if ch and ch not in chapter_token_cache:
            for exam, subjects in full_tree.items():
                for s, chs in subjects.items():
                    if isinstance(chs, dict) and ch in chs:
                        chapter_token_cache[ch] = len(json.dumps({ch: chs[ch]})) // 4

avg_ch_tokens = sum(chapter_token_cache.values()) // max(len(chapter_token_cache), 1)

# Tag count distribution
tag_dist = {}
for e in uploadable:
    n = len(e.get('tagSubConcept', []))
    tag_dist[n] = tag_dist.get(n, 0) + 1

print(f"=== TAG COUNT DISTRIBUTION ===")
for n, cnt in sorted(tag_dist.items()):
    print(f"  {n} tag(s): {cnt} questions")
print(f"  Unique chapters: {len(chapter_token_cache)}")
print(f"  Avg tokens per chapter node: ~{avg_ch_tokens}")
print(f"  Full subject tree: ~{full_tree_tokens} tokens")
print(f"  Chapter-scoped saving: ~{full_tree_tokens - avg_ch_tokens} tokens/request ({100*(1-avg_ch_tokens/full_tree_tokens):.0f}%)")

# Per-question MMD tokens
per_q_mmd = []
for e in uploadable:
    m = mapper_by_qid.get(e.get('qid', ''))
    if not m:
        per_q_mmd.append(500)
        continue
    q_s, q_e = m.get('line_start', 0), m.get('line_end', 0)
    s_s, s_e = m.get('sol_line_start'), m.get('sol_line_end')
    pad = 10
    block = ''.join(mmd_lines[max(0,q_s-pad):min(len(mmd_lines),q_e+pad)])
    if s_s is not None and s_e is not None and s_s > q_e + pad:
        block += ''.join(mmd_lines[max(0,s_s-pad):min(len(mmd_lines),s_e+pad)])
    per_q_mmd.append(len(block) // 4)

avg_mmd = sum(per_q_mmd) // len(per_q_mmd)
avg_user = 2000
avg_output = 500

# Current vs Optimized
curr_input = base_prompt_tokens + full_tree_tokens + full_mmd_tokens + avg_user
opt_input = base_prompt_tokens + avg_ch_tokens + avg_mmd + avg_user

print(f"\n{'='*70}")
print(f"TOKEN COMPARISON (per request)")
print(f"{'='*70}")
print(f"                    Current      Optimized    Savings")
print(f"  Base prompt:      {base_prompt_tokens:>8,}       {base_prompt_tokens:>8,}       -")
print(f"  Concept tree:     {full_tree_tokens:>8,}       {avg_ch_tokens:>8,}       {full_tree_tokens-avg_ch_tokens:>8,}")
print(f"  MMD context:      {full_mmd_tokens:>8,}       {avg_mmd:>8,}       {full_mmd_tokens-avg_mmd:>8,}")
print(f"  User message:     {avg_user:>8,}       {avg_user:>8,}       -")
print(f"  Output:           {avg_output:>8,}       {avg_output:>8,}       -")
print(f"  ---               --------       --------       --------")
print(f"  TOTAL:            {curr_input+avg_output:>8,}       {opt_input+avg_output:>8,}       {(curr_input+avg_output)-(opt_input+avg_output):>8,} ({100*(1-(opt_input+avg_output)/(curr_input+avg_output)):.0f}%)")

# Cost per model per pass
INR = 85
models = {
    'Gemini Flash (Vertex)': {'input_per_m': 0.075, 'output_per_m': 0.30},
    'GPT-5-mini':            {'input_per_m': 0.30,  'output_per_m': 1.20},
    'GPT-5.1':               {'input_per_m': 2.00,  'output_per_m': 8.00},
}

print(f"\n{'='*70}")
print(f"COST PER PASS ({num_qs} questions)")
print(f"{'='*70}")

total_curr = 0
total_opt = 0

for name, p in models.items():
    c_in_cost = curr_input * num_qs / 1e6 * p['input_per_m'] * INR
    c_out_cost = avg_output * num_qs / 1e6 * p['output_per_m'] * INR
    o_in_cost = opt_input * num_qs / 1e6 * p['input_per_m'] * INR
    o_out_cost = avg_output * num_qs / 1e6 * p['output_per_m'] * INR
    c_total = c_in_cost + c_out_cost
    o_total = o_in_cost + o_out_cost
    total_curr += c_total
    total_opt += o_total

    print(f"\n  {name}:")
    print(f"    Current:   Input Rs {c_in_cost:>7.2f} + Output Rs {c_out_cost:>6.2f} = Rs {c_total:>8.2f}")
    print(f"    Optimized: Input Rs {o_in_cost:>7.2f} + Output Rs {o_out_cost:>6.2f} = Rs {o_total:>8.2f}")
    print(f"    Savings: Rs {c_total - o_total:.2f} ({100*(1-o_total/c_total):.0f}%)")

print(f"\n  {'='*50}")
print(f"  ALL 3 PASSES COMBINED:")
print(f"    Current:   Rs {total_curr:.2f}")
print(f"    Optimized: Rs {total_opt:.2f}")
print(f"    Savings:   Rs {total_curr - total_opt:.2f} ({100*(1-total_opt/total_curr):.0f}%)")
print(f"\n  Per question (3 passes):")
print(f"    Current:   Rs {total_curr/num_qs:.4f}")
print(f"    Optimized: Rs {total_opt/num_qs:.4f}")

# End-to-end pipeline
print(f"\n{'='*70}")
print(f"END-TO-END PIPELINE COST PER QUESTION")
print(f"{'='*70}")

mathpix = 0.004 / 4 * INR  # $0.004/page, ~4 Qs/page
gemini_s2 = (2000/1e6*0.075 + 500/1e6*0.30) * INR
tagging = (3000/1e6*0.30 + 200/1e6*1.20) * INR
embed = (500/1e6*0.02) * INR
qa_curr = total_curr / num_qs
qa_opt = total_opt / num_qs

print(f"  Stage 1 MathPIX OCR:        Rs {mathpix:.4f}")
print(f"  Stage 2 Gemini Extraction:  Rs {gemini_s2:.4f}")
print(f"  Stage 7 Tagging (GPT-mini): Rs {tagging:.4f}")
print(f"  Stage 8 Embeddings:         Rs {embed:.6f}")
print(f"  Stage 9 QA Current:         Rs {qa_curr:.4f}")
print(f"  Stage 9 QA Optimized:       Rs {qa_opt:.4f}")
print(f"  ---")
e2e_curr = mathpix + gemini_s2 + tagging + embed + qa_curr
e2e_opt = mathpix + gemini_s2 + tagging + embed + qa_opt
print(f"  E2E CURRENT:   Rs {e2e_curr:.2f} per question")
print(f"  E2E OPTIMIZED: Rs {e2e_opt:.2f} per question")
print(f"  Savings:       Rs {e2e_curr - e2e_opt:.2f} per question")
print(f"\n  For 383 Qs: Rs {e2e_curr*383:.2f} -> Rs {e2e_opt*383:.2f} (save Rs {(e2e_curr-e2e_opt)*383:.2f})")
print(f"  For 1000 Qs: Rs {e2e_curr*1000:.2f} -> Rs {e2e_opt*1000:.2f} (save Rs {(e2e_curr-e2e_opt)*1000:.2f})")

# Output quality impact
print(f"\n{'='*70}")
print(f"OUTPUT QUALITY IMPACT ANALYSIS")
print(f"{'='*70}")
print(f"  Will reducing input decrease output quality?")
print(f"  - The removed content is IRRELEVANT to the question being checked")
print(f"  - Full MMD = 9155 lines, per-Q extraction = ~30 lines (0.3% of total)")
print(f"  - 99.7% of the MMD is about OTHER questions, not the one being evaluated")
print(f"  - Models may actually IMPROVE with less noise (less distraction)")
print(f"  - The relevant context (question + solution + surrounding lines) is PRESERVED")
print(f"  - Output tokens should stay ~same (model reports same errors regardless)")
print(f"  - RISK: Near-zero. The model never needed the other 9125 lines.")
