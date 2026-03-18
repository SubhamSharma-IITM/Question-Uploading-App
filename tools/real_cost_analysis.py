#!/usr/bin/env python3
"""Real cost analysis from actual batch token data + verified pricing."""
import json, sys, glob, os
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

INR = 85

# VERIFIED PRICING (March 2026, batch = 50% of standard)
PRICING = {
    'gemini_flash_batch': {'input': 0.25, 'output': 1.50},    # Vertex AI batch
    'gpt5mini_batch':     {'input': 0.125, 'output': 1.00},   # OpenAI batch
    'gpt51_batch':        {'input': 0.625, 'output': 5.00},   # OpenAI batch
}

# REAL token data
sessions = {
    '35e7d78e': {
        'pass2': {'in': 23354346, 'out': 115851, 'n': 220},
        'pass3': {'in': 23341712, 'out': 11266, 'n': 220},
        'tag':   {'in': 386175, 'out': 27903, 'n': 217},
    },
    '56589a45': {
        'pass2': {'in': 21913080, 'out': 88688, 'n': 194},
        'pass3': {'in': 21903108, 'out': 10147, 'n': 194},
        'tag':   {'in': 419219, 'out': 30359, 'n': 189},
    },
    '09477d7d': {
        'pass2': {'in': 84174692, 'out': 155506, 'n': 375},
        'pass3': {'in': 84157544, 'out': 24115, 'n': 375},
        'tag':   {'in': 816522, 'out': 59715, 'n': 367},
    },
    '73d46401': {
        'pass2': {'in': 27085323, 'out': 92117, 'n': 205},
        'pass3': {'in': 27077777, 'out': 10535, 'n': 205},
        'tag':   {'in': 468561, 'out': 34007, 'n': 209},
    },
    '95e55c16': {
        'pass2': {'in': 3680612, 'out': 44442, 'n': 65},
        'pass3': {'in': 3678143, 'out': 4425, 'n': 65},
        'tag':   {'in': 140230, 'out': 10029, 'n': 62},
    },
}

print("=" * 75)
print("REAL COST ANALYSIS (verified pricing, actual token counts)")
print("=" * 75)
print("Batch pricing (50% of standard):")
print("  Gemini 3 Flash: $0.25/1M in, $1.50/1M out")
print("  GPT-5-mini:     $0.125/1M in, $1.00/1M out")
print("  GPT-5.1:        $0.625/1M in, $5.00/1M out")

grand_total = 0
grand_qs = 0

for sid, ps in sessions.items():
    n = ps['pass2']['n']
    grand_qs += n
    p = PRICING

    # Pass 1 (Gemini) - estimate tokens same as pass2
    p1 = (ps['pass2']['in']/1e6 * p['gemini_flash_batch']['input'] +
          ps['pass2']['out']/1e6 * p['gemini_flash_batch']['output']) * INR
    # Pass 2 (GPT-5-mini)
    p2 = (ps['pass2']['in']/1e6 * p['gpt5mini_batch']['input'] +
          ps['pass2']['out']/1e6 * p['gpt5mini_batch']['output']) * INR
    # Pass 3 (GPT-5.1)
    p3 = (ps['pass3']['in']/1e6 * p['gpt51_batch']['input'] +
          ps['pass3']['out']/1e6 * p['gpt51_batch']['output']) * INR
    # Tagging (GPT-5-mini)
    tg = (ps['tag']['in']/1e6 * p['gpt5mini_batch']['input'] +
          ps['tag']['out']/1e6 * p['gpt5mini_batch']['output']) * INR

    total = p1 + p2 + p3 + tg
    grand_total += total

    print(f"\n{sid} ({n} Qs, avg {ps['pass2']['in']//n:,} prompt tok/q):")
    print(f"  Pass1 Gemini:  Rs {p1:>8.2f}  (Rs {p1/n:.2f}/q)")
    print(f"  Pass2 Mini:    Rs {p2:>8.2f}  (Rs {p2/n:.2f}/q)")
    print(f"  Pass3 5.1:     Rs {p3:>8.2f}  (Rs {p3/n:.2f}/q)")
    print(f"  Tagging:       Rs {tg:>8.2f}  (Rs {tg/n:.4f}/q)")
    print(f"  TOTAL:         Rs {total:>8.2f}  (Rs {total/n:.2f}/q)")

print(f"\n{'='*75}")
print(f"GRAND TOTAL: Rs {grand_total:.2f} for {grand_qs} questions")
print(f"AVG PER Q:   Rs {grand_total/grand_qs:.2f}")

# Cost breakdown by component
total_p1 = sum((ps['pass2']['in']/1e6 * PRICING['gemini_flash_batch']['input'] +
                ps['pass2']['out']/1e6 * PRICING['gemini_flash_batch']['output']) * INR
               for ps in sessions.values())
total_p2 = sum((ps['pass2']['in']/1e6 * PRICING['gpt5mini_batch']['input'] +
                ps['pass2']['out']/1e6 * PRICING['gpt5mini_batch']['output']) * INR
               for ps in sessions.values())
total_p3 = sum((ps['pass3']['in']/1e6 * PRICING['gpt51_batch']['input'] +
                ps['pass3']['out']/1e6 * PRICING['gpt51_batch']['output']) * INR
               for ps in sessions.values())
total_tg = sum((ps['tag']['in']/1e6 * PRICING['gpt5mini_batch']['input'] +
                ps['tag']['out']/1e6 * PRICING['gpt5mini_batch']['output']) * INR
               for ps in sessions.values())

print(f"\nCost breakdown:")
print(f"  Pass 1 (Gemini):  Rs {total_p1:>8.2f} ({total_p1/grand_total*100:.1f}%)")
print(f"  Pass 2 (Mini):    Rs {total_p2:>8.2f} ({total_p2/grand_total*100:.1f}%)")
print(f"  Pass 3 (5.1):     Rs {total_p3:>8.2f} ({total_p3/grand_total*100:.1f}%)")
print(f"  Tagging:          Rs {total_tg:>8.2f} ({total_tg/grand_total*100:.1f}%)")

# OPTIMIZED
print(f"\n{'='*75}")
print(f"OPTIMIZED (per-Q MMD + chapter-scoped tags)")
print(f"{'='*75}")

opt_total = 0
for sid, ps in sessions.items():
    n = ps['pass2']['n']
    curr_avg = ps['pass2']['in'] // n
    opt_prompt = 9700  # base 7K + chapter 300 + MMD 400 + user 2K
    ratio = opt_prompt / curr_avg

    p1 = (ps['pass2']['in']*ratio/1e6 * PRICING['gemini_flash_batch']['input'] +
          ps['pass2']['out']/1e6 * PRICING['gemini_flash_batch']['output']) * INR
    p2 = (ps['pass2']['in']*ratio/1e6 * PRICING['gpt5mini_batch']['input'] +
          ps['pass2']['out']/1e6 * PRICING['gpt5mini_batch']['output']) * INR
    p3 = (ps['pass3']['in']*ratio/1e6 * PRICING['gpt51_batch']['input'] +
          ps['pass3']['out']/1e6 * PRICING['gpt51_batch']['output']) * INR
    tg = (ps['tag']['in']/1e6 * PRICING['gpt5mini_batch']['input'] +
          ps['tag']['out']/1e6 * PRICING['gpt5mini_batch']['output']) * INR
    total = p1 + p2 + p3 + tg
    opt_total += total
    print(f"  {sid}: Rs {total:.2f} (Rs {total/n:.2f}/q)")

print(f"\n  OPTIMIZED TOTAL: Rs {opt_total:.2f}")
print(f"  OPTIMIZED AVG:   Rs {opt_total/grand_qs:.2f}/q")
print(f"  SAVINGS:         Rs {grand_total - opt_total:.2f} ({(1-opt_total/grand_total)*100:.0f}%)")

# E2E
mathpix = 0.004/4 * INR  # per question
gemini_s2 = (2000/1e6*0.50 + 500/1e6*3.00) * INR  # standard rate (realtime)
embed = (500/1e6 * 0.02) * INR

print(f"\n{'='*75}")
print(f"END-TO-END")
print(f"{'='*75}")
print(f"  MathPIX:       Rs {mathpix:.4f}/q")
print(f"  Gemini S2:     Rs {gemini_s2:.4f}/q")
print(f"  Embeddings:    Rs {embed:.6f}/q")
print(f"  S7+S9 current: Rs {grand_total/grand_qs:.2f}/q")
print(f"  S7+S9 optimzd: Rs {opt_total/grand_qs:.2f}/q")
e2e_c = mathpix + gemini_s2 + embed + grand_total/grand_qs
e2e_o = mathpix + gemini_s2 + embed + opt_total/grand_qs
print(f"  E2E CURRENT:   Rs {e2e_c:.2f}/q")
print(f"  E2E OPTIMIZED: Rs {e2e_o:.2f}/q")
print(f"\n  1059 Qs: Rs {e2e_c*1059:.0f} -> Rs {e2e_o*1059:.0f}")
