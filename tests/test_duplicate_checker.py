"""Test duplicate checker: verify 4 exact DB matches + 3 semantic duplicates at 85% threshold.

Reads 4 real questions from the DB, then creates 3 semantically modified versions
to verify they're caught by the trigram similarity engine.
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DB_PATH = r"D:\ACADZA_WORK\acadza-pdf-generator\data\allque030326.json"

# ── Load 4 diverse sample questions from the DB ─────────────────
def load_samples(n=4):
    """Load n sample questions from the DB with usable question text."""
    samples = []
    seen_subjects = set()
    target_subjects = ["Physics", "Chemistry", "Mathematics", "Biology"]

    with open(DB_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(samples) >= n:
                break
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Get subject
            subject = entry.get("subject", "")
            if subject in seen_subjects:
                # Try to get one per subject for diversity
                if len(samples) < len(target_subjects) and subject in target_subjects:
                    continue

            # Extract question text
            for qtype in ["scq", "mcq", "integerQuestion"]:
                qdata = entry.get(qtype, {})
                if isinstance(qdata, dict) and qdata.get("question"):
                    qtext = qdata["question"]
                    if len(qtext) > 50:  # Enough text for meaningful comparison
                        qid = entry.get("qid", entry.get("_id", "?"))
                        if isinstance(qid, dict) and "$oid" in qid:
                            qid = qid["$oid"]
                        samples.append({
                            "qid": str(qid),
                            "subject": subject,
                            "questionType": qtype,
                            qtype: qdata,
                        })
                        seen_subjects.add(subject)
                        break

            if i > 50000 and len(samples) < 2:
                # If we're deep in the file and haven't found enough, relax constraints
                break

    return samples


def create_semantic_duplicates(samples):
    """Create 3 semantically modified versions of sample questions.

    These should be similar enough to trigger at 85% threshold.
    Modifications: synonym swaps, minor rewording, whitespace changes.
    """
    if len(samples) < 3:
        print(f"  Only {len(samples)} samples, need at least 3")
        return []

    dupes = []

    # Dupe 1: Change a few words (synonym swap) from sample[0]
    s0 = samples[0]
    qtype0 = s0["questionType"]
    q0 = s0[qtype0]["question"]
    # Replace common words with synonyms
    modified = q0.replace("the ", "a ").replace("is ", "equals ").replace("find ", "determine ")
    if modified == q0:
        # If no replacements worked, just add minor text
        modified = q0 + " "
    dupes.append({
        "qid": "SEMANTIC_DUPE_1",
        "subject": s0["subject"],
        "questionType": qtype0,
        qtype0: {**s0[qtype0], "question": modified},
    })

    # Dupe 2: Reorder whitespace and add trailing content from sample[1]
    s1 = samples[1]
    qtype1 = s1["questionType"]
    q1 = s1[qtype1]["question"]
    import re
    modified2 = re.sub(r'\s+', ' ', q1).strip() + "  "
    dupes.append({
        "qid": "SEMANTIC_DUPE_2",
        "subject": s1["subject"],
        "questionType": qtype1,
        qtype1: {**s1[qtype1], "question": modified2},
    })

    # Dupe 3: Minor character changes from sample[2]
    s2 = samples[2]
    qtype2 = s2["questionType"]
    q2 = s2[qtype2]["question"]
    modified3 = q2.replace("(A)", "(a)").replace("(B)", "(b)").replace("(C)", "(c)").replace("(D)", "(d)")
    if modified3 == q2:
        modified3 = q2[:len(q2)//2] + " " + q2[len(q2)//2:]
    dupes.append({
        "qid": "SEMANTIC_DUPE_3",
        "subject": s2["subject"],
        "questionType": qtype2,
        qtype2: {**s2[qtype2], "question": modified3},
    })

    return dupes


def test_similarity():
    """Test the trigram similarity function directly."""
    # Import from the stage8 module
    from app.pipeline.stage8_duplicates import _simple_similarity, _strip_html

    print("=" * 60)
    print("DUPLICATE CHECKER VERIFICATION TEST")
    print("=" * 60)

    # 1. Load samples
    print("\n[1/4] Loading 4 sample questions from DB...")
    samples = load_samples(4)
    print(f"  Loaded {len(samples)} samples:")
    for s in samples:
        qtype = s["questionType"]
        qtext = s[qtype].get("question", "")[:80]
        print(f"    {s['qid'][:24]:24s} | {s['subject']:12s} | {qtype:16s} | {qtext}...")

    if len(samples) < 3:
        print("  ERROR: Not enough samples loaded. DB may be malformed.")
        return False

    # 2. Self-match test: each sample against itself should be 1.0
    print("\n[2/4] Self-match test (each question vs itself, expect 1.0)...")
    all_self_pass = True
    for s in samples:
        qtype = s["questionType"]
        text = _strip_html(s[qtype].get("question", ""))
        sim = _simple_similarity(text, text)
        status = "PASS" if sim >= 0.99 else "FAIL"
        if status == "FAIL":
            all_self_pass = False
        print(f"    {s['qid'][:24]:24s} | similarity={sim:.3f} | {status}")

    # 3. Cross-match test: sample[0] vs sample[1] should be LOW (different questions)
    print("\n[3/4] Cross-match test (different questions, expect < 0.85)...")
    all_cross_pass = True
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            qtype_i = samples[i]["questionType"]
            qtype_j = samples[j]["questionType"]
            text_i = _strip_html(samples[i][qtype_i].get("question", ""))
            text_j = _strip_html(samples[j][qtype_j].get("question", ""))
            sim = _simple_similarity(text_i, text_j)
            status = "PASS" if sim < 0.85 else "WARN (false positive?)"
            if sim >= 0.85:
                all_cross_pass = False
            print(f"    {samples[i]['qid'][:12]} vs {samples[j]['qid'][:12]} | sim={sim:.3f} | {status}")

    # 4. Semantic duplicate test: modified versions should be caught at 85%
    print("\n[4/4] Semantic duplicate test (modified versions, expect >= 0.85)...")
    dupes = create_semantic_duplicates(samples)
    all_dupe_pass = True
    for idx, dupe in enumerate(dupes):
        original = samples[idx]
        qtype_o = original["questionType"]
        qtype_d = dupe["questionType"]
        text_o = _strip_html(original[qtype_o].get("question", ""))
        text_d = _strip_html(dupe[qtype_d].get("question", ""))
        sim = _simple_similarity(text_o, text_d)
        status = "PASS" if sim >= 0.85 else "FAIL (not caught!)"
        if sim < 0.85:
            all_dupe_pass = False
        print(f"    {original['qid'][:12]} vs {dupe['qid']:16s} | sim={sim:.3f} | {status}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Self-match (identity):     {'PASS' if all_self_pass else 'FAIL'}")
    print(f"  Cross-match (different):   {'PASS' if all_cross_pass else 'WARN'}")
    print(f"  Semantic dupes (85%):      {'PASS' if all_dupe_pass else 'FAIL'}")
    print(f"  DB path:                   {DB_PATH}")
    print(f"  DB size:                   {os.path.getsize(DB_PATH) / 1024 / 1024:.1f} MB")

    return all_self_pass and all_dupe_pass


if __name__ == "__main__":
    success = test_similarity()
    sys.exit(0 if success else 1)
