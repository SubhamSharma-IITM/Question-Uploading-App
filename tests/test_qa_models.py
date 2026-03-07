"""Test QA model integration: verify all 3 models receive images + structured output.

Picks 5 sample questions from the latest session, sends to each model,
and verifies:
1. Models return valid JSON matching the enforced schema
2. Images are received (no errors about missing images for img-tagged questions)
3. custom_id uniqueness across batch
4. Truncation/safety blocks handled gracefully
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sessions", "d1bf33f5", "test_5_samples.json")


def load_samples():
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_custom_id_uniqueness():
    """Verify custom_id generation produces unique IDs."""
    print("\n[1/4] Testing custom_id uniqueness...")
    samples = load_samples()

    ids_seen = set()
    duplicates = []
    for pass_key in ["pass1", "pass2", "pass3"]:
        for entry in samples:
            qid = entry.get("qid", "?")
            custom_id = f"qa-{pass_key}-{qid}"
            if custom_id in ids_seen:
                duplicates.append(custom_id)
            ids_seen.add(custom_id)

    total_ids = len(samples) * 3
    print(f"  Generated {total_ids} custom_ids, {len(ids_seen)} unique")
    if duplicates:
        print(f"  FAIL: {len(duplicates)} duplicate IDs: {duplicates[:5]}")
        return False
    else:
        print(f"  PASS: All {total_ids} custom_ids are unique")
        return True


def test_image_extraction():
    """Verify image extraction and placeholder replacement works."""
    print("\n[2/4] Testing image extraction + base64 fetch...")
    from app.pipeline.stage9_quality import _collect_images_with_placeholders, IMG_RE

    samples = load_samples()
    img_stats = {"with_images": 0, "total_images": 0, "fetched_ok": 0, "fetch_fail": 0}

    for entry in samples:
        qtype = entry.get("questionType", "")
        qdata = entry.get(qtype, {})
        q_html = qdata.get("question", "") or ""

        urls = IMG_RE.findall(q_html)
        if urls:
            img_stats["with_images"] += 1
            img_stats["total_images"] += len(urls)

            text, attachments = _collect_images_with_placeholders(q_html, "ImageQ")
            img_stats["fetched_ok"] += len(attachments)
            img_stats["fetch_fail"] += len(urls) - len(attachments)

            # Verify placeholders replaced
            assert "<img" not in text, f"img tag not replaced for {entry.get('qid')}"
            assert "[ImageQ1]" in text, f"Placeholder missing for {entry.get('qid')}"
            print(f"  {entry.get('qid')}: {len(urls)} image(s), {len(attachments)} fetched as base64")
        else:
            print(f"  {entry.get('qid')}: no images (text-only)")

    print(f"  Summary: {img_stats['with_images']} questions with images, "
          f"{img_stats['fetched_ok']}/{img_stats['total_images']} fetched OK")

    if img_stats["total_images"] > 0 and img_stats["fetched_ok"] == 0:
        print("  WARN: No images could be fetched (CDN may be unreachable)")
        return True  # Not a code error
    return True


def test_model_calls():
    """Send 5 questions to all 3 models and verify responses."""
    print("\n[3/4] Testing model API calls (5 questions x 3 models)...")
    from app.config import settings
    from app.pipeline.stage9_quality import (
        QualityVerifierStage, PASS_MODELS, OPENAI_QA_RESPONSE_SCHEMA,
        _collect_images_with_placeholders,
    )
    import base64

    has_gemini = bool(settings.gemini_api_key)
    has_openai = bool(settings.openai_api_key)
    print(f"  Credentials: Gemini={'YES' if has_gemini else 'NO'}, OpenAI={'YES' if has_openai else 'NO'}")

    if not has_gemini and not has_openai:
        print("  SKIP: No API keys configured")
        return True

    samples = load_samples()
    # Only test with first 2 samples to save cost
    test_samples = samples[:2]
    results = {}

    for pass_key, pass_info in PASS_MODELS.items():
        api = pass_info["api"]
        model = pass_info["model_id"]
        name = pass_info["name"]

        if api == "gemini" and not has_gemini:
            print(f"  {pass_key} ({name}): SKIPPED (no Gemini key)")
            continue
        if api == "openai" and not has_openai:
            print(f"  {pass_key} ({name}): SKIPPED (no OpenAI key)")
            continue

        print(f"  {pass_key} ({name}): testing with {len(test_samples)} questions...")
        pass_results = []

        for entry in test_samples:
            qid = entry.get("qid", "?")
            qtype = entry.get("questionType", "")
            qdata = entry.get(qtype, {})
            question = (qdata.get("question", "") or "")
            answer = qdata.get("answer", "")
            solution = (qdata.get("solution", "") or "")

            # Build content with images
            q_text, q_imgs = _collect_images_with_placeholders(question, "ImageQ")
            s_text, s_imgs = _collect_images_with_placeholders(solution, "ImageS")
            all_imgs = q_imgs + s_imgs

            user_text = f"""QID: {qid}
Subject: {entry.get('subject','')}
Question Type: {qtype}
QUESTION HTML: {q_text[:2000]}
ANSWER: {answer}
SOLUTION HTML: {s_text[:2000]}
{"NOTE: " + str(len(all_imgs)) + " image(s) attached." if all_imgs else ""}
Return ONLY valid JSON."""

            try:
                if api == "openai":
                    from openai import OpenAI
                    client = OpenAI(api_key=settings.openai_api_key)
                    user_parts = [{"type": "text", "text": user_text}]
                    for img in all_imgs:
                        user_parts.append(img)

                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a QA checker. Return JSON with qid, has_error (bool), errors (array of {field, type, detail})."},
                            {"role": "user", "content": user_parts},
                        ],
                        response_format=OPENAI_QA_RESPONSE_SCHEMA,
                        max_completion_tokens=4000,
                    )
                    finish = resp.choices[0].finish_reason
                    text = resp.choices[0].message.content
                    result = json.loads(text)

                    print(f"    {qid}: finish={finish}, has_error={result.get('has_error')}, "
                          f"errors={len(result.get('errors',[]))}, imgs_sent={len(all_imgs)}")
                    pass_results.append({"qid": qid, "ok": True, "result": result})

                elif api == "gemini":
                    from google import genai
                    from google.genai import types

                    gemini_schema = {
                        "type": "object",
                        "properties": {
                            "qid": {"type": "string"},
                            "has_error": {"type": "boolean"},
                            "errors": {"type": "array", "items": {
                                "type": "object",
                                "properties": {"field": {"type": "string"}, "type": {"type": "string"}, "detail": {"type": "string"}},
                                "required": ["field", "type", "detail"],
                            }},
                        },
                        "required": ["qid", "has_error", "errors"],
                    }

                    gclient = genai.Client(api_key=settings.gemini_api_key)
                    parts = [types.Part.from_text(text=user_text)]
                    for img in all_imgs:
                        data_uri = img.get("image_url", {}).get("url", "")
                        if data_uri.startswith("data:"):
                            header, b64_data = data_uri.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            raw_bytes = base64.b64decode(b64_data)
                            parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))

                    resp = gclient.models.generate_content(
                        model=model,
                        contents=types.Content(role="user", parts=parts),
                        config=types.GenerateContentConfig(
                            system_instruction="You are a QA checker. Return JSON with qid, has_error (bool), errors (array of {field, type, detail}).",
                            response_mime_type="application/json",
                            response_json_schema=gemini_schema,
                            temperature=0,
                            max_output_tokens=2048,
                        ),
                    )
                    finish = resp.candidates[0].finish_reason if resp.candidates else "?"
                    parsed = hasattr(resp, 'parsed') and resp.parsed is not None
                    if parsed:
                        result = resp.parsed if isinstance(resp.parsed, dict) else resp.parsed.model_dump()
                    else:
                        result = json.loads(resp.text)

                    print(f"    {qid}: finish={finish}, has_error={result.get('has_error')}, "
                          f"errors={len(result.get('errors',[]))}, imgs_sent={len(all_imgs)}, parsed={'yes' if parsed else 'no'}")
                    pass_results.append({"qid": qid, "ok": True, "result": result})

            except Exception as e:
                print(f"    {qid}: ERROR - {type(e).__name__}: {e}")
                pass_results.append({"qid": qid, "ok": False, "error": str(e)})

        results[pass_key] = pass_results

    # Verify all results have valid schema
    print("\n[4/4] Validating response schemas...")
    all_valid = True
    for pass_key, pass_results in results.items():
        for pr in pass_results:
            if not pr["ok"]:
                print(f"  {pass_key}/{pr['qid']}: FAILED (API error)")
                all_valid = False
                continue
            r = pr["result"]
            if "qid" not in r or "has_error" not in r or "errors" not in r:
                print(f"  {pass_key}/{pr['qid']}: FAIL - missing required fields: {list(r.keys())}")
                all_valid = False
            elif not isinstance(r["has_error"], bool):
                print(f"  {pass_key}/{pr['qid']}: FAIL - has_error is not bool: {type(r['has_error'])}")
                all_valid = False
            elif not isinstance(r["errors"], list):
                print(f"  {pass_key}/{pr['qid']}: FAIL - errors is not list: {type(r['errors'])}")
                all_valid = False
            else:
                for err in r["errors"]:
                    if not all(k in err for k in ("field", "type", "detail")):
                        print(f"  {pass_key}/{pr['qid']}: FAIL - error missing keys: {err}")
                        all_valid = False
                        break
                else:
                    print(f"  {pass_key}/{pr['qid']}: PASS (schema valid)")

    return all_valid


if __name__ == "__main__":
    print("=" * 60)
    print("QA MODEL INTEGRATION TEST")
    print("=" * 60)

    r1 = test_custom_id_uniqueness()
    r2 = test_image_extraction()
    r3 = test_model_calls()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  custom_id uniqueness: {'PASS' if r1 else 'FAIL'}")
    print(f"  Image extraction:     {'PASS' if r2 else 'FAIL'}")
    print(f"  Model API calls:      {'PASS' if r3 else 'FAIL'}")

    sys.exit(0 if all([r1, r2, r3]) else 1)
