# Bug Fixes Summary — Question Uploading App

**Date:** 2026-06-24
**Files Changed:** 6
**Total Bugs Fixed:** 8

---

## Bug A — Markdown Image Showing as Raw Text in Match Questions

**File:** `app/pipeline/stage4_metadata.py`
**Function:** `_parse_match_latex_tabular()`

**Symptom:** The intro text of a match question showed raw markdown like:
```
![](./images/27d48a96-b45c-4f2f-ac4b-f86a196d5a49-54_205_1647_2470_242.jpg) 48. For the reaction...
```
instead of rendering the image.

**Root Cause:** `_replace_md_images_with_img()` was called on cell contents but never on `intro_html` (the text before `\begin{tabular}`).

**Fix:** Call `_replace_md_images_with_img()` on `intro_raw` before returning it.

---

## Bug B — `\begin{tabular}{l} Reacts with` Leaking into List-II Cells

**File:** `app/pipeline/stage4_metadata.py`
**Function:** `_parse_match_latex_tabular()`

**Symptom:** List-II row (4) showed raw LaTeX:
```
\begin{tabular}{l} Reacts with
```
instead of the actual cell content.

**Root Cause:** MathPix OCR sometimes wraps cell content in a nested `\begin{tabular}...\end{tabular}` block. The cell splitter `re.split(r"(?<!\\)&", row)` split on `&` characters *inside* that nested block, producing wrong cell counts and leaving the `\begin{tabular}{l}` header as the start of the cell content.

**Fix:** Before splitting on `&`, collapse all nested `\begin{tabular}...\end{tabular}` blocks into placeholder tokens. After splitting, restore the collapsed content with inner `&` characters replaced by spaces so the text reads naturally (e.g. "Reacts with ketone followed by acetone to give hexan-2-ol").

---

## Bug C — S Row (Row 4) Missing/Blank in Match Questions

**File:** `app/pipeline/stage4_metadata.py`
**Function:** `_parse_match_latex_tabular()`

**Symptom:** The last row (S / row 4) of a match question was blank or missing entirely on the web.

**Root Cause:** The outer `\end{tabular}` search used `content.find()` which finds the *first* occurrence. When MathPix emitted a nested `\begin{tabular}...\end{tabular}` inside a cell, the first `\end{tabular}` belonged to the inner block — truncating the outer table before the last row.

**Fix:** Replace the naive `find()` with a nesting-depth-aware search that counts `\begin{tabular}` and `\end{tabular}` occurrences to find the correctly matching closing tag for the outermost table.

---

## Bug 1 — Vertex AI QID Extraction Failing for Multi-Part Requests (Stage 9)

**File:** `app/pipeline/stage9_quality.py`
**Function:** `_extract_qid_from_vertex_response()`

**Symptom:** Vertex AI batch responses were being matched to wrong questions, causing QA results to be assigned to incorrect QIDs.

**Root Cause:** The function only searched `parts[0]` for the `QID:` marker. If the request was built with an image part before the text part, `parts[0]` would be an image dict with no `text` key and the QID would be missed entirely, falling back to unreliable line-index mapping.

**Fix:** Iterate over all parts in order and search each one for the `QID:` pattern. Also added `AttributeError` to the except clause for robustness.

---

## Bug 2 — MathML Not Rendering in Stage 8 Duplicate Preview Pane

**File:** `app/templates/stage8.html`

**Symptom:** Mathematical expressions were invisible or showed as raw text in the Stage 8 duplicate review panes.

**Root Cause:** Three separate issues:
1. No MathJax renderer was loaded — a comment said "loads inside iframes" but iframes were never implemented.
2. The two-pane layout used `flex: 1 1 0` which allowed the left pane to shrink to zero width.
3. `MathJax.typesetPromise()` was never called after injecting HTML into the preview divs.

**Fix:**
1. Load MathJax 3 (`mml-chtml`) from CDN in `{% block head %}`.
2. Change layout to `flex: 0 0 calc(50% - 4px)` with explicit `width` so neither pane can collapse.
3. Call `MathJax.typesetPromise()` on all `.math-preview` divs after each HTML injection.

---

## Bug 3 — FAISS Absorption Mutex Timeout Crashing Stage 10

**Files:** `app/pipeline/stage10_upload.py`, `app/templates/completed.html`

**Symptom:** Stage 10 crashed entirely when the FAISS absorber mutex was held by another session, even though DB entries had been safely written.

**Root Cause:** `stage8_duplicates.py` raises `FAISSAbsorptionMutexTimeout` when the absorber mutex is held for over 10 minutes. Stage 10 only caught three other exception types — this one propagated uncaught and crashed the entire absorption step.

**Fix:**
- Import and catch `FAISSAbsorptionMutexTimeout` in Stage 10.
- Added a broad `except Exception` catch-all so no future absorber exception can crash Stage 10.
- Added `mutex_timeout` and `unexpected_error` status handling in `completed.html` UI with human-readable messages.

---

## Bug 4 — Negative `total_ready` Count in Dashboard (Stage 9)

**File:** `app/routes/stage9_quality.py`

**Symptom:** The dashboard showed negative numbers for `total_ready` (e.g. `-23`).

**Root Cause:** The old logic added `quarantined_by_error + quarantined_by_db` for `total_quarantined` then subtracted both from `total_processed`. But `quarantined_by_db` (all DB quarantine records) already included questions physically removed from metadata files — which were also absent from the `total_processed` denominator. This caused double-subtraction.

**Fix:** Split into non-overlapping sets:
- `quarantined_removed` = questions in DB quarantine AND physically removed from metadata
- `quarantined_in_meta` = questions still in metadata with `error >= 3`

`total_processed = len(all_questions) + quarantined_removed`
`total_quarantined = quarantined_removed + quarantined_in_meta`
`total_ready = max(0, total_processed - total_quarantined)`

Also added `max(0, ...)` floor as a safety net.

---

## Bug 5 — "Scan Ready" Button Re-Enabling After Navigating Back to Completed Page

**File:** `app/templates/completed.html`

**Symptom:** If a user navigated away from the Stage 10 completed page during upload and returned, the "Scan Ready" button appeared clickable again instead of showing the completed state.

**Root Cause:** The `DOMContentLoaded` resume path disabled the Upload button but never disabled the Scan Ready button.

**Fix:** Added one line in the resume path to disable the scan button and set its text to "Already Uploaded".

---

## Files Changed

| File | Bugs Fixed |
|------|-----------|
| `app/pipeline/stage4_metadata.py` | Bug A, Bug B, Bug C |
| `app/pipeline/stage9_quality.py` | Bug 1 |
| `app/templates/stage8.html` | Bug 2 |
| `app/pipeline/stage10_upload.py` | Bug 3 |
| `app/templates/completed.html` | Bug 3, Bug 5 |
| `app/routes/stage9_quality.py` | Bug 4 |
