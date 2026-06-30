"""Run the pipeline stages 5‑10 on a specific session and output final HTML.

Usage:
    python scripts/generate_html.py <session_id> <metadata_file>
"""
import sys
from pathlib import Path
import sys
# Add project root to sys.path to ensure 'app' package can be imported
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from app.pipeline.stage5_watermark import WatermarkStage
from app.pipeline.stage6_images import ImageCDNStage
from app.pipeline.stage7_latex_tag import LatexTagStage
from app.pipeline.stage8_duplicates import DuplicateCheckStage
from app.pipeline.stage9_quality import QualityVerifierStage
from app.pipeline.stage10_upload import UploadStage
from app.utils import _safe_print
import json

def run(session_id: str, metadata_path: Path):
    # Session directory (existing from earlier stages)
    from app.config import settings
    session_dir = Path(settings.sessions_dir) / session_id

    # Stage 5 – Watermark removal (ensure consistency)
    WatermarkStage(session_id, session_dir).execute()

    # Stage 6 – Image CDN upload (mocked to avoid network calls)
    from unittest.mock import patch
    with patch("app.pipeline.stage6_images.upload_with_retry", return_value=(True, "https://mock-cdn.example.com/img.png", None)):
        ImageCDNStage(session_id, session_dir).upload_images()

    # Stage 7 – LaTeX & Tagging (use existing batch files, mock download)
    stage7 = LatexTagStage(session_id, session_dir)
    stage7.convert_all_mathml()
    stage7.build_batch()
    stage7.download_results()  # reads local JSONL if present

    # Stage 8 – Duplicate checking (local trigram)
    DuplicateCheckStage(session_id, session_dir).execute(threshold=0.85)

    # Stage 9 – Quality verification (use existing QA batch files, mock download)
    stage9 = QualityVerifierStage(session_id, session_dir)
    stage9.scan()
    stage9.build_qa_batch()
    stage9.download_pass_results()
    stage9.merge_votes()

    # Stage 10 – DB upload (mocked)
    stage10 = UploadStage(session_id, session_dir)
    stage10.scan()
    with patch.object(stage10, "_post_question", return_value=(True, f"mock_id_{session_id}")):
        stage10.upload(dry_run=False)

    # Collect final HTML from metadata_v2 entries
    meta_dir = session_dir / "metadata_v2"
    all_meta = []
    for json_file in sorted(meta_dir.glob("*_metadata_v2.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
            all_meta.extend(entries)
    html_parts = []
    for entry in all_meta:
        qtype = entry.get('questionType')
        # Map the question type to its corresponding field name in the JSON
        field_name = qtype if qtype else ''
        content = entry.get(field_name, {})
        # For passage questions, include the passage and each sub-question's text
        if qtype == 'passageQuestion':
            passage = content.get('passage', '')
            html_parts.append(passage)
            for q in content.get('questions', []):
                # Each question dict may have 'scq', 'mcq', etc.
                for sub_key, sub_val in q.items():
                    html_parts.append(sub_val.get('ques', ''))
                    html_parts.append(sub_val.get('solution', ''))
        else:
            html_parts.append(content.get('question', ''))
            html_parts.append(content.get('solution', ''))
    final_html = "\n".join(html_parts)
    _safe_print("--- FINAL HTML START ---")
    _safe_print(final_html)
    _safe_print("--- FINAL HTML END ---")
    return final_html

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_html.py <session_id> <metadata_v2_json_path>")
        sys.exit(1)
    sess_id = sys.argv[1]
    meta_fp = Path(sys.argv[2])
    run(sess_id, meta_fp)
