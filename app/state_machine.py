"""Session stage transitions + resume logic.

State machine:
  created -> stage1_upload -> stage2_mapping -> stage3_cleaning -> stage4_metadata
  -> stage5_watermark -> stage6_images -> stage7_batch -> stage8_duplicates
  -> stage9_quality -> completed
"""

from typing import Optional

# Ordered list of stages
STAGES = [
    "created",
    "stage1_upload",
    "stage2_mapping",
    "stage3_cleaning",
    "stage4_metadata",
    "stage5_watermark",
    "stage6_images",
    "stage7_batch",
    "stage8_duplicates",
    "stage9_quality",
    "completed",
]

# Display names for the UI
STAGE_LABELS = {
    "created": "New Session",
    "stage1_upload": "1. PDF Upload (MathPIX)",
    "stage2_mapping": "2. LLM Mapper",
    "stage3_cleaning": "3. Cleaner",
    "stage4_metadata": "4. Metadata V2",
    "stage5_watermark": "5. Watermark Removal",
    "stage6_images": "6. Image CDN Upload",
    "stage7_batch": "7. LaTeX + Tagging",
    "stage8_duplicates": "8. Duplicate Check",
    "stage9_quality": "9. Quality Verifier",
    "completed": "Complete",
}

# Short labels for sidebar
STAGE_SHORT = {
    "stage1_upload": "Upload",
    "stage2_mapping": "Map",
    "stage3_cleaning": "Clean",
    "stage4_metadata": "Meta",
    "stage5_watermark": "WM",
    "stage6_images": "Images",
    "stage7_batch": "Batch",
    "stage8_duplicates": "Dupes",
    "stage9_quality": "QA",
}


def stage_index(stage: str) -> int:
    """Get the index of a stage in the pipeline."""
    try:
        return STAGES.index(stage)
    except ValueError:
        return -1


def can_advance(current_stage: str, target_stage: str) -> bool:
    """Check if transition from current to target is valid (forward only)."""
    current_idx = stage_index(current_stage)
    target_idx = stage_index(target_stage)
    if current_idx < 0 or target_idx < 0:
        return False
    return target_idx == current_idx + 1


def next_stage(current_stage: str) -> Optional[str]:
    """Get the next stage after current, or None if completed."""
    idx = stage_index(current_stage)
    if idx < 0 or idx >= len(STAGES) - 1:
        return None
    return STAGES[idx + 1]


def is_stage_accessible(current_stage: str, target_stage: str) -> bool:
    """Check if a stage is accessible (current or any completed stage)."""
    current_idx = stage_index(current_stage)
    target_idx = stage_index(target_stage)
    if current_idx < 0 or target_idx < 0:
        return False
    return target_idx <= current_idx


def get_stage_status(current_stage: str, query_stage: str) -> str:
    """Get the visual status of a stage: completed, active, or locked."""
    current_idx = stage_index(current_stage)
    query_idx = stage_index(query_stage)
    if query_idx < current_idx:
        return "completed"
    elif query_idx == current_idx:
        return "active"
    else:
        return "locked"
