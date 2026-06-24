"""Application configuration via Pydantic Settings + .env file.

Path resolution priority:
  1. Explicit env var (DATA_DIR, SESSIONS_DIR, etc.)
  2. Render persistent disk auto-detect (/data exists)
  3. Local fallback (repo data/ directory)
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import model_validator
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


def _default_data_dir() -> str:
    """Use /data (Render persistent disk) if it exists, else local data/ dir."""
    import platform
    render_disk = Path("/data")
    if platform.system() != "Windows" and render_disk.exists() and render_disk.is_dir():
        return str(render_disk)
    return str(Path(__file__).parent.parent / "data")


class Settings(BaseSettings):
    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Paths ──
    # Auto-detect Render persistent disk at /data, fall back to local data/
    data_dir: str = _default_data_dir()
    sessions_dir: str = ""  # Derived from data_dir if not set
    db_path: str = ""       # Derived from data_dir if not set
    db_index_dir: str = ""  # Derived from data_dir if not set

    # ── OCR Engine Selection ──
    ocr_engine: str = "mineru"

    # ── MathPIX ──
    mathpix_app_id: str = ""
    mathpix_app_key: str = ""

    # ── MinerU ──
    mineru_token: str = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI3NjMwMDEzOSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc4MjI5OTg0MSwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiNjdkNGE4YTItNTIxMS00ZjdmLTllZDItMDAzYTk2ZGMwODYwIiwiZW1haWwiOiJjaGFuZHJhcGFwcjE1MDFAZ21haWwuY29tIiwiZXhwIjoxNzkwMDc1ODQxfQ.yGZrJPZNNb0_z08v8fu1ArmJwFlBaoA8ADmEO2mAso3SoNHuq4ZymysEt7aKOiTAaMFu3HD0NGg4gWCBJkzsUQ"

    # ── Gemini ──
    gemini_api_key: str = ""

    # ── Vertex AI (Gemini batch fallback when AI Studio hits 429) ──
    vertex_ai_credentials: str = ""  # Path to service account JSON
    vertex_ai_project: str = "gen-lang-client-0276374344"
    vertex_ai_location: str = "global"
    vertex_ai_bucket: str = "acadza-qa-batch-0276374344"

    # ── OpenAI ──
    openai_api_key: str = ""

    # ── Anthropic ──
    anthropic_api_key: str = ""

    # ── Acadza API Auth (JWT) ──
    acadza_api_username: str = ""
    acadza_api_password: str = ""

    # ── Video Service API ──
    video_api_token: str = ""

    # ── Stage 10 behavior ──
    # When True, err=3 and err=4 questions from Stage 9 are auto-promoted to
    # err=2 and uploaded (with audit trail in the CSV) instead of being skipped.
    # Faculty review the err=2 queue and fix; nothing dies in quarantine.
    stage10_promote_err34: bool = True

    # ── Stage 11 nightly conversion ──
    # When True, an in-app asyncio sleeper task fires Stage 11 (err=2 -> err=1)
    # at STAGE11_TRIGGER_HOUR_IST:STAGE11_TRIGGER_MIN_IST IST every day.
    # Cohort: previous IST day's err=2 uploads + any failed_retry from history.
    # Floor: STAGE11_FLOOR_DATE (older err=2 NOT swept by nightly).
    stage11_nightly_enabled: bool = False
    stage11_floor_date: str = "2026-05-18"
    stage11_trigger_hour_ist: int = 0
    stage11_trigger_min_ist: int = 30
    stage11_dryrun: bool = False
    # Hard-stop the in-progress nightly cohort once IST wall-clock crosses this
    # time. Operators start arriving at ~09:30 IST; the nightly should be out
    # of the way by then to avoid contending for memory + Acadza backend.
    stage11_hardstop_hour_ist: int = 9
    stage11_hardstop_min_ist: int = 30
    # ThreadPoolExecutor width inside stage11_correction.process_batch. 5 was
    # the conservative default during canary; LLM-API-bound + FIX-I mutex
    # protects FAISS, so we can safely raise to push through bigger cohorts
    # before the 09:30 IST hard-stop. Override per machine via env.
    stage11_max_workers: int = 10

    # ── AWS S3 CDN ──
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_bucket: str = ""
    aws_s3_region: str = "ap-south-1"

    # ── Reference database (for duplicate check + Stage 10 absorption) ──
    # (derived from data_dir if empty)

    # ── Global FAISS index cache (shared across sessions) ──
    # (derived from data_dir if empty)

    # ── Tagging modules (Stage 7) ──
    # Bundled in repo: data/tagging_modules/{subject}_tagging_module/
    concept_tree_path: str = str(Path(__file__).parent.parent / "data" / "acadza_concept_tree.json")
    tagging_modules_dir: str = str(Path(__file__).parent.parent / "data" / "tagging_modules")

    # ── GitHub ──
    github_token: str = ""

    # ── Concurrency ──
    max_concurrent_tasks: int = 3

    @model_validator(mode="after")
    def _derive_paths_from_data_dir(self):
        """Fill in empty path fields from data_dir.

        This runs AFTER env vars override defaults, so data_dir is final.
        Any sub-path not explicitly set via env var gets derived from data_dir.
        """
        base = Path(self.data_dir)
        if not self.sessions_dir:
            self.sessions_dir = str(base / "sessions")
        if not self.db_path:
            self.db_path = str(base / "allque030326.json")
        if not self.db_index_dir:
            self.db_index_dir = str(base / "db_index_cache")
        return self

    model_config = {"env_file": ".env", "env_prefix": "", "extra": "ignore"}


settings = Settings()
