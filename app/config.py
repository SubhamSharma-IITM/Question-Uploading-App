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
    render_disk = Path("/data")
    if render_disk.exists() and render_disk.is_dir():
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

    # ── MathPIX ──
    mathpix_app_id: str = ""
    mathpix_app_key: str = ""

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
