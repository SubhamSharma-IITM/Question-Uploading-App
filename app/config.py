"""Application configuration via Pydantic Settings + .env file."""

from pathlib import Path
from pydantic_settings import BaseSettings
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
    sessions_dir: str = str(Path(_default_data_dir()) / "sessions")

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

    # ── AWS S3 CDN ──
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_bucket: str = ""
    aws_s3_region: str = "ap-south-1"

    # ── Reference database (for duplicate check + Stage 10 absorption) ──
    db_path: str = str(Path(_default_data_dir()) / "allque030326.json")

    # ── Global FAISS index cache (shared across sessions) ──
    db_index_dir: str = str(Path(_default_data_dir()) / "db_index_cache")

    # ── Tagging modules (Stage 7) ──
    # Bundled in repo: data/tagging_modules/{subject}_tagging_module/
    concept_tree_path: str = str(Path(__file__).parent.parent / "data" / "acadza_concept_tree.json")
    tagging_modules_dir: str = str(Path(__file__).parent.parent / "data" / "tagging_modules")

    # ── GitHub ──
    github_token: str = ""

    # ── Concurrency ──
    max_concurrent_tasks: int = 3

    model_config = {"env_file": ".env", "env_prefix": ""}


settings = Settings()
