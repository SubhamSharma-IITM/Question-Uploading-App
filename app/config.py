"""Application configuration via Pydantic Settings + .env file."""

from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class Settings(BaseSettings):
    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Paths ──
    data_dir: str = str(Path(__file__).parent.parent / "data")
    sessions_dir: str = str(Path(__file__).parent.parent / "data" / "sessions")

    # ── MathPIX ──
    mathpix_app_id: str = ""
    mathpix_app_key: str = ""

    # ── Gemini ──
    gemini_api_key: str = ""

    # ── OpenAI ──
    openai_api_key: str = ""

    # ── Anthropic ──
    anthropic_api_key: str = ""

    # ── AWS S3 CDN ──
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_bucket: str = ""
    aws_s3_region: str = "ap-south-1"

    # ── Reference database (for duplicate check) ──
    db_path: str = r"D:\ACADZA_WORK\acadza-pdf-generator\data\allque030326.json"

    # ── Tagging modules ──
    concept_tree_path: str = r"D:\ACADZA_WORK\PCM_Batches\acadza_concept_tree.json"
    tagging_modules_dir: str = r"D:\ACADZA_WORK\PCM_Batches"

    # ── Concurrency ──
    max_concurrent_tasks: int = 3

    model_config = {"env_file": ".env", "env_prefix": ""}


settings = Settings()
