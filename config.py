# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # Project paths
    data_dir: Path = BASE_DIR / "data"
    raw_frames_dir: Path = BASE_DIR / "data" / "raw_frames"
    face_crops_dir: Path = BASE_DIR / "data" / "face_crops"
    results_dir: Path = BASE_DIR / "data" / "results"
    models_dir: Path = BASE_DIR / "models"
    logs_dir: Path = BASE_DIR / "logs"

    # Frame extraction settings
    max_frames: int = Field(default=120, description="Max frames to extract per video")
    min_frames: int = Field(default=20, description="Min frames — short clips still need enough")
    scene_threshold: float = Field(
        default=30.0,
        description="Pixel diff threshold for scene change (0-255). Lower = more sensitive."
    )
    target_fps: float = Field(
        default=4.0,
        description="Target frames per second for uniform sampling fallback"
    )
    frame_size: tuple[int, int] = Field(
        default=(640, 640),
        description="Resize frames to this size before saving"
    )

    # Download settings
    max_video_duration_seconds: int = Field(
        default=600, description="Reject videos longer than 10 minutes"
    )
    max_video_size_mb: int = Field(default=500, description="Reject files larger than 500MB")
    download_timeout_seconds: int = Field(default=120)

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"

    api_keys: list[str] = Field(
    default=["dev-key-change-in-production"],
    description=(
        "List of valid API keys. In production, set via environment variable "
        "DEEPFAKE_API_KEYS='[\"key1\",\"key2\"]'. "
        "The default key is intentionally weak — change it before deploying."
    ),
    )
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for the Celery task queue broker.",
    )
    celery_backend_url: str = Field(
        default="redis://localhost:6379/1",
        description="Redis URL for storing Celery task results.",
    )

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DEEPFAKE_")


settings = Settings()

# Ensure all directories exist at import time
for d in [
    settings.raw_frames_dir,
    settings.face_crops_dir,
    settings.results_dir,
    settings.models_dir,
    settings.logs_dir,
]:
    d.mkdir(parents=True, exist_ok=True)