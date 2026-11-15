"""
Web Application Configuration for MineGuard AI.

Configures database location, camera settings, AI pipeline parameters,
checkpoint inspection thresholds, and Flask runtime options.
"""

import os
from pathlib import Path
import utils.config as base_config

PROJECT_ROOT = Path(__file__).resolve().parent


class AppConfig:
    """Central Flask and System Configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "mineguard-ai-secret-key-2026")

    # Database
    DATABASE_DIR = PROJECT_ROOT / "database"
    DATABASE_PATH = DATABASE_DIR / "mineguard.db"

    # Models
    TRAINED_MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "best.pt"
    PRETRAINED_MODEL_PATH = PROJECT_ROOT / "models" / "pretrained" / "yolo11n.pt"

    # Outputs & Storage
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    VIOLATIONS_DIR = OUTPUTS_DIR / "violations"
    PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
    REPORTS_DIR = OUTPUTS_DIR / "reports"

    # Camera Defaults
    DEFAULT_CAMERA_SOURCE = base_config.CAMERA_INDEX
    DEFAULT_FRAME_WIDTH = base_config.DEFAULT_FRAME_WIDTH
    DEFAULT_FRAME_HEIGHT = base_config.DEFAULT_FRAME_HEIGHT
    TARGET_FPS = base_config.TARGET_FPS

    # AI Detection & Tracking Hyperparameters
    CONFIDENCE_THRESHOLD = base_config.CONFIDENCE_THRESHOLD
    CONF_PERSON = base_config.CONF_PERSON
    CONF_HARDHAT = base_config.CONF_HARDHAT
    CONF_VEST = base_config.CONF_VEST
    IOU_THRESHOLD = base_config.IOU_THRESHOLD
    IMAGE_SIZE = base_config.IMAGE_SIZE
    DEVICE = base_config.DEVICE

    # Checkpoint Inspection Rules
    INSPECTION_ZONE = {
        "x_min": base_config.INSPECTION_ZONE_X_MIN,
        "x_max": base_config.INSPECTION_ZONE_X_MAX,
        "y_min": base_config.INSPECTION_ZONE_Y_MIN,
        "y_max": base_config.INSPECTION_ZONE_Y_MAX,
    }
    MIN_PERSON_HEIGHT_PX = base_config.MIN_PERSON_HEIGHT_PX
    REQUIRED_SAFE_OBSERVATIONS = base_config.REQUIRED_SAFE_OBSERVATIONS
    REQUIRED_VIOLATION_OBSERVATIONS = base_config.REQUIRED_VIOLATION_OBSERVATIONS
    INSPECTION_TIMEOUT_SECONDS = base_config.INSPECTION_TIMEOUT_SECONDS
    COOLDOWN_BETWEEN_WORKERS_SEC = base_config.COOLDOWN_BETWEEN_WORKERS_SEC

    # Server Binding
    HOST = "127.0.0.1"
    PORT = 5000
    DEBUG = False

    @classmethod
    def ensure_directories(cls):
        """Ensures required application directories exist."""
        cls.DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        cls.VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        cls.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
