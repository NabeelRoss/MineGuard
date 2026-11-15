"""
Centralized Configuration Module for MineGuard AI.

Defines project paths, runtime hyperparameters, hardware settings,
checkpoint inspection rules, and per-class confidence thresholds.
Uses pathlib for platform-independent paths.
"""

from pathlib import Path
import torch

# Base Directory Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model Directories & Paths
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
TRAINED_DIR = MODELS_DIR / "trained"

PRETRAINED_MODEL_PATH = PRETRAINED_DIR / "yolo11n.pt"
TRAINED_MODEL_PATH = TRAINED_DIR / "best.pt"

# Active Model Path: Priority to custom trained weights, fallback to pretrained
MODEL_PATH = TRAINED_MODEL_PATH if TRAINED_MODEL_PATH.exists() else PRETRAINED_MODEL_PATH

# Dataset Paths
DATASET_ROOT = PROJECT_ROOT / "dataset" / "atharion-v1.0-public"
DATASET_PATH = PROJECT_ROOT / "training" / "data.yaml"
RAW_DATA_YAML = DATASET_ROOT / "data.yaml"

# Output Directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_REPORTS_DIR = OUTPUTS_DIR / "dataset_reports"
TRAINING_OUTPUTS_DIR = OUTPUTS_DIR / "training"
EVALUATION_OUTPUTS_DIR = TRAINING_OUTPUTS_DIR / "evaluation"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
VIOLATIONS_DIR = OUTPUTS_DIR / "violations"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure all output directories exist
for output_dir in [
    DATASET_REPORTS_DIR,
    TRAINING_OUTPUTS_DIR,
    EVALUATION_OUTPUTS_DIR,
    PREDICTIONS_DIR,
    VIOLATIONS_DIR,
    REPORTS_DIR,
]:
    output_dir.mkdir(parents=True, exist_ok=True)

# Hardware Configuration (Automatically selects CUDA if available)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# Detection & Model Hyperparameters
IMAGE_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 60
IOU_THRESHOLD = 0.45
EARLY_STOPPING_PATIENCE = 15

# Per-Class Detection Confidence Thresholds
# Calibrated for underground coal mine conditions:
# Person: 0.35 (strong body silhouette)
# Hardhat: 0.28 (smaller object, low illumination in mine tunnels, preserves recall)
# Safety Vest: 0.35 (reflective high-vis material)
CONF_PERSON = 0.35
CONF_HARDHAT = 0.28
CONF_VEST = 0.35
CONFIDENCE_THRESHOLD = 0.30  # Baseline cutoff for initial candidate proposals

# Training Worker Processes
MAX_WORKERS = 4

# Camera / Video Ingestion Defaults
CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
TARGET_FPS = 30

# ==============================================================================
# Fixed Checkpoint Inspection Architecture Parameters
# ==============================================================================
# Inspection Zone (Normalized coordinates [0.0 - 1.0] of camera frame)
# Worker must be within this zone to trigger inspection
INSPECTION_ZONE_X_MIN = 0.15
INSPECTION_ZONE_X_MAX = 0.85
INSPECTION_ZONE_Y_MIN = 0.05
INSPECTION_ZONE_Y_MAX = 0.98

# Minimum person height in pixels to ensure worker is close enough for inspection
MIN_PERSON_HEIGHT_PX = 130

# Multi-frame verification thresholds
REQUIRED_SAFE_OBSERVATIONS = 3        # Frames of verified SAFE to confirm PASS
REQUIRED_VIOLATION_OBSERVATIONS = 5   # Frames of verified VIOLATION to confirm FAIL
INSPECTION_TIMEOUT_SECONDS = 15.0      # Maximum seconds before timing out to REINSPECT
COOLDOWN_BETWEEN_WORKERS_SEC = 2.0    # Time to wait after worker departs before next IDLE cycle

# Anatomical Association Geometry (Relative to Person Bounding Box)
HEAD_TOP_OFFSET_RATIO = 0.20     # Allow helmet to extend above person box by 20%
HEAD_BOTTOM_RATIO = 0.35         # Head region spans top 35% of person height
HEAD_HORIZONTAL_EXPANSION = 0.15 # Allow helmet to sit slightly wider than torso
TORSO_TOP_RATIO = 0.18          # Torso starts below upper neck
TORSO_BOTTOM_RATIO = 0.75       # Torso ends above knees
TORSO_HORIZONTAL_EXPANSION = 0.15

# Spatial Association Thresholds
MIN_PPE_OVERLAP_RATIO = 0.22    # Minimum intersection over PPE area
MAX_HORIZONTAL_CENTER_OFFSET = 0.40 # Relative to person width
MAX_INTER_PERSON_IOU = 0.35     # Overlap above this triggers ambiguous REINSPECT

# Expected Target Classes
TARGET_CLASSES = ["Person", "Hardhat", "Safety Vest"]
