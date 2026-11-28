# Global configuration settings and constants for the hiervision image classifier.
# Centralizes all paths and hyperparameters so no YAML or JSON configuration files are needed.
# Simple flat structure accessible across all modules.

from pathlib import Path

# Build paths relative to this file so scripts work from any directory.
BASE_DIR: Path = Path(__file__).parent
PROJECT_ROOT: Path = BASE_DIR

# Directory paths for raw data, saved model checkpoints, and evaluation reports
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
TARGETS_FILE: Path = BASE_DIR / "targets.json"

# Training hyperparameters and hardware settings optimized for RTX 3050 8GB
IMG_SIZE: int = 224
BATCH_SIZE: int = 32
EPOCHS: int = 10
LR: float = 1e-3
THRESHOLD: float = 0.80
SEED: int = 42
NUM_WORKERS: int = 4
SHOW_PROGRESS: bool = True

# Multi-subject detection settings
DETECT_SUBJECTS: bool = True
DETECTION_MIN_SCORE: float = 0.5
MIN_SUBJECT_AREA: float = 0.05
MAX_SUBJECTS: int = 5
HIDE_LABELS: list[str] = ["other"]