import os
from pathlib import Path

# Resolve project root deterministically regardless of working directory
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
STATIC_DIR: Path = PROJECT_ROOT / "static"
DATA_DIR: Path = PROJECT_ROOT / "data"

MODEL_PATH: Path = MODELS_DIR / "txguard_model.pth"
INDEX_HTML_PATH: Path = STATIC_DIR / "index.html"

INPUT_FEATURES: int = 29
HIDDEN_UNITS: int = 64
DROPOUT_RATE: float = 0.3

# Two-Tier Cascading PayPal Authorization Gate Thresholds
STEP_UP_THRESHOLD: float = float(os.getenv("TXGUARD_STEP_UP_THRESHOLD", "0.45"))
AUTO_BLOCK_THRESHOLD: float = float(os.getenv("TXGUARD_AUTO_BLOCK_THRESHOLD", "0.85"))

# External Platform Configuration
PAYPAL_API_BASE: str = os.getenv("PAYPAL_API_BASE", "https://api-m.sandbox.paypal.com").rstrip("/")
PAYPAL_CLIENT_ID: str = os.getenv("PAYPAL_CLIENT_ID", "").strip()
PAYPAL_CLIENT_SECRET: str = os.getenv("PAYPAL_CLIENT_SECRET", "").strip()
PAYPAL_WEBHOOK_ID: str = os.getenv("PAYPAL_WEBHOOK_ID", "").strip()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

