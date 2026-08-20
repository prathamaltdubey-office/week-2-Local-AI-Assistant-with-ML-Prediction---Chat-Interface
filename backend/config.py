"""
Application configuration for the FastAPI inference service.
"""

from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model directory
MODEL_DIR = BASE_DIR / "models"

# Model filenames
LOGISTIC_MODEL = "best_logistic_model.pkl"
RANDOM_FOREST_MODEL = "best_rf_model.pkl"
XGBOOST_MODEL = "best_xgb_model.pkl"

# Default model
DEFAULT_MODEL = "random_forest"

# API information
API_TITLE = "Customer Churn Prediction API"
API_DESCRIPTION = (
    "FastAPI inference service for the Customer Churn " "Prediction ML pipeline."
)
API_VERSION = "1.0.0"
