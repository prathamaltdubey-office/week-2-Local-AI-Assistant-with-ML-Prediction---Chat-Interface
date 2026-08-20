"""
Model loading utilities for the FastAPI inference service.
"""

from pathlib import Path

from joblib import load
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


MODEL_FILES = {
    "logistic_regression": "best_logistic_model.pkl",
    "random_forest": "best_rf_model.pkl",
    "xgboost": "best_xgb_model.pkl",
}


def load_model(model_name: str) -> Pipeline:
    """
    Load a single trained machine learning pipeline.
    """

    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, " f"got {type(model).__name__}")

    return model


def load_all_models() -> dict[str, Pipeline]:
    """
    Load all production models.
    """

    return {model_name: load_model(model_name) for model_name in MODEL_FILES}


def get_model(model_name: str) -> Pipeline:
    """
    Return the requested machine learning model.

    Args:
        model_name: Name of the model to load.

    Returns:
        Trained sklearn Pipeline.
    """

    return load_model(model_name)
