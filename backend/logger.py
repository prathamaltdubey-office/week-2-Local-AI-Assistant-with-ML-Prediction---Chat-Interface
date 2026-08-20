"""
Structured logging configuration for the FastAPI service.
"""

import logging
import sys
from pathlib import Path


def setup_logger() -> logging.Logger:
    """
    Configure and return the application logger.
    """

    logger = logging.getLogger("customer_churn_api")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # ========================================================
    # Application log - INFO and above
    # ========================================================

    app_handler = logging.FileHandler(
        log_dir / "app.log",
        encoding="utf-8",
    )
    app_handler.setLevel(logging.INFO)

    # ========================================================
    # Error log - ERROR and above
    # ========================================================

    error_handler = logging.FileHandler(
        log_dir / "error.log",
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)

    # ========================================================
    # Console
    # ========================================================

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # ========================================================
    # Formatter
    # ========================================================

    formatter = logging.Formatter(
        fmt=("%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # ========================================================
    # Register handlers
    # ========================================================

    logger.addHandler(app_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()
