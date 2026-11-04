"""Configuration de logging partagée (console + fichier dans logs/)."""

import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def get_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # déjà configuré
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
