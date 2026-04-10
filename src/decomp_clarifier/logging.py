from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str, log_file: Path, log_to_console: bool = True) -> logging.Logger:
    logger = logging.getLogger("decomp_clarifier")
    logger.handlers.clear()
    logger.setLevel(level.upper())
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
