# nerd/utils/logging.py

import logging
from pathlib import Path
from rich.logging import RichHandler

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FILE = Path("logs/nerd.log")


def setup_logger(name: str = "nerd", log_to_file: bool = False, log_file: Path = DEFAULT_LOG_FILE, level=DEFAULT_LOG_LEVEL):
    """
    Initialize and return a logger with optional file and rich console handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers in Jupyter or repeated setup
    if logger.handlers:
        return logger

    # Rich console handler
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Optional file logging
    if log_to_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger