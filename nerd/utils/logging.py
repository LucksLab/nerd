# nerd/utils/logging.py

import logging
from pathlib import Path
from rich.logging import RichHandler

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FILE = Path("logs/nerd.log")

# Keep one global logger reference
_LOGGER = None


def setup_logging(level: int = DEFAULT_LOG_LEVEL, log_to_file: bool = False, log_file: Path = DEFAULT_LOG_FILE):
    """
    Configure root logger once with Rich console and optional file logging.
    Call this ONCE (e.g. at CLI startup).
    """
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("nerd")
    logger.setLevel(level)
    logger.propagate = False  # don't duplicate messages to root

    # Rich console handler
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # File handler (DEBUG detail)
    if log_to_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(file_handler)

    _LOGGER = logger
    return logger


def get_logger(name: str = None):
    """
    Get a child logger (after setup_logging has been called).
    Usage: log = get_logger(__name__)
    """
    parent = logging.getLogger("nerd")
    return parent.getChild(name) if name else parent