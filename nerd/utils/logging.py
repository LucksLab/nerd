# nerd/utils/logging.py
"""
This module provides centralized logging configuration for the nerd application.
"""

import logging
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler

<<<<<<< HEAD
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
=======
def setup_logger(logfile: Optional[Path] = None, verbose: bool = False) -> logging.Logger:
    """
    Configures the root logger for the 'nerd' application and returns it.

    - Sets up a RichHandler for console output.
    - Optionally sets up a FileHandler if a logfile path is provided.
    - Log level is set to DEBUG if verbose is True, otherwise INFO.

    Args:
        logfile: Optional path to a file for log output.
        verbose: If True, sets the log level to DEBUG.
        
    Returns:
        The configured 'nerd' root logger instance.
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Get the root logger for the application namespace
    log = logging.getLogger("nerd")
    log.setLevel(level)

    # Prevent propagation to the default root logger
    log.propagate = False
>>>>>>> main

    # Clear any existing handlers to avoid duplicate logs
    if log.hasHandlers():
        log.handlers.clear()

    # --- Console Handler ---
    # Always add a rich handler for beautiful console output
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        log_time_format="[%X]"
    )
    console_handler.setLevel(level)
<<<<<<< HEAD
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
=======
    log.addHandler(console_handler)

    # --- File Handler ---
    # Add a file handler if a path is provided
    if logfile:
        # Ensure logfile is a Path object before using path-specific attributes
        logfile = Path(logfile)

        # Ensure the directory for the log file exists
        logfile.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(level)
        
        # Define a standard format for file logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        log.debug("File logging enabled at: %s", logfile)

    log.debug("Logger configured with level=%s", logging.getLevelName(level))
    return log


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance that is a child of the root 'nerd' logger.

    This ensures that all loggers within the application inherit the
    configuration from setup_logger.

    Args:
        name: The name for the logger, typically __name__.

    Returns:
        A configured logging.Logger instance.
    """
    return logging.getLogger(name)
>>>>>>> main
