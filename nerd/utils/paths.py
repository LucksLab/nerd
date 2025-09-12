# nerd/utils/paths.py
"""
This module provides utilities for creating and managing file paths for runs,
artifacts, and logs, ensuring a consistent directory structure.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from .logging import get_logger

log = get_logger(__name__)

# Assume the project root is two levels above the `utils` directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "test_output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Central log file for the application, managed by the logging setup
NERD_LOG_PATH = LOGS_DIR / "nerd.log"


def get_command_log_path(run_dir: Path) -> Path:
    """Returns the standard path for the command log within a run directory."""
    return run_dir / "command.log"


def label_root(cfg: Dict[str, Any]) -> Path:
    """
    Infers the root directory for a given run label from the configuration.
    The directory will be created if it doesn't exist.

    Args:
        cfg: The application configuration dictionary.

    Returns:
        The path to the label's root directory within the main output directory.
    """
    label = cfg.get("run", {}).get("label")
    if not label:
        raise ValueError("Configuration must contain a 'run.label' to determine the output directory.")

    label_dir = OUTPUT_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    log.debug("Label root directory set to: %s", label_dir)
    return label_dir


def make_run_dir(label_root: Path, task_name: str, suffix: Optional[str] = None) -> Path:
    """
    Creates a unique, timestamped directory for a specific task run.

    Args:
        label_root: The base directory for the label.
        task_name: The name of the task being run (e.g., 'tc_free', 'mut_count').
        suffix: An optional custom suffix for the directory name.

    Returns:
        The path to the newly created run directory.
    """
    # Ensure label_root is a Path object for robust path joining
    label_root = Path(label_root)

    if suffix is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dirname = f"{task_name}_{suffix}"
    run_dir = label_root / task_name / run_dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Created run directory: %s", run_dir)
    return run_dir


def artifact_dir_for(run_dir: Path) -> Path:
    """
    Creates and returns the 'artifacts' subdirectory within a given run directory.

    Args:
        run_dir: The path to the task's run directory.

    Returns:
        The path to the artifacts directory.
    """
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    log.debug("Artifacts directory ensured at: %s", artifacts_dir)
    return artifacts_dir


def update_latest_symlink(label_root: Path, task_name: str, run_dir: Path):
    """
    Creates or updates a 'latest' symlink to point to the most recent run directory.

    This provides a stable path to access the results of the last run for a given task.

    Args:
        label_root: The base directory for the label.
        task_name: The name of the task.
        run_dir: The path to the current run directory to be linked.
    """
    task_dir = label_root / task_name
    latest_link = task_dir / "latest"

    try:
        # Use os.readlink to check if it's a symlink and where it points
        if latest_link.is_symlink():
            log.debug("Removing existing 'latest' symlink at %s", latest_link)
            latest_link.unlink()
        elif latest_link.exists():
            # It's a file or directory, not a symlink. Log a warning.
            log.warning(
                "'%s' exists but is not a symlink. Cannot update 'latest' pointer.",
                latest_link
            )
            return

        # Create the new symlink
        latest_link.symlink_to(run_dir, target_is_directory=True)
        log.info("Updated 'latest' symlink for task '%s' to point to %s", task_name, run_dir)

    except OSError as e:
        log.exception(
            "Failed to create or update 'latest' symlink for task '%s': %s",
            task_name, e
        )
    except Exception as e:
        log.exception("An unexpected error occurred while updating symlink: %s", e)
