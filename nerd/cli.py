# nerd/cli.py
"""
Command-line interface for the nerd application, powered by Typer.
"""

import typer
from pathlib import Path
from typing import Optional
import enum

from nerd.utils.logging import setup_logger, get_logger
from nerd.utils.config import load_config
from nerd.db import api as db_api
from nerd.pipeline.tasks import create, mut_count, tc_free

# Create the main Typer application
app = typer.Typer(
    no_args_is_help=True,
    help="NERD: A data analysis pipeline for RNA engineering.",
    context_settings={"help_option_names": ["-h", "--help"]},
)

# A shared dictionary to store global state from the callback
state = {}

class RunStep(str, enum.Enum):
    """Enum for available pipeline steps."""
    create = "create"
    mut_count = "mut_count"
    tc_free = "tc_free"


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose (DEBUG) logging."
    ),
    db: Path = typer.Option(
        "test_output/nerd_dev.sqlite3",
        "--db",
        help="Path to the SQLite database file.",
        writable=True,
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Path to a file for logging."
    ),
):
    """
    Main callback to set up logging and global state.
    """
    # Store global options in the state dictionary
    state["verbose"] = verbose
    state["db"] = db
    state["log_file"] = log_file

    # Configure the logger
    setup_logger(logfile=log_file, verbose=verbose)
    log = get_logger(__name__)
    log.debug("CLI context initialized. verbose=%s, db=%s", verbose, db)


@app.command()
def run(
    ctx: typer.Context,
    step: RunStep = typer.Argument(..., help="The pipeline step to execute."),
    config_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the run configuration file.",
    ),
):
    """
    Execute a specific step of the data analysis pipeline.
    """
    log = get_logger(__name__)
    log.info("Executing 'run' command for step: '%s'", step.value)

    try:
        cfg = load_config(config_path)
        
        # Establish database connection
        conn = db_api.connect(state["db"])
        db_api.init_schema(conn)

        # Map step name to task class
        task_map = {
            "create": create.CreateTask(),
            "mut_count": mut_count.MutCountTask(),
            "tc_free": tc_free.TimecourseFreeTask(),
        }

        task = task_map.get(step.value)
        if not task:
            log.error("Unknown task step: %s", step.value)
            raise typer.Exit(code=1)

        # Execute the task
        task.exec(conn, cfg, verbose=state["verbose"])

    except Exception as e:
        log.exception("Failed to execute task '%s': %s", step.value, e)
        raise typer.Exit(code=1)
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            log.debug("Database connection closed.")


@app.command()
def ls(
    ctx: typer.Context,
    label: Optional[str] = typer.Option(
        None, "--label", "-l", help="Filter runs by a specific label."
    ),
):
    """
    List available runs and their status.
    """
    log = get_logger(__name__)
    log.info("Executing 'ls' command. Label: %s", label)
    
    # Delegate to a function in main.py
    # list_runs(db_path=state["db"], label=label)
    log.warning("'ls' command is not fully implemented yet.")


if __name__ == "__main__":
    app()