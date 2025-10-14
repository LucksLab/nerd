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
from nerd.utils.hashing import config_hash
from nerd.db import api as db_api
from nerd.pipeline.tasks import TASK_REGISTRY
try:
    from nerd.pipeline.tasks import tc_free as _tc_free_mod
except ImportError:  # pragma: no cover - optional legacy task
    _tc_free_mod = None
else:
    TASK_REGISTRY.setdefault("tc_free", _tc_free_mod.TimecourseFreeTask)
from datetime import datetime

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
    nmr_create = "nmr_create"
    nmr_deg_kinetics = "nmr_deg_kinetics"
    nmr_add_kinetics = "nmr_add_kinetics"
    probe_tc_kinetics = "probe_tc_kinetics"


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose (DEBUG) logging."
    ),
    db: Optional[Path] = typer.Option(
        None,
        "--db",
        help="Path to the SQLite database file. Defaults to run.output_dir from config.",
        writable=True,
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Path to a file for logging. Defaults to run.output_dir/run_logs/<date_time>__cfg-<hash>.log",
    ),
):
    """
    Main callback to set up logging and global state.
    """
    # Store global options in the state dictionary
    state["verbose"] = verbose
    state["db"] = db
    state["log_file"] = log_file

    # Configure the logger (console only for now; file handler will be set after config is loaded)
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

        # Derive defaults from config if not provided at CLI
        output_dir = Path(cfg.get("run", {}).get("output_dir", ".")).resolve()
        # Default DB: <run.output_dir>/nerd.sqlite
        if state.get("db") is None:
            default_db = output_dir / "nerd.sqlite"
            state["db"] = default_db

        # Default log file: <run.output_dir>/run_logs/<date_time>__cfg-<hash>.log
        if state.get("log_file") is None:
            dt_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            cfg_hash = config_hash(cfg)
            log_dir = output_dir / "run_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            default_log = log_dir / f"{dt_str}__cfg-{cfg_hash}.log"
            state["log_file"] = default_log
            # Reconfigure logger to add file handler now that we have a path
            setup_logger(logfile=default_log, verbose=state.get("verbose", False))

        # Establish database connection
        conn = db_api.connect(Path(state["db"]))
        db_api.init_schema(conn)

        task_map = {name: cls() for name, cls in TASK_REGISTRY.items()}

        task = task_map.get(step.value)
        if not task:
            log.error("Task '%s' is not available in this build.", step.value)
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
