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
    drop = "drop"
    probe_tc_kinetics = "probe_tc_kinetics"
    tempgrad_fit = "tempgrad_fit"


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


def _scheduler_connection(config_path: Optional[Path] = None):
    """Open the controller database used by asynchronous scheduler commands."""
    db_path = state.get("db")
    if db_path is None and config_path is not None:
        cfg = load_config(config_path)
        db_path = Path(cfg.get("run", {}).get("output_dir", ".")) / "nerd.sqlite"
    if db_path is None:
        db_path = Path("nerd.sqlite")
    conn = db_api.connect(Path(db_path).resolve())
    db_api.init_schema(conn)
    return conn


def _show_scheduler_row(row) -> None:
    typer.echo("task_id: %s" % row["task_id"])
    typer.echo("task: %s" % row["task_name"])
    typer.echo("task_state: %s" % row["task_state"])
    typer.echo("attempt: %s" % row["try_index"])
    typer.echo("attempt_state: %s" % row["scheduler_state"])
    typer.echo("executor: %s" % row["executor_profile"])
    typer.echo("scheduler_id: %s" % (row["scheduler_id"] or "-"))
    if row["exit_code"] is not None:
        typer.echo("exit_code: %s" % row["exit_code"])
    if row["error"]:
        typer.echo("message: %s" % row["error"])


@app.command()
def submit(
    step: RunStep = typer.Argument(..., help="The scientific task to submit."),
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="Path to the run configuration file."
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Named executor profile from the configuration."
    ),
):
    """Prepare and submit a task without waiting for its command to finish."""
    from nerd.scheduler.service import submit_task

    conn = _scheduler_connection(config_path)
    try:
        row = submit_task(conn, step.value, config_path, profile)
        _show_scheduler_row(row)
    except Exception as exc:
        get_logger(__name__).exception("Task submission failed: %s", exc)
        typer.echo("Submission failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


@app.command()
def status(task_id: int = typer.Argument(..., min=1, help="Controller task ID.")):
    """Reconcile one task with its executor and show durable state."""
    from nerd.scheduler.service import reconcile

    conn = _scheduler_connection()
    try:
        _show_scheduler_row(reconcile(conn, task_id))
    except Exception as exc:
        typer.echo("Status failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


@app.command("logs")
def scheduler_logs(
    task_id: int = typer.Argument(..., min=1, help="Controller task ID."),
    tail: int = typer.Option(100, "--tail", "-n", min=0, help="Number of lines to show."),
):
    """Read local or remote logs for the latest task attempt."""
    from nerd.scheduler.service import task_logs

    conn = _scheduler_connection()
    try:
        typer.echo(task_logs(conn, task_id, tail), nl=True)
    except Exception as exc:
        typer.echo("Logs failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


@app.command()
def cancel(task_id: int = typer.Argument(..., min=1, help="Controller task ID.")):
    """Request cancellation of the latest task attempt."""
    from nerd.scheduler.service import cancel_task

    conn = _scheduler_connection()
    try:
        _show_scheduler_row(cancel_task(conn, task_id))
    except Exception as exc:
        typer.echo("Cancellation failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


@app.command()
def collect(task_id: int = typer.Argument(..., min=1, help="Controller task ID.")):
    """Collect completed output, validate it, and import scientific results."""
    from nerd.scheduler.service import collect_task

    conn = _scheduler_connection()
    try:
        _show_scheduler_row(collect_task(conn, task_id))
    except Exception as exc:
        typer.echo("Collection failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


@app.command()
def retry(task_id: int = typer.Argument(..., min=1, help="Controller task ID.")):
    """Create and submit a new attempt for a failed or cancelled task."""
    from nerd.scheduler.service import retry_task

    conn = _scheduler_connection()
    try:
        _show_scheduler_row(retry_task(conn, task_id))
    except Exception as exc:
        typer.echo("Retry failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


def _container_cli_context(config_path: Path, profile_name: Optional[str]):
    from nerd.containers import container_requested, shapemapper_container_spec
    from nerd.scheduler.profiles import load_executor_profile

    cfg = load_config(config_path)
    block = cfg.get("mut_count") or {}
    if str(block.get("plugin", "")).lower() != "shapemapper":
        raise ValueError("Container readiness currently applies to the ShapeMapper mutcount plugin.")
    tool_cfg = block.get("tool") or {}
    if not container_requested(tool_cfg):
        raise ValueError("ShapeMapper is configured for native/custom execution; no container is required.")
    return shapemapper_container_spec(tool_cfg), load_executor_profile(cfg, profile_name), tool_cfg


def _show_readiness(result) -> None:
    typer.echo("execution_host: %s" % result.execution_host)
    typer.echo("executor: %s (%s)" % (result.profile, result.executor_type))
    for check in result.checks:
        typer.echo("%s  %s: %s" % ("OK" if check["ok"] else "NOT READY", check["name"], check["message"]))
    typer.echo("ready: %s" % ("yes" if result.ready else "no"))


@app.command()
def doctor(
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="ShapeMapper run configuration file."
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Executor profile to inspect."),
):
    """Check ShapeMapper container readiness on the selected execution host."""
    from nerd.containers import inspect_container

    try:
        spec, executor_profile, tool_cfg = _container_cli_context(config_path, profile)
        result = inspect_container(spec, executor_profile, tool_cfg)
        _show_readiness(result)
        if not result.ready:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo("Doctor failed: %s" % exc, err=True)
        raise typer.Exit(code=1)


@app.command("prepare-image")
def prepare_image(
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="ShapeMapper run configuration file."
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Executor profile to prepare."),
):
    """Prepare and smoke-test the immutable ShapeMapper SIF on its execution host."""
    from nerd.containers import prepare_container

    try:
        spec, executor_profile, tool_cfg = _container_cli_context(config_path, profile)
        result = prepare_container(spec, executor_profile, tool_cfg)
        _show_readiness(result)
        typer.echo("sif: %s" % result.sif_path)
        typer.echo("sif_sha256: %s" % result.sif_checksum)
    except Exception as exc:
        typer.echo("Preparation failed: %s" % exc, err=True)
        raise typer.Exit(code=1)


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
    from nerd.scheduler import store

    conn = _scheduler_connection()
    try:
        rows = store.list_tasks(conn, label)
        if not rows:
            typer.echo("No tasks found.")
            return
        typer.echo("ID\tTASK\tLABEL\tSTATE\tATTEMPT\tEXECUTOR\tSCHEDULER ID")
        for row in rows:
            typer.echo("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
                row["id"], row["task_name"], row["label"], row["state"],
                row["try_index"] or "-", row["executor_profile"] or "-",
                row["scheduler_id"] or "-",
            ))
    finally:
        conn.close()


if __name__ == "__main__":
    app()
