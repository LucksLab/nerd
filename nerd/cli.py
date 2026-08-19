"""Command-line interface for NERD, powered by Typer."""

from __future__ import annotations

from datetime import datetime
import enum
from pathlib import Path
from typing import Callable, Optional

import typer

from nerd.db import api as db_api
from nerd.pipeline.tasks import TASK_REGISTRY
from nerd.project import ContextResolutionError, ProjectContext
from nerd.utils.config import load_config
from nerd.utils.hashing import config_hash
from nerd.utils.logging import get_logger, setup_logger


app = typer.Typer(
    no_args_is_help=True,
    help="NERD: run scientific workflows and manage durable tasks.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
task_app = typer.Typer(no_args_is_help=True, help="Inspect and manage durable tasks.")
plugin_app = typer.Typer(no_args_is_help=True, help="Advanced scientific plugin maintenance.")
plugin_doctor_app = typer.Typer(no_args_is_help=True, help="Check plugin readiness.")
image_app = typer.Typer(no_args_is_help=True, help="Inspect and prepare immutable tool images.")
db_app = typer.Typer(no_args_is_help=True, help="Inspect the selected project database.")


class RunStep(str, enum.Enum):
    """Public scientific workflows."""

    create = "create"
    mut_count = "mut_count"
    nmr_create = "nmr_create"
    nmr_kinetic_fit = "nmr_kinetic_fit"
    drop = "drop"
    probe_timecourse = "probe_timecourse"
    tempgrad_fit = "tempgrad_fit"


class ContainerPlugin(str, enum.Enum):
    shapemapper = "shapemapper"


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
    db: Optional[Path] = typer.Option(None, "--db", help="SQLite database path."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory containing nerd.sqlite."),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Write NERD logs to this file."),
):
    """Initialize invocation-scoped project and logging context."""
    ctx.obj = {
        "project_context": ProjectContext(db=db, project=project),
        "verbose": verbose,
        "log_file": log_file,
    }
    setup_logger(logfile=log_file, verbose=verbose)


def _invocation_context(ctx: Optional[typer.Context]) -> dict:
    if ctx is not None and isinstance(ctx.obj, dict):
        return ctx.obj
    return {"project_context": ProjectContext(), "verbose": False, "log_file": None}


def _command_context(ctx: typer.Context, db: Optional[Path], project: Optional[Path]) -> ProjectContext:
    return _invocation_context(ctx)["project_context"].with_overrides(db=db, project=project)


def _scheduler_connection(
    config_path: Optional[Path] = None,
    *,
    context: Optional[ProjectContext] = None,
    read_only: bool = False,
):
    """Open the controller database selected by Phase 1 context resolution."""
    cfg = load_config(config_path) if config_path is not None else None
    db_path = (context or ProjectContext()).resolve_database(
        config=cfg, config_path=config_path, must_exist=read_only
    )
    conn = db_api.connect(db_path)
    if not read_only:
        db_api.init_schema(conn)
    return conn


def _row_value(row, key: str, default=None):
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _next_actions(row) -> list[str]:
    task_id = row["task_id"]
    task_state = row["task_state"]
    attempt_state = row["scheduler_state"]
    actions = ["nerd task logs %s" % task_id]
    if task_state in {"pending", "submitted", "running", "cancel_requested"}:
        actions.extend(["nerd task wait %s" % task_id, "nerd task cancel %s" % task_id])
    elif task_state == "awaiting_collection" or attempt_state == "scheduler_completed":
        actions.extend(["nerd task collect %s" % task_id, "nerd task wait %s --collect" % task_id])
    elif attempt_state in {"scheduler_failed", "submission_failed", "cancelled", "validation_failed"}:
        actions.append("nerd task retry %s" % task_id)
    return actions


def _show_scheduler_row(row, *, detailed: bool = False) -> None:
    """Render stable human fields available before Phase 3's output contract."""
    typer.echo("task_id: %s" % row["task_id"])
    typer.echo("task: %s" % row["task_name"])
    typer.echo("task_state: %s" % row["task_state"])
    typer.echo("attempt: %s" % row["try_index"])
    attempt_id = _row_value(row, "scheduler_attempt_id")
    if attempt_id is not None:
        typer.echo("attempt_id: %s" % attempt_id)
    typer.echo("attempt_state: %s" % row["scheduler_state"])
    typer.echo("executor: %s" % row["executor_profile"])
    executor_type = _row_value(row, "executor_type")
    if executor_type:
        typer.echo("executor_type: %s" % executor_type)
    typer.echo("scheduler_id: %s" % (row["scheduler_id"] or "-"))
    if row["exit_code"] is not None:
        typer.echo("exit_code: %s" % row["exit_code"])
    error = row["error"] or _row_value(row, "task_message")
    if error:
        typer.echo("message: %s" % error)
    if detailed:
        output_dir = _row_value(row, "output_dir")
        if output_dir:
            typer.echo("output_dir: %s" % output_dir)
        try:
            from nerd.scheduler import store
            spec = store.job_spec(row)
            typer.echo("work_dir: %s" % spec.workdir)
        except (KeyError, TypeError, ValueError):
            pass
        remote_workdir = _row_value(row, "remote_workdir")
        if remote_workdir:
            typer.echo("remote_work_dir: %s" % remote_workdir)
        log_path = _row_value(row, "log_path")
        if log_path:
            typer.echo("log_path: %s" % log_path)
        typer.echo("next_actions: %s" % ", ".join(_next_actions(row)))


def _submit_handler(
    ctx: typer.Context, workflow: RunStep, config_path: Path,
    profile: Optional[str], db: Optional[Path], project: Optional[Path],
) -> None:
    from nerd.scheduler.service import submit_task

    conn = None
    try:
        conn = _scheduler_connection(config_path, context=_command_context(ctx, db, project))
        _show_scheduler_row(submit_task(conn, workflow.value, config_path, profile))
    except Exception as exc:
        get_logger(__name__).exception("Task submission failed: %s", exc)
        typer.echo("Submission failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        if conn is not None:
            conn.close()


def _run_sync(
    ctx: typer.Context, workflow: RunStep, config_path: Path,
    db: Optional[Path], project: Optional[Path],
) -> None:
    log = get_logger(__name__)
    conn = None
    try:
        cfg = load_config(config_path)
        output_dir = Path(cfg.get("run", {}).get("output_dir", "."))
        invocation = _invocation_context(ctx)
        db_path = _command_context(ctx, db, project).resolve_database(config=cfg, config_path=config_path)
        if invocation.get("log_file") is None:
            dt_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            log_dir = output_dir / "run_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            setup_logger(
                logfile=log_dir / ("%s__cfg-%s.log" % (dt_str, config_hash(cfg))),
                verbose=invocation.get("verbose", False),
            )
        conn = db_api.connect(db_path)
        db_api.init_schema(conn)
        task_class = TASK_REGISTRY.get(workflow.value)
        if task_class is None:
            raise ValueError("Task '%s' is not available in this build." % workflow.value)
        task_class().exec(conn, cfg, verbose=invocation["verbose"])
    except Exception as exc:
        log.exception("Failed to execute task '%s': %s", workflow.value, exc)
        raise typer.Exit(code=1)
    finally:
        if conn is not None:
            conn.close()


@app.command()
def run(
    ctx: typer.Context,
    workflow: RunStep = typer.Argument(..., help="Scientific workflow to execute."),
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="Run configuration file."
    ),
    detach: bool = typer.Option(False, "--detach", help="Submit a durable task and return without waiting."),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Executor profile used with --detach."),
    db: Optional[Path] = typer.Option(None, "--db", help="SQLite database for this run."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
):
    """Run a scientific workflow synchronously or submit it with --detach."""
    if profile and not detach:
        typer.echo("--profile is only supported with --detach.", err=True)
        raise typer.Exit(code=2)
    if detach:
        _submit_handler(ctx, workflow, config_path, profile, db, project)
    else:
        _run_sync(ctx, workflow, config_path, db, project)


def _open_existing(ctx, db: Optional[Path], project: Optional[Path], failure: str):
    try:
        return _scheduler_connection(context=_command_context(ctx, db, project), read_only=True)
    except ContextResolutionError as exc:
        typer.echo("%s failed: %s" % (failure, exc), err=True)
        raise typer.Exit(code=1)


def _action_handler(
    ctx: typer.Context, task_id: int, db: Optional[Path], project: Optional[Path],
    action: Callable, failure: str, *, detailed: bool = False,
) -> None:
    conn = _open_existing(ctx, db, project, failure)
    try:
        _show_scheduler_row(action(conn, task_id), detailed=detailed)
    except Exception as exc:
        typer.echo("%s failed: %s" % (failure, exc), err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


def _logs_handler(
    ctx: typer.Context, task_id: int, tail: int,
    db: Optional[Path], project: Optional[Path],
) -> None:
    from nerd.scheduler.service import task_logs

    conn = _open_existing(ctx, db, project, "Logs")
    try:
        typer.echo(task_logs(conn, task_id, tail))
    except Exception as exc:
        typer.echo("Logs failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


def _list_handler(
    ctx: typer.Context, label: Optional[str], state: Optional[str], workflow: Optional[RunStep],
    limit: int, db: Optional[Path], project: Optional[Path],
) -> None:
    from nerd.scheduler import store

    conn = _open_existing(ctx, db, project, "List")
    try:
        rows = store.list_tasks(
            conn, label=label, state=state,
            task_name=workflow.value if workflow else None, limit=limit,
        )
        if not rows:
            typer.echo("No tasks found.")
            return
        typer.echo("ID\tTASK\tLABEL\tSTATE\tATTEMPT\tEXECUTOR\tSCHEDULER ID")
        for row in rows:
            typer.echo("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
                row["id"], row["task_name"], row["label"], row["state"],
                row["try_index"] or "-", row["executor_profile"] or "-", row["scheduler_id"] or "-",
            ))
    finally:
        conn.close()


@task_app.command("list")
def task_list(
    ctx: typer.Context,
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Filter by label."),
    state: Optional[str] = typer.Option(None, "--state", "-s", help="Filter by task state."),
    workflow: Optional[RunStep] = typer.Option(None, "--workflow", "--task", "-w", help="Filter by scientific workflow."),
    limit: int = typer.Option(50, "--limit", min=1, help="Maximum tasks to show."),
    db: Optional[Path] = typer.Option(None, "--db", help="Existing controller database."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
):
    """List durable tasks, newest first."""
    _list_handler(ctx, label, state, workflow, limit, db, project)


@task_app.command("show")
def task_show(
    ctx: typer.Context,
    task_id: int = typer.Argument(..., min=1, help="Durable task ID."),
    db: Optional[Path] = typer.Option(None, "--db", help="Existing controller database."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
):
    """Reconcile and show task, attempt, executor, path, and next-action details."""
    from nerd.scheduler.service import reconcile
    _action_handler(ctx, task_id, db, project, reconcile, "Show", detailed=True)


@task_app.command("logs")
def task_logs_command(
    ctx: typer.Context,
    task_id: int = typer.Argument(..., min=1, help="Durable task ID."),
    tail: int = typer.Option(100, "--tail", "-n", min=0, help="Number of lines to show."),
    db: Optional[Path] = typer.Option(None, "--db", help="Existing controller database."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
):
    """Read logs for the latest task attempt."""
    _logs_handler(ctx, task_id, tail, db, project)


@task_app.command("wait")
def task_wait(
    ctx: typer.Context,
    task_id: int = typer.Argument(..., min=1, help="Durable task ID."),
    collect: bool = typer.Option(False, "--collect", help="Collect successful output before returning."),
    poll_interval: float = typer.Option(1.0, "--poll-interval", min=0.05, help="Seconds between checks."),
    db: Optional[Path] = typer.Option(None, "--db", help="Existing controller database."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
):
    """Wait for scheduling to finish; optionally collect successful output."""
    from nerd.scheduler.service import wait_for_task
    action = lambda conn, task_id: wait_for_task(
        conn, task_id, collect=collect, poll_interval=poll_interval
    )
    _action_handler(ctx, task_id, db, project, action, "Wait", detailed=True)


def _lifecycle_command(name: str, action_name: str, help_text: str):
    def command(
        ctx: typer.Context,
        task_id: int = typer.Argument(..., min=1, help="Durable task ID."),
        db: Optional[Path] = typer.Option(None, "--db", help="Existing controller database."),
        project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
    ):
        from nerd.scheduler import service
        _action_handler(ctx, task_id, db, project, getattr(service, action_name), name.title())

    command.__name__ = "task_%s" % name
    command.__doc__ = help_text
    task_app.command(name)(command)
    return command


task_cancel = _lifecycle_command("cancel", "cancel_task", "Request cancellation of a task.")
task_collect = _lifecycle_command("collect", "collect_task", "Collect and validate completed output.")
task_retry = _lifecycle_command("retry", "retry_task", "Retry a failed or cancelled task.")


def _container_cli_context(config_path: Path, profile_name: Optional[str]):
    from nerd.containers import container_requested, shapemapper_container_spec
    from nerd.scheduler.profiles import load_executor_profile

    cfg = load_config(config_path)
    block = cfg.get("mut_count") or {}
    if str(block.get("plugin", "")).lower() != "shapemapper":
        raise ValueError("This command requires the ShapeMapper mut_count plugin.")
    tool_cfg = block.get("tool") or {}
    if not container_requested(tool_cfg):
        raise ValueError("ShapeMapper is configured for native/custom execution; no image is required.")
    return shapemapper_container_spec(tool_cfg), load_executor_profile(cfg, profile_name), tool_cfg


def _show_readiness(result) -> None:
    typer.echo("execution_host: %s" % result.execution_host)
    typer.echo("executor: %s (%s)" % (result.profile, result.executor_type))
    for check in result.checks:
        typer.echo("%s  %s: %s" % (
            "OK" if check["ok"] else "NOT READY", check["name"], check["message"]
        ))
    typer.echo("ready: %s" % ("yes" if result.ready else "no"))


def _inspect_shapemapper(config_path: Path, profile: Optional[str]) -> None:
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
        typer.echo("Inspection failed: %s" % exc, err=True)
        raise typer.Exit(code=1)


def _prepare_shapemapper(config_path: Path, profile: Optional[str]) -> None:
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


@plugin_doctor_app.command("shapemapper")
def plugin_doctor_shapemapper(
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="ShapeMapper run configuration file."
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Executor profile."),
):
    """Check ShapeMapper runtime and immutable-image readiness."""
    _inspect_shapemapper(config_path, profile)


@image_app.command("inspect")
def image_inspect(
    plugin: ContainerPlugin = typer.Argument(..., help="Containerized plugin."),
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="Plugin run configuration file."
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Executor profile."),
):
    """Inspect an immutable plugin image without preparing it."""
    _inspect_shapemapper(config_path, profile)


@image_app.command("prepare")
def image_prepare(
    plugin: ContainerPlugin = typer.Argument(..., help="Containerized plugin."),
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="Plugin run configuration file."
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Executor profile."),
):
    """Prepare and smoke-test an immutable plugin image."""
    _prepare_shapemapper(config_path, profile)


def _resolved_existing_database(ctx: typer.Context, db: Optional[Path], project: Optional[Path]) -> Path:
    try:
        return _command_context(ctx, db, project).resolve_database(must_exist=True)
    except ContextResolutionError as exc:
        typer.echo("Database inspection failed: %s" % exc, err=True)
        raise typer.Exit(code=1)


@db_app.command("path")
def database_path(
    ctx: typer.Context,
    db: Optional[Path] = typer.Option(None, "--db", help="Existing database."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
):
    """Print the resolved existing database path without modifying it."""
    typer.echo(_resolved_existing_database(ctx, db, project))


@db_app.command("info")
def database_info(
    ctx: typer.Context,
    db: Optional[Path] = typer.Option(None, "--db", help="Existing database."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
):
    """Show concise read-only information about the selected database."""
    path = _resolved_existing_database(ctx, db, project)
    conn = db_api.connect(path)
    try:
        task_count = conn.execute("SELECT COUNT(*) FROM core_tasks").fetchone()[0]
        attempt_count = conn.execute("SELECT COUNT(*) FROM core_task_attempts").fetchone()[0]
        typer.echo("path: %s" % path)
        typer.echo("tasks: %s" % task_count)
        typer.echo("attempts: %s" % attempt_count)
    except Exception as exc:
        typer.echo("Database inspection failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


app.add_typer(task_app, name="task")
plugin_app.add_typer(plugin_doctor_app, name="doctor")
app.add_typer(plugin_app, name="plugin")
app.add_typer(image_app, name="image")
app.add_typer(db_app, name="db")


def _deprecated(old: str, replacement: str) -> None:
    typer.echo("Deprecated: 'nerd %s' will be removed; use 'nerd %s'." % (old, replacement), err=True)


@app.command(hidden=True)
def submit(
    ctx: typer.Context,
    workflow: RunStep = typer.Argument(...),
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p"),
    db: Optional[Path] = typer.Option(None, "--db"),
    project: Optional[Path] = typer.Option(None, "--project"),
):
    """Deprecated compatibility wrapper for run --detach."""
    _deprecated("submit", "run WORKFLOW CONFIG --detach")
    _submit_handler(ctx, workflow, config_path, profile, db, project)


@app.command("ls", hidden=True)
def legacy_ls(
    ctx: typer.Context,
    label: Optional[str] = typer.Option(None, "--label", "-l"),
    db: Optional[Path] = typer.Option(None, "--db"),
    project: Optional[Path] = typer.Option(None, "--project"),
):
    """Deprecated compatibility wrapper for task list."""
    _deprecated("ls", "task list")
    _list_handler(ctx, label, None, None, 50, db, project)


@app.command(hidden=True)
def status(
    ctx: typer.Context,
    task_id: int = typer.Argument(..., min=1),
    db: Optional[Path] = typer.Option(None, "--db"),
    project: Optional[Path] = typer.Option(None, "--project"),
):
    """Deprecated compatibility wrapper for task show."""
    from nerd.scheduler.service import reconcile
    _deprecated("status", "task show")
    _action_handler(ctx, task_id, db, project, reconcile, "Status", detailed=True)


@app.command("logs", hidden=True)
def scheduler_logs(
    ctx: typer.Context,
    task_id: int = typer.Argument(..., min=1),
    tail: int = typer.Option(100, "--tail", "-n", min=0),
    db: Optional[Path] = typer.Option(None, "--db"),
    project: Optional[Path] = typer.Option(None, "--project"),
):
    """Deprecated compatibility wrapper for task logs."""
    _deprecated("logs", "task logs")
    _logs_handler(ctx, task_id, tail, db, project)


def _legacy_action(name: str, replacement: str, grouped_command: Callable):
    def command(
        ctx: typer.Context,
        task_id: int = typer.Argument(..., min=1),
        db: Optional[Path] = typer.Option(None, "--db"),
        project: Optional[Path] = typer.Option(None, "--project"),
    ):
        _deprecated(name, replacement)
        grouped_command(ctx, task_id, db, project)

    command.__name__ = name
    command.__doc__ = "Deprecated compatibility wrapper for %s." % replacement
    app.command(name, hidden=True)(command)


_legacy_action("cancel", "task cancel", task_cancel)
_legacy_action("collect", "task collect", task_collect)
_legacy_action("retry", "task retry", task_retry)


@app.command("doctor", hidden=True)
def legacy_doctor(
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p"),
):
    """Deprecated compatibility wrapper for plugin doctor shapemapper."""
    _deprecated("doctor", "plugin doctor shapemapper")
    _inspect_shapemapper(config_path, profile)


@app.command("prepare-image", hidden=True)
def legacy_prepare_image(
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p"),
):
    """Deprecated compatibility wrapper for image prepare shapemapper."""
    _deprecated("prepare-image", "image prepare shapemapper")
    _prepare_shapemapper(config_path, profile)


if __name__ == "__main__":
    app()
