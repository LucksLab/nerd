"""Command-line interface for NERD, powered by Typer."""

from __future__ import annotations

from datetime import datetime, timedelta
import enum
import json
from pathlib import Path
import re
import socket
import time
from typing import Callable, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.text import Text

from nerd.db import api as db_api
from nerd.pipeline.tasks import TASK_REGISTRY
from nerd.project import (
    ContextResolutionError, ProjectConfigError, ProjectContext,
    load_project, render_project_toml, validate_project_name,
)
from nerd.configuration import (
    ConfigValidationError, redact, resolve_config, resolved_view, validate_config,
    write_template,
)
from nerd.utils.hashing import config_hash
from nerd.utils.logging import get_logger, setup_logger
from nerd.reporting.summary import (
    ArtifactReference, TaskIssue, TaskSummary, render_human, render_json,
    summary_exit_code,
)


app = typer.Typer(
    no_args_is_help=True,
    help="NERD: A toolkit for quantitative analysis of RNA reactivity, energetics, and kinetics.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
task_app = typer.Typer(no_args_is_help=True, help="Inspect and manage durable tasks.")
plugin_app = typer.Typer(no_args_is_help=True, help="Advanced scientific plugin maintenance.")
plugin_doctor_app = typer.Typer(no_args_is_help=True, help="Check plugin readiness.")
image_app = typer.Typer(no_args_is_help=True, help="Inspect and prepare immutable tool images.")
db_app = typer.Typer(no_args_is_help=True, help="Inspect the selected project database.")
config_app = typer.Typer(no_args_is_help=True, help="Create, validate, and inspect analysis configs.")
webui_app = typer.Typer(no_args_is_help=True, help="Create, edit, view, or analyze project data in the Web UI.")

WEBUI_DEFAULT_PORTS = {
    "analyze": 8420,
    "create": 8421,
    "edit": 8422,
    "view": 8423,
}


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


WORKFLOW_SUMMARIES: dict[str, str] = {
    "create": "Ingest sequencing samples and reactions from a YAML config.",
    "mut_count": "Count mutations with an external tool and import the profiles.",
    "nmr_create": "Ingest NMR reactions from a YAML config.",
    "nmr_kinetic_fit": "Fit NMR degradation or adduction kinetics.",
    "drop": "Flag sequencing samples so later workflows skip them.",
    "probe_timecourse": "Fit chemical probing timecourses for reaction groups.",
    "tempgrad_fit": "Fit Arrhenius or two-state melt models to temperature gradients.",
}


def _workflow_epilog() -> str:
    """Render the workflow list shown in 'nerd run' help.

    Rich collapses single newlines in an epilog, so each line is its own
    paragraph and no column padding is used: the text stays readable at any
    terminal width.
    """
    lines = ["Workflows:"]
    lines.extend(
        "%s - %s" % (step.value, WORKFLOW_SUMMARIES.get(step.value, ""))
        for step in RunStep
    )
    lines.append("Run 'nerd run WORKFLOW CONFIG_PATH' to execute a workflow, or add "
                 "--detach to submit it as a durable task and return immediately.")
    return "\n\n".join(lines)


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress informational progress."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored logging output."),
    db: Optional[Path] = typer.Option(None, "--db", help="SQLite database path."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project root or direct .nerd/project.toml path."),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Write NERD logs to this file."),
):
    """Initialize invocation-scoped project and logging context."""
    ctx.obj = {
        "project_context": ProjectContext(db=db, project=project),
        "verbose": verbose,
        "log_file": log_file, "quiet": quiet, "no_color": no_color,
    }
    setup_logger(logfile=log_file, verbose=verbose, quiet=quiet, no_color=no_color)


def _invocation_context(ctx: Optional[typer.Context]) -> dict:
    if ctx is not None and isinstance(ctx.obj, dict):
        return ctx.obj
    return {"project_context": ProjectContext(), "verbose": False, "log_file": None,
            "quiet": False, "no_color": False}


def _emit_summary(summary: TaskSummary, json_output: bool = False) -> None:
    typer.echo(render_json(summary) if json_output else render_human(summary), nl=False)


def _set_json_logging(ctx: typer.Context) -> None:
    invocation = _invocation_context(ctx)
    setup_logger(logfile=invocation.get("log_file"), verbose=invocation.get("verbose", False),
                 quiet=invocation.get("quiet", False), no_color=invocation.get("no_color", False),
                 json_mode=True)


def _failure_summary(workflow: str, exc: BaseException, *, started: float,
                     database_path: Optional[str] = None) -> TaskSummary:
    duration = round(time.monotonic() - started, 6)
    ended_at = datetime.now()
    return TaskSummary(
        status="failed", workflow=workflow,
        started_at=(ended_at - timedelta(seconds=duration)).isoformat(),
        ended_at=ended_at.isoformat(), duration_seconds=duration,
        counts={"attempted": 1, "succeeded": 0, "failed": 1, "skipped": 0},
        failures=[TaskIssue("execution_failed", str(exc))],
        database_path=database_path,
    )


def _command_context(ctx: typer.Context, db: Optional[Path], project: Optional[Path]) -> ProjectContext:
    return _invocation_context(ctx)["project_context"].with_overrides(db=db, project=project)


def _scheduler_connection(
    config_path: Optional[Path] = None,
    *,
    context: Optional[ProjectContext] = None,
    read_only: bool = False,
):
    """Open the controller database selected by Phase 1 context resolution."""
    cfg = None
    if config_path is not None:
        cfg, _ = resolve_config(config_path, context=context or ProjectContext())
    db_path = (context or ProjectContext()).resolve_database(
        config=cfg, config_path=config_path, must_exist=read_only
    )
    conn = db_api.connect(db_path)
    # Existing task commands reconcile scheduler state and therefore require a
    # writable, migrated schema even though they never create a missing DB.
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
        actions.extend(["nerd task watch %s" % task_id, "nerd task cancel %s" % task_id])
    elif task_state == "awaiting_collection" or attempt_state == "scheduler_completed":
        actions.extend(["nerd task collect %s" % task_id, "nerd task wait %s --collect" % task_id])
    elif task_state == "partial_success" or attempt_state in {
        "scheduler_failed", "submission_failed", "cancelled", "validation_failed"
    }:
        actions.append("nerd task retry %s" % task_id)
    return actions


def _seconds_between(start, end) -> Optional[float]:
    if not start or not end:
        return None
    try:
        return round((datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))).total_seconds(), 6)
    except (TypeError, ValueError):
        return None


def _scheduler_summary(row, database_path: Optional[str] = None) -> TaskSummary:
    attempt_state = str(row["scheduler_state"])
    task_state = str(row["task_state"])
    statuses = {
        "completed": "completed", "cached": "cached", "failed": "failed",
        "cancelled": "cancelled", "submitted": "submitted", "running": "running",
        "awaiting_collection": "awaiting_collection", "pending": "queued",
    }
    status = statuses.get(task_state, task_state)
    task_message = _row_value(row, "task_message")
    error = row["error"] or task_message
    source_task_id = None
    if status == "cached" and error:
        match = re.search(r"task_id=(\d+)", str(error))
        source_task_id = int(match.group(1)) if match else None
    submitted = _row_value(row, "submitted_at") or _row_value(row, "task_started_at")
    started = _row_value(row, "scheduler_started_at")
    finished = _row_value(row, "scheduler_finished_at")
    collected = _row_value(row, "collected_at") or _row_value(row, "task_ended_at")
    attempt = row["try_index"]
    metrics = {
        "attempt": int(attempt) if attempt is not None else None,
        "attempt_state": attempt_state,
        "executor": row["executor_profile"],
        "executor_type": _row_value(row, "executor_type"),
        "scheduler_id": row["scheduler_id"],
        "exit_code": row["exit_code"],
        "message": task_message,
    }
    timings = {
        "queue_seconds": _seconds_between(submitted, started),
        "execution_seconds": _seconds_between(started, finished),
        "collection_seconds": _seconds_between(finished, collected),
    }
    artifacts = []
    output_dir = _row_value(row, "output_dir")
    if output_dir:
        from nerd.reporting.summary import ArtifactReference
        artifacts.append(ArtifactReference("output_directory", str(output_dir)))
    parent_counts = _row_value(row, "counts") or {}
    attempted = int(_row_value(row, "total_units", 1) or 1)
    return TaskSummary(
        status=status, workflow=str(row["task_name"]), task_id=int(row["task_id"]),
        source_task_id=source_task_id,
        label=_row_value(row, "label"), plugin=_row_value(row, "tool"),
        version=_row_value(row, "tool_version"), started_at=started or submitted,
        ended_at=collected or finished, duration_seconds=_seconds_between(submitted, collected or finished),
        timings=timings,
        counts={"attempted": attempted,
                "succeeded": int(parent_counts.get("completed", 1 if status == "completed" else 0)),
                "failed": int(parent_counts.get("failed", 1 if status in {"failed", "cancelled"} else 0)),
                "skipped": 0},
        metrics=metrics,
        failures=([TaskIssue("scheduler_error", str(error))]
                  if error and status in {"failed", "cancelled"} else []),
        artifacts=artifacts, log_path=_row_value(row, "log_path"),
        database_path=database_path, next_actions=_next_actions(row),
    )


_TASK_STATE_STYLES = {
    "completed": "green",
    "cached": "green",
    "running": "cyan",
    "collecting": "cyan",
    "submitted": "blue",
    "pending": "blue",
    "awaiting_collection": "yellow",
    "partial_success": "yellow",
    "cancel_requested": "yellow",
    "failed": "red",
    "cancelled": "red",
}


def _task_console(no_color: bool = False) -> Console:
    """Return the stdout console used for human task lifecycle output."""
    return Console(no_color=no_color)


def _state_text(state: object) -> Text:
    value = str(state or "unknown")
    label = value.replace("_", " ")
    return Text(label, style=_TASK_STATE_STYLES.get(value, ""))


def _details_table() -> Table:
    return Table.grid(padding=(0, 2))


def _render_next_actions(console: Console, row) -> None:
    actions = _next_actions(row)
    if not actions:
        return
    console.print()
    console.print("[bold]Next[/bold]")
    for action in actions:
        console.print("  [cyan]%s[/cyan]" % action)


def _show_scheduler_row(row, *, detailed: bool = False, no_color: bool = False) -> None:
    """Render concise, terminal-width-aware task lifecycle output."""
    console = _task_console(no_color=no_color)
    console.print("[bold]Task %s[/bold] · %s" % (row["task_id"], row["task_name"]))
    console.print()

    if _row_value(row, "is_parent", False):
        counts = _row_value(row, "counts", {}) or {}
        total = int(_row_value(row, "total_units", 0) or 0)
        completed = int(counts.get("completed", 0))
        failed = int(counts.get("failed", 0)) + int(counts.get("cancelled", 0))

        summary = _details_table()
        summary.add_row("[bold]Status[/bold]", _state_text(row["task_state"]))
        summary.add_row("[bold]Progress[/bold]", "%s/%s complete · %s failed" % (completed, total, failed))
        console.print(summary)
        console.print()

        children = Table(box=None, pad_edge=False, header_style="bold", show_lines=False)
        children.add_column("UNIT", max_width=38, overflow="ellipsis", no_wrap=True)
        children.add_column("STATE", no_wrap=True)
        children.add_column("TASK", justify="right", no_wrap=True)
        children.add_column("SCHEDULER JOB", justify="right", no_wrap=True)
        for child in row["children"]:
            children.add_row(
                str(child.get("unit_label") or child.get("unit_key") or "-"),
                _state_text(child["task_state"]),
                str(child["task_id"]),
                str(child.get("scheduler_id") or "-"),
            )
        console.print(children)
        if detailed:
            paths = _details_table()
            output_dir = _row_value(row, "output_dir")
            if output_dir:
                paths.add_row("[bold]Output[/bold]", str(output_dir))
            if paths.row_count:
                console.print()
                console.print("[bold]Paths[/bold]")
                console.print(paths)
        _render_next_actions(console, row)
        return

    summary = _details_table()
    summary.add_row("[bold]Status[/bold]", _state_text(row["task_state"]))
    scheduler_id = row["scheduler_id"] or "-"
    executor = str(row["executor_profile"] or "-")
    executor_type = _row_value(row, "executor_type")
    if scheduler_id != "-":
        scheduler_label = "Slurm job %s" % scheduler_id if executor_type == "ssh_slurm" else "job %s" % scheduler_id
        executor = "%s · %s" % (executor, scheduler_label)
    summary.add_row("[bold]Executor[/bold]", executor)
    summary.add_row("[bold]Attempt[/bold]", str(row["try_index"] or "-"))
    unit = _row_value(row, "unit_label") or _row_value(row, "unit_key")
    if unit:
        summary.add_row("[bold]Work unit[/bold]", str(unit))
    if row["exit_code"] is not None:
        summary.add_row("[bold]Exit code[/bold]", str(row["exit_code"]))
    error = row["error"] or _row_value(row, "task_message")
    if error and str(error).lower().replace("_", " ") != str(row["task_state"]).lower().replace("_", " "):
        summary.add_row("[bold]Message[/bold]", str(error))
    console.print(summary)

    if detailed:
        internals = _details_table()
        attempt_id = _row_value(row, "scheduler_attempt_id")
        if attempt_id is not None:
            internals.add_row("[bold]Attempt ID[/bold]", str(attempt_id))
        if executor_type:
            internals.add_row("[bold]Executor type[/bold]", str(executor_type))
        attempt_state = _row_value(row, "scheduler_state")
        if attempt_state:
            internals.add_row("[bold]Attempt state[/bold]", str(attempt_state).replace("_", " "))
        if internals.row_count:
            console.print()
            console.print("[bold]Details[/bold]")
            console.print(internals)

        paths = _details_table()
        output_dir = _row_value(row, "output_dir")
        if output_dir:
            paths.add_row("[bold]Output[/bold]", str(output_dir))
        try:
            from nerd.scheduler import store
            spec = store.job_spec(row)
            paths.add_row("[bold]Local work[/bold]", str(spec.workdir))
        except (KeyError, TypeError, ValueError):
            pass
        remote_workdir = _row_value(row, "remote_workdir")
        if remote_workdir:
            paths.add_row("[bold]Remote work[/bold]", str(remote_workdir))
        log_path = _row_value(row, "log_path")
        if log_path:
            paths.add_row("[bold]Log[/bold]", str(log_path))
        if paths.row_count:
            console.print()
            console.print("[bold]Paths[/bold]")
            console.print(paths)
    _render_next_actions(console, row)


def _submit_handler(
    ctx: typer.Context, workflow: RunStep, config_path: Path,
    profile: Optional[str], db: Optional[Path], project: Optional[Path],
    json_output: bool = False,
) -> None:
    from nerd.scheduler.service import submit_task

    conn = None
    try:
        if json_output:
            _set_json_logging(ctx)
        command_context = _command_context(ctx, db, project)
        cfg, project_cfg = resolve_config(config_path, context=command_context, executor=profile)
        conn = _scheduler_connection(config_path, context=command_context)
        if project_cfg is not None:
            row = submit_task(
                conn, workflow.value, config_path, profile, resolved_config=cfg
            )
        else:
            # Retain the historical call shape for third-party integrations.
            row = submit_task(conn, workflow.value, config_path, profile)
        db_path = str(Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve())
        if json_output:
            _emit_summary(_scheduler_summary(row, db_path), True)
        else:
            _show_scheduler_row(
                row, no_color=_invocation_context(ctx).get("no_color", False)
            )
    except Exception as exc:
        get_logger(__name__).exception("Task submission failed: %s", exc)
        if json_output:
            _emit_summary(_failure_summary(workflow.value, exc, started=time.monotonic()), True)
        else:
            typer.echo("Submission failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        if conn is not None:
            conn.close()


def _run_sync(
    ctx: typer.Context, workflow: RunStep, config_path: Path,
    db: Optional[Path], project: Optional[Path], json_output: bool = False,
) -> None:
    log = get_logger(__name__)
    started = time.monotonic()
    conn = None
    selected_log = _invocation_context(ctx).get("log_file")
    try:
        cfg, _ = resolve_config(config_path, context=_command_context(ctx, db, project))
        output_dir = Path(cfg.get("run", {}).get("output_dir", "."))
        invocation = _invocation_context(ctx)
        db_path = _command_context(ctx, db, project).resolve_database(config=cfg, config_path=config_path)
        if invocation.get("log_file") is None:
            dt_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            log_dir = output_dir / "run_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            selected_log = log_dir / ("%s__cfg-%s.log" % (dt_str, config_hash(cfg)))
            setup_logger(
                logfile=selected_log,
                verbose=invocation.get("verbose", False),
                quiet=invocation.get("quiet", False), no_color=invocation.get("no_color", False),
                json_mode=json_output,
            )
        conn = db_api.connect(db_path)
        db_api.init_schema(conn)
        task_class = TASK_REGISTRY.get(workflow.value)
        if task_class is None:
            raise ValueError("Task '%s' is not available in this build." % workflow.value)
        summary = task_class().exec(conn, cfg, verbose=invocation["verbose"])
        # Third-party/legacy Task implementations may still return None.
        if summary is not None:
            if selected_log is not None:
                summary.log_path = str(selected_log)
            _emit_summary(summary, json_output)
            code = summary_exit_code(summary)
            if code:
                raise typer.Exit(code=code)
    except typer.Exit:
        raise
    except (Exception, SystemExit) as exc:
        log.exception("Failed to execute task '%s': %s", workflow.value, exc)
        if json_output:
            db_path_value = None
            if conn is not None:
                db_path_value = str(Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve())
            _emit_summary(_failure_summary(workflow.value, exc, started=started,
                                           database_path=db_path_value), True)
        raise typer.Exit(code=1)
    finally:
        if conn is not None:
            conn.close()


@app.command(epilog=_workflow_epilog())
def run(
    ctx: typer.Context,
    workflow: Optional[RunStep] = typer.Argument(
        None, metavar="WORKFLOW", show_default=False,
        help="Scientific workflow to execute; see the list below."
    ),
    config_path: Optional[Path] = typer.Argument(
        None, metavar="CONFIG_PATH", exists=True, file_okay=True, dir_okay=False,
        readable=True, resolve_path=True, show_default=False, help="Run configuration file."
    ),
    detach: bool = typer.Option(False, "--detach", help="Submit a durable task and return without waiting."),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Executor profile used with --detach."),
    db: Optional[Path] = typer.Option(None, "--db", help="SQLite database for this run."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
    json_output: bool = typer.Option(False, "--json", help="Write only the JSON result to stdout."),
):
    """Run a scientific workflow synchronously or submit it with --detach."""
    if workflow is None or config_path is None:
        # Guide instead of erroring, like the command groups do. A bare 'nerd run'
        # is a request for help (exit 0); a half-typed invocation is incomplete.
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0 if workflow is None and config_path is None else 2)
    if profile and not detach:
        typer.echo("--profile is only supported with --detach.", err=True)
        raise typer.Exit(code=2)
    if detach:
        _submit_handler(ctx, workflow, config_path, profile, db, project, json_output)
    else:
        _run_sync(ctx, workflow, config_path, db, project, json_output)


def _open_existing(ctx, db: Optional[Path], project: Optional[Path], failure: str):
    try:
        return _scheduler_connection(context=_command_context(ctx, db, project), read_only=True)
    except ContextResolutionError as exc:
        typer.echo("%s failed: %s" % (failure, exc), err=True)
        raise typer.Exit(code=1)


def _action_handler(
    ctx: typer.Context, task_id: int, db: Optional[Path], project: Optional[Path],
    action: Callable, failure: str, *, detailed: bool = False, json_output: bool = False,
) -> None:
    if json_output:
        _set_json_logging(ctx)
    conn = _open_existing(ctx, db, project, failure)
    try:
        row = action(conn, task_id)
        if json_output:
            db_path = str(Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve())
            _emit_summary(_scheduler_summary(row, db_path), True)
        else:
            _show_scheduler_row(
                row, detailed=detailed,
                no_color=_invocation_context(ctx).get("no_color", False),
            )
    except Exception as exc:
        if json_output:
            _emit_summary(_failure_summary("task_%s" % failure.lower(), exc,
                                           started=time.monotonic()), True)
        else:
            typer.echo("%s failed: %s" % (failure, exc), err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


def _logs_handler(
    ctx: typer.Context, task_id: int, tail: int,
    db: Optional[Path], project: Optional[Path], unit: Optional[str] = None,
    failed_only: bool = False,
) -> None:
    from nerd.scheduler.service import task_logs

    conn = _open_existing(ctx, db, project, "Logs")
    try:
        typer.echo(task_logs(conn, task_id, tail, unit=unit, failed_only=failed_only))
    except Exception as exc:
        typer.echo("Logs failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()


def _list_handler(
    ctx: typer.Context, label: Optional[str], state: Optional[str], workflow: Optional[RunStep],
    limit: int, db: Optional[Path], project: Optional[Path], json_output: bool = False,
    refresh: bool = True,
) -> None:
    from nerd.scheduler.service import list_tasks

    if json_output:
        _set_json_logging(ctx)
    conn = _open_existing(ctx, db, project, "List")
    try:
        rows = list_tasks(
            conn, label=label, state=state,
            task_name=workflow.value if workflow else None, limit=limit, refresh=refresh,
        )
        if json_output:
            payload = {
                "schema_version": "1.0",
                "tasks": [{
                    "task_id": row["id"], "workflow": row["task_name"], "label": row["label"],
                    "status": row["state"], "attempt": row["try_index"],
                    "executor": row["executor_profile"], "scheduler_id": row["scheduler_id"],
                    "started_at": row["started_at"], "ended_at": row["ended_at"],
                    "completed_units": row["completed_units"], "total_units": row["total_units"],
                } for row in rows],
            }
            typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
            return
        if not rows:
            typer.echo("No tasks found.")
            return
        typer.echo("ID\tTASK\tLABEL\tSTATE\tPROGRESS\tATTEMPT\tEXECUTOR\tSCHEDULER ID")
        for row in rows:
            total = int(row["total_units"] or 0)
            progress = "%s/%s" % (row["completed_units"], total) if total else "-"
            typer.echo("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
                row["id"], row["task_name"], row["label"], row["state"],
                progress, row["try_index"] or "-", row["executor_profile"] or "-", row["scheduler_id"] or "-",
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
    json_output: bool = typer.Option(False, "--json", help="Write only JSON to stdout."),
    no_refresh: bool = typer.Option(False, "--no-refresh", help="Show stored state without contacting executors."),
):
    """List durable tasks, newest first."""
    _list_handler(ctx, label, state, workflow, limit, db, project, json_output,
                  refresh=not no_refresh)


@task_app.command("show")
def task_show(
    ctx: typer.Context,
    task_id: int = typer.Argument(..., min=1, help="Durable task ID."),
    db: Optional[Path] = typer.Option(None, "--db", help="Existing controller database."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
    details: bool = typer.Option(False, "--details", help="Show internal identifiers and filesystem paths."),
    json_output: bool = typer.Option(False, "--json", help="Write only JSON to stdout."),
):
    """Reconcile and show task status and next actions."""
    from nerd.scheduler.service import reconcile
    _action_handler(ctx, task_id, db, project, reconcile, "Show", detailed=details,
                    json_output=json_output)


@task_app.command("logs")
def task_logs_command(
    ctx: typer.Context,
    task_id: int = typer.Argument(..., min=1, help="Durable task ID."),
    tail: int = typer.Option(100, "--tail", "-n", min=0, help="Number of lines to show."),
    db: Optional[Path] = typer.Option(None, "--db", help="Existing controller database."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
    unit: Optional[str] = typer.Option(None, "--unit", help="Work-unit key or label for a batch task."),
    failed_only: bool = typer.Option(False, "--failed", help="Show logs only for failed work units."),
):
    """Read logs for the latest task attempt."""
    _logs_handler(ctx, task_id, tail, db, project, unit=unit, failed_only=failed_only)


@task_app.command("wait")
def task_wait(
    ctx: typer.Context,
    task_id: int = typer.Argument(..., min=1, help="Durable task ID."),
    collect: bool = typer.Option(False, "--collect", help="Collect successful output before returning."),
    poll_interval: float = typer.Option(1.0, "--poll-interval", min=0.05, help="Seconds between checks."),
    db: Optional[Path] = typer.Option(None, "--db", help="Existing controller database."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
    json_output: bool = typer.Option(False, "--json", help="Write only JSON to stdout."),
):
    """Wait for scheduling to finish; optionally collect successful output."""
    from nerd.scheduler.service import wait_for_task
    action = lambda conn, task_id: wait_for_task(
        conn, task_id, collect=collect, poll_interval=poll_interval
    )
    _action_handler(ctx, task_id, db, project, action, "Wait", detailed=False,
                    json_output=json_output)


@task_app.command("watch")
def task_watch(
    ctx: typer.Context,
    task_id: int = typer.Argument(..., min=1, help="Durable task ID."),
    collect: bool = typer.Option(False, "--collect", help="Collect successful work units as they finish."),
    poll_interval: float = typer.Option(2.0, "--poll-interval", min=0.05, help="Seconds between checks."),
    db: Optional[Path] = typer.Option(None, "--db", help="Existing controller database."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
):
    """Watch scheduler progress in the foreground without owning the remote jobs."""
    from nerd.scheduler.service import watch_task

    conn = _open_existing(ctx, db, project, "Watch")
    last_signature = None

    def display(row) -> None:
        nonlocal last_signature
        children = _row_value(row, "children", []) or []
        signature = (
            row["task_state"],
            tuple((child["task_id"], child["task_state"], child["scheduler_state"])
                  for child in children),
        )
        if signature == last_signature:
            return
        if last_signature is not None:
            typer.echo("")
        _show_scheduler_row(
            row, detailed=False,
            no_color=_invocation_context(ctx).get("no_color", False),
        )
        last_signature = signature

    try:
        row = watch_task(
            conn, task_id, collect=collect, poll_interval=poll_interval,
            on_update=display,
        )
    except KeyboardInterrupt:
        typer.echo("\nStopped watching; remote jobs continue running.", err=True)
        raise typer.Exit(code=130)
    except Exception as exc:
        typer.echo("Watch failed: %s" % exc, err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()
    counts = _row_value(row, "counts", {}) or {}
    if row["task_state"] == "partial_success" or int(counts.get("failed", 0)) > 0:
        raise typer.Exit(code=2)
    if row["task_state"] in {"failed", "cancelled"}:
        raise typer.Exit(code=1)


def _lifecycle_command(name: str, action_name: str, help_text: str):
    def command(
        ctx: typer.Context,
        task_id: int = typer.Argument(..., min=1, help="Durable task ID."),
        db: Optional[Path] = typer.Option(None, "--db", help="Existing controller database."),
        project: Optional[Path] = typer.Option(None, "--project", help="Project directory."),
        json_output: bool = typer.Option(False, "--json", help="Write only JSON to stdout."),
    ):
        from nerd.scheduler import service
        _action_handler(ctx, task_id, db, project, getattr(service, action_name), name.title(),
                        json_output=json_output)

    command.__name__ = "task_%s" % name
    command.__doc__ = help_text
    task_app.command(name)(command)
    return command


task_cancel = _lifecycle_command("cancel", "cancel_task", "Request cancellation of a task.")
task_collect = _lifecycle_command("collect", "collect_task", "Collect and validate completed output.")
task_retry = _lifecycle_command("retry", "retry_task", "Retry a failed or cancelled task.")


def _container_cli_context(
    config_path: Path,
    profile_name: Optional[str],
    context: Optional[ProjectContext] = None,
):
    from nerd.containers import container_requested, shapemapper_container_spec
    from nerd.scheduler.profiles import load_executor_profile

    cfg, _ = resolve_config(config_path, context=context, executor=profile_name)
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


def _readiness_summary(result, spec, workflow: str) -> TaskSummary:
    passed = sum(1 for check in result.checks if check["ok"])
    failures = [TaskIssue("container_" + str(check["name"]), str(check["message"]))
                for check in result.checks if not check["ok"]]
    runtime = result.runtime
    return TaskSummary(
        status="success" if result.ready else "failed", workflow=workflow,
        engine=runtime.command if runtime else None,
        version=runtime.version if runtime else None,
        counts={"attempted": len(result.checks), "succeeded": passed,
                "failed": len(result.checks) - passed, "skipped": 0,
                "checks_total": len(result.checks), "checks_passed": passed},
        metrics={"execution_host": result.execution_host, "architecture": result.architecture,
                 "executor": result.profile, "executor_type": result.executor_type,
                 "oci_reference": spec.oci_reference, "image_digest": spec.digest,
                 "sif_checksum": result.sif_checksum},
        failures=failures,
        artifacts=([ArtifactReference("sif", str(result.sif_path), exists=True)]
                   if result.sif_path else []),
        next_actions=(["nerd image prepare shapemapper CONFIG"] if not result.ready else []),
    )


def _inspect_shapemapper(
    config_path: Path,
    profile: Optional[str],
    json_output: bool = False,
    context: Optional[ProjectContext] = None,
) -> None:
    from nerd.containers import inspect_container
    try:
        spec, executor_profile, tool_cfg = _container_cli_context(config_path, profile, context)
        result = inspect_container(spec, executor_profile, tool_cfg)
        if json_output:
            _emit_summary(_readiness_summary(result, spec, "container_inspect"), True)
        else:
            _show_readiness(result)
        if not result.ready:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        if json_output:
            _emit_summary(_failure_summary("container_inspect", exc, started=time.monotonic()), True)
        else:
            typer.echo("Inspection failed: %s" % exc, err=True)
        raise typer.Exit(code=1)


def _prepare_shapemapper(
    config_path: Path,
    profile: Optional[str],
    json_output: bool = False,
    context: Optional[ProjectContext] = None,
) -> None:
    from nerd.containers import prepare_container
    try:
        spec, executor_profile, tool_cfg = _container_cli_context(config_path, profile, context)
        result = prepare_container(spec, executor_profile, tool_cfg)
        if json_output:
            _emit_summary(_readiness_summary(result, spec, "container_prepare"), True)
        else:
            _show_readiness(result)
            typer.echo("sif: %s" % result.sif_path)
            typer.echo("sif_sha256: %s" % result.sif_checksum)
    except Exception as exc:
        if json_output:
            _emit_summary(_failure_summary("container_prepare", exc, started=time.monotonic()), True)
        else:
            typer.echo("Preparation failed: %s" % exc, err=True)
        raise typer.Exit(code=1)


@plugin_doctor_app.command("shapemapper")
def plugin_doctor_shapemapper(
    ctx: typer.Context,
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="ShapeMapper run configuration file."
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Executor profile."),
    json_output: bool = typer.Option(False, "--json", help="Write only JSON to stdout."),
):
    """Check ShapeMapper runtime and immutable-image readiness."""
    _inspect_shapemapper(
        config_path, profile, json_output, _command_context(ctx, None, None)
    )


@image_app.command("inspect")
def image_inspect(
    ctx: typer.Context,
    plugin: ContainerPlugin = typer.Argument(..., help="Containerized plugin."),
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="Plugin run configuration file."
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Executor profile."),
    json_output: bool = typer.Option(False, "--json", help="Write only JSON to stdout."),
):
    """Inspect an immutable plugin image without preparing it."""
    _inspect_shapemapper(
        config_path, profile, json_output, _command_context(ctx, None, None)
    )


@image_app.command("prepare")
def image_prepare(
    ctx: typer.Context,
    plugin: ContainerPlugin = typer.Argument(..., help="Containerized plugin."),
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="Plugin run configuration file."
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Executor profile."),
    json_output: bool = typer.Option(False, "--json", help="Write only JSON to stdout."),
):
    """Prepare and smoke-test an immutable plugin image."""
    _prepare_shapemapper(
        config_path, profile, json_output, _command_context(ctx, None, None)
    )


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


@app.command("init")
def project_init(
    ctx: typer.Context,
    path: Path = typer.Argument(Path("."), metavar="PATH", help="Directory to initialize."),
    name: Optional[str] = typer.Option(
        None, "--name",
        help="Project name (defaults to the directory name; Lucks Lab convention: EKC.07.00.000).",
    ),
    existing: bool = typer.Option(False, "--existing", help="Allow initialization inside an existing non-empty directory."),
    output_dir: str = typer.Option("outputs", "--output-dir", help="Project-root-relative output directory."),
    database: str = typer.Option(".nerd/nerd.sqlite", "--database", help="Project-root-relative database path."),
    json_output: bool = typer.Option(False, "--json", help="Write only JSON to stdout."),
):
    """Initialize a NERD project without overwriting an existing project."""
    if json_output:
        _set_json_logging(ctx)
    target = path.expanduser().resolve()
    try:
        inferred = target.name or None
        if name is None and inferred is None:
            raise ProjectConfigError(
                "Cannot infer a project name from directory %r. Pass --name NAME."
                % target.name
            )
        project_name = validate_project_name(name if name is not None else inferred or "")
        if target.exists() and not target.is_dir():
            raise ProjectConfigError("Initialization target is not a directory: %s" % target)
        if target.exists() and any(target.iterdir()) and not existing:
            raise ProjectConfigError(
                "%s is not empty. Re-run with --existing to add only NERD project assets safely." % target
            )
        project_file = target / ".nerd" / "project.toml"
        if project_file.exists():
            raise ProjectConfigError(
                "A NERD project already exists at %s; refusing to overwrite it." % project_file
            )
        target.mkdir(parents=True, exist_ok=True)
        project_file.parent.mkdir(parents=True, exist_ok=True)
        project_file.write_text(
            render_project_toml(project_name, database=database, output=output_dir),
            encoding="utf-8",
        )
        project_cfg = load_project(project_file)
        project_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        conn = db_api.connect(project_cfg.database)
        try:
            db_api.init_schema(conn)
        finally:
            conn.close()
        payload = {
            "schema_version": "1.0", "status": "initialized",
            "project_id": project_cfg.name, "project_root": str(project_cfg.root),
            "database": str(project_cfg.database), "output_directory": str(project_cfg.output_dir),
            "next_command": "nerd config init create --output configs/create.yaml",
        }
        if json_output:
            typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            typer.echo("project_id: %s" % payload["project_id"])
            typer.echo("project_root: %s" % payload["project_root"])
            typer.echo("database: %s" % payload["database"])
            typer.echo("output_directory: %s" % payload["output_directory"])
            typer.echo("next: %s" % payload["next_command"])
    except (ContextResolutionError, ProjectConfigError, OSError) as exc:
        typer.echo("Initialization failed: %s" % exc, err=True)
        raise typer.Exit(code=1)


def _config_context(ctx: typer.Context, project: Optional[Path]) -> ProjectContext:
    return _command_context(ctx, None, project)


@config_app.command("validate")
def config_validate(
    ctx: typer.Context,
    file: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    workflow: Optional[RunStep] = typer.Option(None, "--workflow", help="Expected workflow."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project root or project.toml."),
    json_output: bool = typer.Option(False, "--json", help="Write only JSON to stdout."),
):
    """Validate an analysis config without creating a database or output files."""
    try:
        cfg, project_cfg = resolve_config(file, context=_config_context(ctx, project))
        chosen = validate_config(cfg, workflow=workflow.value if workflow else None)
        payload = {
            "schema_version": "1.0", "status": "valid", "workflow": chosen,
            "config_file": str(cfg.source_path),
            "project_id": project_cfg.name if project_cfg else None,
        }
        if json_output:
            typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            typer.echo("Valid configuration: %s (%s)" % (cfg.source_path, chosen))
    except (ContextResolutionError, ConfigValidationError, TypeError, ValueError, OSError, yaml.YAMLError) as exc:
        if json_output:
            typer.echo(json.dumps({
                "schema_version": "1.0", "status": "invalid", "errors": [str(exc)]
            }, indent=2, ensure_ascii=False))
        else:
            typer.echo("Configuration invalid: %s" % exc, err=True)
        raise typer.Exit(code=1)


@config_app.command("show")
def config_show(
    ctx: typer.Context,
    file: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    resolved: bool = typer.Option(False, "--resolved", help="Show project defaults and resolved paths."),
    workflow: Optional[RunStep] = typer.Option(None, "--workflow", help="Expected workflow."),
    project: Optional[Path] = typer.Option(None, "--project", help="Project root or project.toml."),
    json_output: bool = typer.Option(False, "--json", help="Write only JSON to stdout."),
):
    """Show authored or fully resolved config without mutating project state."""
    try:
        command_context = _config_context(ctx, project)
        cfg, project_cfg = resolve_config(file, context=command_context)
        chosen = validate_config(cfg, workflow=workflow.value if workflow else None)
        if not resolved:
            payload = redact(cfg.hash_data)
        else:
            payload = resolved_view(cfg, project_cfg, chosen)
            try:
                payload["database"] = str(command_context.resolve_database(
                    config=cfg, config_path=file, must_exist=False
                ))
            except ContextResolutionError:
                pass
        if json_output:
            typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        elif not resolved:
            typer.echo(yaml.safe_dump(payload, sort_keys=False), nl=False)
        else:
            typer.echo("project_id: %s" % (payload["project"]["id"] or "-"))
            typer.echo("project_root: %s" % (payload["project"]["root"] or "-"))
            typer.echo("database: %s" % (payload.get("database") or "-"))
            typer.echo("output_directory: %s" % payload["output_directory"])
            typer.echo("config_base: %s" % payload["config_base"])
            typer.echo("workflow: %s" % payload["workflow"])
            typer.echo("task_label: %s" % payload["task_label"])
            typer.echo("executor: %s (%s)" % (
                payload["executor"]["name"], payload["executor"]["type"]
            ))
            typer.echo("plugin: %s" % (payload["plugin"] or "-"))
            typer.echo("engine: %s" % (payload["engine"] or "-"))
            typer.echo("resolved_input_paths:")
            typer.echo(yaml.safe_dump(payload["resolved_input_paths"], sort_keys=True), nl=False)
            typer.echo("configured_values:")
            typer.echo(yaml.safe_dump(payload["configured"], sort_keys=False), nl=False)
            typer.echo("inherited_defaults:")
            typer.echo(yaml.safe_dump(payload["inherited_defaults"], sort_keys=False), nl=False)
    except (ContextResolutionError, ConfigValidationError, TypeError, ValueError, OSError, yaml.YAMLError) as exc:
        if json_output:
            typer.echo(json.dumps({
                "schema_version": "1.0", "status": "invalid", "errors": [str(exc)]
            }, indent=2, ensure_ascii=False))
        else:
            typer.echo("Configuration show failed: %s" % exc, err=True)
        raise typer.Exit(code=1)


@config_app.command("init")
def config_init(
    workflow: RunStep = typer.Argument(..., help="Workflow starter to generate."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="YAML output path."),
):
    """Generate a safe starter YAML and, for create, a companion sample CSV."""
    destination = output or Path("%s.yaml" % workflow.value)
    try:
        written, companion = write_template(workflow.value, destination)
        cfg, _ = resolve_config(written)
        validate_config(cfg, workflow=workflow.value)
        typer.echo("config: %s" % written)
        if companion:
            typer.echo("sample_sheet: %s" % companion)
        typer.echo("next: nerd config validate %s" % written)
    except (ConfigValidationError, ContextResolutionError, OSError, ValueError) as exc:
        typer.echo("Config initialization failed: %s" % exc, err=True)
        raise typer.Exit(code=1)


app.add_typer(task_app, name="task")
plugin_app.add_typer(plugin_doctor_app, name="doctor")
app.add_typer(plugin_app, name="plugin")
app.add_typer(image_app, name="image")
app.add_typer(db_app, name="db")
app.add_typer(config_app, name="config")
app.add_typer(webui_app, name="webui")



def _ensure_webui_port_available(host: str, port: int, mode: str) -> None:
    """Fail before opening a browser when another Web UI owns the address."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise ContextResolutionError(
            "Port %s is already in use, so the %s Web UI was not started. "
            "Close the existing server or choose another port with --port."
            % (port, mode)
        ) from exc
    finally:
        probe.close()


def _launch_webui(
    ctx: typer.Context,
    mode: str,
    project: Optional[Path],
    db: Optional[Path],
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    """Shared startup for the create, edit, and view workspaces."""
    try:
        import uvicorn
    except ImportError:
        typer.echo(
            "The webui extra isn't installed. Run: pip install -e \".[webui]\"",
            err=True,
        )
        raise typer.Exit(code=1)

    invocation_context = _command_context(ctx, db, project)
    selected_project = invocation_context.project
    try:
        _ensure_webui_port_available(host, port, mode)
        if selected_project is None:
            discovered = invocation_context.resolve_project()
            if discovered is None:
                raise ContextResolutionError(
                    "No NERD project found. Pass --project PATH or run this command inside a Phase 4 project."
                )
            selected_project = discovered.root
        selected_db = invocation_context.db
        if selected_db is not None:
            selected_db = selected_db.expanduser().resolve()

        import nerd.webui.app as webui_module
        webui_module.set_mode(mode)
        info = webui_module.session.connect(
            str(selected_project), str(selected_db) if selected_db else None,
            webui_module.session.label,
            read_only=mode in {"view", "analyze"},
        )
    except (ContextResolutionError, ProjectConfigError, OSError, ValueError) as exc:
        typer.echo("Web UI startup failed: %s" % exc, err=True)
        raise typer.Exit(code=1)

    typer.echo("Serving nerd Web UI (%s) at http://%s:%s" % (mode, host, port))
    typer.echo("project: %s" % info["project_dir"])
    typer.echo("database: %s" % info["db_path"])
    typer.echo("output_directory: %s" % info["output_dir"])

    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()

    server = uvicorn.Server(uvicorn.Config(
        webui_module.app, host=host, port=port, log_level="info",
    ))
    webui_module.set_server(server)
    try:
        server.run()
    finally:
        webui_module.set_server(None)


@webui_app.command("create")
def webui_create(
    ctx: typer.Context,
    project: Optional[Path] = typer.Option(
        None, "--project", "-p",
        help="Project root or direct .nerd/project.toml; defaults to project discovery.",
    ),
    db: Optional[Path] = typer.Option(
        None, "--db", help="Database override; otherwise use project.toml or legacy discovery."
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(WEBUI_DEFAULT_PORTS["create"], "--port"),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser"),
) -> None:
    """Build sample sheets and create configurations in the Web UI."""
    _launch_webui(ctx, "create", project, db, host, port, open_browser)


@webui_app.command("edit")
def webui_edit(
    ctx: typer.Context,
    project: Optional[Path] = typer.Option(
        None, "--project", "-p",
        help="Project root or direct .nerd/project.toml; defaults to project discovery.",
    ),
    db: Optional[Path] = typer.Option(
        None, "--db", help="Database override; otherwise use project.toml or legacy discovery."
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(WEBUI_DEFAULT_PORTS["edit"], "--port"),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser"),
) -> None:
    """Correct existing database entries in the maintenance Web UI."""
    _launch_webui(ctx, "edit", project, db, host, port, open_browser)


@webui_app.command("view")
def webui_view(
    ctx: typer.Context,
    project: Optional[Path] = typer.Option(
        None, "--project", "-p",
        help="Project root or direct .nerd/project.toml; defaults to project discovery.",
    ),
    db: Optional[Path] = typer.Option(
        None, "--db", help="Database override; otherwise use project.toml or legacy discovery."
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(WEBUI_DEFAULT_PORTS["view"], "--port"),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser"),
) -> None:
    """Browse project database metadata in a read-only Web UI."""
    _launch_webui(ctx, "view", project, db, host, port, open_browser)


@webui_app.command("analyze")
def webui_analyze(
    ctx: typer.Context,
    project: Optional[Path] = typer.Option(
        None, "--project", "-p",
        help="Project root or direct .nerd/project.toml; defaults to project discovery.",
    ),
    db: Optional[Path] = typer.Option(
        None, "--db", help="Database override; otherwise use project.toml or legacy discovery."
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(WEBUI_DEFAULT_PORTS["analyze"], "--port"),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser"),
) -> None:
    """Interactively compare probing modification rates and time courses."""
    _launch_webui(ctx, "analyze", project, db, host, port, open_browser)


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
    _action_handler(ctx, task_id, db, project, reconcile, "Status", detailed=False)


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
    ctx: typer.Context,
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p"),
):
    """Deprecated compatibility wrapper for plugin doctor shapemapper."""
    _deprecated("doctor", "plugin doctor shapemapper")
    _inspect_shapemapper(
        config_path, profile, context=_command_context(ctx, None, None)
    )


@app.command("prepare-image", hidden=True)
def legacy_prepare_image(
    ctx: typer.Context,
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p"),
):
    """Deprecated compatibility wrapper for image prepare shapemapper."""
    _deprecated("prepare-image", "image prepare shapemapper")
    _prepare_shapemapper(
        config_path, profile, context=_command_context(ctx, None, None)
    )


if __name__ == "__main__":
    app()
