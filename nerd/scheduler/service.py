"""Scientific-task orchestration over durable asynchronous executors."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import time
import yaml
from typing import Any, Callable, Dict, List, Optional

from nerd.db import api as db_api
from nerd.pipeline.tasks import TASK_REGISTRY
from nerd.pipeline.tasks.base import TaskContext
from nerd.utils.config import load_config
from nerd.utils.hashing import config_hash
from nerd.utils.paths import make_run_dir, update_latest_symlink

from .executors import executor_for
from .models import AttemptState, JobHandle, JobSpec, JobStatus, TaskState
from .profiles import ExecutorProfile, load_executor_profile, profile_resources
from . import store


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(previous))


def _task(step: str):
    task_class = TASK_REGISTRY.get(step)
    if task_class is None:
        raise ValueError("Task '%s' is not available in this build." % step)
    return task_class()


def _context(
    conn: sqlite3.Connection,
    cfg: Dict[str, Any],
    workdir: Path,
    profile: ExecutorProfile,
    resources: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> TaskContext:
    run = cfg.get("run", {}) or {}
    memory = resources.get("memory", run.get("mem_gb", 32))
    try:
        memory_int = int(str(memory).rstrip("GgMm"))
    except ValueError:
        memory_int = 32
    return TaskContext(
        db=conn,
        backend="ssh_slurm" if profile.executor_type == "ssh_slurm" else "local",
        workdir=workdir,
        threads=int(resources.get("cpus", run.get("threads", 8))),
        mem_gb=memory_int,
        time=str(resources.get("time", run.get("time", "02:00:00"))),
        label=str(run["label"]),
        output_dir=output_dir or str(run.get("output_dir", "nerd_output")),
        executor_profile=profile.name,
    )


def _job_spec(task: Any, ctx: TaskContext, cfg: Dict[str, Any], command: str,
              profile: ExecutorProfile, resources: Dict[str, Any],
              provenance: Optional[Dict[str, Any]] = None) -> JobSpec:
    run = cfg.get("run", {}) or {}
    options = profile.options
    remote_workdir = None
    if profile.executor_type == "ssh_slurm":
        base = options.get("remote_base_dir")
        if not base:
            raise ValueError("ssh_slurm profile '%s' requires remote_base_dir." % profile.name)
        remote_workdir = str(PurePosixPath(str(base)) / ctx.workdir.name)
    preamble = options.get("preamble")
    env = {str(k): str(v) for k, v in (run.get("env") or {}).items()}
    env.update({str(k): str(v) for k, v in (options.get("env") or {}).items()})
    stage_in = []
    if hasattr(task, "stage_in_pairs"):
        stage_in = list(task.stage_in_pairs() or [])
    stage_out = []
    configured = options.get("stage_out") or []
    if isinstance(configured, str):
        stage_out.extend(item.strip() for item in configured.split(",") if item.strip())
    else:
        stage_out.extend(str(item) for item in configured)
    stage_out.extend(task.stage_out_patterns() or [])
    return JobSpec(
        command=command,
        workdir=ctx.workdir,
        env=env,
        resources=resources,
        remote_workdir=remote_workdir,
        preamble=preamble,
        stage_in=stage_in,
        stage_out=list(dict.fromkeys(stage_out)),
        controller_cwd=str(getattr(cfg, "base_dir", Path.cwd())),
        provenance=provenance or {},
    )


def _submit_attempt(
    conn: sqlite3.Connection,
    task_id: int,
    spec: JobSpec,
    profile: ExecutorProfile,
    config_path: Path,
) -> sqlite3.Row:
    row = store.create_attempt(conn, task_id, profile.name, profile.executor_type, spec, config_path)
    scheduler_attempt_id = int(row["scheduler_attempt_id"])
    store.transition_attempt(conn, scheduler_attempt_id, AttemptState.SUBMITTING)
    try:
        handle = executor_for(profile).submit(spec)
    except Exception as exc:
        status = JobStatus(AttemptState.SUBMISSION_FAILED, message=str(exc))
        store.transition_attempt(conn, scheduler_attempt_id, AttemptState.SUBMISSION_FAILED, status=status)
        store.transition_task(conn, task_id, TaskState.FAILED, "Submission failed: %s" % exc)
        raise
    store.transition_attempt(conn, scheduler_attempt_id, AttemptState.QUEUED, handle=handle)
    store.transition_task(conn, task_id, TaskState.SUBMITTED)
    return store.get_attempt(conn, scheduler_attempt_id)


def _submit_unit(
    conn: sqlite3.Connection,
    step: str,
    cfg: Dict[str, Any],
    profile: ExecutorProfile,
    resources: Dict[str, Any],
    run_dir: Path,
    *,
    parent_task_id: Optional[int] = None,
    unit_key: str = "main",
    unit_label: str = "main",
    unit_index: int = 0,
    scope_kind: Optional[str] = None,
    scope_id: Optional[int] = None,
) -> sqlite3.Row:
    task = _task(step)
    inputs, params = task.prepare(cfg)
    if hasattr(task, "validate_execution_config"):
        task.validate_execution_config(inputs, profile)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = run_dir / ".nerd-submitted-config.yaml"
    config_snapshot.write_text(yaml.safe_dump(dict(cfg), sort_keys=False), encoding="utf-8")
    output_dir = str(Path(str((cfg.get("run") or {}).get("output_dir", "nerd_output"))).resolve())
    ctx = _context(conn, cfg, run_dir, profile, resources, output_dir=output_dir)
    execution_provenance = None
    if hasattr(task, "prepare_execution"):
        execution_provenance = task.prepare_execution(cfg, profile, ctx, inputs)
    batch_remote_workdir = None
    if parent_task_id is not None and profile.executor_type == "ssh_slurm":
        batch_remote_workdir = str(
            PurePosixPath(str(profile.options.get("remote_base_dir")))
            / run_dir.parent.name / run_dir.name
        )
        container_execution = inputs.get("_container_execution") if isinstance(inputs, dict) else None
        if isinstance(container_execution, dict):
            container_execution["workdir"] = batch_remote_workdir
        if isinstance(execution_provenance, dict):
            execution_provenance["workdir"] = batch_remote_workdir
    scope = task.resolve_scope(ctx, inputs)
    if scope_kind is not None:
        scope.kind = scope_kind
        scope.scope_id = scope_id
        scope.label = unit_label
    unit_cache_key = config_hash(cfg, length=64)
    task_id = db_api.begin_task(
        conn, task.name, scope.kind, scope.scope_id, profile.name, output_dir,
        str((cfg.get("run") or {})["label"]), unit_cache_key,
        tool=task.task_tool(inputs), tool_version=task.task_tool_version(inputs),
    )
    if task_id is None:
        raise RuntimeError("Could not persist the task unit.")
    if parent_task_id is not None:
        store.attach_child(conn, task_id, parent_task_id, unit_key, unit_label, unit_index)
    if scope.members:
        db_api.record_task_scope_members(conn, task_id, scope.members)
    command = task.command(ctx, inputs, params)
    if execution_provenance is not None:
        execution_provenance["command"] = command
        from nerd.containers import write_provenance
        write_provenance(run_dir / ".nerd-container-provenance.json", execution_provenance)
        store.record_container_provenance(conn, task_id, execution_provenance)
    if not command:
        spec = JobSpec(
            command="", workdir=run_dir, resources=resources,
            controller_cwd=str(getattr(cfg, "base_dir", Path.cwd())),
        )
        row = store.create_attempt(conn, task_id, "controller", "controller", spec, config_snapshot)
        scheduler_attempt_id = int(row["scheduler_attempt_id"])
        store.transition_attempt(conn, scheduler_attempt_id, AttemptState.COLLECTING,
                                 handle=JobHandle("controller"))
        try:
            task.consume_outputs(ctx, inputs, params, run_dir, task_id=task_id)
        except Exception as exc:
            status = JobStatus(AttemptState.VALIDATION_FAILED, message=str(exc))
            store.transition_attempt(conn, scheduler_attempt_id, AttemptState.VALIDATION_FAILED,
                                     status=status, collected=True)
            store.transition_task(conn, task_id, TaskState.FAILED,
                                  "Output validation failed: %s" % exc)
            raise
        status = JobStatus(AttemptState.COMPLETED, exit_code=0)
        store.transition_attempt(conn, scheduler_attempt_id, AttemptState.COMPLETED,
                                 status=status, collected=True)
        store.transition_task(conn, task_id, TaskState.COMPLETED)
        return store.get_attempt(conn, scheduler_attempt_id)
    spec = _job_spec(task, ctx, cfg, command, profile, resources, execution_provenance)
    if batch_remote_workdir is not None:
        spec.remote_workdir = batch_remote_workdir
    return _submit_attempt(conn, task_id, spec, profile, config_snapshot)


def submit_task(
    conn: sqlite3.Connection,
    step: str,
    config_path: Path,
    profile_name: Optional[str] = None,
    resolved_config: Optional[Dict[str, Any]] = None,
) -> sqlite3.Row:
    cfg = resolved_config if resolved_config is not None else load_config(config_path)
    run = cfg.get("run", {}) or {}
    if not run.get("label"):
        raise ValueError("Configuration must contain a 'run.label'.")
    configured_output_dir = str(run.get("output_dir", "nerd_output"))
    output_dir = str(Path(configured_output_dir).expanduser().resolve())
    profiles = cfg.get("executors") or run.get("executors") or {}
    for candidate in profiles:
        load_executor_profile(cfg, str(candidate))
    profile = load_executor_profile(cfg, profile_name)
    task = _task(step)
    cache_key = config_hash(cfg, length=64)
    block = cfg.get(task.name) or {}
    force = isinstance(block, dict) and bool(block.get("force_run") or block.get("overwrite"))
    existing = db_api.find_completed_task_by_signature(conn, run["label"], output_dir, cache_key)
    if existing is None and configured_output_dir != output_dir:
        existing = db_api.find_completed_task_by_signature(
            conn, run["label"], configured_output_dir, cache_key
        )
    if existing is not None and not force:
        message = "Identical configuration previously completed as task_id=%s; skipping." % existing["id"]
        cached_id = db_api.record_cached_task(
            conn, task.name, task.scope_kind, None, profile.name, output_dir,
            str(run["label"]), cache_key, message,
            tool=task.task_tool(None), tool_version=task.task_tool_version(None),
        )
        if cached_id is None:
            raise RuntimeError("Could not persist cached task result.")
        cached_dir = Path(output_dir) / str(run["label"]) / task.name / "cached"
        spec = JobSpec(command="", workdir=cached_dir,
                       controller_cwd=str(getattr(cfg, "base_dir", Path.cwd())))
        row = store.create_attempt(conn, cached_id, "controller", "controller", spec, config_path)
        store.transition_attempt(
            conn, int(row["scheduler_attempt_id"]), AttemptState.COMPLETED,
            handle=JobHandle("cache"), status=JobStatus(AttemptState.COMPLETED, exit_code=0),
            collected=True,
        )
        return store.get_attempt(conn, int(row["scheduler_attempt_id"]))
    inputs, params = task.prepare(cfg)
    if hasattr(task, "validate_execution_config"):
        task.validate_execution_config(inputs, profile)
    suffix = "%s__cfg-%s" % (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        config_hash(cfg, length=7),
    )
    run_dir = make_run_dir(Path(output_dir) / str(run["label"]), task.name, suffix=suffix)
    resources = profile_resources(profile, run)
    planning_ctx = _context(conn, cfg, run_dir, profile, resources, output_dir=output_dir)
    units = task.plan_work_units(planning_ctx, cfg, inputs, params)
    if len(units) == 1:
        return _submit_unit(
            conn, step, units[0].config, profile, resources, run_dir,
            scope_kind=units[0].scope_kind, scope_id=units[0].scope_id,
            unit_key=units[0].key, unit_label=units[0].label,
        )

    parent_scope = task.resolve_scope(planning_ctx, inputs)
    parent_id = db_api.begin_task(
        conn, task.name, "work_batch", None, profile.name, output_dir,
        str(run["label"]), cache_key, tool=task.task_tool(inputs),
        tool_version=task.task_tool_version(inputs),
    )
    if parent_id is None:
        raise RuntimeError("Could not persist the parent task.")
    if parent_scope.members:
        db_api.record_task_scope_members(conn, parent_id, parent_scope.members)
    (run_dir / ".nerd-submitted-config.yaml").write_text(
        yaml.safe_dump(dict(cfg), sort_keys=False), encoding="utf-8"
    )
    store.transition_task(conn, parent_id, TaskState.SUBMITTED,
                          "%s work units planned" % len(units))
    for index, unit in enumerate(units):
        unit_dir = run_dir / unit.key
        try:
            _submit_unit(
                conn, step, unit.config, profile, resources, unit_dir,
                parent_task_id=parent_id, unit_key=unit.key,
                unit_label=unit.label, unit_index=index,
                scope_kind=unit.scope_kind, scope_id=unit.scope_id,
            )
        except Exception as exc:
            # Submission failures are already durable once an attempt exists. Keep
            # submitting independent units so one bad group does not block others.
            existing_child = conn.execute(
                "SELECT id FROM core_tasks WHERE parent_task_id=? AND unit_key=?",
                (parent_id, unit.key),
            ).fetchone()
            if existing_child is None:
                failed_id = db_api.begin_task(
                    conn, task.name, unit.scope_kind or "work_unit", unit.scope_id,
                    profile.name, output_dir, str(run["label"]),
                    config_hash(unit.config, length=64), tool=task.task_tool(inputs),
                    tool_version=task.task_tool_version(inputs),
                )
                if failed_id is not None:
                    store.attach_child(
                        conn, failed_id, parent_id, unit.key, unit.label, index
                    )
                    unit_dir.mkdir(parents=True, exist_ok=True)
                    failed_config = unit_dir / ".nerd-submitted-config.yaml"
                    failed_config.write_text(
                        yaml.safe_dump(dict(unit.config), sort_keys=False), encoding="utf-8"
                    )
                    failed_spec = JobSpec(command="", workdir=unit_dir, resources=resources)
                    attempt = store.create_attempt(
                        conn, failed_id, profile.name, profile.executor_type,
                        failed_spec, failed_config,
                    )
                    failed_status = JobStatus(
                        AttemptState.SUBMISSION_FAILED,
                        message="Unit preparation/submission failed: %s" % exc,
                    )
                    store.transition_attempt(
                        conn, int(attempt["scheduler_attempt_id"]),
                        AttemptState.SUBMISSION_FAILED, status=failed_status,
                    )
                    store.transition_task(
                        conn, failed_id, TaskState.FAILED, failed_status.message
                    )
            continue
    if not store.child_task_ids(conn, parent_id):
        store.transition_task(conn, parent_id, TaskState.FAILED,
                              "No work units could be submitted.")
    else:
        store.rollup_parent(conn, parent_id)
    return store.task_overview(conn, parent_id)


def _profile_for_row(row: sqlite3.Row) -> ExecutorProfile:
    cfg = load_config(Path(row["config_path"]))
    return load_executor_profile(cfg, str(row["executor_profile"]))


def _sync_failure_diagnostics(row: sqlite3.Row, status: JobStatus) -> None:
    spec = store.job_spec(row)
    marker = spec.workdir / ".nerd-diagnostics-synced"
    if marker.is_file():
        return
    spec.workdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": int(row["task_id"]),
        "scheduler_id": row["scheduler_id"],
        "state": status.state.value,
        "slurm_state": status.message,
        "exit_code": status.exit_code,
        "signal": status.signal,
        "started_at": status.started_at,
        "finished_at": status.finished_at,
    }
    (spec.workdir / "failure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    executor_for(_profile_for_row(row)).collect_diagnostics(
        JobHandle(str(row["scheduler_id"])), spec
    )
    marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    error_marker = spec.workdir / ".nerd-diagnostics-sync-error"
    if error_marker.exists():
        error_marker.unlink()


def _reconcile_one(conn: sqlite3.Connection, task_id: int) -> sqlite3.Row:
    row = store.latest_attempt_for_task(conn, task_id)
    if row["executor_type"] == "controller" or row["scheduler_state"] in {
        AttemptState.COMPLETED.value, AttemptState.VALIDATION_FAILED.value,
        AttemptState.SUBMISSION_FAILED.value, AttemptState.CANCELLED.value,
    }:
        return row
    spec = store.job_spec(row)
    handle = JobHandle(str(row["scheduler_id"]))
    try:
        status = executor_for(_profile_for_row(row)).status(handle, spec)
    except Exception as exc:
        status = JobStatus(AttemptState.UNKNOWN, message="Status check failed: %s" % exc)
    store.transition_attempt(conn, int(row["scheduler_attempt_id"]), status.state, status=status)
    if status.state in {AttemptState.QUEUED, AttemptState.PENDING, AttemptState.SUBMITTING}:
        store.transition_task(conn, task_id, TaskState.SUBMITTED)
    elif status.state == AttemptState.RUNNING:
        store.transition_task(conn, task_id, TaskState.RUNNING)
    elif status.state == AttemptState.SCHEDULER_COMPLETED:
        store.transition_task(conn, task_id, TaskState.AWAITING_COLLECTION)
    elif status.state == AttemptState.CANCELLED:
        store.transition_task(conn, task_id, TaskState.CANCELLED, status.message)
    elif status.state in {AttemptState.SCHEDULER_FAILED, AttemptState.SUBMISSION_FAILED}:
        diagnostic_error = None
        try:
            _sync_failure_diagnostics(row, status)
        except Exception as exc:
            diagnostic_error = str(exc)
            spec.workdir.mkdir(parents=True, exist_ok=True)
            (spec.workdir / ".nerd-diagnostics-sync-error").write_text(
                diagnostic_error + "\n", encoding="utf-8"
            )
        message = status.message
        if diagnostic_error:
            message = "%s; diagnostics sync failed: %s" % (message or "FAILED", diagnostic_error)
        store.transition_task(conn, task_id, TaskState.FAILED, message)
    return store.latest_attempt_for_task(conn, task_id)


def reconcile(conn: sqlite3.Connection, task_id: int):
    if store.is_parent_task(conn, task_id):
        for child_id in store.child_task_ids(conn, task_id):
            _reconcile_one(conn, child_id)
        store.rollup_parent(conn, task_id)
        return store.task_overview(conn, task_id)
    row = _reconcile_one(conn, task_id)
    parent_id = row["parent_task_id"]
    if parent_id is not None:
        store.rollup_parent(conn, int(parent_id))
    return row


def _logs_for_row(row, tail: int) -> str:
    local_path = Path(row["log_path"])
    if local_path.is_file():
        lines = local_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-tail:]) if tail > 0 else ""
    if row["executor_type"] == "controller":
        return ""
    return executor_for(_profile_for_row(row)).logs(
        JobHandle(str(row["scheduler_id"])), store.job_spec(row), tail=tail
    )


def task_logs(
    conn: sqlite3.Connection, task_id: int, tail: int = 100,
    unit: Optional[str] = None, failed_only: bool = False,
) -> str:
    reconcile(conn, task_id)
    if not store.is_parent_task(conn, task_id):
        return _logs_for_row(store.latest_attempt_for_task(conn, task_id), tail)
    rows = [store.latest_attempt_for_task(conn, child_id)
            for child_id in store.child_task_ids(conn, task_id)]
    if unit is not None:
        rows = [row for row in rows if str(row["unit_key"]) == unit
                or str(row["unit_label"]) == unit]
        if not rows:
            raise ValueError("Task %s has no work unit %r." % (task_id, unit))
    if failed_only:
        rows = [row for row in rows if row["task_state"] in {"failed", "cancelled"}]
    sections = []
    for row in rows:
        text = _logs_for_row(row, tail)
        if text or unit is not None or failed_only:
            sections.append("== %s (task %s) ==\n%s" % (
                row["unit_label"] or row["unit_key"], row["task_id"], text
            ))
    return "\n\n".join(sections)


def cancel_task(conn: sqlite3.Connection, task_id: int):
    if store.is_parent_task(conn, task_id):
        for child_id in store.child_task_ids(conn, task_id):
            row = store.latest_attempt_for_task(conn, child_id)
            if row["task_state"] not in {
                TaskState.COMPLETED.value, TaskState.FAILED.value, TaskState.CANCELLED.value,
            }:
                cancel_task(conn, child_id)
        store.rollup_parent(conn, task_id)
        return store.task_overview(conn, task_id)
    row = store.latest_attempt_for_task(conn, task_id)
    if row["executor_type"] == "controller":
        raise ValueError("Controller-only task is already complete and cannot be cancelled.")
    if row["scheduler_state"] in {
        AttemptState.COMPLETED.value, AttemptState.SCHEDULER_COMPLETED.value,
        AttemptState.SCHEDULER_FAILED.value, AttemptState.SUBMISSION_FAILED.value,
        AttemptState.CANCELLED.value, AttemptState.VALIDATION_FAILED.value,
    }:
        raise ValueError("Task %s is already in terminal attempt state %s." % (
            task_id, row["scheduler_state"]
        ))
    store.transition_attempt(conn, int(row["scheduler_attempt_id"]), AttemptState.CANCEL_REQUESTED)
    executor_for(_profile_for_row(row)).cancel(
        JobHandle(str(row["scheduler_id"])), store.job_spec(row)
    )
    if row["executor_type"] == "local":
        cancelled = JobStatus(AttemptState.CANCELLED, message="Local process terminated.")
        store.transition_attempt(
            conn, int(row["scheduler_attempt_id"]), AttemptState.CANCELLED, status=cancelled
        )
        store.transition_task(conn, task_id, TaskState.CANCELLED, cancelled.message)
    else:
        store.transition_task(conn, task_id, TaskState.CANCEL_REQUESTED, "Cancellation requested.")
    return store.latest_attempt_for_task(conn, task_id)


def collect_task(conn: sqlite3.Connection, task_id: int):
    if store.is_parent_task(conn, task_id):
        reconcile(conn, task_id)
        first_spec = None
        errors = []
        for child_id in store.child_task_ids(conn, task_id):
            child = store.latest_attempt_for_task(conn, child_id)
            if first_spec is None:
                first_spec = store.job_spec(child)
            if child["scheduler_state"] == AttemptState.SCHEDULER_COMPLETED.value:
                try:
                    collect_task(conn, child_id)
                except Exception as exc:
                    errors.append("%s: %s" % (child["unit_label"] or child_id, exc))
        store.rollup_parent(conn, task_id)
        overview = store.task_overview(conn, task_id)
        if first_spec is not None and overview["task_state"] in {
            TaskState.COMPLETED.value, TaskState.PARTIAL_SUCCESS.value,
        }:
            batch_dir = first_spec.workdir.parent
            update_latest_symlink(batch_dir.parent.parent, str(overview["task_name"]), batch_dir)
        if errors:
            raise RuntimeError("Some work units failed collection: %s" % "; ".join(errors))
        return overview
    row = reconcile(conn, task_id)
    state = AttemptState(str(row["scheduler_state"]))
    if state == AttemptState.COMPLETED:
        return row
    if state != AttemptState.SCHEDULER_COMPLETED:
        raise ValueError("Task %s is not ready to collect (attempt state: %s)." % (task_id, state.value))
    scheduler_attempt_id = int(row["scheduler_attempt_id"])
    store.transition_attempt(conn, scheduler_attempt_id, AttemptState.COLLECTING)
    store.transition_task(conn, task_id, TaskState.COLLECTING)
    spec = store.job_spec(row)
    executor = executor_for(_profile_for_row(row))
    status = executor.collect(JobHandle(str(row["scheduler_id"])), spec)
    if status.state != AttemptState.SCHEDULER_COMPLETED or status.exit_code not in (None, 0):
        store.transition_attempt(conn, scheduler_attempt_id, AttemptState.SCHEDULER_FAILED,
                                 status=status, collected=True)
        store.transition_task(conn, task_id, TaskState.FAILED, status.message)
        return store.latest_attempt_for_task(conn, task_id)
    cfg = load_config(Path(row["config_path"]))
    task = _task(str(row["task_name"]))
    profile = _profile_for_row(row)
    ctx = _context(
        conn, cfg, spec.workdir, profile, spec.resources,
        output_dir=str(spec.workdir.parent.parent.parent),
    )
    try:
        with _working_directory(Path(spec.controller_cwd or Path.cwd())):
            inputs, params = task.prepare(cfg)
            task.consume_outputs(ctx, inputs, params, spec.workdir, task_id=task_id)
    except Exception as exc:
        failed = JobStatus(AttemptState.VALIDATION_FAILED, exit_code=status.exit_code,
                           message=str(exc), started_at=status.started_at,
                           finished_at=status.finished_at)
        store.transition_attempt(conn, scheduler_attempt_id, AttemptState.VALIDATION_FAILED,
                                 status=failed, collected=True)
        store.transition_task(conn, task_id, TaskState.FAILED, "Output validation failed: %s" % exc)
        raise
    complete = JobStatus(AttemptState.COMPLETED, exit_code=status.exit_code,
                         started_at=status.started_at, finished_at=status.finished_at)
    store.transition_attempt(conn, scheduler_attempt_id, AttemptState.COMPLETED,
                             status=complete, collected=True)
    store.transition_task(conn, task_id, TaskState.COMPLETED)
    if row["parent_task_id"] is None:
        update_latest_symlink(spec.workdir.parent.parent, task.name, spec.workdir)
    else:
        store.rollup_parent(conn, int(row["parent_task_id"]))
    return store.latest_attempt_for_task(conn, task_id)


def retry_task(conn: sqlite3.Connection, task_id: int):
    if store.is_parent_task(conn, task_id):
        retried = 0
        for child_id in store.child_task_ids(conn, task_id):
            row = store.latest_attempt_for_task(conn, child_id)
            if row["scheduler_state"] in {
                AttemptState.SCHEDULER_FAILED.value, AttemptState.SUBMISSION_FAILED.value,
                AttemptState.CANCELLED.value, AttemptState.VALIDATION_FAILED.value,
            }:
                retry_task(conn, child_id)
                retried += 1
        if not retried:
            raise ValueError("Task %s has no failed work units to retry." % task_id)
        store.rollup_parent(conn, task_id)
        return store.task_overview(conn, task_id)
    row = store.latest_attempt_for_task(conn, task_id)
    if row["scheduler_state"] not in {
        AttemptState.SCHEDULER_FAILED.value, AttemptState.SUBMISSION_FAILED.value,
        AttemptState.CANCELLED.value, AttemptState.VALIDATION_FAILED.value,
    }:
        raise ValueError("Only failed or cancelled attempts can be retried explicitly.")
    profile = _profile_for_row(row)
    spec = store.job_spec(row)
    prior_log = spec.workdir / "command.log"
    if prior_log.is_file():
        archived = spec.workdir / ("command.attempt-%s.log" % row["try_index"])
        prior_log.replace(archived)
    for name in (
        ".nerd-diagnostics-synced", ".nerd-diagnostics-sync-error", "failure.json",
    ):
        path = spec.workdir / name
        if path.exists():
            path.unlink()
    store.transition_task(conn, task_id, TaskState.PENDING, "Explicit retry requested.")
    return _submit_attempt(conn, task_id, spec, profile, Path(row["config_path"]))


def wait_for_task(
    conn: sqlite3.Connection,
    task_id: int,
    *,
    collect: bool = False,
    poll_interval: float = 1.0,
) -> sqlite3.Row:
    """Wait for a durable task to reach a useful terminal/collection state."""
    if poll_interval <= 0:
        raise ValueError("Poll interval must be greater than zero.")
    terminal = {
        TaskState.COMPLETED.value,
        TaskState.PARTIAL_SUCCESS.value,
        TaskState.FAILED.value,
        TaskState.CANCELLED.value,
    }
    while True:
        row = reconcile(conn, task_id)
        if row["task_state"] == TaskState.AWAITING_COLLECTION.value:
            return collect_task(conn, task_id) if collect else row
        if row["task_state"] in terminal:
            return row
        time.sleep(poll_interval)


def list_tasks(
    conn: sqlite3.Connection,
    *,
    label: Optional[str] = None,
    state: Optional[str] = None,
    task_name: Optional[str] = None,
    limit: int = 50,
    refresh: bool = True,
) -> List[sqlite3.Row]:
    """List root tasks, refreshing unfinished scheduler state by default."""
    if refresh:
        active = conn.execute(
            "SELECT id FROM core_tasks WHERE parent_task_id IS NULL "
            "AND state NOT IN ('completed','partial_success','failed','cancelled') "
            "ORDER BY id DESC"
        ).fetchall()
        for item in active:
            reconcile(conn, int(item[0]))
    return store.list_tasks(
        conn, label=label, state=state, task_name=task_name, limit=limit
    )


def watch_task(
    conn: sqlite3.Connection,
    task_id: int,
    *,
    collect: bool = False,
    poll_interval: float = 2.0,
    on_update: Optional[Callable[[Any], None]] = None,
):
    """Watch a task in the foreground; Ctrl-C never cancels remote work."""
    if poll_interval <= 0:
        raise ValueError("Poll interval must be greater than zero.")
    terminal = {
        TaskState.COMPLETED.value, TaskState.PARTIAL_SUCCESS.value,
        TaskState.FAILED.value, TaskState.CANCELLED.value,
    }
    while True:
        row = reconcile(conn, task_id)
        if collect:
            if store.is_parent_task(conn, task_id):
                for child_id in store.child_task_ids(conn, task_id):
                    child = store.latest_attempt_for_task(conn, child_id)
                    if child["scheduler_state"] == AttemptState.SCHEDULER_COMPLETED.value:
                        try:
                            collect_task(conn, child_id)
                        except Exception:
                            pass
                store.rollup_parent(conn, task_id)
                row = store.task_overview(conn, task_id)
            elif row["task_state"] == TaskState.AWAITING_COLLECTION.value:
                row = collect_task(conn, task_id)
        if on_update is not None:
            on_update(row)
        if not collect and row["task_state"] == TaskState.AWAITING_COLLECTION.value:
            return row
        if row["task_state"] in terminal:
            return row
        time.sleep(poll_interval)
