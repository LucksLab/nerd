"""Scientific-task orchestration over durable asynchronous executors."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path, PurePosixPath
import time
import yaml
from typing import Any, Dict, Optional

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


def submit_task(
    conn: sqlite3.Connection,
    step: str,
    config_path: Path,
    profile_name: Optional[str] = None,
) -> sqlite3.Row:
    cfg = load_config(config_path)
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
        raise ValueError(
            "Identical configuration already completed as task %s; set force_run to submit again."
            % existing["id"]
        )
    inputs, params = task.prepare(cfg)
    if hasattr(task, "validate_execution_config"):
        task.validate_execution_config(inputs, profile)
    suffix = "%s__cfg-%s" % (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        config_hash(cfg, length=7),
    )
    run_dir = make_run_dir(Path(output_dir) / str(run["label"]), task.name, suffix=suffix)
    config_snapshot = run_dir / ".nerd-submitted-config.yaml"
    # Persist normalized runtime paths so collection and retry are independent
    # of both the original config and the later controller working directory.
    config_snapshot.write_text(
        yaml.safe_dump(dict(cfg), sort_keys=False), encoding="utf-8"
    )
    resources = profile_resources(profile, run)
    ctx = _context(conn, cfg, run_dir, profile, resources, output_dir=output_dir)
    execution_provenance = None
    if hasattr(task, "prepare_execution"):
        execution_provenance = task.prepare_execution(cfg, profile, ctx, inputs)
    scope = task.resolve_scope(ctx, inputs)
    task_id = db_api.begin_task(
        conn, task.name, scope.kind, scope.scope_id, profile.name, output_dir,
        str(run["label"]), cache_key, tool=task.task_tool(inputs),
        tool_version=task.task_tool_version(inputs),
    )
    if task_id is None:
        raise RuntimeError("Could not persist the task.")
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
        store.transition_attempt(
            conn, scheduler_attempt_id, AttemptState.COLLECTING,
            handle=JobHandle("controller"),
        )
        try:
            task.consume_outputs(ctx, inputs, params, run_dir, task_id=task_id)
        except Exception as exc:
            status = JobStatus(AttemptState.VALIDATION_FAILED, message=str(exc))
            store.transition_attempt(conn, scheduler_attempt_id, AttemptState.VALIDATION_FAILED,
                                     status=status, collected=True)
            store.transition_task(conn, task_id, TaskState.FAILED, "Output validation failed: %s" % exc)
            raise
        status = JobStatus(AttemptState.COMPLETED, exit_code=0)
        store.transition_attempt(conn, scheduler_attempt_id, AttemptState.COMPLETED,
                                 status=status, collected=True)
        store.transition_task(conn, task_id, TaskState.COMPLETED)
        update_latest_symlink(Path(output_dir) / str(run["label"]), task.name, run_dir)
        return store.get_attempt(conn, scheduler_attempt_id)
    spec = _job_spec(task, ctx, cfg, command, profile, resources, execution_provenance)
    return _submit_attempt(conn, task_id, spec, profile, config_snapshot)


def _profile_for_row(row: sqlite3.Row) -> ExecutorProfile:
    cfg = load_config(Path(row["config_path"]))
    return load_executor_profile(cfg, str(row["executor_profile"]))


def reconcile(conn: sqlite3.Connection, task_id: int) -> sqlite3.Row:
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
    if status.state == AttemptState.RUNNING:
        store.transition_task(conn, task_id, TaskState.RUNNING)
    elif status.state == AttemptState.SCHEDULER_COMPLETED:
        store.transition_task(conn, task_id, TaskState.AWAITING_COLLECTION)
    elif status.state == AttemptState.CANCELLED:
        store.transition_task(conn, task_id, TaskState.CANCELLED, status.message)
    elif status.state in {AttemptState.SCHEDULER_FAILED, AttemptState.SUBMISSION_FAILED}:
        store.transition_task(conn, task_id, TaskState.FAILED, status.message)
    return store.latest_attempt_for_task(conn, task_id)


def task_logs(conn: sqlite3.Connection, task_id: int, tail: int = 100) -> str:
    row = store.latest_attempt_for_task(conn, task_id)
    if row["executor_type"] == "controller":
        path = Path(row["log_path"])
        return path.read_text(errors="replace") if path.exists() else ""
    return executor_for(_profile_for_row(row)).logs(
        JobHandle(str(row["scheduler_id"])), store.job_spec(row), tail=tail
    )


def cancel_task(conn: sqlite3.Connection, task_id: int) -> sqlite3.Row:
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


def collect_task(conn: sqlite3.Connection, task_id: int) -> sqlite3.Row:
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
    update_latest_symlink(spec.workdir.parent.parent, task.name, spec.workdir)
    return store.latest_attempt_for_task(conn, task_id)


def retry_task(conn: sqlite3.Connection, task_id: int) -> sqlite3.Row:
    row = store.latest_attempt_for_task(conn, task_id)
    if row["scheduler_state"] not in {
        AttemptState.SCHEDULER_FAILED.value, AttemptState.SUBMISSION_FAILED.value,
        AttemptState.CANCELLED.value, AttemptState.VALIDATION_FAILED.value,
    }:
        raise ValueError("Only failed or cancelled attempts can be retried explicitly.")
    profile = _profile_for_row(row)
    spec = store.job_spec(row)
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
