"""SQLite persistence for scheduler attempts and state transitions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nerd.db import api as db_api
from .models import AttemptState, JobHandle, JobSpec, JobStatus, TaskState


TERMINAL_TASK_STATES = {
    TaskState.COMPLETED.value,
    TaskState.FAILED.value,
    TaskState.CANCELLED.value,
}


def _now() -> str:
    return datetime.now().isoformat()


def transition_task(
    conn: sqlite3.Connection,
    task_id: int,
    state: TaskState,
    message: Optional[str] = None,
) -> None:
    row = conn.execute("SELECT state FROM core_tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise ValueError("Task %s does not exist." % task_id)
    old = str(row[0])
    ended_at = _now() if state.value in TERMINAL_TASK_STATES else None
    with conn:
        conn.execute(
            "UPDATE core_tasks SET state=?, message=?, ended_at=? WHERE id=?",
            (state.value, message, ended_at, task_id),
        )
        conn.execute(
            "INSERT INTO core_state_transitions (entity_kind, entity_id, from_state, to_state, message) "
            "VALUES ('task', ?, ?, ?, ?)",
            (task_id, old, state.value, message),
        )


def create_attempt(
    conn: sqlite3.Connection,
    task_id: int,
    profile_name: str,
    executor_type: str,
    spec: JobSpec,
    config_path: Path,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT COALESCE(MAX(try_index), 0) + 1 FROM core_task_attempts WHERE task_id=?",
        (task_id,),
    ).fetchone()
    try_index = int(row[0])
    attempt_id = db_api.attempt(
        conn,
        task_id,
        try_index,
        spec.command,
        spec.resources,
        spec.workdir / "command.log",
    )
    if attempt_id is None:
        raise RuntimeError("Could not persist the task attempt.")
    now = _now()
    with conn:
        conn.execute(
            """
            INSERT INTO core_scheduler_attempts (
                attempt_id, executor_profile, executor_type, state, remote_workdir,
                log_path, job_spec_json, config_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                profile_name,
                executor_type,
                AttemptState.PENDING.value,
                spec.remote_workdir,
                str(spec.workdir / "command.log"),
                json.dumps(spec.to_dict(), sort_keys=True),
                str(config_path),
                now,
                now,
            ),
        )
        scheduler_id = int(conn.execute(
            "SELECT id FROM core_scheduler_attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()[0])
        conn.execute(
            "INSERT INTO core_state_transitions (entity_kind, entity_id, from_state, to_state) "
            "VALUES ('attempt', ?, NULL, ?)",
            (scheduler_id, AttemptState.PENDING.value),
        )
    return get_attempt(conn, scheduler_id)


def transition_attempt(
    conn: sqlite3.Connection,
    scheduler_attempt_id: int,
    state: AttemptState,
    *,
    message: Optional[str] = None,
    handle: Optional[JobHandle] = None,
    status: Optional[JobStatus] = None,
    collected: bool = False,
) -> None:
    row = get_attempt(conn, scheduler_attempt_id)
    old = str(row["scheduler_state"])
    values: Dict[str, Any] = {
        "state": state.value,
        "updated_at": _now(),
    }
    if handle is not None:
        values["scheduler_id"] = handle.scheduler_id
    if status is not None:
        values.update({
            "exit_code": status.exit_code,
            "signal": status.signal,
            "error": status.message,
            "started_at": status.started_at or row["scheduler_started_at"],
            "finished_at": status.finished_at or row["scheduler_finished_at"],
        })
    if state == AttemptState.QUEUED and not row["submitted_at"]:
        values["submitted_at"] = _now()
    if state == AttemptState.RUNNING and not row["scheduler_started_at"]:
        values["started_at"] = _now()
    if state in {
        AttemptState.SCHEDULER_COMPLETED,
        AttemptState.SCHEDULER_FAILED,
        AttemptState.SUBMISSION_FAILED,
        AttemptState.CANCELLED,
    } and not row["scheduler_finished_at"]:
        values["finished_at"] = _now()
    if collected:
        values["collected_at"] = _now()
    assignments = ", ".join("%s=?" % key for key in values)
    params = list(values.values()) + [scheduler_attempt_id]
    with conn:
        conn.execute("UPDATE core_scheduler_attempts SET %s WHERE id=?" % assignments, params)
        if status is not None and status.exit_code is not None:
            conn.execute(
                "UPDATE core_task_attempts SET exit_code=? WHERE id=?",
                (status.exit_code, row["attempt_id"]),
            )
        conn.execute(
            "INSERT INTO core_state_transitions (entity_kind, entity_id, from_state, to_state, message) "
            "VALUES ('attempt', ?, ?, ?, ?)",
            (scheduler_attempt_id, old, state.value, message or (status.message if status else None)),
        )


def get_attempt(conn: sqlite3.Connection, scheduler_attempt_id: int) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT sa.id AS scheduler_attempt_id, sa.attempt_id, a.task_id, a.try_index,
               sa.executor_profile, sa.executor_type, sa.state AS scheduler_state,
               sa.scheduler_id, sa.remote_workdir, sa.log_path, sa.exit_code,
               sa.signal, sa.error, sa.submitted_at,
               sa.started_at AS scheduler_started_at,
               sa.finished_at AS scheduler_finished_at, sa.collected_at,
               sa.job_spec_json, sa.config_path, t.task_name, t.state AS task_state,
               t.label, t.output_dir, t.message AS task_message
        FROM core_scheduler_attempts sa
        JOIN core_task_attempts a ON a.id=sa.attempt_id
        JOIN core_tasks t ON t.id=a.task_id
        WHERE sa.id=?
        """,
        (scheduler_attempt_id,),
    ).fetchone()
    if row is None:
        raise ValueError("Scheduler attempt %s does not exist." % scheduler_attempt_id)
    return row


def latest_attempt_for_task(conn: sqlite3.Connection, task_id: int) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT sa.id FROM core_scheduler_attempts sa
        JOIN core_task_attempts a ON a.id=sa.attempt_id
        WHERE a.task_id=? ORDER BY a.try_index DESC LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        raise ValueError("Task %s has no asynchronous attempts." % task_id)
    return get_attempt(conn, int(row[0]))


def job_spec(row: sqlite3.Row) -> JobSpec:
    return JobSpec.from_dict(json.loads(row["job_spec_json"]))


def record_container_provenance(conn: sqlite3.Connection, task_id: int,
                                payload: Dict[str, Any]) -> None:
    runtime = payload.get("runtime") or {}
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO core_container_provenance (
               task_id, oci_reference, oci_digest, sif_path, sif_checksum,
               runtime, runtime_version, tool_version, command, execution_host,
               executor_profile, executor_type, recorded_at, provenance_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (task_id, payload.get("oci_reference"), payload.get("oci_digest"),
             payload.get("sif_path"), payload.get("sif_checksum"), runtime.get("command"),
             runtime.get("version"), payload.get("shapemapper_version"), payload.get("command"),
             payload.get("execution_host"), payload.get("executor_profile"),
             payload.get("executor_type"), payload.get("recorded_at"),
             json.dumps(payload, sort_keys=True)),
        )


def list_tasks(
    conn: sqlite3.Connection,
    label: Optional[str] = None,
    state: Optional[str] = None,
    task_name: Optional[str] = None,
    limit: int = 50,
) -> List[sqlite3.Row]:
    """List durable tasks using filters that require no schema changes."""
    if limit < 1:
        raise ValueError("Task list limit must be at least 1.")
    sql = """
        SELECT t.id, t.task_name, t.label, t.state, t.backend, t.started_at, t.ended_at,
               a.try_index, sa.executor_profile, sa.scheduler_id, sa.state AS attempt_state
        FROM core_tasks t
        LEFT JOIN core_task_attempts a ON a.id=(
            SELECT a2.id FROM core_task_attempts a2 WHERE a2.task_id=t.id
            ORDER BY a2.try_index DESC LIMIT 1
        )
        LEFT JOIN core_scheduler_attempts sa ON sa.attempt_id=a.id
    """
    filters = []
    params: List[Any] = []
    if label:
        filters.append("t.label=?")
        params.append(label)
    if state:
        filters.append("t.state=?")
        params.append(state)
    if task_name:
        filters.append("t.task_name=?")
        params.append(task_name)
    if filters:
        sql += " WHERE " + " AND ".join(filters)
    sql += " ORDER BY t.id DESC LIMIT ?"
    params.append(limit)
    return list(conn.execute(sql, params).fetchall())
