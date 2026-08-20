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
    TaskState.PARTIAL_SUCCESS.value,
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
               t.label, t.output_dir, t.message AS task_message,
               t.started_at AS task_started_at, t.ended_at AS task_ended_at,
               t.tool, t.tool_version, t.parent_task_id, t.unit_key,
               t.unit_label, t.unit_index
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


def task_record(conn: sqlite3.Connection, task_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM core_tasks WHERE id=?", (task_id,)).fetchone()
    if row is None:
        raise ValueError("Task %s does not exist." % task_id)
    return row


def attach_child(
    conn: sqlite3.Connection,
    child_id: int,
    parent_id: int,
    unit_key: str,
    unit_label: str,
    unit_index: int,
) -> None:
    with conn:
        conn.execute(
            "UPDATE core_tasks SET parent_task_id=?, unit_key=?, unit_label=?, unit_index=? "
            "WHERE id=?",
            (parent_id, unit_key, unit_label, unit_index, child_id),
        )


def child_task_ids(conn: sqlite3.Connection, parent_id: int) -> List[int]:
    return [
        int(row[0]) for row in conn.execute(
            "SELECT id FROM core_tasks WHERE parent_task_id=? ORDER BY unit_index, id",
            (parent_id,),
        ).fetchall()
    ]


def is_parent_task(conn: sqlite3.Connection, task_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM core_tasks WHERE parent_task_id=? LIMIT 1", (task_id,)
    ).fetchone() is not None


def rollup_parent(conn: sqlite3.Connection, parent_id: int) -> sqlite3.Row:
    children = conn.execute(
        "SELECT state FROM core_tasks WHERE parent_task_id=?", (parent_id,)
    ).fetchall()
    if not children:
        return task_record(conn, parent_id)
    states = [str(row[0]) for row in children]
    terminal = {
        TaskState.COMPLETED.value, TaskState.PARTIAL_SUCCESS.value,
        TaskState.FAILED.value, TaskState.CANCELLED.value,
    }
    successes = sum(state == TaskState.COMPLETED.value for state in states)
    failures = sum(state in {TaskState.FAILED.value, TaskState.CANCELLED.value} for state in states)
    if all(state == TaskState.COMPLETED.value for state in states):
        state = TaskState.COMPLETED
    elif all(item in terminal for item in states):
        if successes and failures:
            state = TaskState.PARTIAL_SUCCESS
        elif failures == len(states):
            state = TaskState.FAILED
        else:
            state = TaskState.COMPLETED
    elif any(item in {TaskState.RUNNING.value, TaskState.COLLECTING.value} for item in states):
        state = TaskState.RUNNING
    elif any(item == TaskState.AWAITING_COLLECTION.value for item in states):
        state = TaskState.AWAITING_COLLECTION
    elif any(item == TaskState.CANCEL_REQUESTED.value for item in states):
        state = TaskState.CANCEL_REQUESTED
    else:
        state = TaskState.SUBMITTED
    message = "%s/%s completed; %s failed" % (successes, len(states), failures)
    current = task_record(conn, parent_id)
    if str(current["state"]) != state.value or str(current["message"] or "") != message:
        transition_task(conn, parent_id, state, message)
    return task_record(conn, parent_id)


def task_overview(conn: sqlite3.Connection, task_id: int) -> Dict[str, Any]:
    task = task_record(conn, task_id)
    child_ids = child_task_ids(conn, task_id)
    if not child_ids:
        return dict(latest_attempt_for_task(conn, task_id))
    rollup_parent(conn, task_id)
    task = task_record(conn, task_id)
    children = [dict(latest_attempt_for_task(conn, child_id)) for child_id in child_ids]
    counts: Dict[str, int] = {}
    for child in children:
        state = str(child["task_state"])
        counts[state] = counts.get(state, 0) + 1
    first = children[0] if children else {}
    return {
        "is_parent": True,
        "task_id": task_id,
        "task_name": task["task_name"],
        "task_state": task["state"],
        "task_message": task["message"],
        "label": task["label"],
        "output_dir": task["output_dir"],
        "task_started_at": task["started_at"],
        "task_ended_at": task["ended_at"],
        "tool": task["tool"],
        "tool_version": task["tool_version"],
        "try_index": None,
        "scheduler_attempt_id": None,
        "scheduler_state": "batch",
        "executor_profile": first.get("executor_profile"),
        "executor_type": first.get("executor_type"),
        "scheduler_id": None,
        "exit_code": None,
        "error": task["message"] if task["state"] in {"failed", "partial_success"} else None,
        "submitted_at": task["started_at"],
        "scheduler_started_at": None,
        "scheduler_finished_at": None,
        "collected_at": task["ended_at"],
        "log_path": None,
        "remote_workdir": None,
        "children": children,
        "counts": counts,
        "total_units": len(children),
    }


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
               a.try_index, sa.executor_profile, sa.scheduler_id, sa.state AS attempt_state,
               (SELECT COUNT(*) FROM core_tasks c WHERE c.parent_task_id=t.id) AS total_units,
               (SELECT COUNT(*) FROM core_tasks c WHERE c.parent_task_id=t.id AND c.state='completed') AS completed_units,
               (SELECT COUNT(*) FROM core_tasks c WHERE c.parent_task_id=t.id AND c.state='failed') AS failed_units
        FROM core_tasks t
        LEFT JOIN core_task_attempts a ON a.id=(
            SELECT a2.id FROM core_task_attempts a2 WHERE a2.task_id=t.id
            ORDER BY a2.try_index DESC LIMIT 1
        )
        LEFT JOIN core_scheduler_attempts sa ON sa.attempt_id=a.id
    """
    filters = ["t.parent_task_id IS NULL"]
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
    sql += " WHERE " + " AND ".join(filters)
    sql += " ORDER BY t.id DESC LIMIT ?"
    params.append(limit)
    return list(conn.execute(sql, params).fetchall())
