# nerd/db/api.py
"""
This module provides a minimal, stateless API for database interactions,
focusing on task tracking and domain-specific data insertion.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from nerd.db.schema import ALL_TABLES, ALL_INDEXES
from nerd.utils.logging import get_logger

log = get_logger(__name__)


def connect(db_path: Path) -> sqlite3.Connection:
    """
    Establishes a connection to the SQLite database.

    Args:
        db_path: The file path to the SQLite database.

    Returns:
        A sqlite3.Connection object.
    """
    try:
        # Ensure the parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        log.debug("Database connection established to %s", db_path)
        return conn
    except sqlite3.Error as e:
        log.exception("Database connection failed: %s", e)
        raise


def init_schema(conn: sqlite3.Connection):
    """
    Initializes the database schema by creating all tables and indexes.

    Args:
        conn: An active sqlite3.Connection object.
    """
    try:
        with conn:
            for table_sql in ALL_TABLES:
                conn.execute(table_sql)
            for index_sql in ALL_INDEXES:
                conn.execute(index_sql)
        log.info("Database schema initialized successfully.")
    except sqlite3.Error as e:
        log.exception("Schema initialization failed: %s", e)
        raise


def begin_task(conn: sqlite3.Connection, name: str, scope_kind: str, scope_id: Optional[int],
               backend: str, output_dir: str, label: str, cache_key: Optional[str]) -> Optional[int]:
    """
    Records the start of a new task.

    Returns:
        The ID of the newly created task, or None on failure.
    """
    sql = """
        INSERT INTO tasks (task_name, scope_kind, scope_id, backend, output_dir, label, cache_key, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with conn:
            cursor = conn.execute(sql, (name, scope_kind, scope_id, backend, output_dir, label, cache_key, datetime.now().isoformat()))
            task_id = cursor.lastrowid
            log.info("Began task '%s' (ID: %s) for %s:%s", name, task_id, scope_kind, scope_id)
            return task_id
    except sqlite3.IntegrityError as e:
        log.error("Failed to begin task '%s' due to integrity error: %s", name, e)
        return None


def attempt(conn: sqlite3.Connection, task_id: int, try_index: int, command: str,
            resources: Dict[str, Any], log_path: Path) -> Optional[int]:
    """
    Records a single attempt to execute a task.

    Returns:
        The ID of the task attempt, or None on failure.
    """
    sql = """
        INSERT INTO task_attempts (task_id, try_index, command, resources_json, log_path)
        VALUES (?, ?, ?, ?, ?)
    """
    try:
        resources_json = json.dumps(resources)
        with conn:
            cursor = conn.execute(sql, (task_id, try_index, command, resources_json, str(log_path)))
            attempt_id = cursor.lastrowid
            log.info("Logged attempt #%d for task ID %d (Attempt ID: %s)", try_index, task_id, attempt_id)
            return attempt_id
    except sqlite3.Error as e:
        log.exception("Failed to record attempt for task %d: %s", task_id, e)
        return None


def finish_task(conn: sqlite3.Connection, task_id: int, state: str, message: Optional[str] = None,
                cache_key: Optional[str] = None):
    """
    Updates a task's final state (e.g., 'completed', 'failed').
    """
    sql = """
        UPDATE tasks
        SET state = ?, message = ?, cache_key = ?, ended_at = ?
        WHERE id = ?
    """
    try:
        with conn:
            conn.execute(sql, (state, message, cache_key, datetime.now().isoformat(), task_id))
        log.info("Finished task ID %d with state '%s'", task_id, state)
    except sqlite3.Error as e:
        log.exception("Failed to finish task %d: %s", task_id, e)


# --- Domain-specific write operations ---

def insert_fmod_calc_run(conn: sqlite3.Connection, run_data: Dict[str, Any]) -> Optional[int]:
    """
    Inserts a new fmod_calc_run record.

    Args:
        run_data: A dictionary with keys matching the fmod_calc_runs table columns.

    Returns:
        The ID of the new record, or None on failure.
    """
    sql = """
        INSERT INTO fmod_calc_runs (software_name, software_version, run_args, run_datetime, output_dir, s_id)
        VALUES (:software_name, :software_version, :run_args, :run_datetime, :output_dir, :s_id)
    """
    try:
        with conn:
            cursor = conn.execute(sql, run_data)
            run_id = cursor.lastrowid
            log.info("Inserted fmod_calc_run for sample ID %s (Run ID: %s)", run_data.get('s_id'), run_id)
            return run_id
    except sqlite3.IntegrityError as e:
        log.error("fmod_calc_run for sample ID %s already exists. %s", run_data.get('s_id'), e)
        return None


def bulk_insert_fmod_vals(conn: sqlite3.Connection, fmod_vals: List[Dict[str, Any]]):
    """
    Performs a bulk insert of fmod values.

    Args:
        fmod_vals: A list of dictionaries, each representing a row in fmod_vals.
    """
    sql = """
        INSERT INTO fmod_vals (nt_id, fmod_calc_run_id, fmod_val, valtype, read_depth, rxn_id)
        VALUES (:nt_id, :fmod_calc_run_id, :fmod_val, :valtype, :read_depth, :rxn_id)
    """
    try:
        with conn:
            conn.executemany(sql, fmod_vals)
        log.info("Bulk inserted %d fmod_vals records.", len(fmod_vals))
    except sqlite3.Error as e:
        log.exception("Bulk insert of fmod_vals failed: %s", e)


def upsert_free_fit(conn: sqlite3.Connection, fit_data: Dict[str, Any]):
    """
    Inserts a new free_tc_fits record or updates it if it already exists
    based on the unique constraint (rg_id, nt_id).
    """
    sql = """
        INSERT INTO free_tc_fits (
            rg_id, nt_id, kobs_val, kobs_err, kdeg_val, kdeg_err,
            fmod0, fmod0_err, r2, chisq, time_min, time_max
        )
        VALUES (
            :rg_id, :nt_id, :kobs_val, :kobs_err, :kdeg_val, :kdeg_err,
            :fmod0, :fmod0_err, :r2, :chisq, :time_min, :time_max
        )
        ON CONFLICT(rg_id, nt_id) DO UPDATE SET
            kobs_val = excluded.kobs_val,
            kobs_err = excluded.kobs_err,
            kdeg_val = excluded.kdeg_val,
            kdeg_err = excluded.kdeg_err,
            fmod0 = excluded.fmod0,
            fmod0_err = excluded.fmod0_err,
            r2 = excluded.r2,
            chisq = excluded.chisq,
            time_min = excluded.time_min,
            time_max = excluded.time_max
    """
    try:
        with conn:
            conn.execute(sql, fit_data)
        log.debug("Upserted free_tc_fit for rg_id %s, nt_id %s", fit_data.get('rg_id'), fit_data.get('nt_id'))
    except sqlite3.Error as e:
        log.exception("Upsert of free_tc_fit failed: %s", e)

