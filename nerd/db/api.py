# nerd/db/api.py
"""
This module provides a minimal, stateless API for database interactions,
focusing on task tracking and domain-specific data insertion.
"""

import sqlite3
import json
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from nerd.db.schema import ALL_TABLES, ALL_INDEXES
from nerd.utils.hashing import config_hash
from nerd.utils.logging import get_logger

log = get_logger(__name__)


_LOCK_RETRY_MESSAGES: tuple[str, ...] = (
    "database is locked",
    "database is busy",
    "database is in use",
)


def _is_lock_error(err: sqlite3.Error) -> bool:
    """Return True if the sqlite error looks like a lock/busy condition."""
    msg = str(err).lower()
    return any(token in msg for token in _LOCK_RETRY_MESSAGES)


def _run_with_retry(
    conn: sqlite3.Connection,
    operation: Callable[[], Any],
    description: str,
    *,
    retries: int = 5,
    initial_delay: float = 0.1,
    backoff: float = 2.0,
) -> Any:
    """
    Execute `operation`, retrying when SQLite reports a lock/busy error.

    Retries are exponential-backoff with jitter-free timing to keep behaviour
    predictable for batch jobs.
    """
    delay = initial_delay
    last_error: Optional[sqlite3.Error] = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except sqlite3.OperationalError as err:
            last_error = err
            if not _is_lock_error(err):
                raise
            if attempt == retries:
                break
            log.warning(
                "SQLite busy during %s (attempt %s/%s); retrying in %.2fs",
                description,
                attempt,
                retries,
                delay,
            )
            try:
                conn.rollback()
            except Exception:
                pass
            time.sleep(delay)
            delay *= backoff
    if last_error is not None:
        raise last_error


def _iter_chunks(items: Iterable[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    """Yield successive chunks from an iterable."""
    chunk: List[Dict[str, Any]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


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
        # Ensure FK constraints (including ON DELETE CASCADE) are enforced
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
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
               backend: str, output_dir: str, label: str, cache_key: Optional[str],
               tool: Optional[str] = None, tool_version: Optional[str] = None) -> Optional[int]:
    """
    Records the start of a new task.

    Returns:
        The ID of the newly created task, or None on failure.
    """
    sql = """
        INSERT INTO core_tasks (
            task_name, scope_kind, scope_id, backend, output_dir, label,
            cache_key, tool, tool_version, config_hash, started_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    def _insert() -> Optional[int]:
        with conn:
            cfg_short = (cache_key or "")[:7]
            cursor = conn.execute(
                sql,
                (
                    name,
                    scope_kind,
                    scope_id,
                    backend,
                    output_dir,
                    label,
                    cache_key,
                    tool,
                    tool_version,
                    cfg_short,
                    datetime.now().isoformat(),
                ),
            )
            task_id = cursor.lastrowid
            log.info(
                "Began task '%s' (ID: %s) for %s:%s (cfg=%s)",
                name,
                task_id,
                scope_kind,
                scope_id,
                cache_key,
            )
            return task_id

    try:
        return _run_with_retry(conn, _insert, "begin_task")
    except sqlite3.IntegrityError as e:
        log.error("Failed to begin task '%s' due to integrity error: %s", name, e)
        # Reuse existing non-cached task row for this signature (task_name,label,output_dir,cache_key)
        try:
            row = _run_with_retry(
                conn,
                lambda: conn.execute(
                    """
                    SELECT id, state FROM core_tasks
                    WHERE task_name = ? AND label = ? AND output_dir = ?
                      AND ((? IS NULL AND cache_key IS NULL) OR cache_key = ?)
                    ORDER BY started_at DESC
                    LIMIT 1
                    """,
                    (name, label, output_dir, cache_key, cache_key),
                ).fetchone(),
                "begin_task reuse lookup",
            )
            if row:
                tid = int(row[0])
                # Reset state to pending and clear message/ended_at for a new attempt
                def _reuse_update():
                    with conn:
                        conn.execute(
                            "UPDATE core_tasks SET state='pending', message=NULL, started_at=datetime('now'), ended_at=NULL WHERE id = ?",
                            (tid,),
                        )

                _run_with_retry(conn, _reuse_update, "begin_task reuse update")
                log.info("Reusing existing task row id=%s for signature (task=%s,label=%s)", tid, name, label)
                return tid
        except Exception as e2:
            log.warning("Could not reuse existing task row after integrity error: %s", e2)
        return None


def attempt(conn: sqlite3.Connection, task_id: int, try_index: int, command: str,
            resources: Dict[str, Any], log_path: Path) -> Optional[int]:
    """
    Records a single attempt to execute a task.

    Returns:
        The ID of the task attempt, or None on failure.
    """
    sql = """
        INSERT INTO core_task_attempts (task_id, try_index, command, resources_json, log_path)
        VALUES (?, ?, ?, ?, ?)
    """
    def _next_try_index(c: sqlite3.Connection, tid: int) -> int:
        row = c.execute(
            "SELECT COALESCE(MAX(try_index), 0) + 1 FROM core_task_attempts WHERE task_id = ?",
            (tid,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 1

    def _insert(idx_val: int, payload_json: str) -> Optional[int]:
        def _op():
            with conn:
                cursor = conn.execute(sql, (task_id, idx_val, command, payload_json, str(log_path)))
                return cursor.lastrowid

        attempt_id = _run_with_retry(conn, _op, f"attempt insert try_index={idx_val}")
        log.info("Logged attempt #%d for task ID %d (Attempt ID: %s)", idx_val, task_id, attempt_id)
        return attempt_id

    resources_json = json.dumps(resources)
    idx = int(try_index) if isinstance(try_index, int) and try_index > 0 else _run_with_retry(
        conn,
        lambda: _next_try_index(conn, task_id),
        "attempt next_try_index",
    )
    try:
        return _insert(idx, resources_json)
    except sqlite3.IntegrityError:
        try:
            idx = _run_with_retry(conn, lambda: _next_try_index(conn, task_id), "attempt retry next_try_index")
            return _insert(idx, json.dumps(resources))
        except sqlite3.Error as e2:
            log.exception("Failed to record attempt for task %d after retry: %s", task_id, e2)
            return None
    except sqlite3.Error as e:
        log.exception("Failed to record attempt for task %d: %s", task_id, e)
        return None


def finish_task(conn: sqlite3.Connection, task_id: int, state: str, message: Optional[str] = None,
                cache_key: Optional[str] = None):
    """
    Updates a task's final state (e.g., 'completed', 'failed').
    If cache_key is provided, updates it; otherwise preserves the existing value.
    """
    def _op():
        with conn:
            if cache_key is None:
                conn.execute(
                    """
                    UPDATE core_tasks
                    SET state = ?, message = ?, ended_at = ?
                    WHERE id = ?
                    """,
                    (state, message, datetime.now().isoformat(), task_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE core_tasks
                    SET state = ?, message = ?, cache_key = ?, ended_at = ?
                    WHERE id = ?
                    """,
                    (state, message, cache_key, datetime.now().isoformat(), task_id),
                )
    try:
        _run_with_retry(conn, _op, "finish_task")
        log.info("Finished task ID %d with state '%s'", task_id, state)
    except sqlite3.Error as e:
        log.exception("Failed to finish task %d: %s", task_id, e)


def find_task_by_signature(conn: sqlite3.Connection, label: str, output_dir: str,
                           cache_key: Optional[str]) -> Optional[sqlite3.Row]:
    """
    Returns the most recent task row matching the same logical signature
    (label, output_dir, cache_key), or None if not found.
    Ignores task_name to account for implementations where it may include timestamps.
    """
    sql = (
        """
        SELECT id, task_name, label, output_dir, cache_key, state, started_at, ended_at
        FROM core_tasks
        WHERE label = ? AND output_dir = ? AND ((? IS NULL AND cache_key IS NULL) OR cache_key = ?)
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    try:
        row = conn.execute(sql, (label, output_dir, cache_key, cache_key)).fetchone()
        return row
    except sqlite3.Error as e:
        log.exception("Failed to query existing task by signature: %s", e)
        return None

def find_completed_task_by_signature(conn: sqlite3.Connection, label: str, output_dir: str,
                                     cache_key: Optional[str]) -> Optional[sqlite3.Row]:
    """
    Returns the most recent COMPLETED task row matching (label, output_dir, cache_key).
    Ignores cached rows and other states to reliably detect a prior success even if
    a newer 'cached' entry exists.
    """
    sql = (
        """
        SELECT id, task_name, label, output_dir, cache_key, state, started_at, ended_at
        FROM core_tasks
        WHERE label = ? AND output_dir = ? AND ((? IS NULL AND cache_key IS NULL) OR cache_key = ?)
          AND state = 'completed'
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    try:
        return conn.execute(sql, (label, output_dir, cache_key, cache_key)).fetchone()
    except sqlite3.Error as e:
        log.exception("Failed to query completed task by signature: %s", e)
        return None


def record_cached_task(conn: sqlite3.Connection, name: str, scope_kind: str, scope_id: Optional[int],
                       backend: str, output_dir: str, label: str, cache_key: Optional[str],
                       message: Optional[str] = None,
                       tool: Optional[str] = None, tool_version: Optional[str] = None) -> Optional[int]:
    """
    Inserts a task row with state='cached' to record a skipped run due to identical config.

    Uses a single INSERT to avoid unique-index conflicts on active states.
    Sets both started_at and ended_at to now.
    """
    sql = """
        INSERT INTO core_tasks (
            task_name, scope_kind, scope_id, backend, output_dir, label,
            cache_key, tool, tool_version, config_hash, started_at, ended_at, state, message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), 'cached', ?)
    """
    def _op() -> Optional[int]:
        with conn:
            cfg_short = (cache_key or "")[:7]
            cur = conn.execute(sql, (name, scope_kind, scope_id, backend, output_dir, label, cache_key, tool, tool_version, cfg_short, message))
            return cur.lastrowid

    try:
        task_id = _run_with_retry(conn, _op, "record_cached_task")
        if task_id is not None:
            log.info("Recorded cached task '%s' (ID: %s) for label '%s'", name, task_id, label)
        return task_id
    except sqlite3.IntegrityError as e:
        log.error("Failed to record cached task due to integrity error: %s", e)
        return None
    except sqlite3.Error as e:
        log.exception("Failed to record cached task: %s", e)
        return None


def record_task_scope_members(
    conn: sqlite3.Connection,
    task_id: int,
    members: Iterable[Any],
) -> None:
    """
    Inserts rows into core_task_scope_members to capture the entities processed by a task run.
    Accepts TaskScopeMember dataclass instances or plain dictionaries with at least a 'kind'.
    """
    rows: List[tuple[Any, Any, Any, Any, Any]] = []
    for member in members or []:
        if member is None:
            continue
        if hasattr(member, "kind"):
            kind = getattr(member, "kind", None)
            ref_id = getattr(member, "ref_id", None)
            label = getattr(member, "label", None)
            extra = getattr(member, "extra", None)
        elif isinstance(member, dict):
            kind = member.get("kind")
            ref_id = member.get("ref_id", member.get("member_id"))
            label = member.get("label", member.get("member_label"))
            extra = member.get("extra", member.get("extra_json"))
        else:
            kind = None
            ref_id = None
            label = None
            extra = None
        if not kind:
            continue
        extra_json = None
        if extra not in (None, "", {}):
            try:
                extra_json = json.dumps(extra)
            except TypeError:
                extra_json = json.dumps(str(extra))
        rows.append((task_id, str(kind), ref_id, label, extra_json))

    if not rows:
        return

    sql = """
        INSERT INTO core_task_scope_members (task_id, member_kind, member_id, member_label, extra_json)
        VALUES (?, ?, ?, ?, ?)
    """

    def _insert() -> None:
        with conn:
            conn.executemany(sql, rows)

    try:
        _run_with_retry(conn, _insert, "record_task_scope_members")
        log.debug("Recorded %d scope member(s) for task_id=%s", len(rows), task_id)
    except sqlite3.Error as e:
        log.exception("Failed to record scope members for task_id=%s: %s", task_id, e)

# --- NMR-specific helpers ---------------------------------------------------

def register_nmr_trace_file(
    conn: sqlite3.Connection,
    *,
    nmr_reaction_id: int,
    role: str,
    path: str,
    checksum: Optional[str] = None,
    task_id: Optional[int] = None,
) -> Optional[int]:
    """
    Insert or update a trace file record for an NMR reaction.
    """
    sql = """
        INSERT INTO nmr_trace_files (nmr_reaction_id, role, path, checksum, task_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(nmr_reaction_id, role) DO UPDATE SET
            path = excluded.path,
            checksum = excluded.checksum,
            task_id = excluded.task_id,
            created_at = datetime('now')
    """

    def _op() -> Optional[int]:
        with conn:
            conn.execute(sql, (nmr_reaction_id, role, path, checksum, task_id))
            row = conn.execute(
                """
                SELECT id FROM nmr_trace_files
                WHERE nmr_reaction_id = ? AND role = ?
                """,
                (nmr_reaction_id, role),
            ).fetchone()
            return int(row["id"]) if row else None

    try:
        return _run_with_retry(conn, _op, "register_nmr_trace_file")
    except sqlite3.Error as e:
        log.exception("Failed to register NMR trace file for reaction_id=%s, role=%s: %s", nmr_reaction_id, role, e)
        return None


def fetch_nmr_trace_files(
    conn: sqlite3.Connection,
    reaction_ids: Iterable[int],
) -> Dict[int, Dict[str, sqlite3.Row]]:
    """
    Return trace files keyed by reaction_id and role for the provided reactions.
    """
    ids = list(reaction_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    sql = f"""
        SELECT *
        FROM nmr_trace_files
        WHERE nmr_reaction_id IN ({placeholders})
    """
    rows = conn.execute(sql, ids).fetchall()
    mapping: Dict[int, Dict[str, sqlite3.Row]] = {}
    for row in rows:
        rid = int(row["nmr_reaction_id"])
        mapping.setdefault(rid, {})
        mapping[rid][row["role"]] = row
    return mapping


def fetch_nmr_reactions(
    conn: sqlite3.Connection,
    *,
    reaction_type: Optional[str] = None,
    reaction_ids: Optional[Iterable[int]] = None,
) -> List[sqlite3.Row]:
    """Return NMR reactions filtered by type and/or specific IDs."""
    clauses: List[str] = []
    params: List[Any] = []
    if reaction_type:
        clauses.append("reaction_type = ?")
        params.append(reaction_type)
    if reaction_ids:
        ids = list(reaction_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        clauses.append(f"id IN ({placeholders})")
        params.extend(ids)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT *
        FROM nmr_reactions
        {where}
        ORDER BY id
    """
    return conn.execute(sql, params).fetchall()


def finish_nmr_fit_run(
    conn: sqlite3.Connection,
    fit_run_id: int,
    status: str,
    message: Optional[str] = None,
) -> None:
    """Update an nmr_fit_runs row when a fit completes."""
    sql = """
        UPDATE nmr_fit_runs
        SET status = ?, message = ?, finished_at = datetime('now')
        WHERE id = ?
    """

    def _op() -> None:
        with conn:
            conn.execute(sql, (status, message, fit_run_id))

    try:
        _run_with_retry(conn, _op, "finish_nmr_fit_run")
    except sqlite3.Error as e:
        log.exception("Failed to finish NMR fit run %s: %s", fit_run_id, e)


def record_nmr_fit_params(
    conn: sqlite3.Connection,
    *,
    fit_run_id: int,
    entries: Iterable[Dict[str, Any]],
) -> None:
    """
    Insert or update tall parameter rows for an NMR fit run.
    Each entry should contain: param_name (required), optional param_numeric,
    param_text, and units.
    """
    rows: List[tuple] = []
    for entry in entries:
        name = entry.get("param_name")
        if not name:
            continue
        rows.append(
            (
                fit_run_id,
                str(name),
                entry.get("param_numeric"),
                entry.get("param_text"),
                entry.get("units"),
            )
        )
    if not rows:
        return

    sql = """
        INSERT INTO nmr_fit_params (fit_run_id, param_name, param_numeric, param_text, units)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(fit_run_id, param_name) DO UPDATE SET
            param_numeric = excluded.param_numeric,
            param_text = excluded.param_text,
            units = excluded.units
    """

    def _op():
        with conn:
            conn.executemany(sql, rows)

    try:
        _run_with_retry(conn, _op, "record_nmr_fit_params")
    except sqlite3.Error as e:
        log.exception("Failed to record NMR fit params for fit_run_id=%s: %s", fit_run_id, e)


def upsert_nmr_reaction(conn: sqlite3.Connection, reaction: Dict[str, Any]) -> Optional[int]:
    """
    Insert or update an NMR reaction record using kinetic_data_dir as the unique key.
    Returns the reaction id.
    """
    required = [
        "reaction_type",
        "temperature",
        "replicate",
        "num_scans",
        "time_per_read",
        "total_kinetic_reads",
        "total_kinetic_time",
        "probe",
        "probe_conc",
        "probe_solvent",
        "substrate",
        "substrate_conc",
        "buffer_id",
        "nmr_machine",
        "kinetic_data_dir",
    ]
    missing = [field for field in required if reaction.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s) for nmr_reactions: {', '.join(missing)}")

    sql = """
        INSERT INTO nmr_reactions (
            reaction_type, temperature, replicate, num_scans, time_per_read,
            total_kinetic_reads, total_kinetic_time, probe, probe_conc, probe_solvent,
            substrate, substrate_conc, buffer_id, nmr_machine, kinetic_data_dir,
            mnova_analysis_dir, raw_fid_dir
        ) VALUES (
            :reaction_type, :temperature, :replicate, :num_scans, :time_per_read,
            :total_kinetic_reads, :total_kinetic_time, :probe, :probe_conc, :probe_solvent,
            :substrate, :substrate_conc, :buffer_id, :nmr_machine, :kinetic_data_dir,
            :mnova_analysis_dir, :raw_fid_dir
        )
        ON CONFLICT(kinetic_data_dir) DO UPDATE SET
            reaction_type = excluded.reaction_type,
            temperature = excluded.temperature,
            replicate = excluded.replicate,
            num_scans = excluded.num_scans,
            time_per_read = excluded.time_per_read,
            total_kinetic_reads = excluded.total_kinetic_reads,
            total_kinetic_time = excluded.total_kinetic_time,
            probe = excluded.probe,
            probe_conc = excluded.probe_conc,
            probe_solvent = excluded.probe_solvent,
            substrate = excluded.substrate,
            substrate_conc = excluded.substrate_conc,
            buffer_id = excluded.buffer_id,
            nmr_machine = excluded.nmr_machine,
            mnova_analysis_dir = excluded.mnova_analysis_dir,
            raw_fid_dir = excluded.raw_fid_dir
    """

    def _op() -> Optional[int]:
        with conn:
            conn.execute(sql, reaction)
            row = conn.execute(
                "SELECT id FROM nmr_reactions WHERE kinetic_data_dir = ?",
                (reaction["kinetic_data_dir"],),
            ).fetchone()
            return int(row[0]) if row else None

    try:
        return _run_with_retry(conn, _op, "upsert_nmr_reaction")
    except sqlite3.Error as e:
        log.exception("Failed to upsert nmr_reactions row for %s: %s", reaction.get("kinetic_data_dir"), e)
        return None


def get_nmr_reaction_id_by_dir(conn: sqlite3.Connection, kinetic_data_dir: str) -> Optional[int]:
    """Resolve an NMR reaction id by kinetic_data_dir."""
    if kinetic_data_dir in (None, ""):
        return None
    sql = "SELECT id FROM nmr_reactions WHERE kinetic_data_dir = ?"
    try:
        row = _run_with_retry(conn, lambda: conn.execute(sql, (kinetic_data_dir,)).fetchone(), "get_nmr_reaction_id_by_dir")
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to lookup nmr_reaction by kinetic_data_dir '%s': %s", kinetic_data_dir, e)
        return None

# --- Domain-specific write operations ---

def insert_probe_fmod_run(conn: sqlite3.Connection, run_data: Dict[str, Any]) -> Optional[int]:
    """
    Inserts a new fmod_calc_run record.

    Args:
        run_data: A dictionary with keys matching the probe_fmod_runs table columns.

    Returns:
        The ID of the new record, or None on failure.
    """
    sql = """
        INSERT INTO probe_fmod_runs (software_name, software_version, run_args, run_datetime, output_dir, s_id)
        VALUES (:software_name, :software_version, :run_args, :run_datetime, :output_dir, :s_id)
    """
    def _op() -> Optional[int]:
        with conn:
            cursor = conn.execute(sql, run_data)
            return cursor.lastrowid

    try:
        run_id = _run_with_retry(conn, _op, "insert_probe_fmod_run")
        if run_id is not None:
            log.info("Inserted probe_fmod_run for sample ID %s (Run ID: %s)", run_data.get('s_id'), run_id)
        return run_id
    except sqlite3.IntegrityError as e:
        log.error("probe_fmod_run for sample ID %s already exists. %s", run_data.get('s_id'), e)
        return None


def bulk_insert_fmod_vals(conn: sqlite3.Connection, fmod_values: List[Dict[str, Any]]):
    """
    Performs a bulk insert of fmod values.

    Args:
        fmod_values: A list of dictionaries, each representing a row in probe_fmod_values.
    """
    sql = """
        INSERT INTO probe_fmod_values (nt_id, fmod_run_id, fmod_val, valtype, read_depth, rxn_id)
        VALUES (:nt_id, :fmod_run_id, :fmod_val, :valtype, :read_depth, :rxn_id)
    """
    if not fmod_values:
        return

    def _insert_chunk(chunk: List[Dict[str, Any]]):
        def _op():
            with conn:
                conn.executemany(sql, chunk)

        _run_with_retry(conn, _op, f"bulk_insert_fmod_vals chunk(size={len(chunk)})")

    try:
        for chunk in _iter_chunks(fmod_values, size=500):
            _insert_chunk(chunk)
        log.info("Bulk inserted %d fmod_vals records.", len(fmod_values))
    except sqlite3.Error as e:
        log.exception("Bulk insert of fmod_vals failed: %s", e)


def delete_probe_tc_fit_runs(
    conn: sqlite3.Connection,
    *,
    fit_kind: str,
    rg_id: Optional[int],
    nt_id: Optional[int] = None,
) -> int:
    """
    Remove existing probe timecourse fit runs (and their params via cascade)
    matching the provided criteria.
    """
    clauses = ["fit_kind = ?"]
    params: List[Any] = [fit_kind]
    if rg_id is not None:
        clauses.append("rg_id = ?")
        params.append(rg_id)
    else:
        clauses.append("rg_id IS NULL")
    if nt_id is None:
        clauses.append("nt_id IS NULL")
    else:
        clauses.append("nt_id = ?")
        params.append(nt_id)
    sql = f"""
        DELETE FROM probe_tc_fit_runs
        WHERE {' AND '.join(clauses)}
    """

    def _op() -> int:
        with conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount or 0

    try:
        return _run_with_retry(conn, _op, "delete_probe_tc_fit_runs")
    except sqlite3.Error as e:
        log.exception(
            "Failed to delete probe_tc_fit_runs for fit_kind=%s, rg_id=%s, nt_id=%s: %s",
            fit_kind,
            rg_id,
            nt_id,
            e,
        )
        return 0


def begin_probe_tc_fit_run(
    conn: sqlite3.Connection,
    *,
    fit_kind: str,
    rg_id: Optional[int],
    nt_id: Optional[int],
    model: Optional[str] = None,
    fmod_run_id: Optional[int] = None,
) -> Optional[int]:
    """
    Insert a row into probe_tc_fit_runs and return its primary key.
    """
    sql = """
        INSERT INTO probe_tc_fit_runs (fit_kind, fmod_run_id, rg_id, nt_id, model)
        VALUES (?, ?, ?, ?, ?)
    """

    def _op() -> Optional[int]:
        with conn:
            cursor = conn.execute(sql, (fit_kind, fmod_run_id, rg_id, nt_id, model))
            return cursor.lastrowid

    try:
        return _run_with_retry(conn, _op, "begin_probe_tc_fit_run")
    except sqlite3.Error as e:
        log.exception(
            "Failed to record probe_tc_fit_run for fit_kind=%s, rg_id=%s, nt_id=%s: %s",
            fit_kind,
            rg_id,
            nt_id,
            e,
        )
        return None


def record_probe_tc_fit_params(
    conn: sqlite3.Connection,
    *,
    fit_run_id: int,
    entries: Iterable[Dict[str, Any]],
) -> None:
    """
    Insert parameter rows for a probe timecourse fit run.
    """
    sql = """
        INSERT INTO probe_tc_fit_params (fit_run_id, param_name, param_numeric, param_text, units)
        VALUES (:fit_run_id, :param_name, :param_numeric, :param_text, :units)
        ON CONFLICT(fit_run_id, param_name) DO UPDATE SET
            param_numeric = excluded.param_numeric,
            param_text = excluded.param_text,
            units = excluded.units
    """

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        name = entry.get("param_name")
        if not name:
            continue
        row = {
            "fit_run_id": fit_run_id,
            "param_name": str(name),
            "param_numeric": entry.get("param_numeric"),
            "param_text": entry.get("param_text"),
            "units": entry.get("units"),
        }
        rows.append(row)
    if not rows:
        return

    def _op() -> None:
        with conn:
            conn.executemany(sql, rows)

    try:
        _run_with_retry(conn, _op, "record_probe_tc_fit_params")
    except sqlite3.Error as e:
        log.exception("Failed to insert probe_tc_fit_params for run_id=%s: %s", fit_run_id, e)


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
    def _op():
        with conn:
            conn.execute(sql, fit_data)

    try:
        _run_with_retry(conn, _op, "upsert_free_fit")
        log.debug("Upserted free_tc_fit for rg_id %s, nt_id %s", fit_data.get('rg_id'), fit_data.get('nt_id'))
    except sqlite3.Error as e:
        log.exception("Upsert of free_tc_fit failed: %s", e)


# --- Inline ingestion helpers (create task) ---
def upsert_construct(conn: sqlite3.Connection, construct: Dict[str, Any]) -> Optional[int]:
    """
    Insert a construct or return existing ID based on unique key, and ensure
    nucleotides are present for the construct.

    Behavior:
    - Inserts into `constructs` using unique key (family, name, version, sequence, disp_name).
    - If successful, upserts nucleotides for this construct:
        - If the caller provides `construct['nt_rows']` (list of {site, base, base_region}),
          use those.
        - Otherwise, generate default per-nt rows from the sequence via
          `generate_nt_rows_default` (caller may customize implementation).
    - If nucleotide upsert fails for a newly inserted construct, the transaction is rolled back
      so the construct is not created.

    Returns the construct id, or None on failure.
    """
    insert_sql = (
        """
        INSERT INTO meta_constructs (family, name, version, sequence, disp_name)
        VALUES (:family, :name, :version, :sequence, :disp_name)
        ON CONFLICT(family, name, version, sequence, disp_name) DO NOTHING
        """
    )
    def _resolve_construct_id() -> int:
        with conn:
            conn.execute(insert_sql, construct)
            row = conn.execute(
                """
                SELECT id FROM meta_constructs
                WHERE family = ? AND name = ? AND version = ? AND sequence = ? AND disp_name = ?
                """,
                (
                    construct.get("family"),
                    construct.get("name"),
                    construct.get("version"),
                    construct.get("sequence"),
                    construct.get("disp_name"),
                ),
            ).fetchone()
        cid_local = int(row[0]) if row else None
        if cid_local is None:
            raise sqlite3.IntegrityError("Failed to resolve construct id after upsert")
        return cid_local

    try:
        cid = _run_with_retry(conn, _resolve_construct_id, "upsert_construct insert")

        nt_rows = construct.get("nt_rows")
        if nt_rows is None:
            nt_rows = generate_nt_rows_default(construct.get("sequence", ""))

        if not isinstance(nt_rows, list):
            raise ValueError("nt_rows must be a list of {site, base, base_region} dicts")

        _bulk_upsert_nucleotides(conn, cid, nt_rows)

        log.debug("Upserted construct '%s' -> id=%s (with %d nucleotides)",
                  construct.get("disp_name"), cid, len(nt_rows))
        return cid
    except Exception as e:
        log.exception("Upsert construct (with nucleotides) failed: %s", e)
        return None


def _bulk_upsert_nucleotides(conn: sqlite3.Connection, construct_id: int, nt_rows: List[Dict[str, Any]]) -> int:
    """
    Bulk upsert nucleotides for a construct.

    Each row must have: site (int), base (str), base_region (str).
    Inserts with ON CONFLICT DO NOTHING to respect unique(site, base, base_region, construct_id).

    Returns number of rows attempted (length of nt_rows).
    """
    rows = []
    for nt in nt_rows:
        try:
            site = int(nt.get("site"))
        except Exception as e:
            raise ValueError(f"Invalid site in nt_rows: {nt}") from e
        base = str(nt.get("base")).strip()
        base_region = str(nt.get("base_region", "")).strip() or "unknown"
        rows.append({
            "site": site,
            "base": base,
            "base_region": base_region,
            "construct_id": construct_id,
        })

    sql = (
        """
        INSERT INTO meta_nucleotides (site, base, base_region, construct_id)
        VALUES (:site, :base, :base_region, :construct_id)
        ON CONFLICT(construct_id, site) DO UPDATE SET
            base = excluded.base,
            base_region = excluded.base_region
        """
    )

    def _insert_chunk(chunk: List[Dict[str, Any]]):
        def _op():
            with conn:
                conn.executemany(sql, chunk)

        _run_with_retry(conn, _op, f"_bulk_upsert_nucleotides chunk(size={len(chunk)})")

    for chunk in _iter_chunks(rows, size=500):
        _insert_chunk(chunk)
    return len(rows)


def generate_nt_rows_default(sequence: str) -> List[Dict[str, Any]]:
    """
    Generate default nucleotide rows from a construct sequence using case-based
    segmentation into primer/target/primer.

    Rules:
    - Scan left→right, starting region_id=0; increment on case switch.
    - If max_region == 2 (lower–upper–lower): keep regions 0/1/2 as-is; ensure region 1 is uppercase.
    - If max_region == 1 (two regions): assign uppercase region → 1; the other region → 0 if
      it is before uppercase, or 2 if it is after uppercase.
    - If max_region == 0 (single region):
        * all uppercase → region 1
        * all lowercase → error (must contain uppercase target segment)
    - Any pattern with >2 switches (max_region > 2) → error, ask user to format as
      lowercase–UPPERCASE–lowercase (or a two-region variant).

    Output rows:
      - site: 1..len(sequence)
      - base: uppercase nucleotide (T→U)
      - base_region: "0", "1", or "2"
    """
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Sequence must be a non-empty string for default nucleotide generation.")

    seq = sequence
    n = len(seq)

    # Validate characters: only A,C,G,T,U allowed (case-insensitive)
    allowed = set("ACGTUacgtu")
    bad_positions: List[int] = [i for i, ch in enumerate(seq) if ch not in allowed]
    if bad_positions:
        preview = ''.join(sorted({seq[i] for i in bad_positions}))
        raise ValueError(
            f"Sequence contains invalid characters at positions {bad_positions[:10]} (unique: {preview}). "
            "Only A,C,G,T,U are allowed."
        )

    # Determine region ids by case switching
    def is_upper(c: str) -> bool:
        return c.isalpha() and c.isupper()

    regions: List[int] = []
    cur_region = 0
    prev_upper = is_upper(seq[0])
    regions.append(cur_region)
    for i in range(1, n):
        u = is_upper(seq[i])
        if u != prev_upper:
            cur_region += 1
            prev_upper = u
        regions.append(cur_region)

    max_region = max(regions) if regions else 0

    # Count case distribution per region
    per_region_upper: Dict[int, int] = {}
    per_region_len: Dict[int, int] = {}
    for i, r in enumerate(regions):
        per_region_len[r] = per_region_len.get(r, 0) + 1
        if is_upper(seq[i]):
            per_region_upper[r] = per_region_upper.get(r, 0) + 1

    # Helper to check region all uppercase
    def region_all_upper(r: int) -> bool:
        return per_region_upper.get(r, 0) == per_region_len.get(r, 0)

    # Helper to check any uppercase overall
    any_upper = any(is_upper(c) for c in seq)

    # Map original region ids to final labels {0,1,2}
    region_map: Dict[int, int]

    if max_region > 2:
        raise ValueError(
            "Sequence contains more than three contiguous case-defined regions. "
            "Please format as lowercase–UPPERCASE–lowercase (or a two-region variant)."
        )

    if max_region == 2:
        # Expect pattern lower–upper–lower and region 1 all uppercase
        if not region_all_upper(1):
            raise ValueError("Default segmentation expects the middle region (1) to be uppercase.")
        region_map = {0: 0, 1: 1, 2: 2}
    elif max_region == 1:
        # Determine which region (0 or 1) is uppercase
        r0_upper = region_all_upper(0)
        r1_upper = region_all_upper(1)
        if r0_upper == r1_upper:
            # Either both upper or both lower — inconsistent two-region pattern
            raise ValueError(
                "Two-region sequence must have exactly one uppercase region."
            )
        if r0_upper:
            # Uppercase first, then lowercase → map to 1 then 2
            region_map = {0: 1, 1: 2}
        else:
            # Lowercase first, then uppercase → keep 0 then 1
            region_map = {0: 0, 1: 1}
    else:  # max_region == 0 (single region)
        if not any_upper:
            # All lowercase — invalid, must have at least target region 1
            raise ValueError(
                "Sequence is all lowercase. Must contain an uppercase target region (base_region=1)."
            )
        # All uppercase → whole sequence is region 1
        region_map = {0: 1}

    # Build output rows
    rows: List[Dict[str, Any]] = []
    for i, ch in enumerate(seq):
        base = ch.upper()
        if base == 'T':
            base = 'U'
        r = region_map[regions[i]]
        rows.append({
            "site": i + 1,
            "base": base,
            "base_region": str(r),
        })

    return rows


def prep_nt_rows(nt_info_path: Path) -> List[Dict[str, Any]]:
    """
    Prepare nt_rows from a CSV file with columns: site, base, base_region.

    - Accepts UTF-8 with BOM.
    - Normalizes headers (trim, lowercase, strip BOM).
    - Validates base characters (A/C/G/T/U), base_region in {0,1,2}, at least one region 1.
    - Ensures base is uppercase and converts T→U.

    Returns a list of dicts: {site:int, base:str, base_region:str}.
    """
    if not nt_info_path.is_file():
        raise FileNotFoundError(f"nt_info file not found at: {nt_info_path}")

    log.info("Reading nt_info from: %s", nt_info_path)

    nt_rows: List[Dict[str, Any]] = []
    with nt_info_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        log.info("nt_info headers detected: %s", fieldnames)

        header_map = { (fn.strip().lstrip("\ufeff")).lower(): fn for fn in fieldnames }
        required = {"site", "base", "base_region"}
        if not required.issubset(set(header_map.keys())):
            raise ValueError(f"nt_info CSV missing required columns: {required}")

        allowed = set("ACGTUacgtu")
        seen_region1 = False
        for row_idx, r in enumerate(reader, start=2):  # +2 accounts for header line
            site_raw = r[header_map["site"]]
            base_raw = r[header_map["base"]]
            region_raw = r[header_map["base_region"]]

            try:
                site_val = int(str(site_raw).strip())
            except Exception as e:
                raise ValueError(f"Invalid site at CSV line {row_idx}: {site_raw}") from e

            base_str = str(base_raw).strip()
            if not base_str or any(ch not in allowed for ch in base_str):
                raise ValueError(
                    f"Invalid base '{base_str}' at CSV line {row_idx}. Only A,C,G,T,U allowed."
                )

            region_str = str(region_raw).strip()
            if region_str not in {"0", "1", "2"}:
                raise ValueError(
                    f"Invalid base_region '{region_str}' at CSV line {row_idx}. Must be one of 0,1,2."
                )
            if region_str == "1":
                seen_region1 = True
                if base_str.isalpha() and not base_str.isupper():
                    raise ValueError(
                        f"Row at CSV line {row_idx} has base_region=1 but base is not uppercase: '{base_str}'."
                    )

            base_up = base_str.upper()
            if base_up == "T":
                base_up = "U"

            nt_rows.append({
                "site": site_val,
                "base": base_up,
                "base_region": region_str,
            })

        if not seen_region1:
            raise ValueError(
                "nt_info CSV has no base_region=1 rows. Must contain an uppercase target region."
            )

    return nt_rows


def upsert_buffer(conn: sqlite3.Connection, buffer: Dict[str, Any]) -> Optional[int]:
    """
    Insert a buffer or return existing ID based on unique key.
    Unique key: (name, pH, composition, disp_name)
    """
    sql = (
        """
        INSERT INTO meta_buffers (name, pH, composition, disp_name)
        VALUES (:name, :pH, :composition, :disp_name)
        ON CONFLICT(name, pH, composition, disp_name) DO NOTHING
        """
    )
    def _resolve_buffer_id() -> Optional[int]:
        with conn:
            conn.execute(sql, buffer)
            row = conn.execute(
                """
                SELECT id FROM meta_buffers
                WHERE name = ? AND pH = ? AND composition = ? AND disp_name = ?
                """,
                (
                    buffer.get("name"),
                    buffer.get("pH"),
                    buffer.get("composition"),
                    buffer.get("disp_name"),
                ),
            ).fetchone()
        return int(row[0]) if row else None

    try:
        bid = _run_with_retry(conn, _resolve_buffer_id, "upsert_buffer")
        log.debug("Upserted buffer '%s' -> id=%s", buffer.get("disp_name"), bid)
        return bid
    except sqlite3.Error as e:
        log.exception("Upsert buffer failed: %s", e)
        return None


def upsert_sequencing_run(conn: sqlite3.Connection, run: Dict[str, Any]) -> Optional[int]:
    """
    Insert a sequencing run or return existing ID based on unique key.
    Unique key: (run_name, date, sequencer, run_manager)
    """
    sql = (
        """
        INSERT INTO sequencing_runs (run_name, date, sequencer, run_manager)
        VALUES (:run_name, :date, :sequencer, :run_manager)
        ON CONFLICT(run_name, date, sequencer, run_manager) DO NOTHING
        """
    )
    def _resolve_run_id() -> Optional[int]:
        with conn:
            conn.execute(sql, run)
            row = conn.execute(
                """
                SELECT id FROM sequencing_runs
                WHERE run_name = ? AND date = ? AND sequencer = ? AND run_manager = ?
                """,
                (
                    run.get("run_name"),
                    run.get("date"),
                    run.get("sequencer"),
                    run.get("run_manager"),
                ),
            ).fetchone()
        return int(row[0]) if row else None

    try:
        rid = _run_with_retry(conn, _resolve_run_id, "upsert_sequencing_run")
        log.debug("Upserted sequencing_run '%s' -> id=%s", run.get("run_name"), rid)
        return rid
    except sqlite3.Error as e:
        log.exception("Upsert sequencing_run failed: %s", e)
        return None


def bulk_upsert_samples(conn: sqlite3.Connection, seqrun_id: int, samples: List[Dict[str, Any]]) -> int:
    """
    Bulk upsert sequencing samples for a given sequencing run.
    On conflict of (seqrun_id, sample_name, fq_dir), updates r1_file, r2_file, to_drop.

    Returns number of rows affected (inserted or updated attempts).
    """
    if not samples:
        return 0

    # Normalize records and default to_drop
    rows = []
    for s in samples:
        rows.append(
            {
                "seqrun_id": seqrun_id,
                "sample_name": s.get("sample_name"),
                "fq_dir": s.get("fq_dir"),
                "r1_file": s.get("r1_file"),
                "r2_file": s.get("r2_file"),
                "to_drop": int(s.get("to_drop", 0)),
            }
        )

    sql = (
        """
        INSERT INTO sequencing_samples (seqrun_id, sample_name, fq_dir, r1_file, r2_file, to_drop)
        VALUES (:seqrun_id, :sample_name, :fq_dir, :r1_file, :r2_file, :to_drop)
        ON CONFLICT(seqrun_id, sample_name, fq_dir) DO UPDATE SET
            r1_file = excluded.r1_file,
            r2_file = excluded.r2_file,
            to_drop = excluded.to_drop
        """
    )
    def _insert_chunk(chunk: List[Dict[str, Any]]):
        def _op():
            with conn:
                conn.executemany(sql, chunk)

        _run_with_retry(conn, _op, f"bulk_upsert_samples chunk(size={len(chunk)})")

    try:
        total = 0
        for chunk in _iter_chunks(rows, size=200):
            _insert_chunk(chunk)
            total += len(chunk)
        log.info("Upserted %d sequencing_samples for seqrun_id=%s", total, seqrun_id)
        return total
    except sqlite3.Error as e:
        log.exception("Bulk upsert samples failed: %s", e)
        return 0


# --- Additional helpers for probing reactions and reaction groups ---
def get_sample_id(conn: sqlite3.Connection, seqrun_id: int, sample_name: str, fq_dir: Optional[str] = None) -> Optional[int]:
    """
    Fetch the id of a sequencing sample by seqrun_id and sample_name.
    If fq_dir is provided, include it in the filter for disambiguation.
    """
    try:
        if fq_dir is None:
            row = conn.execute(
                "SELECT id FROM sequencing_samples WHERE seqrun_id = ? AND sample_name = ?",
                (seqrun_id, sample_name),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM sequencing_samples WHERE seqrun_id = ? AND sample_name = ? AND fq_dir = ?",
                (seqrun_id, sample_name, fq_dir),
            ).fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to fetch sample id for '%s' (seqrun_id=%s): %s", sample_name, seqrun_id, e)
        return None


def get_sample_id_by_name(conn: sqlite3.Connection, sample_name: str) -> Optional[int]:
    """
    Fetch a sequencing sample id by its name across all sequencing runs.
    If multiple rows match, returns None to avoid ambiguity.
    """
    try:
        rows = conn.execute(
            "SELECT id FROM sequencing_samples WHERE sample_name = ?",
            (sample_name,),
        ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            log.error("Sample name '%s' is ambiguous across sequencing runs (found %d). Provide seqrun context.", sample_name, len(rows))
            return None
        return int(rows[0][0])
    except sqlite3.Error as e:
        log.exception("Failed to fetch sample id by name '%s': %s", sample_name, e)
        return None


def get_construct_id_by_disp_name(conn: sqlite3.Connection, disp_name: str) -> Optional[int]:
    """
    Resolve a construct id by its disp_name (case-insensitive).
    """
    if disp_name in (None, ""):
        return None
    query = "SELECT id FROM meta_constructs WHERE LOWER(disp_name) = LOWER(?) LIMIT 1"
    try:
        row = _run_with_retry(
            conn,
            lambda: conn.execute(query, (str(disp_name),)).fetchone(),
            "get_construct_id_by_disp_name",
        )
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to lookup construct by disp_name '%s': %s", disp_name, e)
        return None


def get_buffer_id_by_name_or_disp(conn: sqlite3.Connection, identifier: str) -> Optional[int]:
    """
    Resolve a buffer id by disp_name or name (case-insensitive).
    """
    if identifier in (None, ""):
        return None
    ident = str(identifier).strip()
    queries = [
        "SELECT id FROM meta_buffers WHERE LOWER(disp_name) = LOWER(?) LIMIT 1",
        "SELECT id FROM meta_buffers WHERE LOWER(name) = LOWER(?) LIMIT 1",
    ]
    try:
        for query in queries:
            row = _run_with_retry(
                conn,
                lambda q=query: conn.execute(q, (ident,)).fetchone(),
                "get_buffer_id_by_name_or_disp",
            )
            if row:
                return int(row[0])
        return None
    except sqlite3.Error as e:
        log.exception("Failed to lookup buffer '%s': %s", identifier, e)
        return None


def get_max_reaction_group_id(conn: sqlite3.Connection) -> int:
    """Return the current max rg_id from probe_reaction_groups, or 0 if none."""
    try:
        row = conn.execute("SELECT COALESCE(MAX(rg_id), 0) FROM probe_reaction_groups").fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.Error as e:
        log.exception("Failed to fetch max rg_id: %s", e)
        return 0


def find_sequencing_run_id_by_name(conn: sqlite3.Connection, run_name: str) -> Optional[int]:
    """
    Resolve a sequencing run id by its run_name.
    """
    if run_name in (None, ""):
        return None
    try:
        row = _run_with_retry(
            conn,
            lambda: conn.execute(
                "SELECT id FROM sequencing_runs WHERE run_name = ? LIMIT 1",
                (run_name,),
            ).fetchone(),
            "find_sequencing_run_id_by_name",
        )
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to lookup sequencing run '%s': %s", run_name, e)
        return None


def insert_probing_reaction(conn: sqlite3.Connection, reaction: Dict[str, Any]) -> Optional[int]:
    """
    Insert a probing_reaction and return its id.
    Expected keys in reaction dict:
      temperature, replicate, reaction_time, probe_concentration, probe,
      rt_protocol, done_by, treated, buffer_id, construct_id, rg_id, s_id
    """
    sql = (
        """
        INSERT INTO probe_reactions (
            temperature, replicate, reaction_time, probe_concentration, probe,
            rt_protocol, done_by, treated, buffer_id, construct_id, rg_id, s_id
        ) VALUES (
            :temperature, :replicate, :reaction_time, :probe_concentration, :probe,
            :rt_protocol, :done_by, :treated, :buffer_id, :construct_id, :rg_id, :s_id
        )
        """
    )
    try:
        with conn:
            cur = conn.execute(sql, reaction)
            rxn_id = cur.lastrowid
            log.debug("Inserted probing_reaction id=%s for sample_id=%s (rg_id=%s)", rxn_id, reaction.get("s_id"), reaction.get("rg_id"))
            return rxn_id
    except sqlite3.Error as e:
        log.exception("Failed to insert probing_reaction: %s", e)
        return None


# Deprecated: insert_reaction_group_pair was used when reaction_groups stored per-reaction mappings.
# The schema now normalizes reaction_groups as a catalog (rg_id, rg_label), so this is removed.


def find_rg_id_by_label(conn: sqlite3.Connection, rg_label: str) -> Optional[int]:
    """Return an existing rg_id for a given label, or None if not found."""
    try:
        row = conn.execute(
            "SELECT rg_id FROM probe_reaction_groups WHERE rg_label = ? LIMIT 1",
            (rg_label,),
        ).fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to lookup rg_id by label '%s': %s", rg_label, e)
        return None


def upsert_reaction_group(conn: sqlite3.Connection, rg_id: int, rg_label: Optional[str]) -> Optional[int]:
    """
    Ensure a reaction_groups row exists and sets the label exactly to the provided value.
    If the row exists, updates rg_label to match the YAML.
    Returns the rg_id on success, None on failure.
    """
    try:
        if rg_label is None:
            sql = "INSERT INTO probe_reaction_groups (rg_id) VALUES (?) ON CONFLICT(rg_id) DO NOTHING"
            params = (rg_id,)
        else:
            sql = (
                "INSERT INTO probe_reaction_groups (rg_id, rg_label) VALUES (?, ?) "
                "ON CONFLICT(rg_id) DO UPDATE SET rg_label=excluded.rg_label"
            )
            params = (rg_id, str(rg_label))

        with conn:
            conn.execute(sql, params)
        return rg_id
    except sqlite3.Error as e:
        log.exception("Failed to upsert reaction_group rg_id=%s label=%s: %s", rg_id, rg_label, e)
        return None


# --- Derived samples helpers ---
def upsert_derived_sample(conn: sqlite3.Connection, parent_sample_id: int, child_name: str,
                          kind: str, tool: str, cmd_template: str,
                          params: Optional[Dict[str, Any]] = None,
                          cache_key: Optional[str] = None) -> Optional[int]:
    """
    Insert or update a derived sample definition. On conflict of
    (parent_sample_id, child_name), updates fields.
    Returns the row id.
    """
    if params is None:
        params = {}
    params_json = json.dumps(params, sort_keys=True)
    if cache_key is None:
        # Use a robust 64-char cache key over defining attributes
        cache_key = config_hash({
            "parent_sample_id": parent_sample_id,
            "child_name": child_name,
            "kind": kind,
            "tool": tool,
            "cmd_template": cmd_template,
            "params": params,
        }, length=64)

    sql = """
        INSERT INTO sequencing_derived_samples (
            parent_sample_id, child_name, kind, tool, cmd_template, params_json, cache_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(parent_sample_id, child_name) DO UPDATE SET
            kind = excluded.kind,
            tool = excluded.tool,
            cmd_template = excluded.cmd_template,
            params_json = excluded.params_json,
            cache_key = excluded.cache_key
    """
    try:
        with conn:
            conn.execute(sql, (parent_sample_id, child_name, kind, tool, cmd_template, params_json, cache_key))
        # Fetch id
        row = conn.execute(
            "SELECT id FROM sequencing_derived_samples WHERE parent_sample_id = ? AND child_name = ?",
            (parent_sample_id, child_name),
        ).fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to upsert derived_sample '%s' for parent %s: %s", child_name, parent_sample_id, e)
        return None


def get_derived_by_child_name(conn: sqlite3.Connection, child_name: str) -> Optional[sqlite3.Row]:
    try:
        row = conn.execute(
            "SELECT * FROM sequencing_derived_samples WHERE child_name = ?",
            (child_name,),
        ).fetchone()
        return row
    except sqlite3.Error as e:
        log.exception("Failed to fetch derived sample by child_name '%s': %s", child_name, e)
        return None


def get_derived_for_parent(conn: sqlite3.Connection, parent_sample_id: int) -> list:
    try:
        rows = conn.execute(
            "SELECT * FROM sequencing_derived_samples WHERE parent_sample_id = ?",
            (parent_sample_id,),
        ).fetchall()
        return rows or []
    except sqlite3.Error as e:
        log.exception("Failed to fetch derived samples for parent %s: %s", parent_sample_id, e)
        return []

def _is_lock_error(err: sqlite3.Error) -> bool:
    """Return True if the sqlite error looks like a lock/busy condition."""
    msg = str(err).lower()
    return any(token in msg for token in _LOCK_RETRY_MESSAGES)


def _run_with_retry(
    conn: sqlite3.Connection,
    operation: Callable[[], Any],
    description: str,
    *,
    retries: int = 5,
    initial_delay: float = 0.1,
    backoff: float = 2.0,
) -> Any:
    """
    Execute `operation`, retrying when SQLite reports a lock/busy error.

    Retries are exponential-backoff with jitter-free timing to keep behaviour
    predictable for batch jobs.
    """
    delay = initial_delay
    last_error: Optional[sqlite3.Error] = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except sqlite3.OperationalError as err:
            last_error = err
            if not _is_lock_error(err):
                raise
            if attempt == retries:
                break
            log.warning(
                "SQLite busy during %s (attempt %s/%s); retrying in %.2fs",
                description,
                attempt,
                retries,
                delay,
            )
            try:
                conn.rollback()
            except Exception:
                pass
            time.sleep(delay)
            delay *= backoff
    if last_error is not None:
        raise last_error


def _iter_chunks(items: Iterable[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    """Yield successive chunks from an iterable."""
    chunk: List[Dict[str, Any]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


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
        # Ensure FK constraints (including ON DELETE CASCADE) are enforced
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
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
               backend: str, output_dir: str, label: str, cache_key: Optional[str],
               tool: Optional[str] = None, tool_version: Optional[str] = None) -> Optional[int]:
    """
    Records the start of a new task.

    Returns:
        The ID of the newly created task, or None on failure.
    """
    sql = """
        INSERT INTO core_tasks (
            task_name, scope_kind, scope_id, backend, output_dir, label,
            cache_key, tool, tool_version, config_hash, started_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    def _insert() -> Optional[int]:
        with conn:
            cfg_short = (cache_key or "")[:7]
            cursor = conn.execute(
                sql,
                (
                    name,
                    scope_kind,
                    scope_id,
                    backend,
                    output_dir,
                    label,
                    cache_key,
                    tool,
                    tool_version,
                    cfg_short,
                    datetime.now().isoformat(),
                ),
            )
            task_id = cursor.lastrowid
            log.info(
                "Began task '%s' (ID: %s) for %s:%s (cfg=%s)",
                name,
                task_id,
                scope_kind,
                scope_id,
                cache_key,
            )
            return task_id

    try:
        return _run_with_retry(conn, _insert, "begin_task")
    except sqlite3.IntegrityError as e:
        log.error("Failed to begin task '%s' due to integrity error: %s", name, e)
        # Reuse existing non-cached task row for this signature (task_name,label,output_dir,cache_key)
        try:
            row = _run_with_retry(
                conn,
                lambda: conn.execute(
                    """
                    SELECT id, state FROM core_tasks
                    WHERE task_name = ? AND label = ? AND output_dir = ?
                      AND ((? IS NULL AND cache_key IS NULL) OR cache_key = ?)
                    ORDER BY started_at DESC
                    LIMIT 1
                    """,
                    (name, label, output_dir, cache_key, cache_key),
                ).fetchone(),
                "begin_task reuse lookup",
            )
            if row:
                tid = int(row[0])
                # Reset state to pending and clear message/ended_at for a new attempt
                def _reuse_update():
                    with conn:
                        conn.execute(
                            "UPDATE core_tasks SET state='pending', message=NULL, started_at=datetime('now'), ended_at=NULL WHERE id = ?",
                            (tid,),
                        )

                _run_with_retry(conn, _reuse_update, "begin_task reuse update")
                log.info("Reusing existing task row id=%s for signature (task=%s,label=%s)", tid, name, label)
                return tid
        except Exception as e2:
            log.warning("Could not reuse existing task row after integrity error: %s", e2)
        return None


def attempt(conn: sqlite3.Connection, task_id: int, try_index: int, command: str,
            resources: Dict[str, Any], log_path: Path) -> Optional[int]:
    """
    Records a single attempt to execute a task.

    Returns:
        The ID of the task attempt, or None on failure.
    """
    sql = """
        INSERT INTO core_task_attempts (task_id, try_index, command, resources_json, log_path)
        VALUES (?, ?, ?, ?, ?)
    """
    def _next_try_index(c: sqlite3.Connection, tid: int) -> int:
        row = c.execute(
            "SELECT COALESCE(MAX(try_index), 0) + 1 FROM core_task_attempts WHERE task_id = ?",
            (tid,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 1

    def _insert(idx_val: int, payload_json: str) -> Optional[int]:
        def _op():
            with conn:
                cursor = conn.execute(sql, (task_id, idx_val, command, payload_json, str(log_path)))
                return cursor.lastrowid

        attempt_id = _run_with_retry(conn, _op, f"attempt insert try_index={idx_val}")
        log.info("Logged attempt #%d for task ID %d (Attempt ID: %s)", idx_val, task_id, attempt_id)
        return attempt_id

    resources_json = json.dumps(resources)
    idx = int(try_index) if isinstance(try_index, int) and try_index > 0 else _run_with_retry(
        conn,
        lambda: _next_try_index(conn, task_id),
        "attempt next_try_index",
    )
    try:
        return _insert(idx, resources_json)
    except sqlite3.IntegrityError:
        try:
            idx = _run_with_retry(conn, lambda: _next_try_index(conn, task_id), "attempt retry next_try_index")
            return _insert(idx, json.dumps(resources))
        except sqlite3.Error as e2:
            log.exception("Failed to record attempt for task %d after retry: %s", task_id, e2)
            return None
    except sqlite3.Error as e:
        log.exception("Failed to record attempt for task %d: %s", task_id, e)
        return None


def finish_task(conn: sqlite3.Connection, task_id: int, state: str, message: Optional[str] = None,
                cache_key: Optional[str] = None):
    """
    Updates a task's final state (e.g., 'completed', 'failed').
    If cache_key is provided, updates it; otherwise preserves the existing value.
    """
    def _op():
        with conn:
            if cache_key is None:
                conn.execute(
                    """
                    UPDATE core_tasks
                    SET state = ?, message = ?, ended_at = ?
                    WHERE id = ?
                    """,
                    (state, message, datetime.now().isoformat(), task_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE core_tasks
                    SET state = ?, message = ?, cache_key = ?, ended_at = ?
                    WHERE id = ?
                    """,
                    (state, message, cache_key, datetime.now().isoformat(), task_id),
                )
    try:
        _run_with_retry(conn, _op, "finish_task")
        log.info("Finished task ID %d with state '%s'", task_id, state)
    except sqlite3.Error as e:
        log.exception("Failed to finish task %d: %s", task_id, e)


def find_task_by_signature(conn: sqlite3.Connection, label: str, output_dir: str,
                           cache_key: Optional[str]) -> Optional[sqlite3.Row]:
    """
    Returns the most recent task row matching the same logical signature
    (label, output_dir, cache_key), or None if not found.
    Ignores task_name to account for implementations where it may include timestamps.
    """
    sql = (
        """
        SELECT id, task_name, label, output_dir, cache_key, state, started_at, ended_at
        FROM core_tasks
        WHERE label = ? AND output_dir = ? AND ((? IS NULL AND cache_key IS NULL) OR cache_key = ?)
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    try:
        row = conn.execute(sql, (label, output_dir, cache_key, cache_key)).fetchone()
        return row
    except sqlite3.Error as e:
        log.exception("Failed to query existing task by signature: %s", e)
        return None

def find_completed_task_by_signature(conn: sqlite3.Connection, label: str, output_dir: str,
                                     cache_key: Optional[str]) -> Optional[sqlite3.Row]:
    """
    Returns the most recent COMPLETED task row matching (label, output_dir, cache_key).
    Ignores cached rows and other states to reliably detect a prior success even if
    a newer 'cached' entry exists.
    """
    sql = (
        """
        SELECT id, task_name, label, output_dir, cache_key, state, started_at, ended_at
        FROM core_tasks
        WHERE label = ? AND output_dir = ? AND ((? IS NULL AND cache_key IS NULL) OR cache_key = ?)
          AND state = 'completed'
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    try:
        return conn.execute(sql, (label, output_dir, cache_key, cache_key)).fetchone()
    except sqlite3.Error as e:
        log.exception("Failed to query completed task by signature: %s", e)
        return None


def record_cached_task(conn: sqlite3.Connection, name: str, scope_kind: str, scope_id: Optional[int],
                       backend: str, output_dir: str, label: str, cache_key: Optional[str],
                       message: Optional[str] = None,
                       tool: Optional[str] = None, tool_version: Optional[str] = None) -> Optional[int]:
    """
    Inserts a task row with state='cached' to record a skipped run due to identical config.

    Uses a single INSERT to avoid unique-index conflicts on active states.
    Sets both started_at and ended_at to now.
    """
    sql = """
        INSERT INTO core_tasks (
            task_name, scope_kind, scope_id, backend, output_dir, label,
            cache_key, tool, tool_version, config_hash, started_at, ended_at, state, message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), 'cached', ?)
    """
    def _op() -> Optional[int]:
        with conn:
            cfg_short = (cache_key or "")[:7]
            cur = conn.execute(sql, (name, scope_kind, scope_id, backend, output_dir, label, cache_key, tool, tool_version, cfg_short, message))
            return cur.lastrowid

    try:
        task_id = _run_with_retry(conn, _op, "record_cached_task")
        if task_id is not None:
            log.info("Recorded cached task '%s' (ID: %s) for label '%s'", name, task_id, label)
        return task_id
    except sqlite3.IntegrityError as e:
        log.error("Failed to record cached task due to integrity error: %s", e)
        return None
    except sqlite3.Error as e:
        log.exception("Failed to record cached task: %s", e)
        return None


def record_task_scope_members(
    conn: sqlite3.Connection,
    task_id: int,
    members: Iterable[Any],
) -> None:
    """
    Inserts rows into core_task_scope_members to capture the entities processed by a task run.
    Accepts TaskScopeMember dataclass instances or plain dictionaries with at least a 'kind'.
    """
    rows: List[tuple[Any, Any, Any, Any, Any]] = []
    for member in members or []:
        if member is None:
            continue
        if hasattr(member, "kind"):
            kind = getattr(member, "kind", None)
            ref_id = getattr(member, "ref_id", None)
            label = getattr(member, "label", None)
            extra = getattr(member, "extra", None)
        elif isinstance(member, dict):
            kind = member.get("kind")
            ref_id = member.get("ref_id", member.get("member_id"))
            label = member.get("label", member.get("member_label"))
            extra = member.get("extra", member.get("extra_json"))
        else:
            kind = None
            ref_id = None
            label = None
            extra = None
        if not kind:
            continue
        extra_json = None
        if extra not in (None, "", {}):
            try:
                extra_json = json.dumps(extra)
            except TypeError:
                extra_json = json.dumps(str(extra))
        rows.append((task_id, str(kind), ref_id, label, extra_json))

    if not rows:
        return

    sql = """
        INSERT INTO core_task_scope_members (task_id, member_kind, member_id, member_label, extra_json)
        VALUES (?, ?, ?, ?, ?)
    """

    def _insert() -> None:
        with conn:
            conn.executemany(sql, rows)

    try:
        _run_with_retry(conn, _insert, "record_task_scope_members")
        log.debug("Recorded %d scope member(s) for task_id=%s", len(rows), task_id)
    except sqlite3.Error as e:
        log.exception("Failed to record scope members for task_id=%s: %s", task_id, e)

# --- NMR-specific helpers ---------------------------------------------------

def register_nmr_trace_file(
    conn: sqlite3.Connection,
    *,
    nmr_reaction_id: int,
    role: str,
    path: str,
    checksum: Optional[str] = None,
    task_id: Optional[int] = None,
) -> Optional[int]:
    """
    Insert or update a trace file record for an NMR reaction.
    """
    sql = """
        INSERT INTO nmr_trace_files (nmr_reaction_id, role, path, checksum, task_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(nmr_reaction_id, role) DO UPDATE SET
            path = excluded.path,
            checksum = excluded.checksum,
            task_id = excluded.task_id,
            created_at = datetime('now')
    """

    def _op() -> Optional[int]:
        with conn:
            conn.execute(sql, (nmr_reaction_id, role, path, checksum, task_id))
            row = conn.execute(
                """
                SELECT id FROM nmr_trace_files
                WHERE nmr_reaction_id = ? AND role = ?
                """,
                (nmr_reaction_id, role),
            ).fetchone()
            return int(row["id"]) if row else None

    try:
        return _run_with_retry(conn, _op, "register_nmr_trace_file")
    except sqlite3.Error as e:
        log.exception("Failed to register NMR trace file for reaction_id=%s, role=%s: %s", nmr_reaction_id, role, e)
        return None


def fetch_nmr_trace_files(
    conn: sqlite3.Connection,
    reaction_ids: Iterable[int],
) -> Dict[int, Dict[str, sqlite3.Row]]:
    """
    Return trace files keyed by reaction_id and role for the provided reactions.
    """
    ids = list(reaction_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    sql = f"""
        SELECT *
        FROM nmr_trace_files
        WHERE nmr_reaction_id IN ({placeholders})
    """
    rows = conn.execute(sql, ids).fetchall()
    mapping: Dict[int, Dict[str, sqlite3.Row]] = {}
    for row in rows:
        rid = int(row["nmr_reaction_id"])
        mapping.setdefault(rid, {})
        mapping[rid][row["role"]] = row
    return mapping


def fetch_nmr_reactions(
    conn: sqlite3.Connection,
    *,
    reaction_type: Optional[str] = None,
    reaction_ids: Optional[Iterable[int]] = None,
) -> List[sqlite3.Row]:
    """Return NMR reactions filtered by type and/or specific IDs."""
    clauses: List[str] = []
    params: List[Any] = []
    if reaction_type:
        clauses.append("reaction_type = ?")
        params.append(reaction_type)
    if reaction_ids:
        ids = list(reaction_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        clauses.append(f"id IN ({placeholders})")
        params.extend(ids)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT *
        FROM nmr_reactions
        {where}
        ORDER BY id
    """
    return conn.execute(sql, params).fetchall()


def begin_nmr_fit_run(
    conn: sqlite3.Connection,
    *,
    task_id: int,
    nmr_reaction_id: int,
    plugin: str,
    params: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    species: Optional[str] = None,
) -> Optional[int]:
    """Create an nmr_fit_runs row marking the start of a fit attempt."""
    params_json = json.dumps(params or {})
    sql = """
        INSERT INTO nmr_fit_runs (task_id, nmr_reaction_id, plugin, model, species, params_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    def _op() -> Optional[int]:
        with conn:
            cur = conn.execute(sql, (task_id, nmr_reaction_id, plugin, model, species, params_json))
            return cur.lastrowid

    try:
        return _run_with_retry(conn, _op, "begin_nmr_fit_run")
    except sqlite3.Error as e:
        log.exception("Failed to begin NMR fit run for reaction_id=%s: %s", nmr_reaction_id, e)
        return None


def finish_nmr_fit_run(
    conn: sqlite3.Connection,
    fit_run_id: int,
    status: str,
    message: Optional[str] = None,
) -> None:
    """Update an nmr_fit_runs row when a fit completes."""
    sql = """
        UPDATE nmr_fit_runs
        SET status = ?, message = ?, finished_at = datetime('now')
        WHERE id = ?
    """

    def _op() -> None:
        with conn:
            conn.execute(sql, (status, message, fit_run_id))

    try:
        _run_with_retry(conn, _op, "finish_nmr_fit_run")
    except sqlite3.Error as e:
        log.exception("Failed to finish NMR fit run %s: %s", fit_run_id, e)


def record_nmr_fit_params(
    conn: sqlite3.Connection,
    *,
    fit_run_id: int,
    entries: Iterable[Dict[str, Any]],
) -> None:
    """
    Insert or update tall parameter rows for an NMR fit run.
    Each entry should contain: param_name (required), optional param_numeric,
    param_text, and units.
    """
    rows: List[tuple] = []
    for entry in entries:
        name = entry.get("param_name")
        if not name:
            continue
        rows.append(
            (
                fit_run_id,
                str(name),
                entry.get("param_numeric"),
                entry.get("param_text"),
                entry.get("units"),
            )
        )
    if not rows:
        return

    sql = """
        INSERT INTO nmr_fit_params (fit_run_id, param_name, param_numeric, param_text, units)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(fit_run_id, param_name) DO UPDATE SET
            param_numeric = excluded.param_numeric,
            param_text = excluded.param_text,
            units = excluded.units
    """

    def _op():
        with conn:
            conn.executemany(sql, rows)

    try:
        _run_with_retry(conn, _op, "record_nmr_fit_params")
    except sqlite3.Error as e:
        log.exception("Failed to record NMR fit params for fit_run_id=%s: %s", fit_run_id, e)


def upsert_nmr_reaction(conn: sqlite3.Connection, reaction: Dict[str, Any]) -> Optional[int]:
    """
    Insert or update an NMR reaction record using kinetic_data_dir as the unique key.
    Returns the reaction id.
    """
    required = [
        "reaction_type",
        "temperature",
        "replicate",
        "num_scans",
        "time_per_read",
        "total_kinetic_reads",
        "total_kinetic_time",
        "probe",
        "probe_conc",
        "probe_solvent",
        "substrate",
        "substrate_conc",
        "buffer_id",
        "nmr_machine",
        "kinetic_data_dir",
    ]
    missing = [field for field in required if reaction.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s) for nmr_reactions: {', '.join(missing)}")

    sql = """
        INSERT INTO nmr_reactions (
            reaction_type, temperature, replicate, num_scans, time_per_read,
            total_kinetic_reads, total_kinetic_time, probe, probe_conc, probe_solvent,
            substrate, substrate_conc, buffer_id, nmr_machine, kinetic_data_dir,
            mnova_analysis_dir, raw_fid_dir
        ) VALUES (
            :reaction_type, :temperature, :replicate, :num_scans, :time_per_read,
            :total_kinetic_reads, :total_kinetic_time, :probe, :probe_conc, :probe_solvent,
            :substrate, :substrate_conc, :buffer_id, :nmr_machine, :kinetic_data_dir,
            :mnova_analysis_dir, :raw_fid_dir
        )
        ON CONFLICT(kinetic_data_dir) DO UPDATE SET
            reaction_type = excluded.reaction_type,
            temperature = excluded.temperature,
            replicate = excluded.replicate,
            num_scans = excluded.num_scans,
            time_per_read = excluded.time_per_read,
            total_kinetic_reads = excluded.total_kinetic_reads,
            total_kinetic_time = excluded.total_kinetic_time,
            probe = excluded.probe,
            probe_conc = excluded.probe_conc,
            probe_solvent = excluded.probe_solvent,
            substrate = excluded.substrate,
            substrate_conc = excluded.substrate_conc,
            buffer_id = excluded.buffer_id,
            nmr_machine = excluded.nmr_machine,
            mnova_analysis_dir = excluded.mnova_analysis_dir,
            raw_fid_dir = excluded.raw_fid_dir
    """

    def _op() -> Optional[int]:
        with conn:
            conn.execute(sql, reaction)
            row = conn.execute(
                "SELECT id FROM nmr_reactions WHERE kinetic_data_dir = ?",
                (reaction["kinetic_data_dir"],),
            ).fetchone()
            return int(row[0]) if row else None

    try:
        return _run_with_retry(conn, _op, "upsert_nmr_reaction")
    except sqlite3.Error as e:
        log.exception("Failed to upsert nmr_reactions row for %s: %s", reaction.get("kinetic_data_dir"), e)
        return None


def get_nmr_reaction_id_by_dir(conn: sqlite3.Connection, kinetic_data_dir: str) -> Optional[int]:
    """Resolve an NMR reaction id by kinetic_data_dir."""
    if kinetic_data_dir in (None, ""):
        return None
    sql = "SELECT id FROM nmr_reactions WHERE kinetic_data_dir = ?"
    try:
        row = _run_with_retry(conn, lambda: conn.execute(sql, (kinetic_data_dir,)).fetchone(), "get_nmr_reaction_id_by_dir")
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to lookup nmr_reaction by kinetic_data_dir '%s': %s", kinetic_data_dir, e)
        return None

# --- Domain-specific write operations ---

def insert_probe_fmod_run(conn: sqlite3.Connection, run_data: Dict[str, Any]) -> Optional[int]:
    """
    Inserts a new fmod_calc_run record.

    Args:
        run_data: A dictionary with keys matching the probe_fmod_runs table columns.

    Returns:
        The ID of the new record, or None on failure.
    """
    sql = """
        INSERT INTO probe_fmod_runs (software_name, software_version, run_args, run_datetime, output_dir, s_id)
        VALUES (:software_name, :software_version, :run_args, :run_datetime, :output_dir, :s_id)
    """
    def _op() -> Optional[int]:
        with conn:
            cursor = conn.execute(sql, run_data)
            return cursor.lastrowid

    try:
        run_id = _run_with_retry(conn, _op, "insert_probe_fmod_run")
        if run_id is not None:
            log.info("Inserted probe_fmod_run for sample ID %s (Run ID: %s)", run_data.get('s_id'), run_id)
        return run_id
    except sqlite3.IntegrityError as e:
        log.error("probe_fmod_run for sample ID %s already exists. %s", run_data.get('s_id'), e)
        return None


def bulk_insert_fmod_vals(conn: sqlite3.Connection, fmod_values: List[Dict[str, Any]]):
    """
    Performs a bulk insert of fmod values.

    Args:
        fmod_values: A list of dictionaries, each representing a row in probe_fmod_values.
    """
    sql = """
        INSERT INTO probe_fmod_values (nt_id, fmod_run_id, fmod_val, valtype, read_depth, rxn_id)
        VALUES (:nt_id, :fmod_run_id, :fmod_val, :valtype, :read_depth, :rxn_id)
    """
    if not fmod_values:
        return

    def _insert_chunk(chunk: List[Dict[str, Any]]):
        def _op():
            with conn:
                conn.executemany(sql, chunk)

        _run_with_retry(conn, _op, f"bulk_insert_fmod_vals chunk(size={len(chunk)})")

    try:
        for chunk in _iter_chunks(fmod_values, size=500):
            _insert_chunk(chunk)
        log.info("Bulk inserted %d fmod_vals records.", len(fmod_values))
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
    def _op():
        with conn:
            conn.execute(sql, fit_data)

    try:
        _run_with_retry(conn, _op, "upsert_free_fit")
        log.debug("Upserted free_tc_fit for rg_id %s, nt_id %s", fit_data.get('rg_id'), fit_data.get('nt_id'))
    except sqlite3.Error as e:
        log.exception("Upsert of free_tc_fit failed: %s", e)


# --- Inline ingestion helpers (create task) ---
def upsert_construct(conn: sqlite3.Connection, construct: Dict[str, Any]) -> Optional[int]:
    """
    Insert a construct or return existing ID based on unique key, and ensure
    nucleotides are present for the construct.

    Behavior:
    - Inserts into `constructs` using unique key (family, name, version, sequence, disp_name).
    - If successful, upserts nucleotides for this construct:
        - If the caller provides `construct['nt_rows']` (list of {site, base, base_region}),
          use those.
        - Otherwise, generate default per-nt rows from the sequence via
          `generate_nt_rows_default` (caller may customize implementation).
    - If nucleotide upsert fails for a newly inserted construct, the transaction is rolled back
      so the construct is not created.

    Returns the construct id, or None on failure.
    """
    insert_sql = (
        """
        INSERT INTO meta_constructs (family, name, version, sequence, disp_name)
        VALUES (:family, :name, :version, :sequence, :disp_name)
        ON CONFLICT(family, name, version, sequence, disp_name) DO NOTHING
        """
    )
    def _resolve_construct_id() -> int:
        with conn:
            conn.execute(insert_sql, construct)
            row = conn.execute(
                """
                SELECT id FROM meta_constructs
                WHERE family = ? AND name = ? AND version = ? AND sequence = ? AND disp_name = ?
                """,
                (
                    construct.get("family"),
                    construct.get("name"),
                    construct.get("version"),
                    construct.get("sequence"),
                    construct.get("disp_name"),
                ),
            ).fetchone()
        cid_local = int(row[0]) if row else None
        if cid_local is None:
            raise sqlite3.IntegrityError("Failed to resolve construct id after upsert")
        return cid_local

    try:
        cid = _run_with_retry(conn, _resolve_construct_id, "upsert_construct insert")

        nt_rows = construct.get("nt_rows")
        if nt_rows is None:
            nt_rows = generate_nt_rows_default(construct.get("sequence", ""))

        if not isinstance(nt_rows, list):
            raise ValueError("nt_rows must be a list of {site, base, base_region} dicts")

        _bulk_upsert_nucleotides(conn, cid, nt_rows)

        log.debug("Upserted construct '%s' -> id=%s (with %d nucleotides)",
                  construct.get("disp_name"), cid, len(nt_rows))
        return cid
    except Exception as e:
        log.exception("Upsert construct (with nucleotides) failed: %s", e)
        return None


def _bulk_upsert_nucleotides(conn: sqlite3.Connection, construct_id: int, nt_rows: List[Dict[str, Any]]) -> int:
    """
    Bulk upsert nucleotides for a construct.

    Each row must have: site (int), base (str), base_region (str).
    Inserts with ON CONFLICT DO NOTHING to respect unique(site, base, base_region, construct_id).

    Returns number of rows attempted (length of nt_rows).
    """
    rows = []
    for nt in nt_rows:
        try:
            site = int(nt.get("site"))
        except Exception as e:
            raise ValueError(f"Invalid site in nt_rows: {nt}") from e
        base = str(nt.get("base")).strip()
        base_region = str(nt.get("base_region", "")).strip() or "unknown"
        rows.append({
            "site": site,
            "base": base,
            "base_region": base_region,
            "construct_id": construct_id,
        })

    sql = (
        """
        INSERT INTO meta_nucleotides (site, base, base_region, construct_id)
        VALUES (:site, :base, :base_region, :construct_id)
        ON CONFLICT(construct_id, site) DO UPDATE SET
            base = excluded.base,
            base_region = excluded.base_region
        """
    )

    def _insert_chunk(chunk: List[Dict[str, Any]]):
        def _op():
            with conn:
                conn.executemany(sql, chunk)

        _run_with_retry(conn, _op, f"_bulk_upsert_nucleotides chunk(size={len(chunk)})")

    for chunk in _iter_chunks(rows, size=500):
        _insert_chunk(chunk)
    return len(rows)


def generate_nt_rows_default(sequence: str) -> List[Dict[str, Any]]:
    """
    Generate default nucleotide rows from a construct sequence using case-based
    segmentation into primer/target/primer.

    Rules:
    - Scan left→right, starting region_id=0; increment on case switch.
    - If max_region == 2 (lower–upper–lower): keep regions 0/1/2 as-is; ensure region 1 is uppercase.
    - If max_region == 1 (two regions): assign uppercase region → 1; the other region → 0 if
      it is before uppercase, or 2 if it is after uppercase.
    - If max_region == 0 (single region):
        * all uppercase → region 1
        * all lowercase → error (must contain uppercase target segment)
    - Any pattern with >2 switches (max_region > 2) → error, ask user to format as
      lowercase–UPPERCASE–lowercase (or a two-region variant).

    Output rows:
      - site: 1..len(sequence)
      - base: uppercase nucleotide (T→U)
      - base_region: "0", "1", or "2"
    """
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Sequence must be a non-empty string for default nucleotide generation.")

    seq = sequence
    n = len(seq)

    # Validate characters: only A,C,G,T,U allowed (case-insensitive)
    allowed = set("ACGTUacgtu")
    bad_positions: List[int] = [i for i, ch in enumerate(seq) if ch not in allowed]
    if bad_positions:
        preview = ''.join(sorted({seq[i] for i in bad_positions}))
        raise ValueError(
            f"Sequence contains invalid characters at positions {bad_positions[:10]} (unique: {preview}). "
            "Only A,C,G,T,U are allowed."
        )

    # Determine region ids by case switching
    def is_upper(c: str) -> bool:
        return c.isalpha() and c.isupper()

    regions: List[int] = []
    cur_region = 0
    prev_upper = is_upper(seq[0])
    regions.append(cur_region)
    for i in range(1, n):
        u = is_upper(seq[i])
        if u != prev_upper:
            cur_region += 1
            prev_upper = u
        regions.append(cur_region)

    max_region = max(regions) if regions else 0

    # Count case distribution per region
    per_region_upper: Dict[int, int] = {}
    per_region_len: Dict[int, int] = {}
    for i, r in enumerate(regions):
        per_region_len[r] = per_region_len.get(r, 0) + 1
        if is_upper(seq[i]):
            per_region_upper[r] = per_region_upper.get(r, 0) + 1

    # Helper to check region all uppercase
    def region_all_upper(r: int) -> bool:
        return per_region_upper.get(r, 0) == per_region_len.get(r, 0)

    # Helper to check any uppercase overall
    any_upper = any(is_upper(c) for c in seq)

    # Map original region ids to final labels {0,1,2}
    region_map: Dict[int, int]

    if max_region > 2:
        raise ValueError(
            "Sequence contains more than three contiguous case-defined regions. "
            "Please format as lowercase–UPPERCASE–lowercase (or a two-region variant)."
        )

    if max_region == 2:
        # Expect pattern lower–upper–lower and region 1 all uppercase
        if not region_all_upper(1):
            raise ValueError("Default segmentation expects the middle region (1) to be uppercase.")
        region_map = {0: 0, 1: 1, 2: 2}
    elif max_region == 1:
        # Determine which region (0 or 1) is uppercase
        r0_upper = region_all_upper(0)
        r1_upper = region_all_upper(1)
        if r0_upper == r1_upper:
            # Either both upper or both lower — inconsistent two-region pattern
            raise ValueError(
                "Two-region sequence must have exactly one uppercase region."
            )
        if r0_upper:
            # Uppercase first, then lowercase → map to 1 then 2
            region_map = {0: 1, 1: 2}
        else:
            # Lowercase first, then uppercase → keep 0 then 1
            region_map = {0: 0, 1: 1}
    else:  # max_region == 0 (single region)
        if not any_upper:
            # All lowercase — invalid, must have at least target region 1
            raise ValueError(
                "Sequence is all lowercase. Must contain an uppercase target region (base_region=1)."
            )
        # All uppercase → whole sequence is region 1
        region_map = {0: 1}

    # Build output rows
    rows: List[Dict[str, Any]] = []
    for i, ch in enumerate(seq):
        base = ch.upper()
        if base == 'T':
            base = 'U'
        r = region_map[regions[i]]
        rows.append({
            "site": i + 1,
            "base": base,
            "base_region": str(r),
        })

    return rows


def prep_nt_rows(nt_info_path: Path) -> List[Dict[str, Any]]:
    """
    Prepare nt_rows from a CSV file with columns: site, base, base_region.

    - Accepts UTF-8 with BOM.
    - Normalizes headers (trim, lowercase, strip BOM).
    - Validates base characters (A/C/G/T/U), base_region in {0,1,2}, at least one region 1.
    - Ensures base is uppercase and converts T→U.

    Returns a list of dicts: {site:int, base:str, base_region:str}.
    """
    if not nt_info_path.is_file():
        raise FileNotFoundError(f"nt_info file not found at: {nt_info_path}")

    log.info("Reading nt_info from: %s", nt_info_path)

    nt_rows: List[Dict[str, Any]] = []
    with nt_info_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        log.info("nt_info headers detected: %s", fieldnames)

        header_map = { (fn.strip().lstrip("\ufeff")).lower(): fn for fn in fieldnames }
        required = {"site", "base", "base_region"}
        if not required.issubset(set(header_map.keys())):
            raise ValueError(f"nt_info CSV missing required columns: {required}")

        allowed = set("ACGTUacgtu")
        seen_region1 = False
        for row_idx, r in enumerate(reader, start=2):  # +2 accounts for header line
            site_raw = r[header_map["site"]]
            base_raw = r[header_map["base"]]
            region_raw = r[header_map["base_region"]]

            try:
                site_val = int(str(site_raw).strip())
            except Exception as e:
                raise ValueError(f"Invalid site at CSV line {row_idx}: {site_raw}") from e

            base_str = str(base_raw).strip()
            if not base_str or any(ch not in allowed for ch in base_str):
                raise ValueError(
                    f"Invalid base '{base_str}' at CSV line {row_idx}. Only A,C,G,T,U allowed."
                )

            region_str = str(region_raw).strip()
            if region_str not in {"0", "1", "2"}:
                raise ValueError(
                    f"Invalid base_region '{region_str}' at CSV line {row_idx}. Must be one of 0,1,2."
                )
            if region_str == "1":
                seen_region1 = True
                if base_str.isalpha() and not base_str.isupper():
                    raise ValueError(
                        f"Row at CSV line {row_idx} has base_region=1 but base is not uppercase: '{base_str}'."
                    )

            base_up = base_str.upper()
            if base_up == "T":
                base_up = "U"

            nt_rows.append({
                "site": site_val,
                "base": base_up,
                "base_region": region_str,
            })

        if not seen_region1:
            raise ValueError(
                "nt_info CSV has no base_region=1 rows. Must contain an uppercase target region."
            )

    return nt_rows


def upsert_buffer(conn: sqlite3.Connection, buffer: Dict[str, Any]) -> Optional[int]:
    """
    Insert a buffer or return existing ID based on unique key.
    Unique key: (name, pH, composition, disp_name)
    """
    sql = (
        """
        INSERT INTO meta_buffers (name, pH, composition, disp_name)
        VALUES (:name, :pH, :composition, :disp_name)
        ON CONFLICT(name, pH, composition, disp_name) DO NOTHING
        """
    )
    def _resolve_buffer_id() -> Optional[int]:
        with conn:
            conn.execute(sql, buffer)
            row = conn.execute(
                """
                SELECT id FROM meta_buffers
                WHERE name = ? AND pH = ? AND composition = ? AND disp_name = ?
                """,
                (
                    buffer.get("name"),
                    buffer.get("pH"),
                    buffer.get("composition"),
                    buffer.get("disp_name"),
                ),
            ).fetchone()
        return int(row[0]) if row else None

    try:
        bid = _run_with_retry(conn, _resolve_buffer_id, "upsert_buffer")
        log.debug("Upserted buffer '%s' -> id=%s", buffer.get("disp_name"), bid)
        return bid
    except sqlite3.Error as e:
        log.exception("Upsert buffer failed: %s", e)
        return None


def upsert_sequencing_run(conn: sqlite3.Connection, run: Dict[str, Any]) -> Optional[int]:
    """
    Insert a sequencing run or return existing ID based on unique key.
    Unique key: (run_name, date, sequencer, run_manager)
    """
    sql = (
        """
        INSERT INTO sequencing_runs (run_name, date, sequencer, run_manager)
        VALUES (:run_name, :date, :sequencer, :run_manager)
        ON CONFLICT(run_name, date, sequencer, run_manager) DO NOTHING
        """
    )
    def _resolve_run_id() -> Optional[int]:
        with conn:
            conn.execute(sql, run)
            row = conn.execute(
                """
                SELECT id FROM sequencing_runs
                WHERE run_name = ? AND date = ? AND sequencer = ? AND run_manager = ?
                """,
                (
                    run.get("run_name"),
                    run.get("date"),
                    run.get("sequencer"),
                    run.get("run_manager"),
                ),
            ).fetchone()
        return int(row[0]) if row else None

    try:
        rid = _run_with_retry(conn, _resolve_run_id, "upsert_sequencing_run")
        log.debug("Upserted sequencing_run '%s' -> id=%s", run.get("run_name"), rid)
        return rid
    except sqlite3.Error as e:
        log.exception("Upsert sequencing_run failed: %s", e)
        return None


def bulk_upsert_samples(conn: sqlite3.Connection, seqrun_id: int, samples: List[Dict[str, Any]]) -> int:
    """
    Bulk upsert sequencing samples for a given sequencing run.
    On conflict of (seqrun_id, sample_name, fq_dir), updates r1_file, r2_file, to_drop.

    Returns number of rows affected (inserted or updated attempts).
    """
    if not samples:
        return 0

    # Normalize records and default to_drop
    rows = []
    for s in samples:
        rows.append(
            {
                "seqrun_id": seqrun_id,
                "sample_name": s.get("sample_name"),
                "fq_dir": s.get("fq_dir"),
                "r1_file": s.get("r1_file"),
                "r2_file": s.get("r2_file"),
                "to_drop": int(s.get("to_drop", 0)),
            }
        )

    sql = (
        """
        INSERT INTO sequencing_samples (seqrun_id, sample_name, fq_dir, r1_file, r2_file, to_drop)
        VALUES (:seqrun_id, :sample_name, :fq_dir, :r1_file, :r2_file, :to_drop)
        ON CONFLICT(seqrun_id, sample_name, fq_dir) DO UPDATE SET
            r1_file = excluded.r1_file,
            r2_file = excluded.r2_file,
            to_drop = excluded.to_drop
        """
    )
    def _insert_chunk(chunk: List[Dict[str, Any]]):
        def _op():
            with conn:
                conn.executemany(sql, chunk)

        _run_with_retry(conn, _op, f"bulk_upsert_samples chunk(size={len(chunk)})")

    try:
        total = 0
        for chunk in _iter_chunks(rows, size=200):
            _insert_chunk(chunk)
            total += len(chunk)
        log.info("Upserted %d sequencing_samples for seqrun_id=%s", total, seqrun_id)
        return total
    except sqlite3.Error as e:
        log.exception("Bulk upsert samples failed: %s", e)
        return 0


# --- Additional helpers for probing reactions and reaction groups ---
def get_sample_id(conn: sqlite3.Connection, seqrun_id: int, sample_name: str, fq_dir: Optional[str] = None) -> Optional[int]:
    """
    Fetch the id of a sequencing sample by seqrun_id and sample_name.
    If fq_dir is provided, include it in the filter for disambiguation.
    """
    try:
        if fq_dir is None:
            row = conn.execute(
                "SELECT id FROM sequencing_samples WHERE seqrun_id = ? AND sample_name = ?",
                (seqrun_id, sample_name),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM sequencing_samples WHERE seqrun_id = ? AND sample_name = ? AND fq_dir = ?",
                (seqrun_id, sample_name, fq_dir),
            ).fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to fetch sample id for '%s' (seqrun_id=%s): %s", sample_name, seqrun_id, e)
        return None


def get_sample_id_by_name(conn: sqlite3.Connection, sample_name: str) -> Optional[int]:
    """
    Fetch a sequencing sample id by its name across all sequencing runs.
    If multiple rows match, returns None to avoid ambiguity.
    """
    try:
        rows = conn.execute(
            "SELECT id FROM sequencing_samples WHERE sample_name = ?",
            (sample_name,),
        ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            log.error("Sample name '%s' is ambiguous across sequencing runs (found %d). Provide seqrun context.", sample_name, len(rows))
            return None
        return int(rows[0][0])
    except sqlite3.Error as e:
        log.exception("Failed to fetch sample id by name '%s': %s", sample_name, e)
        return None


def get_construct_id_by_disp_name(conn: sqlite3.Connection, disp_name: str) -> Optional[int]:
    """
    Resolve a construct id by its disp_name (case-insensitive).
    """
    if disp_name in (None, ""):
        return None
    query = "SELECT id FROM meta_constructs WHERE LOWER(disp_name) = LOWER(?) LIMIT 1"
    try:
        row = _run_with_retry(
            conn,
            lambda: conn.execute(query, (str(disp_name),)).fetchone(),
            "get_construct_id_by_disp_name",
        )
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to lookup construct by disp_name '%s': %s", disp_name, e)
        return None


def get_buffer_id_by_name_or_disp(conn: sqlite3.Connection, identifier: str) -> Optional[int]:
    """
    Resolve a buffer id by disp_name or name (case-insensitive).
    """
    if identifier in (None, ""):
        return None
    ident = str(identifier).strip()
    queries = [
        "SELECT id FROM meta_buffers WHERE LOWER(disp_name) = LOWER(?) LIMIT 1",
        "SELECT id FROM meta_buffers WHERE LOWER(name) = LOWER(?) LIMIT 1",
    ]
    try:
        for query in queries:
            row = _run_with_retry(
                conn,
                lambda q=query: conn.execute(q, (ident,)).fetchone(),
                "get_buffer_id_by_name_or_disp",
            )
            if row:
                return int(row[0])
        return None
    except sqlite3.Error as e:
        log.exception("Failed to lookup buffer '%s': %s", identifier, e)
        return None


def get_max_reaction_group_id(conn: sqlite3.Connection) -> int:
    """Return the current max rg_id from probe_reaction_groups, or 0 if none."""
    try:
        row = conn.execute("SELECT COALESCE(MAX(rg_id), 0) FROM probe_reaction_groups").fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.Error as e:
        log.exception("Failed to fetch max rg_id: %s", e)
        return 0


def find_sequencing_run_id_by_name(conn: sqlite3.Connection, run_name: str) -> Optional[int]:
    """
    Resolve a sequencing run id by its run_name.
    """
    if run_name in (None, ""):
        return None
    try:
        row = _run_with_retry(
            conn,
            lambda: conn.execute(
                "SELECT id FROM sequencing_runs WHERE run_name = ? LIMIT 1",
                (run_name,),
            ).fetchone(),
            "find_sequencing_run_id_by_name",
        )
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to lookup sequencing run '%s': %s", run_name, e)
        return None


def insert_probing_reaction(conn: sqlite3.Connection, reaction: Dict[str, Any]) -> Optional[int]:
    """
    Insert a probing_reaction and return its id.
    Expected keys in reaction dict:
      temperature, replicate, reaction_time, probe_concentration, probe,
      rt_protocol, done_by, treated, buffer_id, construct_id, rg_id, s_id
    """
    sql = (
        """
        INSERT INTO probe_reactions (
            temperature, replicate, reaction_time, probe_concentration, probe,
            rt_protocol, done_by, treated, buffer_id, construct_id, rg_id, s_id
        ) VALUES (
            :temperature, :replicate, :reaction_time, :probe_concentration, :probe,
            :rt_protocol, :done_by, :treated, :buffer_id, :construct_id, :rg_id, :s_id
        )
        """
    )
    try:
        with conn:
            cur = conn.execute(sql, reaction)
            rxn_id = cur.lastrowid
            log.debug("Inserted probing_reaction id=%s for sample_id=%s (rg_id=%s)", rxn_id, reaction.get("s_id"), reaction.get("rg_id"))
            return rxn_id
    except sqlite3.Error as e:
        log.exception("Failed to insert probing_reaction: %s", e)
        return None


# Deprecated: insert_reaction_group_pair was used when reaction_groups stored per-reaction mappings.
# The schema now normalizes reaction_groups as a catalog (rg_id, rg_label), so this is removed.


def find_rg_id_by_label(conn: sqlite3.Connection, rg_label: str) -> Optional[int]:
    """Return an existing rg_id for a given label, or None if not found."""
    try:
        row = conn.execute(
            "SELECT rg_id FROM probe_reaction_groups WHERE rg_label = ? LIMIT 1",
            (rg_label,),
        ).fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to lookup rg_id by label '%s': %s", rg_label, e)
        return None


def upsert_reaction_group(conn: sqlite3.Connection, rg_id: int, rg_label: Optional[str]) -> Optional[int]:
    """
    Ensure a reaction_groups row exists and sets the label exactly to the provided value.
    If the row exists, updates rg_label to match the YAML.
    Returns the rg_id on success, None on failure.
    """
    try:
        if rg_label is None:
            sql = "INSERT INTO probe_reaction_groups (rg_id) VALUES (?) ON CONFLICT(rg_id) DO NOTHING"
            params = (rg_id,)
        else:
            sql = (
                "INSERT INTO probe_reaction_groups (rg_id, rg_label) VALUES (?, ?) "
                "ON CONFLICT(rg_id) DO UPDATE SET rg_label=excluded.rg_label"
            )
            params = (rg_id, str(rg_label))

        with conn:
            conn.execute(sql, params)
        return rg_id
    except sqlite3.Error as e:
        log.exception("Failed to upsert reaction_group rg_id=%s label=%s: %s", rg_id, rg_label, e)
        return None


# --- Derived samples helpers ---
def upsert_derived_sample(conn: sqlite3.Connection, parent_sample_id: int, child_name: str,
                          kind: str, tool: str, cmd_template: str,
                          params: Optional[Dict[str, Any]] = None,
                          cache_key: Optional[str] = None) -> Optional[int]:
    """
    Insert or update a derived sample definition. On conflict of
    (parent_sample_id, child_name), updates fields.
    Returns the row id.
    """
    if params is None:
        params = {}
    params_json = json.dumps(params, sort_keys=True)
    if cache_key is None:
        # Use a robust 64-char cache key over defining attributes
        cache_key = config_hash({
            "parent_sample_id": parent_sample_id,
            "child_name": child_name,
            "kind": kind,
            "tool": tool,
            "cmd_template": cmd_template,
            "params": params,
        }, length=64)

    sql = """
        INSERT INTO sequencing_derived_samples (
            parent_sample_id, child_name, kind, tool, cmd_template, params_json, cache_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(parent_sample_id, child_name) DO UPDATE SET
            kind = excluded.kind,
            tool = excluded.tool,
            cmd_template = excluded.cmd_template,
            params_json = excluded.params_json,
            cache_key = excluded.cache_key
    """
    try:
        with conn:
            conn.execute(sql, (parent_sample_id, child_name, kind, tool, cmd_template, params_json, cache_key))
        # Fetch id
        row = conn.execute(
            "SELECT id FROM sequencing_derived_samples WHERE parent_sample_id = ? AND child_name = ?",
            (parent_sample_id, child_name),
        ).fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error as e:
        log.exception("Failed to upsert derived_sample '%s' for parent %s: %s", child_name, parent_sample_id, e)
        return None


def get_derived_by_child_name(conn: sqlite3.Connection, child_name: str) -> Optional[sqlite3.Row]:
    try:
        row = conn.execute(
            "SELECT * FROM sequencing_derived_samples WHERE child_name = ?",
            (child_name,),
        ).fetchone()
        return row
    except sqlite3.Error as e:
        log.exception("Failed to fetch derived sample by child_name '%s': %s", child_name, e)
        return None


def get_derived_for_parent(conn: sqlite3.Connection, parent_sample_id: int) -> list:
    try:
        rows = conn.execute(
            "SELECT * FROM sequencing_derived_samples WHERE parent_sample_id = ?",
            (parent_sample_id,),
        ).fetchall()
        return rows or []
    except sqlite3.Error as e:
        log.exception("Failed to fetch derived samples for parent %s: %s", parent_sample_id, e)
        return []
