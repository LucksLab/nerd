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
import csv
from pathlib import Path

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
        # Ensure FK constraints (including ON DELETE CASCADE) are enforced
        conn.execute("PRAGMA foreign_keys = ON")
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
        INSERT INTO tasks (
            task_name, scope_kind, scope_id, backend, output_dir, label,
            cache_key, config_hash, started_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with conn:
            # Use a short, human-friendly config hash alongside full cache_key
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
                    cfg_short,  # store short 7-char config hash
                    datetime.now().isoformat(),
                ),
            )
            task_id = cursor.lastrowid
            log.info(
                "Began task '%s' (ID: %s) for %s:%s (cfg=%s)",
                name, task_id, scope_kind, scope_id, cache_key
            )
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
    If cache_key is provided, updates it; otherwise preserves the existing value.
    """
    try:
        with conn:
            if cache_key is None:
                conn.execute(
                    """
                    UPDATE tasks
                    SET state = ?, message = ?, ended_at = ?
                    WHERE id = ?
                    """,
                    (state, message, datetime.now().isoformat(), task_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE tasks
                    SET state = ?, message = ?, cache_key = ?, ended_at = ?
                    WHERE id = ?
                    """,
                    (state, message, cache_key, datetime.now().isoformat(), task_id),
                )
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
        FROM tasks
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


def record_cached_task(conn: sqlite3.Connection, name: str, scope_kind: str, scope_id: Optional[int],
                       backend: str, output_dir: str, label: str, cache_key: Optional[str],
                       message: Optional[str] = None) -> Optional[int]:
    """
    Inserts a task row with state='cached' to record a skipped run due to identical config.

    Uses a single INSERT to avoid unique-index conflicts on active states.
    Sets both started_at and ended_at to now.
    """
    sql = """
        INSERT INTO tasks (
            task_name, scope_kind, scope_id, backend, output_dir, label,
            cache_key, config_hash, started_at, ended_at, state, message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), 'cached', ?)
    """
    try:
        with conn:
            cfg_short = (cache_key or "")[:7]
            cur = conn.execute(sql, (name, scope_kind, scope_id, backend, output_dir, label, cache_key, cfg_short, message))
            task_id = cur.lastrowid
            log.info("Recorded cached task '%s' (ID: %s) for label '%s'", name, task_id, label)
            return task_id
    except sqlite3.IntegrityError as e:
        log.error("Failed to record cached task due to integrity error: %s", e)
        return None
    except sqlite3.Error as e:
        log.exception("Failed to record cached task: %s", e)
        return None

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
        INSERT INTO constructs (family, name, version, sequence, disp_name)
        VALUES (:family, :name, :version, :sequence, :disp_name)
        ON CONFLICT(family, name, version, sequence, disp_name) DO NOTHING
        """
    )
    try:
        with conn:
            # Attempt insert (noop on conflict)
            conn.execute(insert_sql, construct)

            # Retrieve id whether inserted or pre-existing
            row = conn.execute(
                """
                SELECT id FROM constructs
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
            cid = int(row[0]) if row else None
            if cid is None:
                raise sqlite3.IntegrityError("Failed to resolve construct id after upsert")

            # Upsert nucleotides for this construct
            nt_rows = construct.get("nt_rows")
            if nt_rows is None:
                # No user-provided CSV parsed rows; generate defaults from sequence
                nt_rows = generate_nt_rows_default(construct.get("sequence", ""))

            if not isinstance(nt_rows, list):
                raise ValueError("nt_rows must be a list of {site, base, base_region} dicts")

            # Normalize and insert
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
        INSERT INTO nucleotides (site, base, base_region, construct_id)
        VALUES (:site, :base, :base_region, :construct_id)
        ON CONFLICT(site, base, base_region, construct_id) DO NOTHING
        """
    )
    with conn:
        conn.executemany(sql, rows)
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
        INSERT INTO buffers (name, pH, composition, disp_name)
        VALUES (:name, :pH, :composition, :disp_name)
        ON CONFLICT(name, pH, composition, disp_name) DO NOTHING
        """
    )
    try:
        with conn:
            conn.execute(sql, buffer)
        row = conn.execute(
            """
            SELECT id FROM buffers
            WHERE name = ? AND pH = ? AND composition = ? AND disp_name = ?
            """,
            (
                buffer.get("name"),
                buffer.get("pH"),
                buffer.get("composition"),
                buffer.get("disp_name"),
            ),
        ).fetchone()
        bid = int(row[0]) if row else None
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
    try:
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
        rid = int(row[0]) if row else None
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
    try:
        with conn:
            conn.executemany(sql, rows)
        log.info("Upserted %d sequencing_samples for seqrun_id=%s", len(rows), seqrun_id)
        return len(rows)
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


def get_max_reaction_group_id(conn: sqlite3.Connection) -> int:
    """Return the current max rg_id from reaction_groups, or 0 if none."""
    try:
        row = conn.execute("SELECT COALESCE(MAX(rg_id), 0) FROM reaction_groups").fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.Error as e:
        log.exception("Failed to fetch max rg_id: %s", e)
        return 0


def insert_probing_reaction(conn: sqlite3.Connection, reaction: Dict[str, Any]) -> Optional[int]:
    """
    Insert a probing_reaction and return its id.
    Expected keys in reaction dict:
      temperature, replicate, reaction_time, probe_concentration, probe,
      RT, done_by, treated, buffer_id, construct_id, rg_id, s_id
    """
    sql = (
        """
        INSERT INTO probing_reactions (
            temperature, replicate, reaction_time, probe_concentration, probe,
            RT, done_by, treated, buffer_id, construct_id, rg_id, s_id
        ) VALUES (
            :temperature, :replicate, :reaction_time, :probe_concentration, :probe,
            :RT, :done_by, :treated, :buffer_id, :construct_id, :rg_id, :s_id
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
            "SELECT rg_id FROM reaction_groups WHERE rg_label = ? LIMIT 1",
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
            sql = "INSERT INTO reaction_groups (rg_id) VALUES (?) ON CONFLICT(rg_id) DO NOTHING"
            params = (rg_id,)
        else:
            sql = (
                "INSERT INTO reaction_groups (rg_id, rg_label) VALUES (?, ?) "
                "ON CONFLICT(rg_id) DO UPDATE SET rg_label=excluded.rg_label"
            )
            params = (rg_id, str(rg_label))

        with conn:
            conn.execute(sql, params)
        return rg_id
    except sqlite3.Error as e:
        log.exception("Failed to upsert reaction_group rg_id=%s label=%s: %s", rg_id, rg_label, e)
        return None
