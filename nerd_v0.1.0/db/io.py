# nerd/db/io.py

import sqlite3
from pathlib import Path
from nerd.db.schema import ALL_TABLES, ALL_INDEXES
from nerd.utils.logging import get_logger
from rich.table import Table
from rich.console import Console
log = get_logger(__name__)
DEFAULT_DB_PATH = Path("nerd.sqlite3")  # fallback if not specified

# ==== Connection + Database Initialization ===================================

def connect_db(db_path: str) -> sqlite3.Connection:
    """Connect to the SQLite3 database."""
    db_file = Path(db_path) if db_path else DEFAULT_DB_PATH
    return sqlite3.connect(db_file)

def init_db(conn: sqlite3.Connection):
    """Create tables and indexes defined in schema.py if they don't exist."""
    cursor = conn.cursor()

    for stmt in ALL_TABLES:
        cursor.executescript(stmt)  # safer for multi-line CREATE TABLE statements

    for index_stmt in ALL_INDEXES:
        cursor.executescript(index_stmt)

    conn.commit()
    
def check_db(conn: sqlite3.Connection, TABLE: str, REQUIRED_COLUMNS: list):
    """Check if a table exists and has the required columns."""
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE,))
    if not cursor.fetchone():
        log.error("Table '%s' does not exist.", TABLE)
        return False

    # Check for required columns
    cursor.execute(f"PRAGMA table_info({TABLE})")
    columns = [row[1] for row in cursor.fetchall()]
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    if missing:
        log.error("Missing columns in %s: %s", TABLE, missing)
        return False
    return True


# === Insert Functions ===

def insert_nmr_reaction(conn, reaction_metadata: dict):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO nmr_reactions (
            reaction_type, substrate, substrate_conc,
            temperature, replicate,
            probe, probe_conc, buffer_id, probe_solvent,
            num_scans, time_per_read, total_kinetic_reads, total_kinetic_time,
            nmr_machine, kinetic_data_dir, mnova_analysis_dir, raw_fid_dir
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        reaction_metadata.get("reaction_type"),
        reaction_metadata.get("substrate"),
        reaction_metadata.get("substrate_conc"),
        reaction_metadata.get("temperature"),
        reaction_metadata.get("replicate"),
        reaction_metadata.get("probe"),
        reaction_metadata.get("probe_conc"),
        reaction_metadata.get("buffer_id"),
        reaction_metadata.get("probe_solvent"),
        reaction_metadata.get("num_scans"),
        reaction_metadata.get("time_per_read"),
        reaction_metadata.get("total_kinetic_reads"),
        reaction_metadata.get("total_kinetic_time"),
        reaction_metadata.get("nmr_machine"),
        reaction_metadata.get("kinetic_data_dir"),
        reaction_metadata.get("mnova_analysis_dir"),
        reaction_metadata.get("raw_fid_dir"),
    ))
    conn.commit()
    return cursor.rowcount > 0

# nmr
def insert_fitted_kinetic_rate(conn, fit_result: dict):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO nmr_kinetic_rates (
            nmr_reaction_id, model, k_value, k_error, r2, chisq, species
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (fit_result.get("nmr_reaction_id"),
            fit_result.get("model"),
            fit_result.get("k_value"),
            fit_result.get("k_error"),
            fit_result.get("r2"),
            fit_result.get("chisq"),
            fit_result.get("species")))
    conn.commit()
    return cursor.rowcount > 0

def insert_arrhenius_fit(conn, fit_result: dict):
    cursor = conn.cursor()
    log.debug("INSERT INTO arrhenius_fits: %s", fit_result)
    # Base columns and values
    columns = ["reaction_type", "data_source", "substrate", "buffer_id", 
               "slope", "slope_err", "intercept", "intercept_err", "r2", "model_file"]
    values = [
        fit_result.get("reaction_type"),
        fit_result.get("data_source"),
        fit_result.get("substrate"),
        fit_result.get("buffer_id"),
        fit_result.get("slope"),
        fit_result.get("slope_err"),
        fit_result.get("intercept"),
        fit_result.get("intercept_err"),
        fit_result.get("r2"),
        fit_result.get("model_file")
    ]
    
    # Optionally add tg_id and nt_id
    if "tg_id" in fit_result:
        columns.append("tg_id")
        values.append(fit_result.get("tg_id"))
    
    if "nt_id" in fit_result:
        columns.append("nt_id")
        values.append(fit_result.get("nt_id"))
    
    # Construct query
    placeholders = ", ".join(["?"] * len(columns))
    columns_str = ", ".join(columns)
    
    query = f"""
        INSERT INTO arrhenius_fits (
            {columns_str}
        ) VALUES (
            {placeholders}
        )
    """
    log.debug("%s %s", query, values)
    cursor.execute(query, values)
    log.info("Inserted %d rows into arrhenius_fits", cursor.rowcount)
    conn.commit()
    return cursor.rowcount > 0


def insert_buffer(conn, buffer_data: dict):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO buffers (
            name, pH, composition, disp_name
        ) VALUES (?, ?, ?, ?)
    """, (
        buffer_data.get("name"),
        buffer_data.get("pH"),
        buffer_data.get("composition"),
        buffer_data.get("disp_name")
    ))
    conn.commit()
    return cursor.rowcount > 0

def insert_construct(conn, construct_data: dict):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO constructs (
            family, name, version, sequence, disp_name
        ) VALUES (?, ?, ?, ?, ?)
    """, (
        construct_data.get("family"),
        construct_data.get("name"),
        construct_data.get("version"),
        construct_data.get("sequence"),
        construct_data.get("disp_name")
    ))

    conn.commit()
    construct_id = cursor.lastrowid
    return cursor.rowcount > 0, construct_id

def insert_nt_info(conn, nt_info_data: dict):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO nucleotides (
            construct_id, site, base, base_region, construct_id
        ) VALUES (?, ?, ?, ?, ?)
    """, (
        nt_info_data.get("construct_id"),
        nt_info_data.get("site"),
        nt_info_data.get("base"),
        nt_info_data.get("base_region"),
        nt_info_data.get("construct_id")
    ))
    conn.commit()
    return cursor.rowcount > 0

def insert_seq_run(conn, run_data: dict):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO sequencing_runs (
            run_name, date, sequencer, run_manager
        ) VALUES (?, ?, ?, ?)
    """, (
        run_data.get("run_name"),
        run_data.get("date"),
        run_data.get("sequencer"),
        run_data.get("run_manager")
    ))
    conn.commit()
    return cursor.rowcount > 0

def insert_seq_sample(conn, sample_data: dict):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO sequencing_samples (
            seqrun_id, sample_name, fq_dir
        ) VALUES (?, ?, ?, ?, ?)
    """, (
        sample_data.get("seqrun_id"),
        sample_data.get("sample_name"),
        sample_data.get("fq_dir")
    ))
    conn.commit()
    return cursor.rowcount > 0

def insert_tc_fit(conn, fit_data: dict, insert_to_table: str):

    cursor = conn.cursor()
    cursor.execute(f"""
        INSERT OR IGNORE INTO {insert_to_table} (
            rg_id, nt_id,
            kobs_val, kobs_err,
            kdeg_val, kdeg_err,
            fmod0, fmod0_err,
            r2, chisq,
            time_min, time_max
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        fit_data.get("rg_id"),
        fit_data.get("nt_id"),
        fit_data.get("kobs_val"),
        fit_data.get("kobs_err"),
        fit_data.get("kdeg_val"),
        fit_data.get("kdeg_err"),
        fit_data.get("fmod0"),
        fit_data.get("fmod0_err"),
        fit_data.get("r2"),
        fit_data.get("chisq"),
        fit_data.get("time_min"),
        fit_data.get("time_max")
    ))
    conn.commit()
    return cursor.rowcount > 0

def insert_fitted_probing_kinetic_rate(conn, fit_result: dict):

    cursor = conn.cursor()

    # Base columns and values
    columns = ["rg_id", "model", "k_value", "k_error", "r2", "chisq", "species"]
    values = [
        fit_result.get("rg_id"),
        fit_result.get("model"),
        fit_result.get("k_value"),
        fit_result.get("k_error"),
        fit_result.get("r2"),
        fit_result.get("chisq"),
        fit_result.get("species")
    ]

    # Optionally add nt_id and rxn_id
    if "nt_id" in fit_result:
        columns.append("nt_id")
        values.append(fit_result.get("nt_id"))
        columns.append("rxn_id")
        values.append(fit_result.get("rxn_id"))

    # Construct query
    placeholders = ", ".join(["?"] * len(columns))
    columns_str = ", ".join(columns)

    query = f"""
        INSERT OR IGNORE INTO probing_kinetic_rates (
            {columns_str}
        ) VALUES (
            {placeholders}
        )
    """

    cursor.execute(query, values)
    conn.commit()

    return cursor.rowcount > 0

# === Query Functions ===

def fetch_all_nmr_samples(conn, reaction_type='deg'):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, temperature, replicate,
            probe, probe_conc, buffer_id, probe_solvent, substrate, substrate_conc,
            nmr_machine, kinetic_data_dir
        FROM nmr_reactions
        WHERE reaction_type = ?
    """, (reaction_type,))
    return cursor.fetchall(), cursor.description

def fetch_kinetic_rates(conn, buffer_id: int, species: str, reaction_type: str):
    cursor = conn.cursor()
    query = """
        SELECT nkr.nmr_reaction_id, nkr.species, nr.temperature, nkr.k_value, nkr.k_error
        FROM nmr_kinetic_rates nkr
        JOIN nmr_reactions nr ON nkr.nmr_reaction_id = nr.id
        WHERE nr.buffer_id = ?
          AND nr.reaction_type = ?
          AND nkr.species = ?
    """
    cursor.execute(query, (buffer_id, reaction_type, species))
    return cursor.fetchall(), cursor.description

# === Display table ===

def display_table(results, column_descriptions, title="Query Results"):
    """
    Prints a formatted Rich table from a SQLite query result.
    
    Parameters:
    - results: list of tuples from cursor.fetchall()
    - column_descriptions: cursor.description (list of column metadata)
    - title: optional table title
    """
    console = Console()
    table = Table(title=title)

    # Extract column names
    column_names = [desc[0] for desc in column_descriptions]

    # Add columns
    for col in column_names:
        table.add_column(col, style="cyan", no_wrap=True)

    # Add rows
    for row in results:
        table.add_row(*[str(cell) if cell is not None else "" for cell in row])

    console.print(table)


def insert_melted_probing_rates(fetched_result: tuple, db_path: str):
    """
    Insert aggregated probing rates into the database.

    Args:
        fetched_result (tuple): Tuple containing (records, description) where records is a list of tuples
                               containing the data, and description is a list of column names.
        db_path (str): Path to the database file.
        console (Console): Rich console for output.
    """
    
    records, description = fetched_result
    column_names = [desc for desc in description]
    # Import probing kinetic rate
    conn = connect_db(db_path)

    count = 0

    for row in records:
        # Create dictionary from row data using column names
        row_dict = dict(zip(column_names, row))
        fit_result = {
            "rg_id": int(row_dict['rg_id']),
            "model": "melted_ind_obs",
            "k_value": row_dict['kobs_val'],
            "k_error": row_dict['kobs_err'],
            "chisq": row_dict['chisq'],
            "r2": row_dict['r2'],
            "species": 'rna_' + row_dict['base'],
            "nt_id":  int(row_dict['id']),
            "rxn_id": int(row_dict['rxn_id'])
        }

        insert_success = insert_fitted_probing_kinetic_rate(conn, fit_result)
        count += insert_success
        
    conn.close()

    if records:
        log.info("Inserted %d probing kinetic rates (%.1f%%)", count, (count / len(records) * 100.0))
    else:
        log.info("Inserted %d probing kinetic rates", count)

    return count

def insert_tempgrad_groups(conn, tg_assignments):
    """
    Insert temperature gradient group assignments into the tempgrad_groups table.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        tg_assignments (list of tuples): Each tuple should contain:
            (tg_id, rg_id, buffer_id, construct_id, RT, probe, probe_concentration, temperature, replicate)
    """
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO tempgrad_groups (
            tg_id, rg_id, buffer_id, construct_id, RT, probe,
            probe_concentration, temperature, replicate
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, tg_assignments)
    conn.commit()


def insert_tempgrad_group(db_path, tempgrad_insert_dict):
    """
    Insert a single temperature gradient group into the tempgrad_groups table.

    Args:
        db_path (str): Path to the SQLite database file.
        tempgrad_insert_dict (dict): Dictionary containing the data to insert.
            Expected keys: 'tg_id', 'rg_id', 'buffer_id', 'construct_id',
                           'RT', 'probe', 'probe_concentration', 'temperature', 'replicate'.

    Returns:
        bool: True if insertion was successful, False otherwise.
    """
    conn = connect_db(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO tempgrad_groups (
            tg_id, rg_id, buffer_id, construct_id, RT, probe,
            probe_concentration, temperature, replicate
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        tempgrad_insert_dict['tg_id'],
        tempgrad_insert_dict['rg_id'],
        tempgrad_insert_dict['buffer_id'],
        tempgrad_insert_dict['construct_id'],
        tempgrad_insert_dict['RT'],
        tempgrad_insert_dict['probe'],
        tempgrad_insert_dict['probe_concentration'],
        tempgrad_insert_dict['temperature'],
        tempgrad_insert_dict['replicate']
    ))

    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    
    return success

# === Miscellaneous (might incorporate) ===
from collections import defaultdict

def group_fam_tg_ids(db_path: str) -> dict[int, dict]:
    """
    Group tg_ids by construct family and probing conditions.
    
    Returns:
        dict: fam_tg_id (int) -> dict with 'family' and 'tg_ids' keys
    """
    conn = connect_db(db_path)
    cursor = conn.cursor()

    # Fetch construct family + probing conditions for all tg_ids
    cursor.execute("""
        SELECT 
            tg.tg_id,
            tg.buffer_id,
            tg.RT,
            tg.probe,
            tg.probe_concentration,
            c.family
        FROM tempgrad_groups tg
        JOIN constructs c ON tg.construct_id = c.id
    """)

    rows = cursor.fetchall()
    conn.close()

    # Group tg_ids by (family, buffer_id, RT, probe, probe_concentration)
    group_dict = defaultdict(set)  # use set to deduplicate tg_ids
    for tg_id, buffer_id, RT, probe, probe_conc, family in rows:
        group_key = (family, buffer_id, RT, probe, probe_conc)
        group_dict[group_key].add(tg_id)  # set prevents duplicates

    # Convert to indexed dict with family name
    fam_tg_dict = {}
    for i, (group_key, tg_set) in enumerate(group_dict.items(), start=1):
        family, _, _, _, _ = group_key
        fam_tg_dict[i] = {
            "family": family,
            "tg_ids": sorted(tg_set)  # sort for consistent order
        }
    print(fam_tg_dict)
    return fam_tg_dict


def insert_melt_fit(db_path, fit_data: dict):
    """
    Insert a melt curve fit result into the probing_melt_fits table.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        fit_data (dict): Dictionary containing the melt fit data.
            Expected keys: 'tg_id', 'nt_id',
                           'a', 'a_err', 'b', 'b_err',
                           'c', 'c_err', 'd', 'd_err',
                           'f', 'f_err', 'g', 'g_err',
                           'r2', 'chisq'.
    """
    conn = connect_db(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO probing_melt_fits (
            tg_id, nt_id,
            a, a_err, b, b_err,
            c, c_err, d, d_err,
            f, f_err, g, g_err,
            r2, chisq
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        fit_data['tg_id'], fit_data['nt_id'],
        fit_data['a'], fit_data['a_err'], fit_data['b'], fit_data['b_err'],
        fit_data['c'], fit_data['c_err'], fit_data['d'], fit_data['d_err'],
        fit_data['f'], fit_data['f_err'], fit_data['g'], fit_data['g_err'],
        fit_data['r2'], fit_data['chisq']
    ))
    conn.commit()

    return cursor.rowcount > 0

# def append_csv_to_sqlite(csv_file, table_name, db_file):
#     """Appends a CSV file to a given SQLite table with matching column names."""
    
#     # Load CSV into a DataFrame
#     df = pd.read_csv(csv_file)

#     # Connect to SQLite database
#     conn = sqlite3.connect(db_file)
#     cursor = conn.cursor()

#     # Get existing table column names
#     cursor.execute(f"PRAGMA table_info({table_name})")
#     existing_columns = {row[1] for row in cursor.fetchall()}  # Column names from DB

#     # Filter DataFrame to only include matching columns
#     df = df[[col for col in df.columns if col in existing_columns]]

#     if df.empty:
#         print("No matching columns found. Nothing to insert.")
#     else:
#         # Append DataFrame to the table
#         df.to_sql(table_name, conn, if_exists="append", index=False)
#         print(f"Successfully appended {len(df)} rows to {table_name}.")

#     conn.close()