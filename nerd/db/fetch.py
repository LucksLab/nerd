import sqlite3
from typing import Optional
from collections import defaultdict

def fetch_run_name(db_path: str, sample_name: str) -> Optional[str]:
    """
    Fetch the sequencing run name for a given sample name.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT sr.run_name from sequencing_samples ss
        JOIN sequencing_runs sr ON sr.id = ss.seqrun_id
        WHERE ss.sample_name = ?
    """, (sample_name,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]
    else:
        return None
    

def fetch_all_nt(db_path: str, rg_id: int) -> list:
    """
    Fetch all nt_ids for a given reaction group ID (rg_id).
    
    1. Find the unique construct_id from reaction_groups → probing_reactions.
    2. Use that construct_id to get all nt_ids from the nucleotides table.
    
    Raises:
        ValueError: if multiple or no construct_ids found for the given rg_id.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Step 1: Get all construct_ids associated with the rg_id
    cursor.execute("""
        SELECT DISTINCT pr.construct_id
        FROM reaction_groups rg
        JOIN probing_reactions pr ON rg.rxn_id = pr.id
        WHERE rg.rg_id = ?
    """, (rg_id,))
    
    construct_ids = [row[0] for row in cursor.fetchall()]

    if len(construct_ids) == 0:
        conn.close()
        raise ValueError(f"No construct_id found for rg_id {rg_id}")
    if len(construct_ids) > 1:
        conn.close()
        raise ValueError(f"Multiple construct_ids found for rg_id {rg_id}: {construct_ids}")

    construct_id = construct_ids[0]

    # Step 2: Get all nt_ids for that construct_id
    cursor.execute("""
        SELECT id FROM nucleotides
        WHERE construct_id = ?
    """, (construct_id,))
    
    nt_ids = [row[0] for row in cursor.fetchall()]

    conn.close()

    if not nt_ids:
        raise ValueError(f"No nucleotides found for construct_id {construct_id}")

    return nt_ids

def fetch_timecourse_data(db_path: str, nt_id: int, rg_id: int) -> tuple:
    """
    Fetch timecourse data for a given nt_id in a reaction group.
    Returns:
        time_data: list of adjusted reaction times (0 if untreated)
        fmod_data: list of corresponding fmod values
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT pr.reaction_time, fv.fmod_val, pr.treated
        FROM fmod_vals fv
        JOIN probing_reactions pr ON fv.rxn_id = pr.id
        JOIN reaction_groups rg ON rg.rxn_id = pr.id
        WHERE rg.rg_id = ?
        AND fv.nt_id = ?
        AND fv.fmod_val IS NOT NULL
    """

    cursor.execute(query, (rg_id, nt_id))
    rows = cursor.fetchall()
    conn.close()

    # Adjust time by treated status
    time_data = [reaction_time * treated for reaction_time, _, treated in rows]
    fmod_data = [fmod_val for _, fmod_val, _ in rows]

    return time_data, fmod_data


def fetch_reaction_temp(db_path: str, rg_id: int) -> float:
    """
    Fetch the reaction temperature for a given reaction group ID (rg_id).
    
    Args:
        db_path (str): Path to the database file.
        rg_id (int): Reaction group ID.
    
    Returns:
        float: Reaction temperature in Celsius.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT pr.temperature
        FROM reaction_groups rg
        JOIN probing_reactions pr ON rg.rxn_id = pr.id
        WHERE rg.rg_id = ?
    """, (rg_id,))
    
    temperatures = [row[0] for row in cursor.fetchall()]

    if len(temperatures) == 0:
        conn.close()
        raise ValueError(f"No temperature found for rg_id {rg_id}")
    if len(temperatures) > 1:
        conn.close()
        raise ValueError(f"Multiple temperatures found for rg_id {rg_id}: {temperatures}")

    return temperatures[0]

def fetch_reaction_pH(db_path: str, rg_id: int) -> float:
    """
    Fetch the reaction pH for a given reaction group ID (rg_id).
    
    Args:
        db_path (str): Path to the database file.
        rg_id (int): Reaction group ID.
    
    Returns:
        float: Reaction pH.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT b.pH
        FROM reaction_groups rg
        JOIN probing_reactions pr ON rg.rxn_id = pr.id
        JOIN buffers b ON pr.buffer_id = b.id
        WHERE rg.rg_id = ?
    """, (rg_id,))
    
    pH_values = [row[0] for row in cursor.fetchall()]

    if len(pH_values) == 0:
        conn.close()
        raise ValueError(f"No pH found for rg_id {rg_id}")
    if len(pH_values) > 1:
        conn.close()
        raise ValueError(f"Multiple pH values found for rg_id {rg_id}: {pH_values}")

    return pH_values[0]

def fetch_buffer_id(buffer_name: str, db_path: str) -> Optional[int]:
    """
    Fetch the buffer ID for a given buffer name.
    
    Args:
        buffer_name (str): Name of the buffer.
        db_path (str): Path to the database file.
    
    Returns:
        Optional[int]: Buffer ID if found, otherwise None.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id FROM buffers WHERE name = ?
    """, (buffer_name,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]
    else:
        return None
    

def fetch_arrhenius_closest_pH(db_path: str, reaction_type: str, species: str, pH: float) -> list:
    """
    Fetch Arrhenius parameters for the buffer with the closest pH.

    Args:
        db_path (str): Path to the database file.
        reaction_type (str): Type of reaction (e.g., "deg", "add").
        species (str): Species involved in the reaction (data_source column).
        pH (float): Target pH to find the closest match in buffers.

    Returns:
        list: List of tuples containing Arrhenius parameters.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT af.slope, af.slope_err, af.intercept, af.intercept_err, b.pH
        FROM arrhenius_fits af
        JOIN buffers b ON af.buffer_id = b.id
        WHERE af.reaction_type = ? AND af.substrate = ?
        ORDER BY ABS(b.pH - ?) ASC
        LIMIT 1
    """, (reaction_type, species, pH))

    result = cursor.fetchall()
    conn.close()
    return result


def fetch_all_rg_ids(db_path: str) -> list:
    """
    Fetch all reaction group IDs from the database.
    
    Args:
        db_path (str): Path to the database file.
    
    Returns:
        list: List of reaction group IDs.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT reaction_groups.rg_id 
        FROM reaction_groups
        JOIN probing_reactions pr ON reaction_groups.rxn_id = pr.id
        JOIN sequencing_samples ss ON pr.s_id = ss.id
        WHERE ss.to_drop = 0
    """)
    rg_ids = [row[0] for row in cursor.fetchall()]
    
    conn.close()


    return rg_ids


def fetch_rxn_ids(db_path: str, rg_id: int) -> list:
    """
    Fetch all reaction IDs associated with a given reaction group ID (rg_id).
    
    Args:
        db_path (str): Path to the database file.
        rg_id (int): Reaction group ID.
    
    Returns:
        list: List of reaction IDs.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT pr.reaction_time, pr.treated, pr.s_id
        FROM reaction_groups rg
        JOIN probing_reactions pr ON rg.rxn_id = pr.id
        JOIN sequencing_samples ss ON pr.s_id = ss.id
        WHERE rg.rg_id = ? AND ss.to_drop = 0
    """, (rg_id,))
    results = cursor.fetchall()
        
    conn.close()
    
    return results

def fetch_fmod_vals(db_path: str, s_id: int) -> Optional[list]:
    """
    Fetch all fmod_vals for a given sequencing sample ID (s_id).
    
    Args:
        db_path (str): Path to the database file.
        s_id (int): Sequencing sample ID.
    
    Returns:
        Optional[list]: List of tuples containing fmod_val and reaction_time if found, otherwise None.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT fmod_val, pr.reaction_time
        FROM fmod_vals
        JOIN probing_reactions pr ON fmod_vals.rxn_id = pr.id
        WHERE pr.s_id = ?
    """, (s_id,))
    
    result = cursor.fetchall()
    conn.close()
    
    if result:
        return result  # Return fmod_val and reaction_time
    else:
        return None

def fetch_timecourse_records(db_path: str, rg_id: int) -> list:
    """
    Fetch raw timecourse records for a given reaction group ID.

    Args:
        db_path (str): Path to the SQLite database.
        rg_id (int): Reaction group ID.

    Returns:
        list: List of tuples for each record.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT fv.fmod_val, pr.reaction_time, n.site, n.base, pr.treated,
               fv.read_depth, pr.RT, fv.valtype, pr.temperature,
               c.disp_name, rg.rg_id, pr.buffer_id, sr.id
        FROM probing_reactions pr
        JOIN fmod_vals fv ON pr.id = fv.rxn_id
        JOIN nucleotides n ON fv.nt_id = n.id
        JOIN constructs c ON pr.construct_id = c.id
        JOIN sequencing_samples ss ON pr.s_id = ss.id
        JOIN reaction_groups rg ON rg.rxn_id = pr.id
        JOIN sequencing_runs sr ON ss.seqrun_id = sr.id
        WHERE rg.rg_id = ?
        AND fv.fmod_val IS NOT NULL
    """, (rg_id,))
    records = cursor.fetchall()
    conn.close()
    return records


def fetch_filtered_freefit_sites(db_path: str, rg_id: int, bases: list = [], r2_thres: float = 0.8) -> list:
    """
    Fetch nucleotides that passed the free fit with a specified R2 threshold and specified bases.

    Args:
        db_path (str): Path to the database file.
        rg_id (int): Reaction group ID.
        bases (list): List of bases to filter on (e.g., ['A', 'C']).
        r2_thres (float): Minimum R-squared value threshold.

    Returns:
        list: List of nt_ids that match the filter criteria.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if bases:
        placeholders = ','.join(['?'] * len(bases))
        query = f"""
            SELECT DISTINCT n.id
            FROM free_tc_fits ft
            JOIN nucleotides n ON ft.nt_id = n.id
            WHERE ft.rg_id = ?
            AND ft.r2 > ?
            AND n.base IN ({placeholders})
        """
        params = [rg_id, r2_thres] + bases
    else:
        query = """
            SELECT DISTINCT n.id
            FROM free_tc_fits ft
            JOIN nucleotides n ON ft.nt_id = n.id
            WHERE ft.rg_id = ?
            AND ft.r2 > ?
        """
        params = [rg_id, r2_thres]

    cursor.execute(query, params)
    nt_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return nt_ids


def fetch_dataset_fmods(db_path: str, rg_id: int, nt_ids: list) -> tuple:
    """
    Fetch the dataset for fmod values and reaction times for a given reaction group ID (rg_id),
    grouped by nt_id.
    
    Args:
        db_path (str): Path to the database file.
        rg_id (int): Reaction group ID.
        nt_ids (list): List of nucleotide IDs to filter the results.
    
    Returns:
        tuple: (time_data_list, fmod_data_list), each a list of lists, grouped by nt_id.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Prepare SQL placeholders
    placeholders = ','.join('?' * len(nt_ids))

    cursor.execute(f"""
        SELECT pr.reaction_time, fv.fmod_val, fv.nt_id, pr.treated
        FROM fmod_vals fv
        JOIN probing_reactions pr ON fv.rxn_id = pr.id
        JOIN reaction_groups rg ON rg.rxn_id = pr.id
        WHERE rg.rg_id = ?
        AND fv.fmod_val IS NOT NULL
        AND fv.nt_id IN ({placeholders})
        ORDER BY fv.nt_id, pr.reaction_time
    """, (rg_id, *nt_ids))

    records = cursor.fetchall()
    conn.close()

    # time = time * treated (untreated time = 0)
    records = [(time * treated, fmod, nt_id) for time, fmod, nt_id, treated in records]

    # Group by nt_id
    time_data_dict = defaultdict(list)
    fmod_data_dict = defaultdict(list)

    for time, fmod, nt_id in records:
        time_data_dict[nt_id].append(time)
        fmod_data_dict[nt_id].append(fmod)

    # Return as list of lists in the order of nt_ids provided
    time_data = [time_data_dict[nt_id] for nt_id in nt_ids]
    fmod_data = [fmod_data_dict[nt_id] for nt_id in nt_ids]

    return time_data, fmod_data

def fetch_dataset_freefit_params(db_path: str, rg_id: int, nt_ids: list) -> tuple:
    """
    Fetch the free fit parameters for a given reaction group ID (rg_id).
    
    Args:
        db_path (str): Path to the database file.
        rg_id (int): Reaction group ID.
        nt_ids (list): List of nucleotide IDs to filter the results. If empty, fetch all.
    
    Returns:
        tuple: (kappa_array, kdeg_array, fmod0_array) where each is a list of corresponding values.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT kobs_val, kdeg_val, fmod0
        FROM free_tc_fits
        WHERE rg_id = ?
        AND nt_id IN ({})
        ORDER BY nt_id ASC
    """.format(','.join('?' * len(nt_ids))), (rg_id, *nt_ids))
    
    records = cursor.fetchall()
    conn.close()
    
    kappa_array = [record[0] for record in records]
    kdeg_array = [record[1] for record in records]
    fmod0_array = [record[2] for record in records]
    
    return kappa_array, kdeg_array, fmod0_array

def fetch_global_kdeg_val(db_path: str, rg_id: int, model: str, species: str) -> float:
    """
    Fetch the global kdeg value for a given reaction group ID (rg_id).
    
    Args:
        db_path (str): Path to the database file.
        rg_id (int): Reaction group ID.
    
    Returns:
        float: Global kdeg value.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT k_value
        FROM probing_kinetic_rates
        WHERE rg_id = ?
        AND model = ?
        AND species = ?
    """, (rg_id, model, species))

    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]
    else:
        raise ValueError(f"No global kdeg value found for rg_id {rg_id}")
    

def fetch_melted_probing_data(db_path: str, temp_thres: float = 60) -> Optional[tuple]:
    """
    Fetch melted probing fit param (kobs) for a given reaction group ID (rg_id).
    
    Args:
        db_path (str): Path to the database file.
        temp_thres (float): Minimum temperature threshold to filter results.
    
    Returns:
        tuple: (records, description) where records is a list of tuples containing the data,
               and description is a list of column names.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT rg.rg_id, rg.rxn_id, cf.kobs_val, cf.kobs_err, cf.chisq, cf.r2, n.id, n.base,
                   pr.temperature, pr.replicate,
                   pr.RT, pr.done_by, pr.buffer_id, pr.construct_id
        FROM constrained_tc_fits cf
        JOIN reaction_groups rg ON cf.rg_id = rg.rg_id
        JOIN probing_reactions pr ON rg.rxn_id = pr.id
        JOIN nucleotides n ON cf.nt_id = n.id
        JOIN sequencing_samples ss ON pr.s_id = ss.id
        AND pr.temperature >= ?
        AND cf.kobs_err > 0
        AND ss.to_drop = 0
    """, (temp_thres,))

    records = cursor.fetchall()
    description = [desc[0] for desc in cursor.description]
    conn.close()
    
    return records, description


def fetch_probing_kinetic_rate(conn: sqlite3.Connection, buffer_id: int, species: str, reaction_type: str) -> Optional[float]:
    """
    Fetch the probing kinetic rate (kobs) for a given reaction group ID (rg_id) and nucleotide ID (nt_id).
    
    Args:
        conn (sqlite3.Connection): SQLite connection object.
        buffer_id (int): Buffer ID to filter the results.
        species (str): Species involved in the reaction.
        reaction_type (str): Type of reaction (e.g., "deg", "add").
    
    Returns:
        Optional[tuple]: Probing kinetic rate k_value, k_err and description if found, otherwise None.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT kr.k_value, kr.k_error
        FROM probing_kinetic_rates kr
        JOIN reaction_groups rg ON kr.rg_id = rg.rg_id
        JOIN probing_reactions pr ON rg.rxn_id = pr.id
        WHERE pr.buffer_id = ?
        AND kr.species = ?
        AND kr.model = ?
    """, (buffer_id, species, reaction_type))

    result = cursor.fetchone()
    return result[0] if result else None


def fetch_distinct_tempgrad_group(db_path: str) -> list:
    """
    Fetch distinct temperature gradient groups from the database.

    Args:
        db_path (str): Path to the database file.

    Returns:
        list: List of tuples containing distinct temperature gradient groups.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT
            pr.rg_id,
            pr.buffer_id,
            pr.construct_id,
            pr.RT,
            pr.probe,
            pr.probe_concentration
        FROM probing_reactions pr
        WHERE pr.treated = 1
    """)

    results = cursor.fetchall()
    conn.close()
    return results