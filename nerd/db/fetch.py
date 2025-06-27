# nerd/db/fetch.py

import sqlite3
from pathlib import Path
from nerd.db.schema import ALL_TABLES
from typing import Optional

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
        JOIN probing_reactions pr ON fv.id = pr.rxn_id
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