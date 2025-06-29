import sqlite3
from pathlib import Path
from nerd.db.schema import ALL_TABLES

def update_sample_to_drop(db_path: str, s_id: int, rg_id: int, qc_comment: str) -> bool:
    """
    Updates the sequencing_samples table to mark a sample as to_drop = 1.

    Args:
        db_path (str): Path to the SQLite database.
        s_id (int): Sequencing sample ID to update.
        rg_id (int): Reaction group ID (not currently used in query).
        qc_comment (str): QC annotation (not currently used in query).

    Returns:
        bool: True if the update affected any rows, False otherwise.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE sequencing_samples SET to_drop = 1 WHERE id = ?", (s_id,))
    update_success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return update_success

def update_sample_fmod_val(db_path: str, s_id: int, avg_fmod_val: float) -> bool:
    """
    Updates the fmod_val of a sequencing sample in the fmod_vals table via rxn_id join.

    Args:
        db_path (str): Path to the SQLite database.
        s_id (int): Sequencing sample ID whose fmod_val should be updated.
        avg_fmod_val (float): The new average fmod_val to set.

    Returns:
        bool: True if the update affected any rows, False otherwise.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE fmod_vals
        SET fmod_val = ?
        WHERE rxn_id IN (
            SELECT pr.id
            FROM probing_reactions pr
            WHERE pr.s_id = ?
        )
    """, (avg_fmod_val, s_id))
    update_success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return update_success