import sqlite3
from pathlib import Path
from nerd.db.schema import ALL_TABLES
from nerd.db.io import insert_tempgrad_group
from collections import defaultdict

import sqlite3

def assign_tempgrad_groups(db_path: str, results: list) -> int:
    """
    Assign a tg_id to each rg_id in the reaction_groups table based on shared condition sets.

    Args:
        db_path (str): Path to the database file.
        results (list): List of tuples (buffer_id, construct_id, RT, probe, probe_concentration)

    Returns:
        list of tuples: (tg_id, rg_id, buffer_id, construct_id, RT, probe, probe_concentration)
    """
    # Step 1: Create mapping from condition tuple to tg_id
    condition_to_tg = {}
    tg_counter = 1

    for cond in results:
        if cond not in condition_to_tg:
            condition_to_tg[cond] = tg_counter
            tg_counter += 1

    print(condition_to_tg)

    # Step 2: Loop through each rg_id and fetch its condition tuple
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT rg.rg_id, pr.buffer_id, pr.construct_id, pr.RT, pr.probe, pr.probe_concentration, pr.temperature, pr.replicate
        FROM reaction_groups rg
        JOIN probing_reactions pr ON rg.rxn_id = pr.id
        WHERE pr.treated = 1
    """)

    count = 0
    for rg_id, buffer_id, construct_id, RT, probe, probe_concentration, temp, rep in cursor.fetchall():
        cond_tuple = (buffer_id, construct_id, RT, probe, probe_concentration)
        tg_id = condition_to_tg.get(cond_tuple)
        print(tg_id)
        if tg_id is not None:
            # insert into tempgrad_groups table
            tempgrad_insert_dict = {
                'tg_id': tg_id,
                'rg_id': rg_id,
                'buffer_id': buffer_id,
                'construct_id': construct_id,
                'RT': RT,
                'probe': probe,
                'probe_concentration': probe_concentration,
                'temperature': temp,
                'replicate': rep
            }

            insert_success = insert_tempgrad_group(db_path, tempgrad_insert_dict)
            count += insert_success
    conn.close()
    return count

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