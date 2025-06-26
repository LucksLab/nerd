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