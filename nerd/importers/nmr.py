# nerd/importers/nmr.py

import csv
from pathlib import Path
from rich.console import Console
from nerd.db.io import connect_db, init_db, check_db, insert_nmr_reaction

console = Console()

# List of required columns for NMR reaction metadata import.
REQUIRED_COLUMNS = [
    "reaction_type",         # Type of reaction, e.g., "deg" or "add"
    "temperature",           # Temperature at which the reaction was run (°C)
    "replicate",             # Replicate identifier
    "num_scans",             # Number of NMR scans performed
    "time_per_read",         # Time per read (minutes)
    "total_kinetic_reads",   # Total number of kinetic reads
    "total_kinetic_time",    # Total kinetic time (seconds)
    "probe",                 # Probe used in the experiment
    "probe_conc",            # Concentration of the probe (M)
    "probe_solvent",         # Solvent used for the probe
    "substrate",             # Substrate used in the reaction
    "substrate_conc",        # Concentration of the substrate (M)
    "buffer",                # Buffer used in the reaction
    "nmr_machine",           # NMR machine identifier or name
    "kinetic_data_dir"       # Directory containing kinetic data files
]

def run(csv_path: str, db_path: str = None):
    """
    Import NMR reaction metadata from a CSV into the nmr_reactions table.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        console.print(f"[red]Error:[/red] NMR sample sheet not found: {csv_file}")
        return

    conn = connect_db(db_path)

    # Check if the database is initialized and has the required columns
    if not check_db(conn, "nmr_reactions", REQUIRED_COLUMNS):
        console.print(f"[red]Error:[/red] Database initialization failed or missing required columns.")
        conn.close()
        return
    else:
        console.print(f"[green]✓ Database check passed and ready for NMR import[/green]")

    with open(csv_file, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        # Check for required columns
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            console.print(f"[red]Error:[/red] Missing required columns: {missing}")
            return

        count = 0
        for row in reader:
            try:
                insert_success = insert_nmr_reaction(conn, row)
                count += insert_success

            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Skipped row due to error: {e}")

        # Close the connection after all inserts
        conn.close()

        console.print(f"[green]✓ Imported {count} NMR reactions[/green]")