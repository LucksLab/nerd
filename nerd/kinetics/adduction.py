# nerd/kinetics/adduction.py

import pandas as pd
from pathlib import Path
from rich.console import Console
from nerd.db.io import connect_db, insert_fitted_kinetic_rate, check_db
from nerd.utils.param_aggregation import process_aggregation
from nerd.utils.fit_models import fit_ODE_probe
from nerd.db.fetch import fetch_melted_probing_data

console = Console()

# List of required columns for NMR reaction metadata import.
REQUIRED_COLUMNS = [
    "nmr_reaction_id",  # Foreign key to nmr_reactions table
    "model",            # Model used for fitting (e.g. "exponential", "ode")
    "k_value",          # Fitted kinetic rate value
    "k_error",          # Fitted kinetic rate std error
    "r2",               # R-squared value of the fit
    "chisq",            # Chi-squared value of the fit
    "species",          # Species for which the rate is calculated (e.g. "dms", "atp-c8", etc.)
]

def fit_kinetic_trace(peak_perc_csv: str, dms_perc_csv: str, ntp_conc: float):
    """
    Fit adduction kinetics from 2 CSV file containing time and fraction_species 
    data for DMS and some reporter atom for NTP (ex. C8 for ATP).
    Returns the fitted rate constant and its error.
    """

    # Load CSV files
    peak_perc = pd.read_csv(peak_perc_csv)
    dms_perc = pd.read_csv(dms_perc_csv)

    # Check if the required columns are present in both CSV files
    if "time" not in peak_perc or "peak" not in peak_perc:
        raise ValueError("CSV for peak must contain 'time' and 'peak' columns")
    if "time" not in dms_perc or "peak" not in dms_perc:
        raise ValueError("CSV for DMS must contain 'time' and 'peak' columns")

    # Perform fit
    fit_params, r2, chisq = fit_ODE_probe(dms_perc, peak_perc, ntp_conc)

    k = fit_params["k_add"].value
    k_err = fit_params["k_add"].stderr / (fit_params["k_add"].value ** 2) if fit_params["k_add"].stderr is not None else None

    return k, k_err, r2, chisq

def process_param_aggregation():

    # Fetch all fitted kobs params (melted temps >60C)
    try:
        result = fetch_melted_probing_data(
            db_path='test_output/nerd_dev.sqlite3',
            temp_thres=60
            )
        if result is not None:
            records, description = result
            process_aggregation(records, description, db_path='test_output/nerd_dev.sqlite3', console=console)
        else:
            console.print("[red]Error:[/red] No data returned from fetch_melted_probing_data")
    except Exception as e:
        console.print(f"[red]Error fetching data:[/red] {e}")


def run(all_samples: list, select_id: list = [], db_path: str = ''):
    """
    Main CLI entrypoint: fits adduction curve to ODE and stores result to DB.
    """

    # if select_id is None, run on all samples
    # else, filter samples by select_id - only process reaction_id in select_id
    count = 0
    for sample in all_samples:
        reaction_id = sample[0]
        kinetic_trace_dir = sample[-1]
        substrate = sample[-4]
        ntp_conc = sample[-3]

        if select_id is not None and reaction_id not in select_id:
            continue

        # Split csv by comma and confirm there are exactly 2 files
        csv_paths = kinetic_trace_dir.split(",")
        if len(csv_paths) != 2:
            raise ValueError("Exactly two CSV files are required: one for DMS and one for NTP reporter atom")

        peak_perc_csv = Path(f'test_data/{csv_paths[0]}')
        dms_perc_csv = Path(f'test_data/{csv_paths[1]}')

        if not peak_perc_csv.exists() or not dms_perc_csv.exists():
            console.print(f"[red]Error:[/red] One or both CSV files not found: {csv_paths}")
            continue

        console.print(f"[blue]Processing:[/blue] {peak_perc_csv.name} + {dms_perc_csv.name}")
        
        try:
            conn = connect_db(db_path)

            # Check if the database is initialized and has the required columns
            if not check_db(conn, "nmr_kinetic_rates", REQUIRED_COLUMNS):
                console.print(f"[red]Error:[/red] Database initialization failed or missing required columns.")
                conn.close()
                return
            else:
                console.print(f"[green]✓ Database check passed and ready for NMR import[/green]")

            # Fit the kinetic trace
            k, k_err, rsq, chisq = fit_kinetic_trace(peak_perc_csv, dms_perc_csv, ntp_conc)
            console.print(f"[green]✓ Fit complete[/green]: k = {k:.4f} ± {k_err:.4f}, R² = {rsq:.4f}, χ² = {chisq:.4f}")

            # Prepare reaction metadata
            reaction_metadata = {
                "nmr_reaction_id": reaction_id,
                "model": "ODE",
                "k_value": k,
                "k_error": k_err,
                "r2": rsq,
                "chisq": chisq,
                "species": substrate  # Assuming species is in the second column
            }

            # Insert the fitted kinetic rate into the database
            insert_success = insert_fitted_kinetic_rate(conn, reaction_metadata)
            count += insert_success
            conn.close()

        except Exception as e:
            console.print(f"[red]Fit failed:[/red] {e}")

    console.print(f"[green]✓ Imported kinetic fits for {count} NMR adduction samples[/green]")