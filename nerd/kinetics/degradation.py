# nerd/kinetics/degradation.py

import pandas as pd
from pathlib import Path
from rich.console import Console
from nerd.db.io import connect_db, insert_fitted_kinetic_rate, check_db
from nerd.utils.fit_models import fit_exp_decay
from nerd.db.fetch import fetch_arrhenius_closest_pH

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

def fit_kinetic_trace(csv_path: str):
    """
    Fit degradation kinetics from a CSV file containing time and signal data.
    Returns the fitted rate constant and its error.
    """
    df = pd.read_csv(csv_path)
    if "time" not in df or "peak_integral" not in df:
        raise ValueError("CSV must contain 'time' and 'peak_integral' columns")

    # Perform fit
    result = fit_exp_decay(df["time"], df["peak_integral"])
    
    # Decay is tau = 1/k, so k = 1/decay
    k = 1 / result.params["decay"].value

    # Error handling for inverse of decay constant
    k_err = result.params["decay"].stderr / (result.params["decay"].value ** 2) if result.params["decay"].stderr is not None else None
    rsq = result.rsquared
    chisq = result.chisqr

    return k, k_err, rsq, chisq

def global_fit_probing(rg_id: int, db_path: str):
    """
    Fit degradation kinetics for a reaction group (rg_id) and store results in the database.
    This function is a placeholder for future global fitting logic.
    """
    console.print(f"[yellow]Warning:[/yellow] Global fitting for rg_id {rg_id} is not implemented yet.")
    return None

def run(all_samples: list, select_id: list = None, db_path: str = None):
    """
    Main CLI entrypoint: fits degradation kinetics and stores result to DB.
    """

    # if select_id is None, run on all samples
    # else, filter samples by select_id - only process reaction_id in select_id
    count = 0
    for sample in all_samples:
        reaction_id = sample[0]
        kinetic_trace_dir = sample[-1]

        if select_id is not None and reaction_id not in select_id:
            continue

        csv_file = Path(f'test_data/{kinetic_trace_dir}')

        if not csv_file.exists():
            console.print(f"[red]Error:[/red] File not found: {kinetic_trace_dir}")
            continue

        console.print(f"[blue]Processing:[/blue] {csv_file.name}")
        
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
            k, k_err, rsq, chisq = fit_kinetic_trace(f'test_data/{kinetic_trace_dir}')
            console.print(f"[green]✓ Fit complete[/green]: k = {k:.4f} ± {k_err:.4f}, R² = {rsq:.4f}, χ² = {chisq:.4f}")

            # Prepare reaction metadata
            reaction_metadata = {
                "nmr_reaction_id": reaction_id,
                "model": "exponential",  # Assuming exponential decay model
                "k_value": k,
                "k_error": k_err,
                "r2": rsq,
                "chisq": chisq,
                "species": 'dms'  # Assuming species is in the second column
            }

            # Insert the fitted kinetic rate into the database
            insert_success = insert_fitted_kinetic_rate(conn, reaction_metadata)
            count += insert_success
            conn.close()

        except Exception as e:
            console.print(f"[red]Fit failed:[/red] {e}")

    console.print(f"[green]✓ Imported kinetic fits for {count} NMR degradation samples[/green]")

from numpy import exp

def calc_kdeg(temp, pH):
    """
    Find NMR arrhenius at closest pH then calculate kdeg using the Arrhenius equation.
    """

    # Fetch Arrhenius parameters for the given temperature and pH (deg)
    arrhenius_params = fetch_arrhenius_closest_pH(
        db_path='test_output/nerd_dev.sqlite3',
        reaction_type="deg",
        species="dms",
        pH=pH
    )
    if not arrhenius_params:
        console.print(f"[red]Error:[/red] No Arrhenius parameters found for pH {pH}")
        return None
    
    slope, slope_err, intercept, intercept_err, pH_value = arrhenius_params[0]

    kdeg =  exp(slope / (temp + 273.15) + intercept)

    return kdeg