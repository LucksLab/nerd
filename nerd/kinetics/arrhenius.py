# nerd/kinetics/arrhenius.py

import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console

from nerd.db.io import connect_db, fetch_kinetic_rates, check_db, insert_arrhenius_fit
from nerd.utils.fit_models import fit_linear
from nerd.db.fetch import fetch_buffer_id

console = Console()

REQUIRED_COLUMNS = [
    "reaction_type",  # Type of reaction (e.g. "deg")
    "data_source",    # Source of data (e.g. "nmr")
    "slope",          # Slope of the Arrhenius fit
    "slope_err",      # Standard error of the slope
    "intercept",      # Intercept of the Arrhenius fit
    "intercept_err",  # Standard error of the intercept
    "r2",            # R-squared value of the fit
    "model_file"      # Optional file path to model data (e.g. pickle, json)
]

def rows_to_dicts(rows, description):
    column_names = [desc[0] for desc in description]
    return [dict(zip(column_names, row)) for row in rows]

def run(reaction_type="deg", data_source="nmr", species = "dms", 
        buffer = "Schwalbe_bistris", db_path: str = None):
    """
    CLI entrypoint for Arrhenius fitting.
    """
    
    count = 0

    if data_source == "nmr":
        conn = connect_db(db_path)  # row_factory set inside
        buffer_id = fetch_buffer_id(buffer, db_path)

        data, description = fetch_kinetic_rates(conn, buffer_id, species, reaction_type)
        conn.close()

        if not data:
            console.print(f"[red]No kinetic rates found for reaction type '{reaction_type}' with species '{species}' and buffer '{buffer}'[/red]")
            return

        data = rows_to_dicts(data, description)
        x_vals = np.array([1 / r["temperature"] for r in data])
        y_vals = np.array([np.log(r["k_value"]) for r in data])

    try:
        result = fit_linear(x_vals, y_vals)

        slope = result.params["slope"].value
        slope_err = result.params["slope"].stderr
        intercept = result.params["intercept"].value
        intercept_err = result.params["intercept"].stderr
        r2 = result.rsquared

        fit_result = {
            "reaction_type": reaction_type,
            "data_source": data_source,
            "substrate": species,
            "slope": slope,
            "slope_err": slope_err,
            "intercept": intercept,
            "intercept_err": intercept_err,
            "r2": r2,
            "model_file": None  # Optional, can be set to a file path if needed
        }

        console.print(f"[green]✓ Arrhenius fit complete[/green]: slope = {slope:.4f} ± {slope_err:.4f}, intercept = {intercept:.4f} ± {intercept_err:.4f}, r² = {r2:.4f}")

        conn = connect_db(db_path)

        # Check if the database is initialized and has the required columns
        if not check_db(conn, "arrhenius_fits", REQUIRED_COLUMNS):
            console.print(f"[red]Error:[/red] Database initialization failed or missing required columns.")
            conn.close()
            return
        else:
            console.print(f"[green]✓ Database check passed and ready for Arrhenius fit import[/green]")


        insert_success = insert_arrhenius_fit(conn, fit_result)
        count += insert_success
        conn.close()

    except Exception as e:
        console.print(f"[red]Fit failed:[/red] {e}")

    console.print(f"[green]Imported Arrhenius fits for {count} samples[/green]")