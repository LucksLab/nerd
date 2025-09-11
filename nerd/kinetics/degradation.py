# nerd/kinetics/degradation.py

import pandas as pd
from pathlib import Path
from nerd.utils.logging import get_logger
from nerd.db.io import connect_db, insert_fitted_kinetic_rate, check_db
from nerd.utils.fit_models import fit_exp_decay
from nerd.db.fetch import fetch_arrhenius_closest_pH

log = get_logger(__name__)

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
    log.warning("Global fitting for rg_id %s is not implemented yet.", rg_id)


    return None

def run(all_samples: list, select_id: list = [], db_path: str  = '.'):
    """
    Main CLI entrypoint: fits degradation kinetics and stores result to DB.
    """

    # if select_id is None, run on all samples
    # else, filter samples by select_id - only process reaction_id in select_id
    count = 0
    for sample in all_samples:
        reaction_id = sample[0]
        kinetic_trace_dir = sample[-1]

        # If select_id is provided, only process samples with reaction_id in select_id
        # If select_id is empty, process all samples
        if (len(select_id) > 0) and (reaction_id not in select_id):
            continue

        csv_file = Path(f'test_data/{kinetic_trace_dir}')

        if not csv_file.exists():
            log.error("File not found: %s", kinetic_trace_dir)
            continue

        log.info("Processing: %s", csv_file.name)
        
        try:
            conn = connect_db(db_path)

            # Check if the database is initialized and has the required columns
            if not check_db(conn, "nmr_kinetic_rates", REQUIRED_COLUMNS):
                log.error("Database initialization failed or missing required columns.")
                conn.close()
                return
            else:
                log.info("Database check passed and ready for NMR import")

            # Fit the kinetic trace
            k, k_err, rsq, chisq = fit_kinetic_trace(f'test_data/{kinetic_trace_dir}')
            log.info("Fit complete: k = %.4f ± %.4f, R² = %.4f, χ² = %.4f", k, k_err or 0.0, rsq, chisq)

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
            log.exception("Fit failed: %s", e)

    log.info("Imported kinetic fits for %d NMR degradation samples", count)

from numpy import exp

def calc_kdeg(temp, pH, db_path):
    """
    Find NMR arrhenius at closest pH then calculate kdeg using the Arrhenius equation.
    """

    # Fetch Arrhenius parameters for the given temperature and pH (deg)
    arrhenius_params = fetch_arrhenius_closest_pH(
        db_path=db_path,
        reaction_type="deg",
        species="dms",
        pH=pH
    )
    if not arrhenius_params:
        log.error("No Arrhenius parameters found for pH %s", pH)
        return None
    
    slope, slope_err, intercept, intercept_err, pH_value = arrhenius_params[0]

    kdeg =  exp(slope / (temp + 273.15) + intercept)

    return kdeg