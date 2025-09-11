# nerd/kinetics/adduction.py

import pandas as pd
from pathlib import Path
from nerd.db.io import connect_db, insert_fitted_kinetic_rate, check_db, insert_melted_probing_rates
from nerd.utils.param_aggregation import process_aggregation
from nerd.utils.fit_models import fit_ODE_probe
from nerd.db.fetch import fetch_melted_probing_data
from nerd.utils.logging import get_logger

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

def process_melted_param_aggregation(db_path: str = 'test_output/nerd_dev.sqlite3'):

    # Fetch all fitted kobs params (melted temps >60C)
    try:
        result = fetch_melted_probing_data(
            db_path=db_path,
            temp_thres=60
            )
        if result is not None:
            records, description = result
            process_aggregation(records, description, db_path=db_path)
        else:
            log.error("No data returned from fetch_melted_probing_data")
    except Exception as e:
        log.exception("Error fetching data: %s", e)


def process_melted_ind_params(db_path: str = 'test_output/nerd_dev.sqlite3'):
    try:
        # Fetch all melted probing data
        result = fetch_melted_probing_data(db_path=db_path, temp_thres=60)

        if result:
            success_inserts = insert_melted_probing_rates(result, db_path=db_path)
            log.info("Inserted melted probing rates for %d records", success_inserts)
        else:
            log.error("No data returned from fetch_melted_probing_data")
    except Exception as e:
        log.exception("Error extracting melted individual parameters: %s", e)

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

        # If select_id is provided, only process samples with reaction_id in select_id
        # If select_id is empty, process all samples
        if (len(select_id) > 0) and (reaction_id not in select_id):
            continue

        # Split csv by comma and confirm there are exactly 2 files
        csv_paths = kinetic_trace_dir.split(",")
        if len(csv_paths) != 2:
            raise ValueError("Exactly two CSV files are required: one for DMS and one for NTP reporter atom")

        peak_perc_csv = Path(f'test_data/{csv_paths[0]}')
        dms_perc_csv = Path(f'test_data/{csv_paths[1]}')

        if not peak_perc_csv.exists() or not dms_perc_csv.exists():
            log.error("One or both CSV files not found: %s", csv_paths)
            continue

        log.info("Processing: %s + %s", peak_perc_csv.name, dms_perc_csv.name)

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
            k, k_err, rsq, chisq = fit_kinetic_trace(str(peak_perc_csv), str(dms_perc_csv), ntp_conc)
            log.info("Fit complete: k = %.4f ± %.4f, R² = %.4f, χ² = %.4f", k, (k_err or 0.0), rsq, chisq)

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
            log.exception("Fit failed: %s", e)
    log.info("Imported kinetic fits for %d NMR adduction samples", count)