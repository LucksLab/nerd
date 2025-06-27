# nerd/kinetics/timecourse.py

import pandas as pd
import numpy as np
from rich.console import Console
from lmfit.models import ExponentialModel, ConstantModel
from nerd.db.io import connect_db, init_db, check_db, insert_tc_fit
from nerd.utils.fit_models import fit_timecourse 
from nerd.db.fetch import fetch_all_nt, fetch_timecourse_data, fetch_reaction_temp, fetch_reaction_pH

console = Console()

def free_fit(rg_id, db_path):
    """
    Free fit to a rg_id, loops over each site
    """

    REQUIRED_COLUMNS = [
        "rg_id",        # Foreign key to reaction_groups table
        "nt_id",        # Foreign key to nucleotides table
        "kobs_val",     # Observed rate constant value
        "kobs_err",     # Error in observed rate constant
        "kdeg_val",     # Degradation rate constant value
        "kdeg_err",     # Error in degradation rate constant
        "fmod0",        # Initial fraction modified
        "fmod0_err",    # Error in initial fraction modified
        "r2",           # R-squared value of the fit
        "chisq",        # Chi-squared value of the fit
        "time_min",     # Minimum time (in seconds) used in fit
        "time_max"      # Maximum time (in seconds) used in fit
    ]

    # Guess kdeg0 using NMR deg with closest pH

    temp = fetch_reaction_temp(db_path, rg_id)
    pH = fetch_reaction_pH(db_path, rg_id)
    kdeg0 = calc_kdeg(temp, pH)

    # Fetch all nt (nt_ids)
    nt_ids = fetch_all_nt(db_path, rg_id)

    count = 0
    for nt_id in nt_ids:

        time_data, fmod_data = fetch_timecourse_data(db_path, nt_id, rg_id)

        # Actual
        result, outlier = fit_timecourse(time_data, fmod_data, kdeg0)

        # Construct fit_result dict for insertion
        fit_result = {
            "rg_id": rg_id,
            "nt_id": nt_id,
            "kobs_val": result.params["log_kappa"].value,
            "kobs_err": result.params["log_kappa"].stderr,
            "kdeg_val": result.params["log_kdeg"].value,
            "kdeg_err": result.params["log_kdeg"].stderr,
            "fmod0": result.params["log_fmod_0"].value,
            "fmod0_err": result.params["log_fmod_0"].stderr,
            "r2": result.rsquared,
            "chisq": result.chisqr,
            "time_min": min(time_data),
            "time_max": max(time_data)
        }

        console.print(f"[green]✓ Free timecourse fit complete[/green]: slope = {slope:.4f} ± {slope_err:.4f}, intercept = {intercept:.4f} ± {intercept_err:.4f}, r² = {r2:.4f}")

        conn = connect_db(db_path)

        # Check if the database is initialized and has the required columns
        if not check_db(conn, "free_tc_fits", REQUIRED_COLUMNS):
            console.print(f"[red]Error:[/red] Database initialization failed or missing required columns.")
            conn.close()
            return
        else:
            console.print(f"[green]✓ Database check passed and ready for Arrhenius fit import[/green]")

        insert_success = insert_tc_fit(conn, fit_result)
        count += insert_success
        conn.close()
    
    console.print(f"[green]✓ Free timecourse fits imported successfully[/green]: {count} fits inserted into the database.")


def run(reaction_group_id: str, db_path: str = None):
    """
    Fit independent time-courses for each site in a reaction group.
    """
    conn = connect_db(db_path)
    init_db(conn)

    # === Placeholder: load time-course data ===
    # Replace this with actual query logic:
    # Example data structure:
    # {
    #   10: {"times": [0, 1, 2], "signals": [0.1, 0.4, 0.6]},
    #   11: {"times": [...], "signals": [...]}
    # }
    dummy_data = {
        10: {"times": [0, 1, 2, 4], "signals": [0.05, 0.32, 0.52, 0.68]},
        11: {"times": [0, 1, 2, 4], "signals": [0.02, 0.22, 0.38, 0.60]},
    }

    for site, ts in dummy_data.items():
        try:
            result = fit_site_timecourse(np.array(ts["times"]), np.array(ts["signals"]))
            k_obs = result.params["exp_decay"].value
            k_err = result.params["exp_decay"].stderr

            console.print(f"[green]✓ Site {site}[/green]: k_obs = {k_obs:.4f} ± {k_err:.4f}")

            insert_probe_kinetic_rate(conn, nmr_reaction_id=f"{reaction_group_id}_nt{site}",
                                      k_value=k_obs, k_error=k_err, species="DMS")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Fit failed for site {site}: {e}")