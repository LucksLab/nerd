# nerd/kinetics/timecourse.py

import pandas as pd
import numpy as np
from rich.console import Console
from lmfit.models import ExponentialModel, ConstantModel
from nerd.db.io import connect_db, init_db, check_db, insert_tc_fit
from nerd.utils.fit_models import fit_timecourse 
from nerd.db.fetch import fetch_all_nt, fetch_timecourse_data, fetch_reaction_temp, fetch_reaction_pH
from nerd.kinetics.degradation import calc_kdeg

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
        try:
            # Fetch timecourse data for this nt_id and rg_id
            time_data, fmod_data = fetch_timecourse_data(db_path, nt_id, rg_id)

            # Actual fitting
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

            console.print(
                f"[green]✓ Free timecourse fit complete[/green]: "
                f"kobs = {fit_result['kobs_val']:.4f} ± {fit_result['kobs_err']:.4f}, "
                f"kdeg = {fit_result['kdeg_val']:.4f} ± {fit_result['kdeg_err']:.4f}, "
                f"fmod0 = {fit_result['fmod0']:.4f} ± {fit_result['fmod0_err']:.4f}, "
                f"r² = {fit_result['r2']:.4f}"
            )

            conn = connect_db(db_path)

            # Check if the database is initialized and has the required columns
            if not check_db(conn, "free_tc_fits", REQUIRED_COLUMNS):
                console.print(f"[red]Error:[/red] Database initialization failed or missing required columns.")
                conn.close()
                return
            else:
                console.print(f"[green]✓ Database check passed and ready for Arrhenius fit import[/green]")


            console.print(f"[green]✓ Free timecourse fit success [/green]: {fit_result['rsquared']:.4f} R²")
            insert_success = insert_tc_fit(conn, fit_result)
            count += insert_success
            conn.close()


        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Free timecourse fit failed: {e}")

    console.print(f"[green]✓ Free timecourse fits imported successfully[/green]: {count} fits inserted into the database.")