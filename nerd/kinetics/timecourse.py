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

            if time_data is None or fmod_data is None:
                console.print(f"[yellow]Warning:[/yellow] No timecourse data found for rg_id {rg_id}, nt_id {nt_id}. Skipping.")
                continue
            elif len(time_data) < 3 or len(fmod_data) < 3:
                console.print(f"[yellow]Warning:[/yellow] Not enough data points for fitting for rg_id {rg_id}, nt_id {nt_id}. Skipping.")
                continue
            # Actual fitting
            result, outlier = fit_timecourse(time_data, fmod_data, kdeg0)
            # Construct fit_result dict for insertion
            fit_result = {
                "rg_id": rg_id,
                "nt_id": nt_id,
                "kobs_val": result.params["log_kappa"].value,
                "kobs_err": result.params["log_kappa"].stderr if result.params["log_kappa"].stderr is not None else -1,
                "kdeg_val": result.params["log_kdeg"].value,
                "kdeg_err": result.params["log_kdeg"].stderr if result.params["log_kdeg"].stderr is not None else -1,
                "fmod0": result.params["log_fmod_0"].value,
                "fmod0_err": result.params["log_fmod_0"].stderr if result.params["log_fmod_0"].stderr is not None else -1,
                "r2": result.rsquared,
                "chisq": result.chisqr,
                "time_min": min(time_data),
                "time_max": max(time_data)
            }

            def fmt(val):
                return f"{val:.4f}" if val is not None else "None"

            console.print(
                f"[green]✓ Free timecourse fit complete[/green]: "
                f"kobs = {fmt(fit_result['kobs_val'])} ± {fmt(fit_result['kobs_err'])}, "
                f"kdeg = {fmt(fit_result['kdeg_val'])} ± {fmt(fit_result['kdeg_err'])}, "
                f"fmod0 = {fmt(fit_result['fmod0'])} ± {fmt(fit_result['fmod0_err'])}, "
                f"r² = {fmt(fit_result['r2'])}"
            )

            conn = connect_db(db_path)

            # Check if the database is initialized and has the required columns
            if not check_db(conn, "free_tc_fits", REQUIRED_COLUMNS):
                console.print(f"[red]Error:[/red] Database initialization failed or missing required columns.")
                conn.close()
                return
            else:
                console.print(f"[green]✓ Database check passed and ready for free timecourse fit import[/green]")


            console.print(f"[green]✓ Free timecourse fit success [/green]: {fit_result['r2']:.4f} R²")
            insert_success = insert_tc_fit(conn, fit_result)
            count += insert_success
            conn.close()


        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Free timecourse fit failed: {e}")

    console.print(f"[green]✓ Free timecourse fits imported successfully[/green]: {count} fits inserted into the database.")


def run()