# nerd/kinetics/timecourse.py

import pandas as pd
import numpy as np
from rich.console import Console
from lmfit.models import ExponentialModel, ConstantModel
from nerd.db.io import connect_db, init_db, insert_probe_kinetic_rate  # may use a separate insert_kobs table
# from nerd.db.io import get_timecourse_data  # implement this if pulling from db

console = Console()

def fit_site_timecourse(time_array, signal_array):
    """
    Fit y(t) = A * (1 - exp(-k * t)) + C to time-course data for one site.
    """
    model = (-1 * ExponentialModel(prefix="exp_")) + ConstantModel(prefix="const_")
    params = model.make_params()
    params["exp_amplitude"].set(value=1.0, min=0)
    params["exp_decay"].set(value=0.5, min=0)
    params["const_c"].set(value=np.min(signal_array))

    result = model.fit(signal_array, params, x=time_array)
    return result

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