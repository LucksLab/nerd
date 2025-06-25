# nerd/energy/meltfit.py

import numpy as np
import pandas as pd
from lmfit import Model
from rich.console import Console

console = Console()

R = 1.987e-3  # kcal/mol·K

def two_state_K(T, dH, dS):
    """
    Compute equilibrium constant K(T) from ΔH and ΔS via van't Hoff.
    """
    return 1 / (1 + np.exp((dH - T * dS) / (R * T)))


def fit_two_state_melt(df: pd.DataFrame):
    """
    Fit K(T) to a 2-state van’t Hoff model using lmfit.
    Assumes df has columns: 'temperature' (C or K), 'k_obs'
    """
    if "temperature" not in df.columns or "k_obs" not in df.columns:
        raise ValueError("Input DataFrame must contain 'temperature' and 'k_obs' columns.")

    # Convert temp to Kelvin if needed
    if df["temperature"].max() < 150:
        df["T"] = df["temperature"] + 273.15
    else:
        df["T"] = df["temperature"]

    # Normalize k_obs to estimate relative K
    kobs_norm = df["k_obs"] / df["k_obs"].max()

    model = Model(two_state_K)
    params = model.make_params(dH=10.0, dS=0.03)
    result = model.fit(kobs_norm, params, T=df["T"])

    return result


def run(csv_path: str):
    """
    CLI entrypoint: fit 2-state melt curve from CSV of k_obs vs. temperature.
    """
    try:
        df = pd.read_csv(csv_path)
        result = fit_two_state_melt(df)

        dH = result.params["dH"].value
        dS = result.params["dS"].value
        dH_err = result.params["dH"].stderr
        dS_err = result.params["dS"].stderr

        console.print(f"[green]✓ Melt fit complete[/green]")
        console.print(f"  ΔH = {dH:.2f} ± {dH_err:.2f} kcal/mol")
        console.print(f"  ΔS = {dS:.4f} ± {dS_err:.4f} kcal/mol·K")

        # Optionally store result to DB or file

    except Exception as e:
        console.print(f"[red]Fit failed:[/red] {e}")