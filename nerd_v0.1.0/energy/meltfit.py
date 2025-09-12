# nerd/energy/meltfit.py

import numpy as np
import pandas as pd
from rich.console import Console
from nerd.db.fetch import fetch_all_tempgrad_groups, fetch_all_kobs
from nerd.db.io import insert_melt_fit
from nerd.utils.fit_models import fit_meltcurve
console = Console()
from collections import defaultdict

def run(db_path: str):
    """
    CLI entrypoint: fit 2-state melt curve from CSV of k_obs vs. temperature.
    """
    try:
        tg_ids = fetch_all_tempgrad_groups(db_path = db_path)
        if not tg_ids:
            console.print("[red]No temperature gradient groups found in the database.[/red]")
            return
        console.print(f"[green]Found {len(tg_ids)} temperature gradient groups to process.[/green]")

        # Step 1: Collect all kobs data across temperature gradient groups
        all_kobs_data = []

        for tg_entry in tg_ids:
            tg_id = tg_entry[0]
            temp = tg_entry[1]
            buffer_id = tg_entry[2]
            construct_id = tg_entry[3]

            console.print(f"[blue]Processing temperature gradient group ID: {tg_id}[/blue]")
            console.print(f"Temperature: {temp}, Buffer ID: {buffer_id}, Construct ID: {construct_id}")

            # Fetch k_obs data for this temperature group
            kobs_data = fetch_all_kobs(db_path, tg_id)
            if not kobs_data:
                console.print(f"[red]No k_obs data found for TG ID: {tg_id}[/red]")
                continue

            all_kobs_data.extend(kobs_data)

        console.print(f"[green]Fetched total {len(all_kobs_data)} k_obs records across all TGs[/green]")

        # Step 2: Group by unique nucleotide (e.g., by nt_id or (nt_id, site))
        grouped_by_nt = defaultdict(list)
        for row in all_kobs_data:
            # Unpack: (kobs_val, kobs_err, chisq, r2, nt_id, base, site, temperature)
            kobs_val, kobs_err, chisq, r2, nt_id, base, site, temperature = row
            grouped_by_nt[(nt_id, site)].append((kobs_val, kobs_err, temperature))


        count = 0
        # Step 3: Fit kobs vs 1/T for each nucleotide
        for (nt_id, site), records in grouped_by_nt.items():
            console.print(f"\nProcessing nt_id={nt_id} (site={site}) with {len(records)} records...")
            if len(records) < 3:
                console.print(f"[yellow]Skipping nt_id={nt_id} (site={site}): only {len(records)} temperatures[/yellow]")
                continue

            # Extract values
            temps = np.array([rec[2] for rec in records])        # temperature
            kobs_vals = np.array([rec[0] for rec in records])    # k_obs
            kobs_errs = np.array([rec[1] for rec in records])    # error in k_obs

            # Convert temperature to 1/T in Kelvin
            inv_T = 1 / (temps + 273.15)

            try:
                result = fit_meltcurve(inv_T, kobs_vals)
                fit_data = {
                    'tg_id': tg_id,
                    'nt_id': nt_id,
                    'a': result.params['a'].value,
                    'a_err': result.params['a'].stderr,
                    'b': result.params['b'].value,
                    'b_err': result.params['b'].stderr,
                    'c': result.params['c'].value,
                    'c_err': result.params['c'].stderr,
                    'd': result.params['d'].value,
                    'd_err': result.params['d'].stderr,
                    'f': result.params['f'].value,
                    'f_err': result.params['f'].stderr,
                    'g': result.params['g'].value,
                    'g_err': result.params['g'].stderr,
                    'r2': result.rsquared if hasattr(result, 'rsquared') else compute_r2(result),  # fallback if not stored
                    'chisq': result.chisqr
                }
                insert_success = insert_melt_fit(db_path, fit_data)
                count += insert_success

            except Exception as e:
                console.print(f"[red]Error fitting melt curve for nt_id={nt_id} (site={site}): {e}[/red]")
                continue
            console.print(f"[green]Fitting result for nt_id={nt_id} (site={site}):[/green]")
            console.print(f" {result.fit_report()}")

        # Optionally store result to DB or file
        console.print(f"[green]Successfully inserted {count} melt fit records into the database.[/green]")
    except Exception as e:
        console.print(f"[red]Fit failed:[/red] {e}")