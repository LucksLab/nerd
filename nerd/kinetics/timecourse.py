#/kinetics/timecourse.py

import pandas as pd
import numpy as np
from rich.console import Console
from lmfit.models import ExponentialModel, ConstantModel
from nerd.db.io import connect_db, init_db, check_db, insert_tc_fit
from nerd.utils.fit_models import fit_timecourse 
from nerd.db.fetch import fetch_all_nt, fetch_timecourse_data, fetch_reaction_temp, fetch_reaction_pH, fetch_all_rg_ids, fetch_rxn_ids, fetch_fmod_val
from nerd.kinetics.degradation import calc_kdeg
from nerd.db.update import update_sample_to_drop, update_sample_fmod_val
import sqlite3

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

def mark_samples_to_drop(qc_csv_path, db_path):
    """
    Marks sequencing samples as 'to_drop' based on qc annotations from a CSV.
    The CSV should have columns: rg_id, qc_comment
    Accepted qc_comment values: drop_tp0, drop_tp1, drop_tp2, ..., bad_rg
    """
    import csv

    # Load QC annotations
    qc_annotations = []
    with open(qc_csv_path, 'r', encoding = 'utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qc_annotations.append({'rg_id': int(row['rg_id']), 'qc_comment': row['qc_comment']})

    for annotation in qc_annotations:
        rg_id = annotation['rg_id']
        qc_comment = annotation['qc_comment']

        # Step 1: Get relevant reactions in the reaction group
        results = fetch_rxn_ids(db_path, rg_id)

        # Multiply treated by reaction time and sort by time
        time_sids = [(rt * treated, s_id) for rt, treated, s_id in results]
        time_sids.sort(key=lambda x: x[0])
        unique_times = sorted(set(t for t, _ in time_sids))
        print(unique_times)
        # Map each unique time to a tp label
        tp_map = {t: f'tp{i}' for i, t in enumerate(unique_times)}
        tp_sids = {tp_map[t]: s_id for t, s_id in time_sids if t in tp_map}
        print(tp_map)
        print(tp_sids)
        if qc_comment == 'bad_rg':
            s_ids = [s_id for _, s_id in time_sids]
        elif qc_comment.startswith('drop_tp'):
            target_tp = qc_comment.split('_')[1]
            s_ids = [tp_sids.get(target_tp)] if tp_sids.get(target_tp) else []
        elif qc_comment == 'average_tp2':
            tp2 = tp_sids.get('tp2')
            tp3 = tp_sids.get('tp3')
            if tp2 is None or tp3 is None:
                console.print(f"[yellow]Warning:[/yellow] tp2 or tp3 missing for rg_id {rg_id}")
                continue

            result2 = fetch_fmod_val(db_path, tp2)
            result3 = fetch_fmod_val(db_path, tp3)

            if result2 is None or result3 is None:
                print(f"[yellow]Warning:[/yellow] Could not retrieve fmod_vals for tp2 or tp3 (rg_id {rg_id})")
                continue

            val2, rxn_time2 = result2
            val3, rxn_time3 = result3

            print(result2, result3)

            assert rxn_time2 == rxn_time3, "rxn_times for tp2 and tp3 must match"

            avg_val = (val2 + val3) / 2
            drop_success = update_sample_to_drop(db_path, tp3, rg_id, qc_comment)
            update_success = update_sample_fmod_val(db_path, tp2, avg_val)

            if drop_success:
                print(f"[INFO] Dropped tp3 (s_id {tp3}) for rg_id {rg_id}")
            else:
                print(f"[yellow]Warning:[/yellow] Failed to drop tp3 (s_id {tp3}) for rg_id {rg_id}")

            if update_success:
                print(f"[green]Success:[/green] Updated tp2 (s_id {tp2}) with avg fmod_val {avg_val:.4f}")
            else:
                print(f"[yellow]Warning:[/yellow] Failed to update tp2 (s_id {tp2}) for rg_id {rg_id}")
        else:
            print(f"[yellow]Warning:[/yellow] Unknown qc_comment '{qc_comment}' for rg_id {rg_id}")
            continue

        # Mark each s_id in sequencing_samples as to_drop = 1
        for s_id in s_ids:
            if s_id is not None:
                update_success = update_sample_to_drop(db_path, s_id, rg_id, qc_comment)
                if update_success:
                    print(f"[green]Success:[/green] Marked s_id {s_id} (rg_id {rg_id}, {qc_comment}) as to_drop")
                else:
                    print(f"[yellow]Warning:[/yellow] Failed to mark s_id {s_id} (rg_id {rg_id}, {qc_comment}) as to_drop")


def run(db_path):
    """
    Main CLI entrypoint: fits free timecourse data and stores result to DB.
    """
    console.print("[blue]Running free timecourse fits...[/blue]")
    
    all_rg_ids = fetch_all_rg_ids(db_path)

    for rg_id in all_rg_ids:
        console.print(f"[blue]Processing reaction group ID:[/blue] {rg_id}")
        
        try:
            free_fit(rg_id, db_path)
        except Exception as e:
            console.print(f"[red]Error processing rg_id {rg_id}:[/red] {e}")

    console.print("[green]✓ Free timecourse fits completed successfully[/green]")