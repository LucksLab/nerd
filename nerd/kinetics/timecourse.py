#/kinetics/timecourse.py

import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from lmfit.models import ExponentialModel, ConstantModel
from nerd.db.io import connect_db, init_db, check_db, insert_tc_fit, insert_fitted_probing_kinetic_rate
from nerd.utils.fit_models import fit_timecourse, global_fit_timecourse
from nerd.db.fetch import fetch_all_nt, fetch_timecourse_data, fetch_reaction_temp, fetch_reaction_pH, fetch_all_rg_ids, fetch_rxn_ids, fetch_fmod_vals, fetch_filtered_freefit_sites, fetch_dataset_freefit_params, fetch_dataset_fmods, fetch_global_kdeg_val
from nerd.kinetics.degradation import calc_kdeg
from nerd.db.update import update_sample_to_drop, update_sample_fmod_val
import sqlite3
from nerd.utils.plotting import plot_aggregate_timecourse


console = Console()

def plot_all_aggregated_timecourses(db_path):
    # get all rg_ids and run plot_aggregate_timecourse for each
    all_rg_ids = fetch_all_rg_ids(db_path)
    for rg_id in all_rg_ids:
        console.print(f"[blue]Plotting aggregated fits for rg_id {rg_id}...[/blue]")
        try:
            plot_aggregate_timecourse(rg_id, db_path)
            console.print(f"[green]✓ Aggregated fit plot for rg_id {rg_id} completed successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error plotting aggregated fit for rg_id {rg_id}:[/red] {e}")


from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

def single_fit(rg_id, db_path, constrained_kdeg=False):
    """
    Free fit to a rg_id, loops over each site
    """

    REQUIRED_COLUMNS = [
        "rg_id", "nt_id", "kobs_val", "kobs_err",
        "kdeg_val", "kdeg_err", "fmod0", "fmod0_err",
        "r2", "chisq", "time_min", "time_max"
    ]

    temp = fetch_reaction_temp(db_path, rg_id)
    pH = fetch_reaction_pH(db_path, rg_id)

    console.print()
    if constrained_kdeg:
        kdeg0 = fetch_global_kdeg_val(db_path, rg_id, "global_deg", "dms")
        console.print(f"[blue]Using constrained kdeg value from probing_kinetic_rates table: {kdeg0:.4f} s^-1[/blue]")
        fit_mode = "Constrained kdeg"
        insert_to_table = "constrained_tc_fits"
    else:
        kdeg0 = calc_kdeg(temp, pH, db_path)
        console.print(f"[blue]Using calculated kdeg value from NMR: {kdeg0:.4f} s^-1[/blue]")
        fit_mode = "Free"
        insert_to_table = "free_tc_fits"
    console.print()

    nt_ids = fetch_all_nt(db_path, rg_id)
    count = 0

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Fitting {len(nt_ids)} nts in rg_id {rg_id}...", total=len(nt_ids))

        for nt_id in nt_ids:
            try:
                time_data, fmod_data = fetch_timecourse_data(db_path, nt_id, rg_id)

                if time_data is None or fmod_data is None:
                    console.print(f"  [yellow]Warning:[/yellow] No timecourse data for rg_id {rg_id}, nt_id {nt_id}. Skipping.")
                    progress.advance(task)
                    continue
                elif len(time_data) < 3:
                    console.print(f"  [yellow]Warning:[/yellow] Not enough data for rg_id {rg_id}, nt_id {nt_id}. Skipping.")
                    progress.advance(task)
                    continue

                result, outlier = fit_timecourse(time_data, fmod_data, kdeg0, constrained_kdeg)

                fit_result = {
                    "rg_id": rg_id,
                    "nt_id": nt_id,
                    "kobs_val": result.params["log_kappa"].value,
                    "kobs_err": result.params["log_kappa"].stderr or -1,
                    "kdeg_val": result.params["log_kdeg"].value,
                    "kdeg_err": result.params["log_kdeg"].stderr or -1,
                    "fmod0": result.params["log_fmod_0"].value,
                    "fmod0_err": result.params["log_fmod_0"].stderr or -1,
                    "r2": result.rsquared,
                    "chisq": result.chisqr,
                    "time_min": min(time_data),
                    "time_max": max(time_data)
                }

                def fmt(val): return f"{val:.4f}" if val is not None else "None"

                console.print(
                    f"  [green]✓ Fit[/green] nt_id {nt_id}: "
                    f"kobs = {fmt(fit_result['kobs_val'])} ± {fmt(fit_result['kobs_err'])}, "
                    f"kdeg = {fmt(fit_result['kdeg_val'])} ± {fmt(fit_result['kdeg_err'])}, "
                    f"fmod0 = {fmt(fit_result['fmod0'])} ± {fmt(fit_result['fmod0_err'])}, "
                    f"r² = {fmt(fit_result['r2'])}"
                )

                conn = connect_db(db_path)
                if not check_db(conn, insert_to_table, REQUIRED_COLUMNS):
                    console.print(f"  [red]Error:[/red] Database missing required columns.")
                    conn.close()
                    return
                else:
                    console.print(f"  [green]✓ DB ready for {fit_mode} import[/green]")

                insert_success = insert_tc_fit(conn, fit_result, insert_to_table)
                count += insert_success
                conn.close()

            except Exception as e:
                console.print(f"  [yellow]Warning:[/yellow] {fit_mode} fit failed for nt_id {nt_id}: {e}")
            finally:
                progress.advance(task)

    console.rule(f"[bold green]✓ {fit_mode} timecourse fits imported[/bold green]")
    console.print(f"[bold green]{count} fits inserted into the database.[/bold green]")
    console.print()

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
        if rg_id == 2:
            console.print(f"[yellow]Warning:[/yellow] Skipping rg_id {rg_id} as it is not applicable for this operation")
            continue
        qc_comment = annotation['qc_comment']

        # Step 1: Get relevant reactions in the reaction group
        results = fetch_rxn_ids(db_path, rg_id)

        # Multiply treated by reaction time and sort by time
        time_sids = [(rt * treated, s_id) for rt, treated, s_id in results]
        time_sids.sort(key=lambda x: x[0])
        unique_times = sorted(set(t for t, _ in time_sids))
        # Map each unique time to a tp label
        tp_map = {t: f'tp{i}' for i, t in enumerate(unique_times)}
        from collections import defaultdict
        tp_sids = defaultdict(list)
        for t, s_id in time_sids:
            if t in tp_map:
                tp_sids[tp_map[t]].append(s_id)
        tp_sids = dict(tp_sids)  # convert back to regular dict

        if qc_comment == 'bad_rg':
            s_ids = [s_id for _, s_id in time_sids]
        elif qc_comment.startswith('drop_tp'):
            target_tp = qc_comment.split('_')[1]
            s_ids = tp_sids.get(target_tp, [])
        elif qc_comment == 'average_tp2':
            continue
            # tp2 = tp_sids.get('tp2', [])[0]
            # tp2b = tp_sids.get('tp2', [])[1]
            # if tp2 is None or tp2b is None:
            #     console.print(f"[yellow]Warning:[/yellow] tp2 or tp2b missing for rg_id {rg_id}")
            #     continue
            # print(f"tp2: {tp2}, tp2b: {tp2b}")
            # result2 = fetch_fmod_vals(db_path, tp2)
            # result3 = fetch_fmod_vals(db_path, tp2b)

            # if result2 is None or result3 is None:
            #     console.print(f"[yellow]Warning:[/yellow] Could not retrieve fmod_vals for tp2 or tp2b (rg_id {rg_id})")
            #     continue

            # print(result2, result3)


            # val2, rxn_time2 = result2
            # val3, rxn_time3 = result3


            # assert rxn_time2 == rxn_time3, "rxn_times for tp2 and tp2b must match"

            # avg_val = (val2 + val3) / 2
            # drop_success = update_sample_to_drop(db_path, tp2b, rg_id, qc_comment)
            # update_success = update_sample_fmod_val(db_path, tp2, avg_val)

            # if drop_success:
            #     console.print(f"[INFO] Dropped tp2b (s_id {tp2b}) for rg_id {rg_id}")
            # else:
            #     console.print(f"[yellow]Warning:[/yellow] Failed to drop tp2b (s_id {tp2b}) for rg_id {rg_id}")

            # if update_success:
            #     console.print(f"[green]Success:[/green] Updated tp2 (s_id {tp2}) with avg fmod_val {avg_val:.4f}")
            # else:
            #     console.print(f"[yellow]Warning:[/yellow] Failed to update tp2 (s_id {tp2}) for rg_id {rg_id}")
        else:
            console.print(f"[yellow]Warning:[/yellow] Unknown qc_comment '{qc_comment}' for rg_id {rg_id}")
            continue

        # Mark each s_id in sequencing_samples as to_drop = 1
        for s_id in s_ids:
            if s_id is not None:
                update_success = update_sample_to_drop(db_path, s_id, rg_id, qc_comment)
                if update_success:
                    console.print(f"[green]Success:[/green] Marked s_id {s_id} (rg_id {rg_id}, {qc_comment}) as to_drop")
                else:
                    console.print(f"[yellow]Warning:[/yellow] Failed to mark s_id {s_id} (rg_id {rg_id}, {qc_comment}) as to_drop")

def global_fit(rg_id, db_path, mode):
    """
    Placeholder for global fitting logic.
    This function is not implemented yet.
    require free fit to be done first
    """

    # need logic to select what to global fit
    # Mode 1: all nucleotides with R2 > threshold
    # Mode 2: A's and C's with R2 > threshold

    try:

        if mode == 'all':
            select_nts = fetch_filtered_freefit_sites(db_path, rg_id, bases = [], r2_thres = 0.8)
        elif mode == 'ac_only':
            select_nts = fetch_filtered_freefit_sites(db_path, rg_id, bases = ['A', 'C'], r2_thres = 0.8)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'all' or 'ac_only'.")
        
        if not select_nts:
            console.print(f"[yellow]Warning:[/yellow] No suitable nucleotides found for global fitting in rg_id {rg_id} with mode {mode}.")
            return None
        console.print(f"[blue]Selected nucleotides for global fitting in rg_id {rg_id} ({mode} mode): {select_nts}[/blue]")
        # Fetch initial guess parameters for global fit
        # kappa, kdeg, fmod0
        kappa_array, kdeg_array, fmod0_array = fetch_dataset_freefit_params(db_path, rg_id, select_nts) # for use initial guess
        temp = fetch_reaction_temp(db_path, rg_id)
        pH = fetch_reaction_pH(db_path, rg_id)

        # if want to use nmr,
        #kdeg0 = np.log(calc_kdeg(temp, pH, db_path))

        # change kdeg_array to be the same value for all sites
        #kdeg_array = [kdeg0] * len(kdeg_array)  # needs to be in log scale

        # time and fmod data for global fit
        time_data_array, fmod_data_array = fetch_dataset_fmods(db_path, rg_id, select_nts)

        # check if all lens are the same across all arrays (kappa, kdeg, fmod0, time, fmod)
        if not (len(kappa_array) == len(kdeg_array) == len(fmod0_array) == len(time_data_array) == len(fmod_data_array)):
            console.print("[red]Error:[/red] Length mismatch in arrays for global fit. Please check the data.")
            return None

        console.print(f"[blue]Global fitting with {len(kappa_array)} nucleotides selected...[/blue]")



        # perform global fit
        kdeg_val, kdeg_err, chiseq, r2 = global_fit_timecourse(time_data_array, fmod_data_array, kappa_array, kdeg_array, fmod0_array)

        console.print(f"[green]✓ Global fit for rg_id {rg_id} completed successfully[/green]: "
                      f"kdeg = {kdeg_val:.4f} ± {kdeg_err:.4f}, "
                      f"chisq = {chiseq:.4f}, r² = {r2:.4f}")
        
        # insert kdeg val to probing_kinetic_rates table
        fit_result = {
            "rg_id": rg_id,
            "model": "global_deg",
            "k_value": kdeg_val,
            "k_error": kdeg_err,
            "chisq": chiseq,
            "r2": r2,
            "species": "dms"
        }

        conn = connect_db(db_path)
        insert_success = insert_fitted_probing_kinetic_rate(conn, fit_result)
        conn.close()
        if insert_success:
            console.print(f"[green]✓ Global fit for rg_id {rg_id} inserted successfully[/green]")
        else:
            console.print(f"[red]Error:[/red] Failed to insert global fit for rg_id {rg_id}")
            
    except Exception as e:
        console.print(f"[yellow]Warning: Error during global fit or insert for rg_id {rg_id}:[/yellow] {e}")
        return None

def run(db_path):
    """
    Main CLI entrypoint: fits free timecourse data and stores result to DB.
    """
    console.print("[blue]Running free timecourse fits...[/blue]")

    all_rg_ids = fetch_all_rg_ids(db_path)

    for rg_id in all_rg_ids:
        # Step 0: Mark samples to drop based on QC annotations
        try:
            mark_samples_to_drop(qc_csv_path='test_data/probing_data/rg_qc_annotations.csv', db_path=db_path)
        except Exception as e:
            console.print(f"[red]Error marking samples to drop for rg_id {rg_id}:[/red] {e}")
            continue

        console.print(f"[blue]Processing reaction group ID:[/blue] {rg_id}")
        
        # Step 1: free fit
        try:
            single_fit(rg_id, db_path)
        except Exception as e:
            console.print(f"[red]Error processing rg_id {rg_id}:[/red] {e}")
        
        # Step 2: global deg fit
        try:
            global_fit(rg_id, db_path, mode='all')
        except Exception as e:
            console.print(f"[red]Error performing global fit for rg_id {rg_id}:[/red] {e}")

        # Step 3: refit with constrained kdeg
        try:
            single_fit(rg_id, db_path, constrained_kdeg=True)
        except Exception as e:
            console.print(f"[red]Error refitting with constrained kdeg for rg_id {rg_id}:[/red] {e}")

    console.print("[green]✓ Free timecourse fits completed successfully[/green]")