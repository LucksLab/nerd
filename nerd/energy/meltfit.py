# nerd/energy/meltfit.py

import numpy as np
import pandas as pd
from nerd.utils.logging import get_logger
from nerd.db.fetch import fetch_all_tempgrad_groups, fetch_all_kobs
from nerd.db.io import insert_melt_fit
from nerd.utils.fit_models import fit_meltcurve
log = get_logger(__name__)
from collections import defaultdict

def run(db_path: str):
    """
    CLI entrypoint: fit 2-state melt curve from CSV of k_obs vs. temperature.
    """
    try:
        tg_ids = fetch_all_tempgrad_groups(db_path = db_path)
        if not tg_ids:
            log.error("No temperature gradient groups found in the database.")
            return
        log.info("Found %d temperature gradient groups to process.", len(tg_ids))

        # Step 1: Collect all kobs data across temperature gradient groups
        all_kobs_data = []

        for tg_entry in tg_ids:
            tg_id = tg_entry[0]
            temp = tg_entry[1]
            buffer_id = tg_entry[2]
            construct_id = tg_entry[3]

            log.info("Processing temperature gradient group ID: %s", tg_id)
            log.debug("Temperature: %s, Buffer ID: %s, Construct ID: %s", temp, buffer_id, construct_id)

            # Fetch k_obs data for this temperature group
            kobs_data = fetch_all_kobs(db_path, tg_id)
            if not kobs_data:
                log.error("No k_obs data found for TG ID: %s", tg_id)
                continue

            all_kobs_data.extend(kobs_data)

        log.info("Fetched total %d k_obs records across all TGs", len(all_kobs_data))

        # Step 2: Group by unique nucleotide (e.g., by nt_id or (nt_id, site))
        grouped_by_nt = defaultdict(list)
        for row in all_kobs_data:
            # Unpack: (kobs_val, kobs_err, chisq, r2, nt_id, base, site, temperature)
            kobs_val, kobs_err, chisq, r2, nt_id, base, site, temperature = row
            grouped_by_nt[(nt_id, site)].append((kobs_val, kobs_err, temperature))


        count = 0
        # Step 3: Fit kobs vs 1/T for each nucleotide
        for (nt_id, site), records in grouped_by_nt.items():
            log.info("Processing nt_id=%s (site=%s) with %d records...", nt_id, site, len(records))
            if len(records) < 3:
                log.warning("Skipping nt_id=%s (site=%s): only %d temperatures", nt_id, site, len(records))
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
                    # Compute R² if not provided on result
                    'r2': getattr(result, 'rsquared', None) if hasattr(result, 'rsquared') else None,
                    'chisq': result.chisqr
                }
                # If rsquared missing, compute from best fit vs observed
                if fit_data['r2'] is None:
                    y_pred = result.best_fit
                    ss_res = np.sum((kobs_vals - y_pred) ** 2)
                    ss_tot = np.sum((kobs_vals - np.mean(kobs_vals)) ** 2)
                    fit_data['r2'] = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
                insert_success = insert_melt_fit(db_path, fit_data)
                count += insert_success
            except Exception as e:
                log.exception("Error fitting melt curve for nt_id=%s (site=%s): %s", nt_id, site, e)
                continue
            log.debug("Fitting report for nt_id=%s (site=%s):\n%s", nt_id, site, result.fit_report())

        # Optionally store result to DB or file
        log.info("Successfully inserted %d melt fit records into the database.", count)
    except Exception as e:
        log.exception("Fit failed: %s", e)