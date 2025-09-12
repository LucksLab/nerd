# nerd/kinetics/arrhenius.py

import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console

from nerd.db.io import connect_db, fetch_kinetic_rates, check_db, insert_arrhenius_fit, group_fam_tg_ids
from nerd.db.fetch import fetch_construct_disp_name, fetch_probing_kinetic_rates
from nerd.utils.fit_models import fit_linear
from nerd.db.fetch import fetch_buffer_id

console = Console()

REQUIRED_COLUMNS = [
    "reaction_type",  # Type of reaction (e.g. "deg")
    "data_source",    # Source of data (e.g. "nmr")
    "substrate",      # Substrate used in the reaction (e.g. "dms")
    "buffer_id",      # ID of the buffer used in the reaction
    "slope",          # Slope of the Arrhenius fit
    "slope_err",      # Standard error of the slope
    "intercept",      # Intercept of the Arrhenius fit
    "intercept_err",  # Standard error of the intercept
    "r2",            # R-squared value of the fit
    "model_file"      # Optional file path to model data (e.g. pickle, json)
]

def rows_to_dicts(rows, description):
    column_names = [desc[0] for desc in description]
    return [dict(zip(column_names, row)) for row in rows]


def run(reaction_type="deg", data_source="nmr", species = "dms", 
        buffer = "Schwalbe_bistris", db_path: str = ''):
    """
    CLI entrypoint for Arrhenius fitting.
    """
    
    count = 0

    if data_source == "nmr":
        conn = connect_db(db_path)  # row_factory set inside
        buffer_id = fetch_buffer_id(buffer, db_path)

        data, description = fetch_kinetic_rates(conn, buffer_id, species, reaction_type)
        conn.close()

        if not data:
            console.print(f"[red]No kinetic rates found for reaction type '{reaction_type}' with species '{species}' and buffer '{buffer}'[/red]")
            return

        data = rows_to_dicts(data, description)
        x_vals = np.array([1 / r["temperature"] for r in data])
        y_vals = np.array([np.log(r["k_value"]) for r in data])
        y_errs = np.array([r["k_error"] / r["k_value"] for r in data]) if "k_error" in description else None
        weights = 1 / y_errs**2 if y_errs is not None else None

    else:
        console.print(f"[red]Unsupported data source: {data_source}[/red]")
        return

    try:
        result = fit_linear(x_vals, y_vals, weights = weights)

        slope = result.params["slope"].value
        slope_err = result.params["slope"].stderr
        intercept = result.params["intercept"].value
        intercept_err = result.params["intercept"].stderr
        r2 = result.rsquared

        fit_result = {
            "reaction_type": reaction_type,
            "data_source": data_source,
            "substrate": species,
            "buffer_id": buffer_id,
            "rg_id": None, # only for probing data
            "slope": slope,
            "slope_err": slope_err,
            "intercept": intercept,
            "intercept_err": intercept_err,
            "r2": r2,
            "model_file": None  # Optional, can be set to a file path if needed
        }

        console.print(f"[green]✓ Arrhenius fit complete[/green]: slope = {slope:.4f} ± {slope_err:.4f}, intercept = {intercept:.4f} ± {intercept_err:.4f}, r² = {r2:.4f}")

        conn = connect_db(db_path)

        # Check if the database is initialized and has the required columns
        if not check_db(conn, "arrhenius_fits", REQUIRED_COLUMNS):
            console.print(f"[red]Error:[/red] Database initialization failed or missing required columns.")
            conn.close()
            return
        else:
            console.print(f"[green]✓ Database check passed and ready for Arrhenius fit import[/green]")


        insert_success = insert_arrhenius_fit(conn, fit_result)
        count += insert_success
        conn.close()

    except Exception as e:
        console.print(f"[red]Fit failed:[/red] {e}")

    console.print(f"[green]Imported Arrhenius fits for {count} samples[/green]")



def run_probing(reaction_type="global_deg", species = "dms", db_path: str = ''):
    """
    CLI entrypoint for Arrhenius fitting for probing data.

    reaction_type - "model" in probing_kinetic_rates table ("global_deg", "melted_agg_obs", "melted_ind_obs", "melted_agg_obs_by_fam")
    species - "dms", "rna_A", "rna_C", "rna_T", "rna_G" ...
    """

    data_source = "probing"
    
    model = reaction_type
    if 'by_fam' in model:
        # For family-based models, we need to fetch the family name
        model = "melted_agg_obs"

    res, description = fetch_probing_kinetic_rates(
        db_path=db_path,
        model=model,
        species=species
    )

    # get unique tg_ids from results
    unique_tg_ids = list(set(row[0] for row in res))

    data = rows_to_dicts(res, description)

    count = 0



    if 'ind' in reaction_type:
        # loop over unique tg_id and nt_id
        for tg_id in unique_tg_ids:
            filtered_results = [row for row in data if row['tg_id'] == tg_id]
            unique_nt_ids = list(set(row['nt_id'] for row in filtered_results))
            for nt_id in unique_nt_ids:
                nt_filtered_results = [row for row in filtered_results if row['nt_id'] == nt_id]

                x_vals = np.array([1 / row['temperature'] for row in nt_filtered_results])
                y_vals = np.array([row['k_value'] for row in nt_filtered_results])
                y_errs = np.array([row['k_error'] for row in nt_filtered_results])
                weights = 1 / y_errs**2 if y_errs is not None else None
                # Get buffer_id from first row
                buffer_id = nt_filtered_results[0]['buffer_id']
                construct_id = nt_filtered_results[0]['construct_id']
                construct_name = fetch_construct_disp_name(db_path, construct_id)
                nt_id = nt_filtered_results[0]['nt_id']

                console.print(f"[blue]Processing tg_id: {tg_id}, nt_id: {nt_id} for reaction type '{reaction_type}' and species '{species}'[/blue]")
                console.print(f"[blue]Buffer ID: {buffer_id}, Construct: {construct_name}[/blue]")
                try:
                    result = fit_linear(x_vals, y_vals, weights = weights)

                    slope = result.params["slope"].value
                    slope_err = result.params["slope"].stderr
                    intercept = result.params["intercept"].value
                    intercept_err = result.params["intercept"].stderr
                    r2 = result.rsquared

                    fit_result = {
                        "reaction_type": reaction_type,
                        "data_source": data_source,
                        "substrate": species,
                        "buffer_id": buffer_id,
                        "tg_id": tg_id,
                        "nt_id": nt_id,
                        "slope": slope,
                        "slope_err": slope_err,
                        "intercept": intercept,
                        "intercept_err": intercept_err,
                        "r2": r2,
                        "model_file": None  # Optional, can be set to a file path if needed
                    }

                    console.print(f"[green]✓ Arrhenius fit complete[/green]: slope = {slope:.4f} ± {slope_err:.4f}, intercept = {intercept:.4f} ± {intercept_err:.4f}, r² = {r2:.4f}")

                    conn = connect_db(db_path)

                    # Check if the database is initialized and has the required columns
                    if not check_db(conn, "arrhenius_fits", REQUIRED_COLUMNS):
                        console.print(f"[red]Error:[/red] Database initialization failed or missing required columns.")
                        conn.close()
                        return
                    else:
                        console.print(f"[green]✓ Database check passed and ready for Arrhenius fit import[/green]")

                    insert_success = insert_arrhenius_fit(conn, fit_result)
                    count += insert_success
                    conn.close()
                except Exception as e:
                    console.print(f"[red]Fit failed:[/red] {e}")

        console.print(f"[green]Imported Arrhenius fits for {count} samples[/green]")
    
    elif "agg_obs_by_fam" in reaction_type:
        # group unique tg_ids by construct_family

        fam_tg_groups = group_fam_tg_ids(db_path)

        for fam_tg_id, group in fam_tg_groups.items():
            family_name = group["family"]
            tg_ids = group["tg_ids"]
            
            console.print(f"[cyan]Group {fam_tg_id} — Family: {family_name}, tg_ids: {tg_ids}[/cyan]")

            filtered_results = [row for row in data if row['tg_id'] in tg_ids]

            if len(filtered_results) < 3:
                console.print(f"[red]Not enough data for family '{family_name}' with tg_ids {tg_ids}[/red]")
                continue
            x_vals = np.array([1 / row['temperature'] for row in filtered_results])
            y_vals = np.array([row['k_value'] for row in filtered_results])
            y_errs = np.array([row['k_error'] for row in filtered_results])
            weights = 1 / y_errs**2 if y_errs is not None else None
            # Get buffer_id from first row
            buffer_id = filtered_results[0]['buffer_id']
            construct_id = filtered_results[0]['construct_id']
            construct_name = fetch_construct_disp_name(db_path, construct_id)

            console.print(f"[blue]Processing reaction type '{reaction_type}' and species '{species}'[/blue]")
            console.print(f"[blue]Buffer ID: {buffer_id}, Construct: {construct_name}[/blue]")
            try:
                result = fit_linear(x_vals, y_vals, weights = weights)

                slope = result.params["slope"].value
                slope_err = result.params["slope"].stderr
                intercept = result.params["intercept"].value
                intercept_err = result.params["intercept"].stderr
                r2 = result.rsquared

                fit_result = {
                    "reaction_type": reaction_type,
                    "data_source": data_source,
                    "substrate": species,
                    "buffer_id": buffer_id,
                    "tg_id": tg_ids[0],  # Use the first tg_id as representative
                    "construct_family": family_name,
                    "slope": slope,
                    "slope_err": slope_err,
                    "intercept": intercept,
                    "intercept_err": intercept_err,
                    "r2": r2,
                    "model_file": None  # Optional, can be set to a file path if needed
                }

                console.print(f"[green]✓ Arrhenius fit complete[/green]: slope = {slope:.4f} ± {slope_err:.4f}, intercept = {intercept:.4f} ± {intercept_err:.4f}, r² = {r2:.4f}")

                conn = connect_db(db_path)

                # Check if the database is initialized and has the required columns
                if not check_db(conn, "arrhenius_fits", REQUIRED_COLUMNS):
                    console.print(f"[red]Error:[/red] Database initialization failed or missing required columns.")
                    conn.close()
                    return
                else:
                    console.print(f"[green]✓ Database check passed and ready for Arrhenius fit import[/green]")

                insert_success = insert_arrhenius_fit(conn, fit_result)
                count += insert_success
                conn.close()

            except Exception as e:
                console.print(f"[red]Fit failed:[/red] {e}")

        console.print(f"[green]Imported Arrhenius fits for {count} samples[/green]")


    elif 'agg' in reaction_type:
        for tg_id in unique_tg_ids:

            filtered_results = [row for row in data if row['tg_id'] == tg_id]
            # row['temperature'] - temperature
            # row['k_value'] - k_value
            # row['k_error'] - k_error

            x_vals = np.array([1 / row['temperature'] for row in filtered_results])
            y_vals = np.array([row['k_value'] for row in filtered_results])
            y_errs = np.array([row['k_error'] for row in filtered_results])
            weights = 1 / y_errs**2 if y_errs is not None else None

            # Get buffer_id from first row
            buffer_id = filtered_results[0]['buffer_id']
            construct_id = filtered_results[0]['construct_id']
            construct_name = fetch_construct_disp_name(db_path, construct_id)

            console.print(f"[blue]Processing tg_id: {tg_id} for reaction type '{reaction_type}' and species '{species}'[/blue]")
            console.print(f"[blue]Buffer ID: {buffer_id}, Construct: {construct_name}[/blue]")
            try:
                result = fit_linear(x_vals, y_vals, weights = weights)

                slope = result.params["slope"].value
                slope_err = result.params["slope"].stderr
                intercept = result.params["intercept"].value
                intercept_err = result.params["intercept"].stderr
                r2 = result.rsquared

                fit_result = {
                    "reaction_type": reaction_type,
                    "data_source": data_source,
                    "substrate": species,
                    "buffer_id": buffer_id,
                    "tg_id": tg_id,
                    "slope": slope,
                    "slope_err": slope_err,
                    "intercept": intercept,
                    "intercept_err": intercept_err,
                    "r2": r2,
                    "model_file": None  # Optional, can be set to a file path if needed
                }

                console.print(f"[green]✓ Arrhenius fit complete[/green]: slope = {slope:.4f} ± {slope_err:.4f}, intercept = {intercept:.4f} ± {intercept_err:.4f}, r² = {r2:.4f}")

                conn = connect_db(db_path)

                # Check if the database is initialized and has the required columns
                if not check_db(conn, "arrhenius_fits", REQUIRED_COLUMNS):
                    console.print(f"[red]Error:[/red] Database initialization failed or missing required columns.")
                    conn.close()
                    return
                else:
                    console.print(f"[green]✓ Database check passed and ready for Arrhenius fit import[/green]")

                insert_success = insert_arrhenius_fit(conn, fit_result)
                count += insert_success
                conn.close()

            except Exception as e:
                console.print(f"[red]Fit failed:[/red] {e}")

        console.print(f"[green]Imported Arrhenius fits for {count} samples[/green]")

