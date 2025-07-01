# /utils/param_aggregation.py


import pandas as pd
import numpy as np
from rich.console import Console

def compute_weighted_mean(kobs_vals: pd.Series, kobs_errs: pd.Series) -> float:
    """
    Compute the weighted mean of kobs_vals using inverse square of kobs_errs as weights.
    """
    weights = 1 / (kobs_errs ** 2)
    return (kobs_vals * weights).sum() / weights.sum()

def compute_weighted_stderr(kobs_errs: pd.Series) -> float:
    """
    Compute the propagated standard error of the weighted mean.
    """
    weights = 1 / (kobs_errs ** 2)
    return np.sqrt(1 / weights.sum())

def summarize_group(group: pd.DataFrame) -> pd.Series:
    """
    Compute summary statistics for a group of kobs values.

    Includes both unweighted (mean, SEM) and weighted estimates using inverse-variance weighting.
    Weighted values are appropriate because kobs_val entries come from independent fits
    with differing uncertainties (kobs_err). In this case, the weighted mean provides the 
    most precise estimate, and the propagated error reflects the combined confidence 
    based on individual fit uncertainties.

    Returns:
        pd.Series: Summary statistics including weighted mean, propagated stderr,
                unweighted mean, SEM, and count.
    """

    return pd.Series({
        'rg_id': group['rg_id'].iloc[0],
        'mean_val_weighted': compute_weighted_mean(group['kobs_val'], group['kobs_err']),
        'stderr_propagate': compute_weighted_stderr(group['kobs_err']),
        'mean_val': group['kobs_val'].mean(),
        'stderr_val': group['kobs_val'].sem(),
        'count': len(group)
    })

def aggregate_probing_data(records, description):
    """
    Aggregate probing data from multiple records into a DataFrame and compute summary statistics.

    Args:
        records (list): List of tuples containing probing data.
        description (list): List of column descriptions for the records.

    Returns:
        pd.DataFrame: Aggregated DataFrame with summary statistics.
    """
    
    # Convert records to DataFrame
    df = pd.DataFrame(records, columns=[desc for desc in description])

    to_group_by = ['base', 'temperature', 'RT', 'done_by', 
                    'buffer_id', 'construct_id']

    agg_df = df.groupby(to_group_by).apply(summarize_group).reset_index()
    agg_df['species'] = 'rna_' + agg_df['base']

    return agg_df

def insert_aggregated_probing_rates(agg_df: pd.DataFrame, db_path: str, console: Console):
    """
    Insert aggregated probing rates into the database.

    Args:
        agg_df (pd.DataFrame): DataFrame containing aggregated probing rates.
        db_path (str): Path to the database file.
    """
    
    from nerd.db.io import connect_db, insert_fitted_probing_kinetic_rate

    # Import probing kinetic rate
    conn = connect_db(db_path)

    for _, row in agg_df.iterrows():
        fit_result = {
            "rg_id": int(row['rg_id']),
            "model": "melted_agg_obs",
            "k_value": row['mean_val_weighted'],
            "k_error": row['stderr_propagate'],
            "chisq": -1,
            "r2": -1,
            "species": row['species']
        }

        insert_success = insert_fitted_probing_kinetic_rate(conn, fit_result)
        
        if insert_success:
            console.print(f"[green]✓ Inserted agg_add rate for rg_id {fit_result['rg_id']} ({fit_result['species']})[/green]")
        else:
            console.print(f"[red]✗ Failed to insert agg_add rate for rg_id {fit_result['rg_id']} ({fit_result['species']})[/red]")

    conn.close()

def process_aggregation(records, description, db_path: str, console: Console):
    """
    Process aggregation of probing data and insert into the database.

    Args:
        records (list): List of tuples containing probing data.
        description (list): List of column descriptions for the records.
        db_path (str): Path to the database file.
        console (Console): Rich console for output.
    """
    
    agg_df = aggregate_probing_data(records, description)
    insert_aggregated_probing_rates(agg_df, db_path, console)