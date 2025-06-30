import matplotlib.pyplot as plt
from rich.table import Table
from rich.console import Console
import numpy as np
from nerd.db.fetch import fetch_timecourse_records
import pandas as pd
import sqlite3

console = Console()


# === 1. Plot fit overlay ===
def plot_fit_with_data(x, y, result, title="Fit", xlabel="x", ylabel="y", save_path=None):
    """
    Plot raw data and lmfit model fit.
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Data", color="black")
    ax.plot(x, result.best_fit, label="Fit", color="blue")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


# === 2. Arrhenius plot: ln(k) vs 1/T ===
def plot_arrhenius(x, y, result, save_path=None):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="black", label="Data")
    ax.plot(x, result.best_fit, color="red", label="Linear fit")

    ax.set_xlabel("1 / Temperature (1/K)")
    ax.set_ylabel("ln(k)")
    ax.set_title("Arrhenius Plot")
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


# === 3. Melt curve: normalized kobs vs T ===
def plot_melt_curve(T, k_norm, result=None, save_path=None):
    fig, ax = plt.subplots()
    ax.scatter(T, k_norm, color="black", label="Data")

    if result:
        T_fit = np.linspace(min(T), max(T), 100)
        k_fit = result.eval(T=T_fit)
        ax.plot(T_fit, k_fit, color="blue", label="Fit")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Normalized k_obs")
    ax.set_title("Two-State Melting Fit")
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


# === 4. Rich table display for parameters ===
def display_fit_params(result, title="Fit Results"):
    table = Table(title=title)
    table.add_column("Parameter")
    table.add_column("Value", justify="right")
    table.add_column("± Error", justify="right")

    for name, param in result.params.items():
        table.add_row(name, f"{param.value:.4g}", f"{param.stderr:.2g}" if param.stderr else "—")

    console.print(table)


def plot_aggregate_timecourse(rg_id, db_path):
    columns = ['fmod_val', 'reaction_time', 'site', 'base', 'treated', 'read_depth', 'RT', 'valtype',
               'temperature', 'disp_name', 'rg_id', 'buffer_id', 'id']
    data = fetch_timecourse_records(db_path, rg_id)
    rg_df = pd.DataFrame(data, columns=columns)
    # Filter RT == 'MRT' and valtype == 'modrate'
    rg_df = rg_df[(rg_df['RT'] == 'MRT') & (rg_df['valtype'] == 'modrate')]

    # if "hiv" in disp_name and temperature = 25, drop treated == 0
    rg_df = rg_df[~((rg_df['disp_name'].str.contains('hiv')) & (rg_df['temperature'] == 25) & (rg_df['treated'] == 0))]

    rg_df['reaction_time'] = rg_df['reaction_time'] * rg_df['treated']
    rg_df = rg_df.sort_values(['site', 'reaction_time'])

    # print disp_name temperature, buffer_id
    construct = rg_df['disp_name'].unique()
    temp = rg_df['temperature'].unique()
    buffer = rg_df['buffer_id'].unique()

    rg_df_grouped = rg_df.groupby(['reaction_time', 'id']).sum()
    rg_df_grouped = rg_df_grouped.reset_index()
    rg_df_grouped['reaction_time'] = rg_df_grouped['reaction_time'].astype(int)
    rg_df_grouped = rg_df_grouped.sort_values(['reaction_time'])
    # print(rg_df_grouped)

    temp_mode = rg_df_grouped['temperature'].mode()
    print(temp_mode.values[0])
    rg_df_grouped['adjust_factor'] =  temp_mode.values[0] / rg_df_grouped['temperature']
    rg_df_grouped['adj_fmod_val'] = rg_df_grouped['fmod_val'] * rg_df_grouped['adjust_factor']

    
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.scatter(rg_df_grouped['reaction_time'], rg_df_grouped['adj_fmod_val'])
    # try:
    #     plt.scatter(rg_df_filtered['reaction_time'], rg_df_filtered['adj_fmod_val'], color='red', marker='x')
    # except:
    #     pass
    plt.xlabel('Reaction Time (s)')
    plt.ylabel('Fraction Modified')
    plt.title(f'rg {rg_id} - {construct[0]} - temp {temp[0]} C - buff{buffer[0]}')
    # create directory if it does not exist f'test_output/plots/aggregate_timecourses'
    import os
    os.makedirs('test_output/plots/aggregate_timecourses', exist_ok=True)
    
    plt.savefig(f'test_output/plots/aggregate_timecourses/rg_{rg_id}_agg_timecourse.png', bbox_inches='tight')
    return