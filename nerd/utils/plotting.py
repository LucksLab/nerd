# nerd/utils/plotting.py

import matplotlib.pyplot as plt
from rich.table import Table
from rich.console import Console
import numpy as np

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


def plot_aggregate_timecourse(rg_id, rxns_to_remove):
    db_path = '/projects/b1044/Computational_Output/EKC/EKC.01_SHAPE_standardization/EKC.01.060.developing_DB_input/new.db'
    # Define the query to get fmod_val and reaction_time with a join between the necessary tables
    query = f"""
        SELECT fv.fmod_val, pr.reaction_time, n.site, n.base, pr.treated, fv.read_depth, pr.RT, fv.valtype, pr.temperature, c.disp_name, rg.rg_id, pr.buffer_id, sr.id
        FROM probing_reactions pr
        JOIN fmod_vals fv ON pr.id = fv.rxn_id
        JOIN nucleotides n ON fv.nt_id = n.id
        JOIN constructs c ON pr.construct_id = c.id
        JOIN sequencing_samples ss ON pr.s_id = ss.id
        JOIN reaction_groups rg ON rg.rxn_id = pr.id
        JOIN sequencing_runs sr on ss.seqrun_id = sr.id
        WHERE rg.rg_id = {rg_id}
        AND fv.fmod_val IS NOT NULL
    """

    # Import data into dataframe
    conn = sqlite3.connect(db_path)
    rg_df = pd.read_sql(query, conn)
    conn.close()
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

    # apply comment
    if rg_id not in rg_qc_comments:
        print(f'rg {rg_id} passed manual qc')
    else:
        print(f'rg {rg_id} failed manual qc')
        qc_comment = rg_qc_comments[rg_id].split(',')
        print(qc_comment)

        mask = len(rg_df_grouped) * [1]
        for comment in qc_comment:
            print(comment)
            if 'drop' in comment:
                which_tp = int(comment.split('drop_tp')[1])
                print(which_tp)
                mask[which_tp] = 0
            elif 'average_tp2' in comment:
                reaction_times_with_counts = rg_df_grouped['reaction_time'].value_counts()
                reaction_times_with_counts = reaction_times_with_counts[reaction_times_with_counts > 1]
                print(reaction_times_with_counts)
        print(mask)
        # apply mask
        rg_df_grouped['mask'] = mask

        # filter mask
        rg_df_filtered = rg_df_grouped[rg_df_grouped['mask'] == 0]
        rxns_to_remove[rg_id] = list(rg_df_filtered['reaction_time'])
        rg_df_grouped = rg_df_grouped[rg_df_grouped['mask'] == 1]
        # filtered data
    
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.scatter(rg_df_grouped['reaction_time'], rg_df_grouped['adj_fmod_val'])
    try:
        plt.scatter(rg_df_filtered['reaction_time'], rg_df_filtered['adj_fmod_val'], color='red', marker='x')
    except:
        pass
    plt.xlabel('Reaction Time (s)')
    plt.ylabel('Fraction Modified')
    plt.title(f'rg {rg_id} - {construct[0]} - temp {temp[0]} C - buff{buffer[0]}')
    plt.show()
    return rg_df, rg_df_grouped, rxns_to_remove