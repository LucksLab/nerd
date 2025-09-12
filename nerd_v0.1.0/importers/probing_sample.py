# nerd/importers/sample.py

import csv
from pathlib import Path
from rich.console import Console
from nerd.db.io import connect_db, check_db, insert_buffer, insert_construct, insert_nt_info, insert_seq_run
from nerd.db.fetch import fetch_distinct_tempgrad_group
from nerd.db.update import assign_tempgrad_groups
import sqlite3
import pandas as pd
import numpy as np
from rich.table import Table
from rich.prompt import Prompt


console = Console()

# Define required columns (match schema)
BUFFER_REQUIRED_COLUMNS = [
    "name", "pH", "composition", "disp_name"
]

CONSTRUCT_REQUIRED_COLUMNS = [
    "family", "name", "version", "sequence", "disp_name"
]

SEQRUN_REQUIRED_COLUMNS = [
    "run_name", "date", "sequencer", "run_manager"
]

# need to import buffer, construct (and nt_info), sequencing runs, then probing samples
def import_buffer(csv_path: str, db_path: str = ''):
    """
    Import buffer data into the buffers table.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        console.print(f"[red]Error:[/red] Buffer CSV file not found: {csv_file}")
        return

    conn = connect_db(db_path)
    check_db(conn, "buffers", BUFFER_REQUIRED_COLUMNS)

    with open(csv_file, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing = [col for col in BUFFER_REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            console.print(f"[red]Error:[/red] Missing required columns: {missing}")
            return

        count = 0
        for row in reader:
            try:
                insert_success = insert_buffer(conn, row)
                count += insert_success
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Skipped row due to error: {e}")

        console.print(f"[green]✓ Imported {count} buffers[/green]")


def import_construct(csv_path: str, db_path: str = None):
    """
    Import construct data into the constructs table.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        console.print(f"[red]Error:[/red] Construct CSV file not found: {csv_file}")
        return

    conn = connect_db(db_path)
    check_db(conn, "constructs", CONSTRUCT_REQUIRED_COLUMNS)

    with open(csv_file, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing = [col for col in CONSTRUCT_REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            console.print(f"[red]Error:[/red] Missing required columns: {missing}")
            return

        count = 0
        for row in reader:
            try:
                insert_success, construct_id = insert_construct(conn, row)
                count += insert_success

                if insert_success:
                    # check if nt_info_csv is provided
                    nt_csv = row.get('nt_info_csv')
                    if nt_csv not in (None, ''):
                        nt_count = 0
                        # Insert nt_info if provided
                        with open(f'test_data/{nt_csv}', newline='', encoding='utf-8-sig') as nt_file:
                            nt_reader = csv.DictReader(nt_file)
                            for nt_row in nt_reader:
                                # Add construct_id to the nt_row
                                nt_row['construct_id'] = construct_id
                                try:
                                    insert_success_nt = insert_nt_info(conn, nt_row)
                                    nt_count += insert_success_nt
                                except Exception as e:
                                    console.print(f"[yellow]Warning:[/yellow] Skipped NT info row due to error: {e}")
                        console.print(f"[green]✓ Imported {nt_count} nt info records for construct {row['disp_name']}[/green]")

            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Skipped row due to error: {e}")

        console.print(f"[green]✓ Imported {count} constructs[/green]")


def import_seqrun(csv_path: str, db_path: str = None):
    conn = connect_db(db_path)
    check_db(conn, "sequencing_runs", SEQRUN_REQUIRED_COLUMNS)

    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing = [col for col in SEQRUN_REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            console.print(f"[red]Error:[/red] Missing required columns: {missing}")
            return

        count = 0
        for row in reader:
            try:
                insert_success = insert_seq_run(conn, row)
                count += insert_success
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Skipped row due to error: {e}")

        console.print(f"[green]✓ Imported {count} sequencing runs[/green]")


# Connect to database
def connect_db(db_path):
    return sqlite3.connect(db_path)

# Get max id helper
def get_max_id(cursor, table, id_col):
    cursor.execute(f"SELECT MAX({id_col}) FROM {table}")
    max_id = cursor.fetchone()[0]
    return max_id + 1 if max_id else 1

# Handle buffer selection/creation
def handle_buffer(b, console, cursor):
    cursor.execute("SELECT id, name FROM buffers")
    buffers = cursor.fetchall()

    if buffers:
        table = Table(title="Available Buffers", show_lines=True)
        table.add_column("ID", style="cyan", justify="center")
        table.add_column("Name", style="green")

        for existing_b in buffers:
            table.add_row(str(existing_b[0]), existing_b[1])

        console.print(table)

    buffer_id = Prompt.ask(
        f"[bold yellow]For buffer '{b}', enter Buffer ID or 'new' to create[/bold yellow]", 
        choices=[str(buf[0]) for buf in buffers] + ["new"]
    )

    if buffer_id.lower() == 'new':
        buffer_name = Prompt.ask("[bold cyan]Enter buffer name[/bold cyan]")

        # Validate numeric input for pH
        pH = Prompt.ask("[bold cyan]Enter buffer pH[/bold cyan] (numeric only)", default="8.0")

        composition = Prompt.ask("[bold cyan]Enter buffer composition[/bold cyan]")
        disp_name = Prompt.ask("[bold cyan]Enter buffer display name[/bold cyan]")

        cursor.execute(
            "INSERT INTO buffers (name, pH, composition, disp_name) VALUES (?, ?, ?, ?)", 
            (buffer_name, pH, composition, disp_name)
        )

        console.print(f"[green]Buffer '{buffer_name}' created![/green]")
        return cursor.lastrowid

    return int(buffer_id)

# Handle construct selection/creation
def handle_construct(c, console, cursor):
    cursor.execute("SELECT id, family, name, disp_name FROM constructs")
    constructs = cursor.fetchall()

    if constructs:
        table = Table(title="Available Constructs", show_lines=True)
        table.add_column("ID", style="cyan", justify="center")
        table.add_column("Family", style="green")
        table.add_column("Name", style="green")
        table.add_column("Display name", style="green")

        for existing_c in constructs:
            table.add_row(str(existing_c[0]), existing_c[1], existing_c[2], existing_c[3])

        console.print(table)

    construct_id = Prompt.ask(
        f"[bold yellow]For construct '{c}', enter Construct ID or 'new' to create[/bold yellow]", 
        choices=[str(construct[0]) for construct in constructs] + ["new"]
    )

    if construct_id.lower() == 'new':
        family = Prompt.ask("[bold cyan]Enter construct family[/bold cyan]")
        name = Prompt.ask("[bold cyan]Enter construct name[/bold cyan]")
        version = Prompt.ask("[bold cyan]Enter construct version (numeric only)[/bold cyan]", default="1")
        sequence = Prompt.ask("[bold cyan]Enter construct sequence[/bold cyan]")
        disp_name = Prompt.ask("[bold cyan]Enter construct display name[/bold cyan]")

        cursor.execute(
            "INSERT INTO constructs (family, name, version, sequence, disp_name) VALUES (?, ?, ?, ?, ?)", 
            (family, name, version, sequence, disp_name)
        )
        current_construct_id = cursor.lastrowid
        console.print(f"[green]Construct '{name}' created![/green]")

        # Prompt generate nucleotides via prompt or csv file
        nucleotide_fill_mode = Prompt.ask("[bold cyan]Generate nucleotides?[/bold cyan] (yes/no)", choices=["yes", "no"])

        if nucleotide_fill_mode == "yes":
         # Generate nucleotides for this construct
            generate_nucleotides(sequence, current_construct_id, console, cursor)
        else:
            # Prompt for nucleotide file
            nucleotide_file = Prompt.ask("[bold cyan]Enter nucleotide file path[/bold cyan]")
            nt_df = pd.read_csv(nucleotide_file)
            for i, row in nt_df.iterrows():
                cursor.execute("INSERT INTO nucleotides (site, base, base_region, construct_id) VALUES (?, ?, ?, ?)", (row['site'], row['base'], row['base_region'], current_construct_id))

        # return lastrowid in constructs table
        return current_construct_id
    
    return int(construct_id)


# Handle sequencing run selection/creatio
def handle_seqrun(sr, console, cursor):
    cursor.execute("SELECT id, run_name FROM sequencing_runs")
    seq_runs = cursor.fetchall()

    if seq_runs:
        table = Table(title="Available Sequencing Runs", show_lines=True)
        table.add_column("ID", style="cyan", justify="center")
        table.add_column("Run Name", style="green")

        for existing_sr in seq_runs:
            table.add_row(str(existing_sr[0]), existing_sr[1])

        console.print(table)

    seq_run_id = Prompt.ask(
        f"[bold yellow]For sequencing run '{sr}', enter Sequencing Run ID or 'new' to create[/bold yellow]", 
        choices=[str(seq_run[0]) for seq_run in seq_runs] + ["new"]
    )

    if seq_run_id.lower() == 'new':
        run_name = Prompt.ask("[bold cyan]Enter run name[/bold cyan]")
        date = Prompt.ask("[bold cyan]Enter run date (YYMMDD)[/bold cyan]")
        sequencer = Prompt.ask("[bold cyan]Enter sequencer name[/bold cyan]")
        run_manager = Prompt.ask("[bold cyan]Enter run manager name[/bold cyan]")

        cursor.execute(
            "INSERT INTO sequencing_runs (run_name, date, sequencer, run_manager) VALUES (?, ?, ?, ?)", 
            (run_name, date, sequencer, run_manager)
        )

        console.print(f"[green]Sequencing run '{run_name}' created![/green]")
        return cursor.lastrowid

    return int(seq_run_id)

def generate_nucleotides(seq, construct_id, console, cursor):
    ### Create nucleotides

    # create nucleotides tables
    # columns: nt_id, site, base, base_region, construct_id

    def convert_seq(seq):
        # convert row['sequence'] to 0's and 1's depending on case
        return [1 if x.isupper() else 0 for x in seq]

    def collapse_seq(seq):
        # collapse into contiguous regions of 0's and 1's
        # ex. 00001111000011110000 -> (0, 4), (1, 4), (0, 4), (1, 4)
        collapsed = []
        count = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                count += 1
            else:
                collapsed.append((seq[i-1], count))
                count = 1
        collapsed.append((seq[-1], count))
        return collapsed

    # base region: 0 for 5' end, 1 for internal ROI with data, 2 for 3' end
    nucleotides = []

    collapsed = collapse_seq(convert_seq(seq))

    console.print(f"[green]Processing seq   '{seq}'...[/green]")
    console.print(f"[green]Assigning region '{collapsed}...'[/green]")

    if len(collapsed) == 2:
        sites = np.arange(1, len(seq) + 1)
        base_regions = [1] * collapsed[0][1] + [2] * collapsed[1][1]
    elif len(collapsed) == 3:
        len_5prime = collapsed[0][1]
        sites_upstream = np.arange(-len_5prime, 0)
        
        # query for site adjustment downstream of 5' primer region
        start_site = int(Prompt.ask("[bold cyan]Enter site number following the 5'-primer region[/bold cyan]", default=1))
        sites_downstream = np.arange(start_site, len(seq) - len_5prime + start_site)
        sites = np.concatenate([sites_upstream, sites_downstream])
        base_regions = [0] * len_5prime + [1] * collapsed[1][1] + [2] * collapsed[2][1]

    for j in range(len(seq)):
        cursor.execute("INSERT INTO nucleotides (site, base, base_region, construct_id) VALUES (?, ?, ?, ?)", (int(sites[j]), seq[j], base_regions[j], construct_id))

    console.print(f"[green]Nucleotides for construct '{construct_id}' created![/green]")

    return nucleotides

# Main function to handle sample import
def import_samples(csv_path: str, db_path: str = None):
    print(f"Importing samples to {db_path} from {csv_path}...")
    conn = connect_db(db_path)
    cursor = conn.cursor()
    
    # Process unique buffers, constructs, and sequencing runs
    unique_buffers = pd.read_csv(csv_path)["buffer"].unique()
    unique_constructs = pd.read_csv(csv_path)["construct"].unique()
    unique_seq_runs = pd.read_csv(csv_path)["sequencing_run"].unique()
    console = Console()

    print(f"Detected the following unique buffers: {unique_buffers}")
    print("Trying to get buffer_id or create new buffer...")
    buffer_dict = {}
    for b in unique_buffers:
        buffer_id = handle_buffer(b, console, cursor)
        buffer_dict[b] = buffer_id

    print(f"Detected the following unique constructs: {unique_constructs}")
    print("Trying to get construct_id or create new construct...")
    construct_dict = {}
    for c in unique_constructs:
        construct_id = handle_construct(c, console, cursor)
        construct_dict[c] = construct_id

    print(f"Detected the following unique sequencing_runs: {unique_seq_runs}")
    print("Trying to get sequencing_run_id or create new sequencing_run...")
    seq_run_dict = {}
    for sr in unique_seq_runs:
        sr_id = handle_seqrun(sr, console, cursor)
        seq_run_dict[sr] = sr_id

    # Get max id for reaction_groups and probing_reactions
    rxn_group_maxid = get_max_id(cursor, "reaction_groups", "rg_id")
    rxn_maxid = get_max_id(cursor, "probing_reactions", "id")

    unique_rxn_groups = pd.read_csv(csv_path)["rxn_group"].unique()
    adjusted_rxn_groups_id = np.arange(rxn_group_maxid, rxn_group_maxid + len(unique_rxn_groups))
    rxn_groups_dict = {rg: adjusted_rxn_groups_id[i] for i, rg in enumerate(unique_rxn_groups)}
    
    # Get max id for sequencing_samples
    s_maxid = get_max_id(cursor, "sequencing_samples", "id")

    main_df = pd.read_csv(csv_path)

    # adjust buffer to buffer_id
    main_df["buffer"] = main_df["buffer"].map(buffer_dict)
    # adjust construct to construct_id
    main_df["construct"] = main_df["construct"].map(construct_dict)
    # adjust sequencing_run to seq_run_id
    main_df["sequencing_run"] = main_df["sequencing_run"].map(seq_run_dict)
    # adjust rxn_groups_id
    main_df["rxn_group"] = main_df["rxn_group"].map(rxn_groups_dict)

    for i, row in main_df.iterrows():
        # insert new samples
        cursor.execute(
            "INSERT INTO sequencing_samples (seqrun_id, sample_name, fq_dir) VALUES (?, ?, ?)",
            (row['sequencing_run'], row['sample_name'], row['fq_dir'])
        )
        # get sample_id from cursor
        sample_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO probing_reactions (temperature, replicate, reaction_time, probe_concentration, probe, buffer_id, construct_id, RT, done_by, treated, rg_id, s_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row['temperature'], row['replicate'], row['reaction_time'], row['probe_concentration'], row['probe'], row['buffer'], row['construct'], row['RT'], row['done_by'], row['treated'], row['rxn_group'], sample_id)
        )
        rxn_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO reaction_groups (rg_id, rxn_id) VALUES (?, ?)",
            (row['rxn_group'], rxn_id)
        )
    conn.commit()
    conn.close()
    print("Sample import completed!")


### Automate tempgrad assignment

def process_tempgrad_groups(db_path: str) -> int:
    """
    Assign a tg_id to each rg_id in the reaction_groups table based on shared condition sets.
    This function fetches distinct conditions from the probing_reactions table and assigns a unique tg_id
    to each unique set of conditions. It then updates the tempgrad_groups table with these assignments
    and returns the count of successful insertions.

    Args:
        db_path (str): Path to the database file.
    Returns:
        int: Count of successful insertions into the tempgrad_groups table.
    This function assumes that the probing_reactions table has the following columns:
        - buffer_id
        - construct_id 
        - RT (reaction time)
        - probe
        - probe_concentration
    """

    tempgrad_conditions = fetch_distinct_tempgrad_group(db_path)
    count_insert_success = assign_tempgrad_groups(db_path, tempgrad_conditions)
    console.print(f"[green]✓ Imported {count_insert_success} tempgrad groups[/green]")

    return count_insert_success
