# nerd/importers/shapemapper.py

from pathlib import Path
from rich.console import Console
import glob
import sqlite3
import pandas as pd
from nerd.db.io import connect_db, init_db  # You’ll define insert function later
from nerd.db.fetch import fetch_run_name
from rich.progress import Progress

console = Console()


def extract_info_from_log(log_file, sample_name, run0420 = None):
    """
    Extracts information from a ShapeMapper log file.

    Parameters:
        log_file (str): Path to the ShapeMapper log file.
        sample_name (str): Name of the sample to check against the R1 file.

    Returns:
        tuple: Contains the following elements:
            - run_datetime (str): The datetime when the ShapeMapper run started.
            - version (str): The version of ShapeMapper used.
            - r1_file (str): The R1 file used in the run.
            - untreated (int): Indicates if the sample was untreated (1 if untreated, 0 otherwise).
            - denatured (int): Indicates if the sample was denatured (1 if denatured, 0 otherwise).
            - sample_check (bool): Indicates if the sample name matches the R1 file.
    """

    with open(log_file) as f:
        lines = f.readlines()

    # find all lines containing "Started ShapeMapper" and get index of most recent one
    detect_shapemapper_runs = [i for i, line in enumerate(lines) if 'Started ShapeMapper' in line]
    assert len(detect_shapemapper_runs) > 0, 'No ShapeMapper runs detected in log file'

    most_recent_run = detect_shapemapper_runs[-1]
    lines = lines[most_recent_run:]

    # check shapemapper success
    run_completed = [i for i, line in enumerate(lines) if ('ShapeMapper run completed' in line) or ('ShapeMapper run successfully completed' in line)]
    if len(run_completed) == 0:
        console.print(f'ShapeMapper run not completed successfully in log file: {log_file}')
        return None

    #print(log_file)
    # extract date and version from:  "Started ShapeMapper v2.2.0 at 2023-04-22 17:19:59"
    version_date_line = lines[0]
    run_datetime = version_date_line.split(' at ')[1].rstrip()
    version = version_date_line.split(' ')[2]
    run_args = lines[2]
    assert 'args: ' in run_args, 'args line not found in log file'
    
    # get index of 'modified'
    modified_index = run_args.split(' --').index('modified')
    assert modified_index > 0, 'modified not found in run_args'

    # extract R1 file
    r1_file = run_args.split(' --')[modified_index + 1].split(' ')[-1]
    assert (r1_file is not None) or (r1_file == ''), 'R1 file not found in run_args'

    untreated = 0
    denatured = 0
    # check if untreated sample provided
    if 'untreated' in run_args.split(' --'):
        untreated_index = run_args.split(' --').index('untreated')
        untreated_r1_file = run_args.split(' --')[untreated_index + 1].split(' ')[-1]
        assert (untreated_r1_file is not None) or (untreated_r1_file == ''), 'R1 file not found in run_args'
        untreated = untreated_r1_file
    elif 'denatured' in run_args.split(' --'):
        denatured_index = run_args.split(' --').index('denatured')
        den_r1_file = run_args.split(' --')[denatured_index + 1].split(' ')[-1]
        assert (den_r1_file is not None) or (den_r1_file == ''), 'R1 file not found in run_args'
        denatured = den_r1_file

    # confirm sample_name matches r1_file
    # remove .fastq.gz from both if they exist
    if len(r1_file.split('/')) > 2:
        r1_file = r1_file.split('/')[-1]
        r1_file_check = r1_file.replace('...', '')
        sample_name_check = sample_name[:len(r1_file_check)]
        sample_check = (sample_name_check == r1_file_check)
        return run_datetime, run_args, version, r1_file, untreated, denatured, sample_check
    if r1_file.endswith('.fastq.gz'):
        r1_file_check = r1_file[:-9]
    elif r1_file.endswith('.fastq'):
        r1_file_check = r1_file[:-6]
    else:
        r1_file_check = r1_file
    if sample_name.endswith('.fastq.gz'):
        sample_name_check = sample_name[:-9]
    elif sample_name.endswith('.fastq'):
        sample_name_check = sample_name[:-6]
    else:
        sample_name_check = sample_name
    if r1_file.startswith('./'):
        r1_file_check = r1_file_check[2:]
    if sample_name_check.startswith('YYYR'):
        r1_file_check = r1_file_check[5:]
        sample_name_check = sample_name_check[5:]
    elif sample_name_check.startswith('etOH'):
        #DMS-150-WTII_S8_L001_R1_001 etOH-150-WTII_S16_L001_R1_001
        r1_file_check = '-'.join(r1_file_check.split('_')[0].split('-')[1:])
        sample_name_check = '-'.join(sample_name_check.split('_')[0].split('-')[1:])

    sample_check = (sample_name_check == r1_file_check)

    # override mistaken r1 file name
    if 'WT-33c-b-6' in r1_file_check:
        sample_check = True
        return run_datetime, run_args, version, r1_file, untreated, denatured, sample_check

    if sample_check == False:
        # try removing all underscores and compare again
        r1_file_check = r1_file_check.replace('_', '')
        sample_name_check = sample_name_check.replace('_', '')
        sample_check = (sample_name_check == r1_file_check)
        if sample_check == False:
            console.print('upper', r1_file_check, sample_name_check)

    if run0420:
        name_index = run_args.split(' --')[1]
        assert 'name' in name_index, 'name not found in run_args'
        name_check = name_index.split(' ')[-1]
        sample_check = (name_check in sample_name) or (r1_file_check == sample_name_check)
        if sample_check:
            console.print('Rechecking on name successful')
        else:
            console.print('lower', r1_file_check, sample_name_check)

    return run_datetime, run_args, version, r1_file, untreated, denatured, sample_check

def fetch_s_id(db_path, sample_name):
    """
    Fetches the ID of a sample from the sequencing_samples table.

    Parameters:
        db_path (str): Path to the database file.
        sample_name (str): Name of the sample to fetch the ID for.

    Returns:
        int: The ID of the sample.
    """
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id FROM sequencing_samples WHERE sample_name = ?', (sample_name,))
    result = c.fetchall()  # Fetch only one row
    conn.close()

    if result is None:
        raise ValueError(f"No sample found with name: {sample_name}")
    elif len(result) > 1:
        raise ValueError(f"Multiple samples found with name: {sample_name}")
    else:
        return result[0][0]  # Extract ID from tuple

def get_max_id(db_path, table, id_col):
    """
        Fetches the maximum ID from a specified table and column.

        Parameters:
            db_path (str): Path to the database file.
            table (str): Name of the table to query.
            id_col (str): Name of the ID column to find the maximum value.

        Returns:
            int: The maximum ID value plus one, or 1 if the table is empty.
        """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f"SELECT MAX({id_col}) FROM {table}")
    max_id = c.fetchone()[0]
    return max_id + 1 if max_id else 1

def fetch_construct_seq(db_path, s_id):
    """
    Fetches the construct sequence for a given sample ID.

    Parameters:
        db_path (str): Path to the database file.
        s_id (int): Sample ID to fetch the construct sequence for.

    Returns:
        str: The construct sequence with T's converted to U's.
    """

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT construct_id FROM probing_reactions WHERE s_id = ?', (s_id,))
    construct_id = c.fetchone()[0]
    c.execute('SELECT sequence FROM constructs WHERE id = ?', (construct_id,))
    construct_seq = c.fetchone()[0]
    dict_convertTU = {'T': 'U', 't': 'u'}
    construct_seq = ''.join([dict_convertTU.get(base, base) for base in construct_seq])
    conn.close()
    return construct_seq

def fetch_rxn_id(db_path, s_id):
    """
    Fetches the reaction ID for a given sample ID.

    Parameters:
        db_path (str): Path to the database file.
        s_id (int): Sample ID to fetch the reaction ID for.

    Returns:
        int: The reaction ID.
    """

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, treated FROM probing_reactions WHERE s_id = ?', (s_id,))
    result = c.fetchone()
    conn.close()
    
    rxn_id = result[0]
    treated = result[1]
    return rxn_id, treated

def fetch_nt_ids(db_path, s_id):
    """
    Fetches the nucleotide IDs and sequence for a given sample ID.

    Parameters:
        db_path (str): Path to the database file.
        s_id (int): Sample ID to fetch the nucleotide IDs and sequence for.

    Returns:
        tuple: A tuple containing a list of nucleotide IDs and the nucleotide sequence with T's converted to U's.
    """
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT construct_id FROM probing_reactions WHERE s_id = ?', (s_id,))
    construct_id = c.fetchone()[0]
    c.execute('SELECT id, base FROM nucleotides WHERE construct_id = ?', (construct_id,))
    selected_nts = sorted(c.fetchall())
    conn.close()

    nt_ids = [nt[0] for nt in selected_nts]
    nt_seq = ''.join([nt[1] for nt in selected_nts])
    dict_convertTU = {'T': 'U', 't': 'u'}
    nt_seq = ''.join([dict_convertTU.get(base, base) for base in nt_seq])
    return nt_ids, nt_seq


def construct_fmod_calc_run(sample_name, fmod_dir, db_path, run0420 = None):

    run = glob.glob(f'{fmod_dir}/*shapemapper_log*')[0]
    run_datetime, run_args, version, r1_file, untreated, denatured, sample_check = extract_info_from_log(run, sample_name, run0420)
    s_id = fetch_s_id(db_path, sample_name)

    # get potential fmod_calc id but do not add until fmod vals are good

    # Tentative fmod_calc id (pending fmod_vals check)
    fmod_calc_id = get_max_id(db_path, 'fmod_calc_runs', 'id')

    profile_txt = glob.glob(f'{fmod_dir}/**/*_profile.txt', recursive=True)
    # exclude shapemapper_temp
    profile_txt = [x for x in profile_txt if 'shapemapper_temp' not in x]

    # choose profile with "reanalyzed"
    if len(profile_txt) > 1:
        profile_txt = [x for x in profile_txt if 'reanalyzed' in x]
        #print(fmod_dir, profile_txt)
    assert len(profile_txt) == 1, 'Multiple or no profile.txt files found'
    profile_txt = profile_txt[0]
    
    
    # process GAmodrate
    profile_txtga = glob.glob(f'{fmod_dir}/**/*_profile.txtga', recursive=True)

    # exclude shapemapper_temp
    profile_txtga = [x for x in profile_txtga if 'shapemapper_temp' not in x]
    # choose profile with "reanalyzed"
    if len(profile_txtga) > 1:
        profile_txtga = [x for x in profile_txtga if 'reanalyzed' in x]
        #print(fmod_dir, profile_txt)
    elif len(profile_txtga) == 0:
        profile_txtga = None
    else:
        profile_txtga = profile_txtga[0]

    # handle untreated or denatured
    rxn_id, rxn_treated = fetch_rxn_id(db_path, s_id)

    use_untreated_calc = False

    if (untreated != 0) & (rxn_treated == 0):
        r1_file = untreated
        use_untreated_calc = True
    elif (denatured != 0) & (rxn_treated == 1):
        r1_file = denatured

    return run_datetime, run_args, version, use_untreated_calc, r1_file, sample_check, s_id, fmod_calc_id, profile_txt, profile_txtga

def construct_fmod_vals(profile_txt, db_path, s_id, fmod_calc_id, use_untreated_calc):
    # read the csv file
    df = pd.read_csv(profile_txt, sep='\t')
    seq_from_profile = ''.join(df['Sequence'].values)

    construct_seq = fetch_construct_seq(db_path, s_id)
    assert construct_seq.upper() == seq_from_profile.upper(), 'Construct sequence does not match profile.txt sequence'

    nt_ids, nt_seq = fetch_nt_ids(db_path, s_id)

    assert nt_seq.upper() == seq_from_profile.upper(), 'Nt sequence does not match profile.txt sequence'

    rxn_id, rxn_treated = fetch_rxn_id(db_path, s_id)

    if use_untreated_calc:
        #print('using untreated')
        fmod_vals = df['Untreated_rate'].values
        read_depths = df['Untreated_read_depth'].values
    else:
        fmod_vals = df['Modified_rate'].values
        read_depths = df['Modified_read_depth'].values

    fmod_vals_df = pd.DataFrame({'nt_id': nt_ids, 'fmod_calc_run_id': fmod_calc_id, 'fmod_val': fmod_vals, 'valtype': 'modrate', 'read_depth': read_depths, 'rxn_id': rxn_id})
    return fmod_vals_df


def import_single_fmod_calc(sample_name: str, fmod_dir: str, db_path: str = None, console = None):
    """
    Import metadata from a shapemapper output directory.
    """
    if 'stop' in sample_name:
        console.print(f"[red]Error:[/red] Sample name '{sample_name}' contains 'stop'. Skipping import.")
        return

    # get text inside single quote '
    if "'" in fmod_dir:
        fmod_dir = fmod_dir.split("'")[1]

    base_path = Path(fmod_dir).resolve()
    if not base_path.exists():
        console.print(f"[red]Error:[/red] SHAPEMapper directory not found: {base_path}")
        return

    # get sequencing run date for given sample name
    seqrun_name = fetch_run_name(db_path, sample_name)

    if seqrun_name == '230420_M05164_0141_000000000-KV9HJ_RRRY-YYYR_demultiplexed':
        run0420 = True
    else:
        run0420 = False    

    run_datetime, run_args, version, use_untreated_calc, r1_file, sample_check, s_id, fmod_calc_id, profile_txt, profile_txtga = construct_fmod_calc_run(sample_name, fmod_dir, db_path, run0420)

    if sample_check:
        fmod_vals_df = construct_fmod_vals(profile_txt, db_path, s_id, fmod_calc_id, use_untreated_calc)

        if profile_txtga is not None:
            fmod_vals_df_ga = construct_fmod_vals(profile_txtga, db_path, s_id, fmod_calc_id, use_untreated_calc)
            fmod_vals_df_ga['valtype'] = 'GAmodrate'
            fmod_vals_df = pd.concat([fmod_vals_df, fmod_vals_df_ga])
        
        # Append fmod_calc_run fmod_vals to db
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('INSERT OR IGNORE INTO fmod_calc_runs (id, s_id, software_name, software_version, run_args, run_datetime, output_dir) VALUES (?, ?, ?, ?, ?, ?, ?)', (fmod_calc_id, s_id, 'shapemapper', version, run_args, run_datetime, r1_file))
        fmod_vals_df.to_sql('fmod_vals', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()

    else:
        console.print(f"[red]Error:[/red] Sample name '{sample_name}' does not match R1 file '{r1_file}' in SHAPEMapper run.")
        return        

    console.print(f"[green]✓ Successfully imported sample output for '{sample_name}'[/green]")


def run(fmod_calc_csv: str = None, db_path: str = None):
    """
    Run the fmod_calc importer for runs specified in a CSV file
    """

    base_path = Path(fmod_calc_csv).resolve()
    if not base_path.exists():
        console.print(f"[red]Error:[/red] fmod_calc_runs CSV file not found: {base_path}")
        return

    df = pd.read_csv(fmod_calc_csv)

    with Progress() as progress:
        task = progress.add_task("[cyan]Importing fmod_calc runs...", total=len(df))
        for index, row in df.iterrows():
            sample_name = row['sample_name']
            fmod_dir = row['fmod_runs']
            import_single_fmod_calc(sample_name, fmod_dir, db_path, progress.console)
            progress.advance(task)