from nerd.importers import fmod_calc, probing_sample, nmr
from nerd.kinetics import degradation, adduction, arrhenius, timecourse
#from nerd.energy import meltfit, calc_K
from nerd.db import io, fetch, update
from nerd.utils import plotting


# === Paths to example input files ===
SAMPLE_SHEET = "test_data/sample_sheet.csv"
SHAPEMAPPER_DIR = "test_data/shapemapper_run/"
NMR_SHEET = "test_data/nmr_sheet.csv"
DEGRADATION_CSV = "test_data/nmr_deg.csv"
ADDUCTION_CSV = "test_data/nmr_add.csv"
ARRHENIUS_CSV = "test_data/arrhenius.csv"
MELT_CURVE_CSV = "test_data/melt_curve.csv"

# === Dev DB path ===
DB_PATH = "nerd_dev.sqlite3"


def test_all():
    print("\n[IMPORT NMR DEGRADATION SAMPLES]")
    # Initialize the database
    conn = io.connect_db('test_output/nerd_dev.sqlite3')
    io.init_db(conn)
    conn.close()

    # Import buffers
    nmr.import_buffer(
        csv_path="test_data/nmr_buffers.csv",
        db_path='test_output/nerd_dev.sqlite3'
    )

    # Run the NMR importer with a sample CSV file
    nmr.run('test_data/nmr_degradation_samples.csv', db_path='test_output/nerd_dev.sqlite3')

    # Display samples to analyze
    conn = io.connect_db('test_output/nerd_dev.sqlite3')
    samples, columns = io.fetch_all_nmr_samples(conn, 'deg')
    conn.close()

    #io.display_table(samples, columns, title="All NMR Degradation Samples")

    # TODO display 10, then click for next page

    print("\n[DEGRADATION NMR]")
    user_input = 'all'
    #user_input = '1, 2, 3'
    #user_input = '1'

    degradation.run(samples, [], db_path='test_output/nerd_dev.sqlite3')

    arrhenius.run(reaction_type="deg", data_source="nmr", species="dms",
                  buffer="Schwalbe_bistris", db_path='test_output/nerd_dev.sqlite3')
    
    print("\n[IMPORT NMR ADDUCTION SAMPLES]")

    # Run the NMR importer with a sample CSV file
    nmr.run('test_data/nmr_adduction_samples.csv', db_path='test_output/nerd_dev.sqlite3')

    print("\n[ADDUCTION NMR]")

    # Display samples to analyze
    conn = io.connect_db('test_output/nerd_dev.sqlite3')
    add_samples, add_columns = io.fetch_all_nmr_samples(conn, 'add')
    conn.close()

    #io.display_table(add_samples, add_columns, title="All NMR Adduction Samples")

    adduction.run(add_samples, [], db_path='test_output/nerd_dev.sqlite3')

    arrhenius.run(reaction_type="add", data_source="nmr", species="ATP",
                  buffer="Schwalbe_bistris", db_path='test_output/nerd_dev.sqlite3')

    arrhenius.run(reaction_type="add", data_source="nmr", species="CTP",
                  buffer="Schwalbe_bistris", db_path='test_output/nerd_dev.sqlite3')

    arrhenius.run(reaction_type="add", data_source="nmr", species="GTP",
                  buffer="Schwalbe_bistris", db_path='test_output/nerd_dev.sqlite3')


    print("\n[IMPORT PROBING SAMPLES]")

    probing_sample.import_buffer(
        csv_path="test_data/probing_data/buffers.csv",
        db_path='test_output/nerd_dev.sqlite3'
    )

    probing_sample.import_construct(
        csv_path="test_data/probing_data/constructs.csv",
        db_path='test_output/nerd_dev.sqlite3'
    )

    probing_sample.import_seqrun(
        csv_path="test_data/probing_data/sequencing_runs.csv",
        db_path='test_output/nerd_dev.sqlite3'
    )

    probing_sample.import_samples(
        csv_path="test_data/probing_data/probing_samples.csv",
        db_path='test_output/nerd_dev.sqlite3'
    )

    print("\n[IMPORT FMOD CALC RUNS]")

    # Import fmod_calc runs
    fmod_calc.run(
        fmod_calc_csv="test_data/probing_data/fmod_calc_runs.csv",
        db_path='test_output/nerd_dev.sqlite3'
    )

    print("\n[PROBING SAMPLES]")
    timecourse.run(db_path='test_output/nerd_dev.sqlite3')


    print("\n[ADDUCTION FITTING]")
    adduction.process_melted_param_aggregation()
    adduction.process_melted_ind_params(db_path='test_output/nerd_dev.sqlite3')    

if __name__ == "__main__":
    #est_all()
    #timecourse.run(db_path='test_output/nerd_dev.sqlite3')
    #adduction.process_melted_param_aggregation()
    #adduction.process_melted_ind_params(db_path='test_output/nerd_dev.sqlite3') 
    tempgrad_conditions = fetch.fetch_distinct_tempgrad_group('test_output/nerd_dev.sqlite3')
    rg_to_tg = update.assign_tempgrad_groups('test_output/nerd_dev.sqlite3', tempgrad_conditions)

    print(rg_to_tg)