from nerd.importers import probing_sample, shapemapper, nmr
from nerd.kinetics import degradation, adduction, arrhenius
#from nerd.energy import meltfit, calc_K
from nerd.db import io


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
    print("\n[IMPORT]")
    probing_sample.run(SAMPLE_SHEET, db_path=DB_PATH)
    shapemapper.run(SHAPEMAPPER_DIR, db_path=DB_PATH)
    nmr.run(NMR_SHEET, db_path=DB_PATH)

    print("\n[DEGRADATION]")
    degradation.run(csv_path=DEGRADATION_CSV, reaction_id="rxn_deg_test", db_path=DB_PATH)

    print("\n[ADDUCTION]")
    adduction.run(csv_path=ADDUCTION_CSV, reaction_id="rxn_add_test", db_path=DB_PATH)

    print("\n[ARRHENIUS]")
    arrhenius.run(csv_path=ARRHENIUS_CSV,
                  reaction_type="degradation",
                  data_source="nmr",
                  db_path=DB_PATH)

    print("\n[TIMECOURSE]")
    timecourse.run(reaction_group_id="rg_test", db_path=DB_PATH)

    print("\n[CALC ENERGY: Two-State]")
    meltfit.run(MELT_CURVE_CSV)

    print("\n[CALC ENERGY: Single K]")
    calc_K.run(["0.015", "0.005", "0.002"])



if __name__ == "__main__":
    #test_all()
    # Initialize the database
    conn = io.connect_db('test_output/nerd_dev.sqlite3')
    io.init_db(conn)
    conn.close()

    # Run the NMR importer with a sample CSV file
    nmr.run('test_data/nmr_degradation_samples.csv', db_path='test_output/nerd_dev.sqlite3')

    # Display samples to analyze
    conn = io.connect_db('test_output/nerd_dev.sqlite3')
    samples, columns = io.fetch_all_nmr_samples(conn, 'deg')
    conn.close()

    #io.display_table(samples, columns, title="All NMR Degradation Samples")

    # TODO display 10, then click for next page

    # Run degradation fit
    user_input = 'all'
    #user_input = '1, 2, 3'
    #user_input = '1'

    degradation.run(samples, None, db_path='test_output/nerd_dev.sqlite3')

    arrhenius.run(reaction_type="deg", data_source="nmr", species="none",
                  buffer="schwalbe_buffer", db_path='test_output/nerd_dev.sqlite3')
    

    # Run the NMR importer with a sample CSV file
    nmr.run('test_data/nmr_adduction_samples.csv', db_path='test_output/nerd_dev.sqlite3')

    # Display samples to analyze
    conn = io.connect_db('test_output/nerd_dev.sqlite3')
    add_samples, add_columns = io.fetch_all_nmr_samples(conn, 'add')
    conn.close()

    #io.display_table(add_samples, add_columns, title="All NMR Adduction Samples")

    adduction.run(add_samples, None, db_path='test_output/nerd_dev.sqlite3')

    arrhenius.run(reaction_type="add", data_source="nmr", species="ATP",
                  buffer="schwalbe_buffer", db_path='test_output/nerd_dev.sqlite3')

    arrhenius.run(reaction_type="add", data_source="nmr", species="CTP",
                  buffer="schwalbe_buffer", db_path='test_output/nerd_dev.sqlite3')

    arrhenius.run(reaction_type="add", data_source="nmr", species="GTP",
                  buffer="schwalbe_buffer", db_path='test_output/nerd_dev.sqlite3')

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