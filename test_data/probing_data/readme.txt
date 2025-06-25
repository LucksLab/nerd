1 - initialize db

sqlite3 new.db < initial_migration.sql

2 - import buffers and constructs

python append_csv_to_sqlite.py ./initial_import/buffers.csv buffers ./new.db

python append_csv_to_sqlite.py ./initial_import/constructs.csv constructs ./new.db

python append_csv_to_sqlite.py ./initial_import/sequencing_runs.csv sequencing_runs ./new.db

python append_csv_to_sqlite.py ./initial_import/all_nts.csv nucleotides ./new.db

3 - run interactive import script

python import_csv.py ./new.db initial_import/samples_import.csv

follow interactive prompt