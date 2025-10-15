# `nerd create`

`nerd create` ingests metadata (constructs, buffers, sequencing runs, samples) into a NERD SQLite database. It is typically the first task you run when setting up a new analysis.

````bash
nerd create --config PATH/TO/config.yaml --db PATH/TO/nerd.sqlite
````

If `--db` is omitted, the CLI creates/updates `nerd.sqlite` inside `run.output_dir`.

---

## Configuration Layout

Every config has two top-level sections:

```yaml
run:
  label: create_all_samples         # used in output paths
  output_dir: examples              # base directory for logs & artifacts
  backend: local                    # optional: slurm / ssh

create:
  buffers: buffers.csv
  constructs: constructs.csv
  sequencing_runs: sequencing_runs.csv
  samples: probing_samples.csv
  derived_samples: derived.csv      # optional
```

- **`run` block** – standard NERD metadata (label, output directory, execution backend, resources).
- **`create` block** – describes what to import. Values can be inline objects _or_ CSV filenames relative to the config (or resolved via `search_roots`).

### CSV Mode

When a value is a string (e.g., `samples: probing_samples.csv`), `nerd create` treats it as a sheet:

- `buffers.csv` → inserted into `meta_buffers`.
- `constructs.csv` → inserted into `meta_constructs` (optionally with `nt_info` paths).
- `sequencing_runs.csv` → inserted into `sequencing_runs`.
- `probing_samples.csv` → inserted into `sequencing_samples`, plus derived `probe_reactions`/`probe_reaction_groups`.

Columns should match the spreadsheet headers (see `examples/create_all_samples/to_import/`). Extra columns are ignored; missing required columns raise errors.

### YAML Mode

You can inline dictionaries instead of CSV files:

```yaml
create:
  buffers:
    - name: Schwalbe_bistris
      pH: 6.5
      composition: Tris/Bicine
      disp_name: pH6.5_A
  constructs:
    - family: HIV
      name: fourU_new
      version: 2
      sequence: ACTG...
      disp_name: 4U_wt
      nt_info: fourU_wt_nt.csv     # optional relative path
```

Inline mode is best for small test cases; CSV mode scales better for large studies.

---

## Sample Ingestion Details

When loading samples from CSV:

- `sequencing_run_name` is used to look up (or create) sequencing runs.
- `construct` and `buffer` names are resolved against the database (using either `disp_name` or `name`). Missing references throw errors before any reactions are inserted.
- `reaction_group` labels are turned into `probe_reaction_groups`, and each sample becomes a row in `probe_reactions` with the associated metadata (`temperature`, `reaction_time`, etc.).

If you omit constructs or buffers in the sheet, you may supply defaults in the YAML; otherwise, `nerd create` requires every sample to specify a valid reference.

---

## Trace of Inserted Objects

The task writes two log files in the run directory:

- `created_objects.json`
- `created_objects.log`

Both summarize how many buffers, constructs, sequencing runs, and samples were created or updated. Use them to confirm the import before running downstream tasks.

---

## Common Flags

| Option | Description |
|--------|-------------|
| `--config` | Path to YAML file (required). |
| `--db` | SQLite database path; defaults to `<run.output_dir>/nerd.sqlite`. |
| `--verbose` | Enables DEBUG logging. |
| `--log-file` | Write logs to a custom path. |

Because `nerd create` runs inside the CLI’s task framework, you can also supply `--backend`, `--threads`, `--mem-gb`, etc., in the `run` section to scale the job.

---

## Examples

````bash
# Minimal inline config
nerd create --config examples/create_minimal.yaml --db nerd.sqlite

# Full CSV import (buffers, constructs, sequencing runs, samples)
nerd create --config examples/create_all_samples/create_all.yaml --db nerd.sqlite

# Separate probing sample import
nerd create --config examples/create_all_samples/create_probing_samples.yaml \
            --db nerd.sqlite
````

Once metadata is ingested, you’re ready to run `nmr_*` or `probe_timecourse` tasks on the same database.
