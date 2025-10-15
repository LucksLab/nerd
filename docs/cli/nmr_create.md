# `nerd nmr_create`

`nerd nmr_create` registers NMR reactions and their trace files in the database, enabling downstream kinetics tasks (`nmr_deg_kinetics`, `nmr_add_kinetics`). You can inline reactions in YAML or point to CSV sheets.

```
nerd nmr_create --config PATH/TO/config.yaml --db PATH/TO/nerd.sqlite
```

If `--db` is omitted, `<run.output_dir>/nerd.sqlite` is used.

---

## Configuration Layout

```yaml
run:
  label: nmr_create_deg
  output_dir: examples

nmr_create:
  search_roots:
    - ../../test_data
  reactions:
    - reaction_type: deg
      temperature: 25
      replicate: 1
      probe: dms
      probe_conc: 0.01585
      buffer: Schwalbe_bistris
      substrate: none
      substrate_conc: 0
      num_scans: 64
      time_per_read: 5
      total_kinetic_reads: 96
      total_kinetic_time: 28800
      nmr_machine: A600
      trace_files:
        decay_trace: nmr_degradation_data/dms_schwalbe_25C_rep1.csv
```

### Key Fields

| Field | Description |
|-------|-------------|
| `search_roots` | Extra directories to resolve trace file paths. |
| `reactions` | List (or CSV filename) describing each NMR reaction to ingest. |
| `trace_files` | Mapping of trace roles (e.g., `decay_trace`, `peak_trace`, `dms_trace`) to file paths. Can be `{path, species}` for adduction traces. |

Optional: `kinetic_data_dir`, `mnova_analysis_dir`, `raw_fid_dir` for bookkeeping.

---

## CSV Mode

If `reactions` is a string (e.g., `reactions: nmr_degradation_samples.csv`), NERD loads that CSV. Headers should match the YAML fields above. Use `search_roots` so relative trace paths resolve correctly.

---

## Outputs

- Rows in `nmr_reactions` (temperature, replicate, buffer, construct association).
- Trace metadata in `nmr_trace_files` for each role/path/species.
- Run artifacts and logs under `<output_dir>/<label>/nmr_create_latest/`.

---

## Examples

```bash
# Import degradation reactions from CSV
nerd nmr_create --config examples/nmr_create/nmr_create_deg.yaml --db nerd.sqlite

# Inline adduction entries
nerd nmr_create --config configs/nmr_create_add.yaml --db nerd.sqlite
```

After `nmr_create`, run `nmr_deg_kinetics` or `nmr_add_kinetics` to fit the registered reactions.
