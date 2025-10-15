# `nerd tempgrad_fit`

`nerd tempgrad_fit` performs temperature-gradient analyses (Arrhenius or two-state melt) using the pluggable engines under `pipeline/plugins/tempgrad`. It can consume NMR fits (`data_source: nmr`) or probe timecourse fits (`data_source: probe_tc`) and stores results in `tempgrad_fit_runs`/`tempgrad_fit_params`.

```
nerd tempgrad_fit --config PATH/TO/config.yaml --db PATH/TO/nerd.sqlite
```

If `--db` is omitted, NERD falls back to `<run.output_dir>/nerd.sqlite`.

---

## Configuration Layout

```yaml
run:
  label: tempgrad_demo
  output_dir: examples

tempgrad_fit:
  mode: arrhenius                 # or two_state_melt
  data_source: probe_tc           # nmr or probe_tc
  overwrite: true

  filters:
    construct: 4U_wt
    base: A
    melt_threshold_c: 60
    fit_kind: round3_constrained

  engine_options:
    temperature_unit: c
    weighted: true
```

### Key Fields

| Field | Description |
|-------|-------------|
| `mode` | Selects the engine (`arrhenius`, `two_state_melt`). |
| `data_source` | `nmr` (uses `nmr_fit_runs`) or `probe_tc` (uses `probe_tc_fit_params`). |
| `filters` | Constraints on construct, buffer, base, nt, melt threshold, etc. |
| `engine_options` | Passed to the engine; e.g., `weighted`, `temperature_unit`, `melt_threshold_c`. |
| `overwrite` | Removes existing rows in `tempgrad_fit_runs/tempgrad_fit_params` before inserting new ones. |

For `probe_tc`, grouping happens automatically: reactions are grouped by construct, buffer, probe, concentration, and RT protocol; temperatures are filtered using `melt_threshold_c` before fitting each nucleotide.

---

## Supported Data Sources

1. **NMR Arrhenius (`data_source: nmr`)**  
   Pulls completed fits from `nmr_fit_runs`, using `k_value`/`k_error` to produce log-linear fits.

2. **Probe Arrhenius (`data_source: probe_tc`)**  
   Aggregates `probe_tc` fits on the fly. You can filter by constructs, buffers, bases, reaction-group IDs, etc. Weighted regression uses `log_kobs_err` when available.

3. **Two-state melt (`mode: two_state_melt`)**  
   A placeholder stub exists; hook in your custom engine or supply `series` overrides until implemented.

---

## Outputs

- A run record per series in `tempgrad_fit_runs` (fit kind, scope, data source, metadata).
- Tall-format parameters and diagnostics in `tempgrad_fit_params`.
- JSON artifacts under `<output_dir>/<label>/tempgrad_fit_latest/results/tempgrad_result.json`.

---

## Examples

```bash
# NMR degradation Arrhenius (species: dms)
nerd tempgrad_fit --config manuscript_pipeline/03_nmr_arrhenius/tempgrad_deg.yaml \
                  --db manuscript_pipeline/nerd.sqlite

# Probe Arrhenius, melted-only ATP nucleotides
nerd tempgrad_fit --config examples/probe_tempgrad_arrhenius/config.yaml \
                  --db examples/nerd.sqlite

# Manual series override
nerd tempgrad_fit --config configs/tempgrad_manual.yaml --db nerd.sqlite
```

After running `tempgrad_fit`, downstream notebooks or reports can summarize activation energies, global slope/intercept, and melt diagnostics directly from the tall-format tables.
