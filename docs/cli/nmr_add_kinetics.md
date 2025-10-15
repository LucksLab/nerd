# `nerd nmr_add_kinetics`

`nerd nmr_add_kinetics` fits NMR-based adduction kinetics (k_add) for reactions registered via `nmr_create`. It relies on trace files (peak and DMS traces) staged in the database and records results in `nmr_fit_runs`/`nmr_fit_params`.

```
nerd nmr_add_kinetics --config PATH/TO/config.yaml --db PATH/TO/nerd.sqlite
```

If `--db` is omitted, the CLI uses `<run.output_dir>/nerd.sqlite`.

---

## Configuration Layout

```yaml
run:
  label: nmr_add_demo
  output_dir: examples
  backend: local

nmr_add_kinetics:
  search_roots:
    - ../../test_data
  plugin: ode_lsq_ntp_add
  plugin_options:
    k_add_init: 0.002
    k_deg_init: 0.001
  fit_params:
    S_factor: 1.0
  species:
    - ATP_C8
```

### Key Fields

| Field | Description |
|-------|-------------|
| `search_roots` | Directories to resolve trace files if `trace_files.path` is relative. |
| `plugin` | Registered NMR fit engine (defaults to `ode_lsq_ntp_add`). |
| `plugin_options` | Passed to the engine constructor (e.g., initial guesses). |
| `fit_params` | Forwarded to the plugin’s `fit` call (e.g., scaling factors). |
| `species` | One or more trace species to fit (e.g., `ATP_C8`). If omitted, defaults to `substrate` or `ntp_probe`. |

Optional filters:

- `reaction_ids`: explicit list of reactions to fit.
- `substrates`: restrict to specific substrates (e.g., `ATP`, `CTP`).
- `buffer`, `construct`, `temperature`, etc., can be enforced via custom pre-filters (deprecated in favor of species/substrate lists).

---

## Prerequisites

1. Run `nerd nmr_create` with `trace_files` entries:
   ```yaml
   trace_files:
     peak_trace:
       path: nmr_ntp_adduction_data/ATP_peak_percentages/20_1_peak8.csv
       species: ATP_C8
     dms_trace:
       path: nmr_ntp_adduction_data/ATP_peak_percentages/20_1_peakDMS.csv
       species: ATP_DMS
   ```
2. Ensure constructs, buffers, and sequencing runs are already in the database (`nerd create`).

---

## Outputs

- New rows in `nmr_fit_runs` capturing plugin name, model, species, and run metadata.
- Associated parameters in `nmr_fit_params` (`k_value`, `k_error`, `r2`, `chisq`, plus plugin diagnostics).
- JSON artifacts under `<output_dir>/<label>/nmr_add_kinetics_latest/results/` describing each reaction fit.

---

## Examples

```bash
# Fit ATP C8 adduction kinetics
nerd nmr_add_kinetics --config examples/nmr_fit_add/configs/fit_add.yaml                       --db examples/nerd.sqlite

# Run with multiple species (C8 + DMS)
nerd nmr_add_kinetics --config configs/fit_add_multi.yaml --db nerd.sqlite
```

After the fits complete, downstream tasks (e.g., `tempgrad_fit` with `data_source: nmr`) can consume the stored rates for Arrhenius analysis.
