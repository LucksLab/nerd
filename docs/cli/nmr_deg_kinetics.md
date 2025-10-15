# `nerd nmr_deg_kinetics`

`nerd nmr_deg_kinetics` fits first-order degradation rates (k_deg) for NMR reactions registered via `nmr_create`. Each reaction needs a `decay_trace` file recorded in the database trace list.

```
nerd nmr_deg_kinetics --config PATH/TO/config.yaml --db PATH/TO/nerd.sqlite
```

If `--db` is omitted, the CLI uses `<run.output_dir>/nerd.sqlite`.

---

## Configuration Layout

```yaml
run:
  label: nmr_deg_demo
  output_dir: examples
  backend: local

nmr_deg_kinetics:
  search_roots:
    - ../../test_data
  plugin: lmfit_deg
  plugin_options:
    model: exponential
  fit_params:
    normalize: true
  reaction_ids: [1, 2, 3]
```

### Key Fields

| Field | Description |
|-------|-------------|
| `search_roots` | Directories to resolve trace paths when they are relative. |
| `plugin` | Registered degradation fit engine (`lmfit_deg` is the default). |
| `plugin_options` | Passed to the engine constructor (e.g., choose a different model). |
| `fit_params` | Forwarded to the plugin’s `fit` call; allows run-specific overrides. |
| `reaction_ids` | Optional explicit reactions to fit. Leave empty to fit every degradation reaction in the database. |

---

## Prerequisites

1. Ingest reactions via `nerd nmr_create`, including trace mappings:
   ```yaml
   trace_files:
     decay_trace: nmr_degradation_data/dms_schwalbe_25C_rep1.csv
   ```
2. Ensure constructs, buffers, and sequencing runs exist in the database (`nerd create`).

---

## Outputs

- Rows in `nmr_fit_runs` with plugin, model, and reaction metadata.
- Associated `nmr_fit_params` capturing `k_value`, `k_error`, `r2`, `chisq`, etc.
- JSON summaries under `<output_dir>/<label>/nmr_deg_kinetics_latest/results/`.

---

## Examples

```bash
# Fit all degradation reactions
nerd nmr_deg_kinetics --config examples/nmr_fit_deg/configs/fit_deg.yaml                        --db examples/nerd.sqlite

# Fit a subset of reaction IDs
nerd nmr_deg_kinetics --config configs/fit_deg_subset.yaml --db nerd.sqlite
```

Once the fits are recorded, Arrhenius analysis can reuse them via `tempgrad_fit` with `data_source: nmr`.
