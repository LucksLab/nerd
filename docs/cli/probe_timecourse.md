# `nerd probe_timecourse`

`nerd probe_timecourse` fits chemical probing time-course data stored in `probe_fmod_values`, producing per-nucleotide parameters (free fits, global deg fits, constrained refits). Results are written to `probe_tc_fit_runs`/`probe_tc_fit_params` and JSON artifacts for each reaction group.

```
nerd probe_timecourse --config PATH/TO/config.yaml --db PATH/TO/nerd.sqlite
```

Omitting `--db` defaults to `<run.output_dir>/nerd.sqlite`.

---

## Configuration Layout

```yaml
run:
  label: probe_tc_demo
  output_dir: examples

probe_timecourse:
  engine: python_baseline
  rounds:
    - round1_free
    - round2_global
    - round3_constrained
  rg_ids: [101, 102]
  valtype: GAmodrate
  min_points: 3
  overwrite: true

  engine_options:
    global_selection: ac_only
    global_filters:
      r2_threshold: 0.75
    initial_kdeg: 0.0025
```

### Key Fields

| Field | Description |
|-------|-------------|
| `engine` | Registered timecourse engine (`python_baseline` is the default). |
| `rounds` | Subset of the three-stage workflow (free, global, constrained). |
| `rg_ids` | Reaction-group IDs to process; leave empty to process all. |
| `valtype` | Which value type to use (`modrate`, `GAmodrate`, etc.). |
| `min_points` | Minimum points required per nucleotide (default 3). |
| `overwrite` | When true, existing fits in `probe_tc_fit_runs` are replaced. |
| `engine_options` | Passed to the engine; e.g., global selection filters, initial guesses. |

Additional filters: `nt_ids`, `species`, `buffer`, `construct`, `fit_kind`, etc. (ignored unless supported by the engine). See the example configs under `examples/probing_timecourse_fits/`.

---

## Workflow Overview

1. **Round 1 (`round1_free`)** – independent per-nucleotide fits (k_obs, k_deg, f_mod0).
2. **Round 2 (`round2_global`)** – global k_deg using filtered nucleotides (A/C only, R² threshold, etc.).
3. **Round 3 (`round3_constrained`)** – refit each nucleotide with k_deg fixed from round 2.

You can run any subset of rounds depending on the goal (e.g., skip global and only produce free fits by listing `round1_free`).

---

## Outputs

- Per-nucleotide parameters and diagnostics in `probe_tc_fit_params` (tall format).
- Run-level metadata in `probe_tc_fit_runs` (fit kind, rg_id, nt_id, timestamp).
- JSON summaries under `<output_dir>/<label>/probe_timecourse_latest/results/`.

---

## Examples

```bash
# Full three-round fit for selected reaction groups
nerd probe_timecourse --config examples/probing_timecourse_fits/config.yaml                       --db examples/nerd.sqlite

# Only run free fits
nerd probe_timecourse --config configs/probe_tc_free.yaml --db nerd.sqlite

# Constrained refits using existing global kdeg
nerd probe_timecourse --config configs/probe_tc_constrained.yaml --db nerd.sqlite
```

Once fits are completed, downstream tasks (e.g., `tempgrad_fit` with `data_source: probe_tc`) can reuse the stored kinetics for Arrhenius analysis.
