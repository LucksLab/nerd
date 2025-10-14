---
title: "Refactoring plan"
geometry: margin=2cm
output:
  pdf_document:
    path: /01.065.refactoring_data_analysis/refactoring_plan.pdf
---

# Refactoring plan

### Objective

Want to restructure HDX model code for publication. Need to it be modular (i.e. reuse models and functions) and readable.

### Utility code

- lmfit fitting Model for time-course
- lmfit fitting Model for 2-state melt
- lmfit fitting Model for aggregate fit of kadd
- Arrhenius Linear model for kdeg independent
- global fit to get kdeg

## Main `nerd`

Nucleotide energetics from reactivity data
Tools to extract nucleotide energy from chemical probing experiments.

### `nerd import`

1. Sample import - from csv, import into sqlite3 db. Includes sequence info, nucleotide info, etc.
    input: sample sheet
    output: sqlite3 db
3. Import previously ran shapemapper folder
    input: shapemapper directory
    output: sqlite3 db (add directory)
4. Sample import (NMR)
    input: sample sheet
    output: sqlite3 db (nmr_reactions)

### `nerd fmod_calc`

1. Run shapemapper
    input: sample, shapemapper config
    output: sqlite3 db (with new entry)
2. Run spats
    input: XX
    output: XX
3. Run other tools fmod_calc tools

### `nerd degradation`

1. NMR deg measurements exponential timecourse
    input: NMR peak integrals (csv)
    output: exponential fit
2. Global time-course fits for probing data
    input: timecourses (rg_id, specify sites)
    output: global kdeg fit
3. Store to db (kinetic params) - need to do if want to do arrhenius

### `nerd adduction`

1. Independent kadd measurements
    input: NMR peak integrals (csv)
    output: ODE fit (add values to table)
2. Aggregate kadd from DMS
    input: kobs values at melted temps
    output: aggregated kobs values at melted temps (raw data points for arrhenius, add to table)
3. Store to db (kinetic params) - need to do if want to do arrhenius

### `nerd arrhenius`

1. Linear fit
    input: array of k values, corresponding temperatures
    output: LinearModel or params with errors
2. Store to db (Arrhenius)

### `nerd timecourse`

1. Independent time-course fits
    input: reaction group id
    output: non-linear fit or params
2. Refit with constrained deg
    input: reaction group id
    output: non-linear fit or params
3. Store to db (kinetic params, kobs)

#### Pipeline task `probe_timecourse`

- Configuration keys:
  - `engine` (default `python_baseline`) selects a registered timecourse engine.
  - `rounds` allows subsetting the legacy three-round workflow (`round1_free`, `round2_global`, `round3_constrained`).
  - `rg_ids` lists reaction-group identifiers to process; optional filters include `nt_ids`, `valtype`, `fmod_run_id`, and `min_points`.
  - `engine_options` and `global_metadata` pass through tunables such as `global_filters.r2_threshold` or precomputed `kdeg_initial`.
  - `overwrite` toggles whether previously stored fits in `probe_tc_fit_runs`/`probe_tc_fit_params` are replaced.
- Output artefacts:
  - JSON summaries are written to the task run directory (`results/rg_<rg_id>.json`) in a normalized schema (`engine_metadata`, `rounds`, `artifacts`).
  - Fit parameters and QC metrics persist to `probe_tc_fit_runs`/`probe_tc_fit_params`, using per-round fit kinds and optional global entries.
- Plugin architecture:
  - `python_baseline` reproduces the three-round pipeline with per-nt fits, optional global filtering, and constrained refits.
  - Stub engines `r_integration` and `ode_fit` register placeholders for future R/ODE implementations while emitting skipped-round metadata.

### `nerd tempgrad_fit`

- Modes:
  - `arrhenius`: log-linear fit for temperature-dependent rate constants (NMR or probing-derived).
  - `two_state_melt`: two-state melt curve fitting from temperature gradient data.
- Engines live under `pipeline/plugins/tempgrad` so alternative fitters (e.g., external R integration) can be swapped in.
- Results persist to `tempgrad_fit_runs`/`tempgrad_fit_params` using tall parameter storage; per-series fits include diagnostics and metadata captured from the request.
- Config exposes `series` overrides or DB-backed filters (e.g., buffer, reaction_type, species) and allows custom engine options such as temperature units or initial guesses.
- Probe timecourse data can be grouped dynamically (matching construct, buffer, probe, concentration, and RT conditions), filtered by melt threshold, and fit via weighted Arrhenius per nucleotide using `data_source: probe_tc`.

### `nerd calc_energy`

1. 2-state melting fit - K linear curve
    input: array of kobs values from probe_kinetic rates
    output: 2-state melt params
2. Single K calc
    input: kobs, kadd
    output: K value

## New sqlite tables

### sqlite table: probe_kinetic_rates

- Columns:
  - nmr_reaction_id
  - k value
  - k error
  - species (DMS, C8, etc.)

### sqlite table: nmr_reactions

- Columns:
  - reaction type (deg, add)
  - NMR machine (A600, A400, etc.)
  - temperature
  - scans
  - read time
  - probe
  - probe_conc
  - probe_solvent
  - substrate
  - substrate_conc
  - buffer
  - csv directory

### sqlite table: arrhenius_fits

- Columns:
  - Reaction type (degradation, adduction, kobs, K)
  - Data source (nmr, probe_global, probe_free)
  - slope
  - slope_err
  - intercept
  - intercept_err
  - rsq
  - dir to LinearModel (contains data)

## Example Usage

1. Import NMR deg samples with `nerd import`
2. Run exponential `nerd degradation`
3. Run deg arrhenius `nerd arrhenius`
4. Import DMS samples `nerd import`
5. Run time-course free fits `nerd timecourse`
6. Run global fits for deg `nerd degradation`
7. Run time-course refit with deg `nerd timecourse`
8. Run store kobs values `nerd timecourse store`
9. Run import NMR add samples with `nerd import`
10. Run ODE `nerd adduction`
11. Run add arrhenius `nerd arrhenius`
12. Run aggregate arrhenius on data `nerd arrhenius`
13. Run 2 state melt
14. (optional) Run K calc with given kadd value - HIV or P4P6
