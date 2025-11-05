# Sample Organization & Database Layout

NERD persists every piece of metadata, analysis configuration, and fit result inside a single SQLite database (`nerd.sqlite`). A lightweight relational store gives us a few big wins:

- Transactions keep inserts and updates consistent, even when multiple CLI tasks run in sequence.
- Foreign keys model the lab reality—constructs, buffers, reaction groups, and fit runs link explicitly.
- The file is portable: copy `nerd.sqlite` alongside figures and configs to reproduce an analysis.

Think of the database as the canonical record of an experiment. CSVs define starting metadata, FASTQs supply raw reads, fits generate parameters, but the SQLite file ties everything together.

---

## Schema at a Glance

Tables are grouped by function:

- **core\_*** – task orchestration (`core_tasks`, `core_task_members`).
- **meta\_*** – reusable metadata (constructs, buffers, nucleotides).
- **sample creation** – sequencing runs, samples, reaction groups, and probe reactions.
- **probe\_fmod\*** – outputs from mutational counting pipelines (ShapeMapper, etc.).
- **probe\_tc\*** – free/global/constrained time-course fit runs and parameters.
- **nmr\_*** – NMR reaction definitions, trace files, and kinetics fits.
- **tempgrad\_*** – Arrhenius and melt fits derived from probe or NMR data.

The sections below sketch how each area fits together so you can quickly orient in the database browser of your choice.

---

## Core Task Tables

| Table | Purpose |
|-------|---------|
| `core_tasks` | One row per CLI task execution (e.g., `create`, `mut_count`). Stores status, config hash, timestamps. |
| `core_task_members` | Links a task to specific entities (sample IDs, reaction groups) to capture scope. |
| `core_cached_tasks` | Records when a task reuses prior results via the caching mechanism. |

These tables are the audit trail. If the CLI reports that a run completed, you’ll see a matching row with `state='completed'`. Failures, reruns, and cached executions are all tracked here.

---

## Metadata (`meta_…`)

| Table | Key Columns | Notes |
|-------|-------------|-------|
| `meta_constructs` | `disp_name`, `family`, `version` | Unique construct definitions. |
| `meta_nucleotides` | `construct_id`, `site`, `base` | Per-nucleotide metadata; generated if not provided. |
| `meta_buffers` | `name`, `pH`, `composition` | Buffer recipes referenced by reactions. |

Constructs and buffers are shared resources. When you import new samples, the CLI resolves construct/display names back into these tables. Deleting a construct cascades to its nucleotides thanks to foreign keys.

---

## Sample Creation & Organization

| Table | Purpose |
|-------|---------|
| `sequencing_runs` | One row per instrument run (MiSeq, NovaSeq, etc.). |
| `sequencing_samples` | Parent samples pointing to FASTQ directories/files. |
| `sequencing_derived_samples` | Virtual samples produced by subsampling/filtering operations. |
| `probe_reaction_groups` | Labels that group related probe reactions (e.g., time course replicates). |
| `probe_reactions` | Connects a sequencing sample to constructs, buffers, probe chemistry, and reaction conditions. |

Typical flow: `create` inserts sequencing runs and samples, `probe_reaction_groups` tracks the lab-defined grouping (e.g., `65_1`), and `probe_reactions` attaches experimental context (temperature, probe concentration). Derived samples record how a `mut_count` task generated filtered FASTQs; parent-child relationships let downstream tasks trace provenance.

Foreign-key highlights:

- `probe_reactions.s_id → sequencing_samples.id`
- `probe_reactions.construct_id → meta_constructs.id`
- `probe_reactions.buffer_id → meta_buffers.id`

---

## Mutational Counting (`probe_fmod_*`)

| Table | Purpose |
|-------|---------|
| `probe_fmod_runs` | Each invocation of a mutational counting tool for a sample. |
| `probe_fmod_values` | Per-nucleotide reactivity (e.g., `modrate`) plus depth and QC flags. |

When you run `nerd run mut_count`, the task registers a `probe_fmod_runs` row, inserts per-nucleotide values, and marks outliers. `probe_fmod_values.nt_id` points back to `meta_nucleotides`, so all reactivity data stays linked to constructs.

---

## Probe Time-Course Fits (`probe_tc_*`)

| Table | Purpose |
|-------|---------|
| `probe_tc_fit_runs` | Catalog of free/global/constrained fits. Captures round type, reaction group, valtype, and optional fmod run linkage. |
| `probe_tc_fit_params` | Tall table storing fit parameters, errors, or QC metrics keyed by `fit_run_id`. |

Each time-course task can spawn multiple rounds; the CLI writes one `probe_tc_fit_runs` row per round or scope. Parameters (e.g., `log_kobs`, `log_kdeg`) are stored as rows in `probe_tc_fit_params`, making it easy to extend with new metrics.

---

## NMR Experiments (`nmr_*`)

| Table | Purpose |
|-------|---------|
| `nmr_reactions` | Experiment definition (reaction type, substrate, buffer, replicate info). |
| `nmr_trace_files` | File registry per reaction and role (decay trace, peak trace, DMS trace). |
| `nmr_fit_runs` | Each kinetic model execution, including plugin name and status. |
| `nmr_fit_params` | Tall parameter storage for kinetics outputs (`k_value`, `k_error`, R², etc.). |

The workflow:

1. `nmr_create` registers trace files and reaction metadata.
2. `nmr_deg_kinetics` / `nmr_add_kinetics` draw inputs, stage trace files into a run directory, and execute the named plugin.
3. Results land in `nmr_fit_runs` (one row per reaction fit) with parameters normalized into `nmr_fit_params`.

---

## Temperature-Gradient Fits (`tempgrad_*`)

| Table | Purpose |
|-------|---------|
| `tempgrad_fit_runs` | Top-level record for each Arrhenius or two-state fit execution. |
| `tempgrad_series` | Series metadata (construct, buffer, probe) for individual fit curves. |
| `tempgrad_series_params` | Parameters per series (activation energy, slopes, intercepts). |
| `tempgrad_series_diagnostics` | Diagnostics like R², RMSE, weight usage. |

Arrhenius runs use per-series regression; two-state fits may share baselines and store additional free energy estimates. Every series references the reaction group or source data so you can trace the input pipeline.

---

## Navigating the Database

Open `nerd.sqlite` in your favorite viewer (DB Browser for SQLite, Datasette, TablePlus). A few orientation tips:

- **Start with `core_tasks`** to see what ran. Use `task_id` to jump to downstream artifacts (`nmr_fit_runs.task_id`, provenance logs under the run directory).
- **Follow foreign keys**. Most tables use integer IDs with descriptive columns (`rg_label`, `sample_name`) for human readability.
- **Tall parameter tables** (`*_fit_params`) include both numeric and text columns. Filter by `param_name` to extract specific values.
- **JSON artifacts**. Tasks also dump companion JSON files in `run/output_dir/<label>/<task>/create___cfg-.../results`. The SQLite entries are designed to point back to those files via paths stored in metadata tables.

With this structure your analyses are reproducible: rerunning a task updates entries, derived tables stay in sync, and every parameter ties back to the sample and construct that produced it. Use the CLI to orchestrate; use the database to audit, explore, and share.
