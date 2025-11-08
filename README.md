# NERD: Extensible Toolkit for Organization and Kinetic Analysis of Chemical Probing Data

NERD (**N**ucleic acid **E**nergetics from **R**eactivity **D**ata) is a reproducible, extensible analysis pipeline for transforming high-dimensional chemical probing datasets, such as kinetic time-courses and temperature-gradient experiments, into quantitative models with underlying RNA energetics. Designed to integrate every stage of analysis, from raw sequencing (FASTQ) or independent kinetic rate measurements to complex multi-parameter fits, NERD provides a unified framework for data curation, model fitting, and visualization.

Check out <https://luckslab.github.io/nerd/> for detailed CLI documentation and high-level description of key components.

---

## Why NERD?

- **Integrated data backbone** – every construct, buffer, condition, and fit is tracked within a single SQLite database, enabling reproducible analyses and effortless comparisons across experiments.
- **Pluggable fit engines** – swap between Python, R, or bespoke models without changing orchestration.
- **Scientist-friendly inputs** – CSV importers mirror bench sheets, while YAML configs capture analysis choices (outliers, filters, engine options) so decisions stay versioned and reproducible.

---

## How to install

```bash
# clone the repository
git clone https://github.com/<lab>/NERD.git
cd NERD

# optional: use a fresh virtual environment
python -m venv .venv
source .venv/bin/activate

# install in editable mode with extras
pip install -e .[dev]
```

NERD ships with a lightweight dependency set (Typer, SQLite, NumPy, Pandas, lmfit). For heavy fits you can point the CLI at SLURM or SSH targets, but the default runs locally.

---

## Core Nerd CLI Functions

1. **Sample creation & organization** – `nerd run create` ingests constructs, buffers, reaction conditions, and FASTQ paths into the SQLite backbone. Declare `derived_samples` in the same config to spin up subsampled or filtered sequencing inputs on the fly.
2. **NMR kinetic analysis** – Pair `nerd run nmr_create` (trace registration) with `nerd run nmr_deg_kinetics` or `nerd run nmr_add_kinetics` to generate degradation and adduction fits.
3. **Mutational counting pipeline** – `nerd run mut_count` stages the FASTQs, dispatches the selected counter (ShapeMapper supported today), and writes counts plus QC metadata back to the database, including any derived samples.
4. **Probe time-course fitting** – `nerd run probe_timecourse` executes free, global, and constrained kinetic rounds, centralizing fit metadata so results stay linked to reaction groups and nucleotides.
5. **Temperature-gradient analysis** – `nerd run tempgrad_fit` consumes NMR or probe outputs to fit melted Arrhenius or two-state models across constructs, buffers, and bases.

Every command follows the same pattern:

```bash
nerd run <step> path/to/config.yaml
```

Each config shares a small `run` header for bookkeeping:

```yaml
run:
  label: my_analysis
  output_dir: results
  backend: local  # or slurm / ssh / custom
```

Add a task-specific block (e.g., `create`, `mut_count`, `probe_timecourse`) to declare inputs and engine options, and NERD handles staging, logging, and database updates.

---

## Quick Start Workflow

1. **Register samples and metadata**

   Sequence-probing and NMR inputs are declared via YAML configs. Each run logs to SQLite and creates a reproducible record of constructs, buffers, and raw inputs.

   ```bash
   # chemical probing samples
   nerd run create demo_folder/01_create_samples/configs/create_meta.yaml

   # NMR samples (adduction & degradation)
   nerd run create demo_folder/01_create_samples/configs/create_nmr_add_samples.yaml
   nerd run create demo_folder/01_create_samples/configs/create_nmr_deg_samples.yaml
   ```

2. **Fit independent kinetic measurements (NMR)**

   Estimate degradation and adduction rate constants, then perform Arrhenius fits across temperatures:

   ```bash
   nerd run nmr_deg_kinetics demo_folder/02_nmr_kinetics/fit_deg.yaml
   nerd run nmr_deg_kinetics demo_folder/02_nmr_kinetics/fit_add.yaml

   # temperature dependence
   nerd run nmr_deg_kinetics demo_folder/03_nmr_arrhenius/tempgrad_deg.yaml
   nerd run nmr_deg_kinetics demo_folder/03_nmr_arrhenius/tempgrad_atp_c8.yaml
   ```

3. **Count mutations from sequencing data**

   ```bash
   nerd run mut_count demo_folder/04_run_mutcounts/configs/mut_count_shapemapper.yaml
   ```

   ShapeMapper runs out‑of‑the‑box; configs can swap counters or tweak QC rules.

4. **Fit chemical‑probe time‑courses**

   ```bash
   nerd run probe_timecourse demo_folder/05_probe_tc_kinetics/configs/probe_tc.yaml
   ```

   Executes free, global, and constrained fits; all metadata is written to SQLite.

5. **Fit temperature gradients for probe data**

   ```bash
   nerd run tempgrad_fit demo_folder/06_probe_tempgrad_fit/configs/config.yaml
   ```

   Groups time‑course fits, filters by melt temperature, and performs weighted Arrhenius analysis per nucleotide.

Outputs are written to `output_dir/label/<task>/latest/results`, and all database insertions are tracked in `created_objects.log`. Browse the SQLite file with `sqlitebrowser`, `datasette`, or the built‑in CLI tooling.

---

## Configuration Tips

- **CSV vs YAML**: The `create` task automatically treats strings like `samples: probing_samples.csv` as sheets. It resolves constructs, buffers, and sequencing runs by name, and validates every reference before inserting reactions.
- **Trace metadata**: When registering NMR traces, you can tag species directly:

  ```yaml
  trace_files:
    peak_trace:
      path: nmr_ntp_adduction_data/ATP_peak_percentages/20_1_peak8.csv
      species: ATP_C8
    dms_trace:
      path: nmr_ntp_adduction_data/ATP_peak_percentages/20_1_peakDMS.csv
      species: ATP_DMS
  ```

- **Grouping flexibility**: `tempgrad_fit` honors `filters` like `construct`, `buffer`, `probe`, `base`, and `nt_id`. Future options (`group_by`, `aggregate`) will let you combine nucleotides by base or across constructs.

---

## Getting Help & Contributing

- **Examples**: `demo_folder/` contains ready-to-run configs for NMR degradation, adduction, probe timecourse, and temperature-gradient fits.
- **Issues & enhancements**: File GitHub issues or PRs; the maintainers welcome field-specific engines (R, Bayesian) via the plugin registry.
- **Extending tasks**: New CLI steps simply subclass `Task` and leverage the shared logging, SQLite helpers, and runner infrastructure.

## License & Copyright

NERD is released under the [Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International License](LICENSE.md).  
© Lucks Lab, 2025. For attribution details and allowed uses, see `LICENSE.md`.
