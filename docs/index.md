# NERD: A Reproducible Pipeline for RNA Chemical Probing, Kinetic Modeling, and Energetic Analysis


NERD (Nucleic acid Energetics from reactivity data) is a reproducible, modular analysis pipeline for transforming high-dimensional chemical probing datasets, such as kinetic time-courses and temperature-gradient experiments, into quantitative models of RNA energetics. Designed to integrate every stage of analysis, from raw sequencing (FASTQ) or independent kinetic rate measurements to complex multi-parameter fits, NERD provides a unified framework for data curation, model fitting, and visualization.

---

## Why NERD?

- **Integrated data backbone** – every construct, buffer, condition, and fit is tracked within a single SQLite database, enabling reproducible analyses and effortless comparisons across experiments.
- **Pluggable fit engines** – swap between Python, R, or bespoke models without changing orchestration.
- **Scientist-friendly inputs** – CSV importers mirror bench sheets; automated QC catches missing references and inconsistent metadata before they propagate.

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

## CLI Overview

All commands are namespaced under `NERD`. The primary entry points are:

| Level | Purpose | Example |
|-------|---------|---------|
| `NERD create` | Populate metadata tables (constructs, buffers, reactions, samples). | `NERD create examples/create.yaml` |
| `NERD nmr_deg_kinetics` / `NERD nmr_add_kinetics` | Fit NMR degradation or adduction kinetics using registered trace files. | `NERD nmr_deg_kinetics configs/deg.yaml` |
| `NERD probe_timecourse` | Run free / global / constrained probing fits with Arrhenius hooks. | `NERD probe_timecourse configs/probe_tc.yaml` |
| `NERD tempgrad_fit` | Arrhenius or melt fits from NMR runs or probe timecourse outputs. | `NERD tempgrad_fit configs/tempgrad.yaml` |

All tasks share a standard YAML schema:

```yaml
run:
  label: my_analysis
  output_dir: results
  backend: local  # or slurm / ssh / custom

tempgrad_fit:
  mode: arrhenius
  data_source: probe_tc
  filters:
    construct: 4U_wt
    base: A
  engine_options:
    melt_threshold_c: 60
    weighted: true
```

The `run` block controls logging, execution backend, and output directories. The task block (here `tempgrad_fit`) defines domain-specific options.

---

## Quick Start Workflow

1. **Create metadata**  

   ```bash
   NERD create --config manuscript_pipeline/01_create_samples/create_meta.yaml \
     --db manuscript_pipeline/NERD.sqlite
   ```

   Optional follow-up configs (`create_probing_samples.yaml`, `create_nmr_add_samples.yaml`, etc.) add domain-specific rows.

2. **Register NMR traces**  
   Use `nmr_create` with `trace_files` entries so each reaction knows where its CSV traces live.

3. **Run fits**  
   - `NERD nmr_deg_kinetics …` for degradation rates.  
   - `NERD nmr_add_kinetics …` for NTP adduction curves (species like `ATP_C8` derived from trace metadata).
   - `NERD probe_timecourse …` for free / global / constrained kinetic parameters per nucleotide.

4. **Temperature-gradient analysis**  

   ```bash
   NERD tempgrad_fit --config examples/probe_tempgrad_arrhenius/config.yaml \
     --db manuscript_pipeline/NERD.sqlite
   ```

   This groups probe timecourse fits on the fly, filters by melt temperature, and performs weighted Arrhenius fits per nucleotide.

5. **Inspect results**  
   Every task writes JSON artifacts under `output_dir/label/<task>/latest/results` and logs insertions in `created_objects.log`. Explore the SQLite database with your favorite viewer (e.g., `sqlitebrowser` or `datasette`).

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

- **Examples**: `examples/` contains ready-to-run configs for NMR degradation, adduction, probe timecourse, and temperature-gradient fits.
- **Issues & enhancements**: File GitHub issues or PRs; the maintainers welcome field-specific engines (R, Bayesian) via the plugin registry.
- **Extending tasks**: New CLI steps simply subclass `Task` and leverage the shared logging, SQLite helpers, and runner infrastructure.

NERD is designed to evolve with your experiments. Start with the examples, keep everything in version control, and you’ll have a reproducible kinetic analysis pipeline ready for the next manuscript.
