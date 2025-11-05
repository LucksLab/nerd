# NERD

NERD (**N**ucleic acid **E**nergetics from **R**eactivity **D**ata) is a unified toolkit for extracting nucleotide-level thermodynamic and kinetic information from chemical probing experiments. It streamlines sample registration, mutation-count processing, and kinetic/melt-fit analysis, providing an organized path from raw sequencing data to time-course and temperature-gradient energetics.

## Quick Start

```bash
# create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# install NERD
pip install nerd-pipeline

# initialize a project from an example config
nerd run create demo_folder/01_create_samples/configs/create_meta.yaml

# execute a probe timecourse fit
nerd run probe_timecourse demo_folder/05_probe_tc_kinetics/configs/probe_tc.yaml
```

## Key Features

- Unified SQLite backbone linking constructs, buffers, reactions, fits, and artifacts
- CLI-based execution with consistent logging, caching, and reproducibility guarantees
- Pluggable fitting engines for probe timecourses, Arrhenius, and two-state melt models
- Rich filtering (construct, buffer, probe, nt_id, base) and outlier management in configs
- Automated metadata ingestion from CSVs and lab notebooks
- Portable outputs (JSON artifacts, logs, database entries) for downstream visualization
- Local and remote execution modes (local CPU, SLURM, SSH)
- Tutorial notebooks and guides for sample organization, NMR kinetics, probe analyses, and tempgrad fits

## Documentation Map

- **Getting Started**: [Sample organization & database layout](guides/sample-organization.md)
- **Guides**:
  - [Probe timecourse workflow](guides/probe-timecourse.md)
  - [NMR kinetics workflow](guides/nmr-kinetics.md)
  - [Temperature-gradient fitting](guides/tempgrad-fit.md)
- **CLI Reference**: `nerd run …`
  - [create](cli/create.md)
  - [mut_count](cli/mut_count.md)
  - [nmr_create](cli/nmr_create.md)
  - [nmr_deg_kinetics](cli/nmr_deg_kinetics.md)
  - [nmr_add_kinetics](cli/nmr_add_kinetics.md)
  - [probe_timecourse](cli/probe_timecourse.md)
  - [tempgrad_fit](cli/tempgrad_fit.md)
- **Configuration & Workflow Guide**: See guides above plus example YAML in `demo_folder/`
- **Architecture Overview**: [Sample organization & database layout](guides/sample-organization.md)
- **Examples**: Explore `demo_folder/` configs, `examples/nerd.sqlite`, and accompanying notebooks

## How It Works
>
> _[Pipeline diagram here: sample CSVs & raw data → NERD CLI tasks → SQLite database → fit engines → JSON/plots]_

1. Register constructs, buffers, reaction groups, and traces with `nerd run create`.
2. Quantify sequencing or NMR data via task-specific runners (`mut_count`, `nmr_*`).
3. Execute probe timecourse fits to derive kinetic parameters per nucleotide.
4. Run `tempgrad_fit` to combine kinetics across temperatures with Arrhenius or two-state models.
5. Visualize results using bundled notebooks or your own data science stack.

## Example Output (Optional)

```
manuscript_pipeline/05_probe_tempgrad_fit/tempgrad_fit/latest/results/tempgrad_result.json
```

## Citation
>
> _Please cite the NERD pipeline as: [placeholder—add citation here]._

## Links

- [GitHub Repository](https://github.com/luckslab/nerd)
- [Issue Tracker](https://github.com/luckslab/nerd/issues)
- [License](https://github.com/luckslab/nerd/blob/main/LICENSE)
