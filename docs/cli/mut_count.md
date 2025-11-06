# `nerd mut_count`

`nerd mut_count` orchestrates mutation counting on sequencing samples (e.g., FASTQ files). It stages FASTQs, runs the configured plugin (default SHAPEMapper), and records per-nucleotide mutation rates into `probe_fmod_values`.

```
nerd mut_count --config PATH/TO/config.yaml --db PATH/TO/nerd.sqlite
```

Without `--db`, `<run.output_dir>/nerd.sqlite` is used.

---

## Configuration Layout

```yaml
run:
  label: mutcount_demo
  output_dir: examples
  backend: local
  threads: 8

mut_count:
  plugin: shapemapper
  params:
    n_proc: 8
  samples:
    - sample_name: 001-EKC-fourU-37C-30s
      fq_dir: /data/runs/2024-09-01
      r1_file: sample_R1.fastq.gz
      r2_file: sample_R2.fastq.gz
      sequencing_run_name: 2024-09-01_SHAPE
      construct: 4U_wt
      buffer: pH6.5_A
      reaction_group: 37_1
      probe: dms
      probe_concentration: 0.016
      rt_protocol: SSIII
      temperature: 37
      replicate: 1
      reaction_time: 30
      treated: 2
```

`mut_count.samples` can be inline (as above) or a CSV path. Each row must reference existing constructs, buffers, and sequencing runs, or NERD will resolve them via `create` imports.

### Key Fields

| Field | Description |
|-------|-------------|
| `plugin` | Registered mutation-count plugin (e.g., `shapemapper`). |
| `params` | Plugin-specific options (e.g., `n_proc`, `amplicon`). |
| `samples` | List/CSV describing parent (and optional derived) samples to process. |
| `dry_run` | Skip plugin execution and only stage metadata (optional). |

Derived samples can be specified through `derived_samples` when you need to subsample or filter hits before counting mutations.

---

## Workflow Summary

1. Resolve sequencing runs, constructs, buffers, and builds FASTA targets.
2. Stage FASTQs (locally or remotely) and run the plugin command.
3. Parse the resulting mutation profiles and insert them into `probe_fmod_values`/`probe_fmod_runs`.

Probing reactions (rg labels) are linked automatically if `reaction_group` is supplied.

---

## Outputs

- Entries in `probe_fmod_runs` (run metadata, software version, arguments).
- Mutation rates in `probe_fmod_values` (per nt_id, reaction, valtype).
- Optional per-read histograms and staging artifacts under `<output_dir>/<label>/mut_count_latest/`.

---

## Examples

```bash
# Run SHAPEMapper on all samples listed in CSV
nerd mut_count --config examples/create_all_samples/create_samples.yaml --db nerd.sqlite

# Dry run to inspect staging
nerd mut_count --config configs/mutcount.yaml --db nerd.sqlite --dry-run
```

After `mut_count`, you can proceed to `probe_timecourse` or Arrhenius tasks using the inserted mutation rates.
