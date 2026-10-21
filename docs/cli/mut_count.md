# `nerd run mut_count`

`nerd mut_count` orchestrates mutation counting on sequencing samples (e.g., FASTQ files). It stages FASTQs, runs the configured plugin (default SHAPEMapper), and records per-nucleotide mutation rates into `probe_fmod_values`.

```
nerd mut_count --config PATH/TO/config.yaml --db PATH/TO/nerd.sqlite
```

Without `--db`, `<run.output_dir>/nerd.sqlite` is used.

---


## Discover groups and run samples without results

List all available reaction-group IDs and labels, with total sample counts and
counts of eligible samples without recorded mutation-count runs:

```bash
nerd db reaction-groups --db /path/to/nerd.sqlite
```

Use a listed ID or label in `mut_count.reaction_group` (or a list of IDs/labels).
To select all samples with no existing row in `probe_fmod_runs`, use:

```yaml
mut_count:
  plugin: shapemapper
  missing_only: true
  params:
    n_proc: 8
```

Run this configuration with `nerd run mut_count config.yaml`. Database and
execution settings follow the normal project/configuration rules. With no explicit
sample or group selection, pending samples are automatically scheduled as one
independent job per reaction group. Samples without a reaction group run in a
separate ungrouped job; groups with no pending samples are omitted. Samples marked
for dropping are excluded. Existing runs from any mutation-count plugin count as
already processed. If `samples` or `reaction_group` is also supplied,
`missing_only` filters that selection instead of selecting the entire database.

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
      fq_source: local
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

## Count mutations using one read

Select either read from a sample's FASTQ pair with `unpaired_mode`:

```yaml
mut_count:
  plugin: shapemapper
  reaction_group: 1
  unpaired_mode: R1  # R1 or R2
  params:
    n_proc: 8
```

ShapeMapper receives `--modified --U <selected FASTQ>` instead of
`--R1 <R1 FASTQ> --R2 <R2 FASTQ>`. Omit `unpaired_mode` (or set it to
`null`) to keep paired-read counting. The option may also be placed under
`params`; that value takes precedence. Values other than `R1`, `R2`, or
`null` are rejected. Sample metadata remains unchanged. Derived-sample
preparation still uses both parent reads; mutation counting uses the selected
derived read.
