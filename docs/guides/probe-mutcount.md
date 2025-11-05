# Probe Mutational Counting

Time-course kinetics and temperature-gradient fits rely on accurate per-nucleotide reactivities. NERD’s `mut_count` task turns raw FASTQ files into modification rates by orchestrating an RT-MaP pipeline (ShapeMapper today, additional engines in the future).

---

## From FASTQ to Reactivity

1. **Start with parent sequencing samples** registered via `nerd run create`. Each sample knows where its paired FASTQs live and which reaction group it belongs to.
2. **Optionally define derived samples** (e.g., subsampled reads or single-hit filters) in the same config. The CLI will materialize these derivatives before counting.
3. **Run mutational counting** with ShapeMapper through:

   ```bash
   nerd run mut_count path/to/mutcount.yaml
   ```

   The config specifies which samples to process, ShapeMapper arguments, and staging options.

---

## RT-Stop vs RT-MaP

Two broad strategies exist for chemical probing readouts:

- **RT-stop**: quantify cDNA truncations (not yet implemented in NERD).
- **RT-MaP (Mutation Profiling)**: quantify mismatches and deletions introduced during reverse transcription. NERD currently supports this via ShapeMapper, enabling reactivity calculation for each nucleotide.

The abstraction keeps room for alternate tools—swap engines in future configs without rewriting the orchestrator.

---

## Deployment Options

Mutational counting jobs can run:

- **Locally** (default): stage FASTQs into a temporary working directory and execute ShapeMapper on the same machine.
- **HPC (SLURM)**: leverage the shared task runner to dispatch jobs to a SLURM cluster. Configure submission scripts in the `run` block or runner settings.

Regardless of execution backend, the CLI collects outputs, logs provenance, and imports results into SQLite.

---

## Derived Samples

Derived samples help control read depth or enforce per-read quality constraints before counting:

- **Subsampled**: use `seqtk sample` under the hood to create downsampled FASTQs (`kind: subsample`).
- **Single-hit filters**: call ShapeMapper once to identify reads with ≤1 modification, then use `seqtk subseq` to retain only those reads (`kind: filter_singlehit`).

Derived definitions live in the `derived_samples` section of the `create` config. Once defined, they can be referenced just like parent sample names in the `mut_count` config.

---

## Configuration Skeleton

```yaml
run:
  label: mutcount_run
  output_dir: outputs

mut_count:
  samples:
    - fourU_WT_65c_rep1_tp1
    - fourU_WT_65c_rep1_tp1__subsample-n10000-s42
  engine: shapemapper
  engine_options:
    shapemapper_path: /opt/shapemapper/bin/ShapeMapper.py
    threads: 8
  stage:
    include_traces: false
  filters:
    reaction_groups: [65_1]
```

*Key fields*:

- `samples`: parent or derived sample names.
- `engine`: currently `shapemapper`; future values can point to other RT-MaP implementations.
- `engine_options`: forwarded to the runner (binary path, threads, etc.).
- `stage`: optional staging tweaks (e.g., include or skip trace outputs).
- `filters`: constrain by reaction group, sample metadata, or derived-sample attributes.

---

## Database Storage

Mutational counting results land in two tables documented in the sample-organization guide:

### `probe_fmod_runs`

| Column | Meaning |
|--------|---------|
| `id` | Primary key. |
| `software_name`, `software_version` | Engine metadata (ShapeMapper version). |
| `output_dir` | Directory where raw tool outputs were staged. |
| `s_id` | Foreign key to `sequencing_samples` (parent sample). |
| `created_at` | Timestamp for the run. |

Each entry represents a single execution of the counting tool for a sample (parent or derived).

### `probe_fmod_values`

| Column | Meaning |
|--------|---------|
| `nt_id` | Foreign key to `meta_nucleotides` (construct + position). |
| `fmod_run_id` | Links back to `probe_fmod_runs`. |
| `rxn_id` | Optional foreign key to `probe_reactions` (for cross-referencing reaction conditions). |
| `valtype` | Type of value stored (`modrate`, `fmod`, etc.). |
| `fmod_val` | Reactivity or modification metric. |
| `read_depth` | Effective depth for that nucleotide. |
| `outlier` | Boolean flag for outlier handling. |

#### Example Query

```sql
SELECT
  m.construct_id,
  m.site,
  v.fmod_val,
  v.read_depth,
  v.outlier
FROM probe_fmod_values v
JOIN meta_nucleotides m ON m.id = v.nt_id
JOIN probe_fmod_runs r ON r.id = v.fmod_run_id
WHERE r.software_name = 'ShapeMapper'
  AND v.valtype = 'modrate'
  AND r.s_id = (
    SELECT id FROM sequencing_samples WHERE sample_name = 'fourU_WT_65c_rep1_tp1'
  );
```

This query pulls the per-nucleotide `modrate` series for a given sample, alongside read depth and outlier flags.

---

## Best Practices

- **Keep FASTQs organized** under the run-label directory (`outputs/<label>/fastqs`) for faster staging.
- **Validate derived sample definitions**—ensure parent sample names match those inserted by the `create` task.
- **Monitor log files** in `outputs/<label>/run_logs` for ShapeMapper warnings or read filtering summaries.
- **Re-run selectively** by filtering on reaction groups or sample subsets to avoid redundant processing.
- **Version bump**: include ShapeMapper version in configs so `probe_fmod_runs` records remain traceable.

Once mutation rates are in SQLite, you can pivot directly into probe timecourse fitting, temperature gradients, or custom analysis notebooks, confident that raw reads, derived samples, and reactivities are linked by shared identifiers.
