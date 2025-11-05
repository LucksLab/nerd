# Probe Timecourse Fitting

Time-course probing captures RNA reactivity at multiple reaction times (e.g., 15 s, 30 s, …). Each time point is a quenched sample whose reactivity (`fmod`, `modrate`, or related metrics) is tracked per nucleotide. NERD’s timecourse engine converts those trajectories into kinetic parameters that summarize how quickly each nucleotide reacts.

- **Inputs:** per-nucleotide reactivity values across time points (often `modrate` from mutational counting).
- **Outputs:** `k_obs` (composite rate), `k_deg`, and baseline offsets (`fmod0`) stored in the SQLite database and JSON artifacts.

A separate guide covers how reactivity values are computed; this page focuses on the fitting workflow.

---

## Kinetic Model

Every nucleotide is fit to the standardized equation:

\[
r(t) = 1 - \exp\!\left[-\frac{k_{\text{obs}}}{k_{\text{deg}}} \left(1 - e^{-k_{\text{deg}} t}\right)\right] + r_0
\]

where:

- \(r(t)\) is the observed reactivity at time \(t\).
- \(k_{\text{obs}}\) is the composite rate \(\left(\frac{K}{K+1} \cdot \frac{k_{\text{add}} [P]_0}{k_{\text{deg}}}\right)\).
- \(k_{\text{deg}}\) is the degradation rate of the probe (shared across nucleotides within a molecule).
- \(r_0\) (`fmod0` in code) is a baseline offset capturing background signal.

The free round fits all three parameters per nucleotide; subsequent rounds constrain \(k_{\text{deg}}\) to enforce consistency.

---

## Three-Round Workflow

| Round | Purpose | Key Details |
|-------|---------|-------------|
| **Round 1 – Free** | Fit each nucleotide independently. | Provides initial estimates of \(k_{\text{obs}}\), \(k_{\text{deg}}\), \(r_0\); caches results for reuse. |
| **Round 2 – Global** | Share \(k_{\text{deg}}\) across the molecule. | Combines the best free fits (default R² cutoff ≈ 0.8) and performs a global nonlinear regression. |
| **Round 3 – Constrained** | Refit all nucleotides with \(k_{\text{deg}}\) fixed. | Produces the final parameter set used downstream. |

Why so many rounds?

1. **Stability:** Free fits bootstrap good starting points and flag problematic traces.
2. **Physical consistency:** Within the same molecule and condition, degradation should be common. The global round enforces that intuition.
3. **Final reporting:** Constrained fits yield a clean, per-nucleotide parameter set with a shared \(k_{\text{deg}}\).

You can request one or more rounds in the config (`rounds: [round1_free, round3_constrained]`), but the full sequence is most common.

---

## Engines & Extensibility

NERD ships with a Python baseline engine (`python_baseline`) that uses `lmfit`. Other engines are available or planned:

- `r_integration` (experimental): delegates fitting to R/NLME.
- Custom engines: register new backends by extending the `TimecourseEngine` interface.

Each engine plugs into the same CLI interface, so swapping is as simple as changing `engine: r_integration` in the config.

---

## Configuration Example

```yaml
run:
  label: tc_fit_4U
  output_dir: outputs

probe_timecourse:
  engine: python_baseline
  rounds: [round1_free, round2_global, round3_constrained]
  rg_ids: [42]             # Reaction group IDs from `create`
  valtype: modrate         # or fmod, etc.
  min_points: 3            # Require at least three time points
  engine_options:
    r2_threshold: 0.8      # Minimum R² to include in global round
```

**Key fields:**

- `rg_ids`: which reaction groups to fit.
- `valtype`: the reactivity metric to use. Defaults to `modrate`.
- `engine_options.r2_threshold`: filter for global fits (Tuning this is important for challenging data).
- `rounds`: any subset of `round1_free`, `round2_global`, `round3_constrained`.

The task reads per-nucleotide time series from the database (loaded by `mut_count` and `create` tasks) and submits them to the engine.

---

## Outputs & Interpretation

Each round generates:

- `nucleotides/<round>/...` entries in the JSON artifact with parameters and diagnostics.
- Database inserts into `probe_tc_fit_runs` (one row per round + nucleotide set) and `probe_tc_fit_params` (per-parameter rows for `kobs`, `kdeg`, `fmod0`, errors, QC metrics).

To inspect results:

1. Open the JSON artifact under `outputs/<label>/probe_timecourse/.../results`.
2. Query SQLite:
   ```sql
   SELECT rg_id, nt_id, param_name, param_numeric
   FROM probe_tc_fit_params
   WHERE param_name IN ('kobs', 'kdeg')
     AND fit_run_id IN (
       SELECT id FROM probe_tc_fit_runs
       WHERE fit_kind = 'round3_constrained'
     );
   ```
3. Join with `meta_nucleotides` or `probe_reactions` to annotate constructs, positions, or conditions.

Remember: the headline metric is **one \(k_{\text{obs}}\) per nucleotide per reaction group**. These composite rates feed directly into downstream analyses such as temperature-gradient fits or global kinetic comparisons.

---

## Best Practices

- Use derived samples (subsampling, filtering) to generate cleaner mutational profiles before fitting.
- Inspect free-fit diagnostics (`r2`, `chisq`) to spot nucleotides with noisy or non-monotonic traces.
- Provide prior knowledge when available:
  - Seed `log_kdeg_initial` in the config if degradation rates are known.
  - Restrict `nt_ids` or `valtype` to focus on specific bases.
- Consider running alternate engines (Python vs R) to validate difficult datasets.

With a consistent modeling approach, time-course fits become a reliable bridge from raw reactivities to interpretable kinetic fingerprints of your RNA construct.
