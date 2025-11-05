# NMR Kinetics

NERD supports two complementary NMR kinetic experiments that measure probe chemistry independent of sequencing:

- **Probe + water/buffer (degradation)** – quantifies how fast the probe hydrolyzes or self-reacts in the assay buffer.
- **Probe + nucleotide (adduction)** – quantifies formation of probe–NTP adducts that compete with probing.

Both assays monitor the integral of known chemical-shift peaks (for example, DMS at ~4 ppm, NTP C8 around 8 ppm). CSV traces capture peak integrals as a function of time, and the CLI converts those traces into rate constants that feed downstream models.

---

## Data Prep & Registration

Use `nerd run nmr_create` to register:

- The NMR reaction metadata (reaction type, substrate, buffer, concentrations).
- Trace files for each role:
  - `decay_trace` for degradation.
  - `peak_trace` (NTP reporter) and `dms_trace` for adduction.

Each trace is a CSV with at least:

```csv
time,peak_integral
0,0.98
10,0.91
...
```

For adduction, the column header is `peak` (representing normalized integral). Multiple traces must share the same time points; the task will intersect the times if one trace has additional rows.

---

## Degradation Fits (`nmr_deg_kinetics`)

Degradation experiments expose probe to buffer (no nucleotide). The measured peak integral decays exponentially as reactive probe is consumed. Internally, the default `lmfit_deg` plugin fits:

\[
I(t) = A \, e^{-t / \tau}
\]

where:

- \(I(t)\) is the integrated peak height (e.g., DMS).
- \(A\) is the initial amplitude.
- \(\tau\) is the decay time constant.

The first-order degradation rate is reported as:

\[
k_{\text{deg}} = \frac{1}{\tau}
\]

### Configuration Example

```yaml
run:
  label: water_deg
  output_dir: outputs

nmr_deg_kinetics:
  reaction_ids: [12]        # IDs from nmr_create
  plugin: lmfit_deg
  species: dms              # Optional override; defaults to reaction probe
```

### Outputs

- A JSON artifact under `outputs/<label>/nmr_deg_kinetics/.../results`.
- Database rows in `nmr_fit_runs` (`plugin = lmfit_deg`) and `nmr_fit_params` (parameters such as `k_value`, `tau`, `chisq`).

---

## Adduction Fits (`nmr_add_kinetics`)

Adduction experiments mix probe with a nucleotide triphosphate (NTP). Two peaks are tracked:

1. **NTP C8** – decreases as adduct forms.
2. **DMS (probe)** – decreases via both adduction and degradation pathways.

The model treats the system with three state variables:

- \(U(t)\): unreacted NTP (scaled from the C8 peak).
- \(S(t)\): free probe (scaled from the DMS peak).
- \(M(t)\): adduct product (derived from the complement to \(U\)).

The underlying ODEs (see `ode_lsq_ntp_add` plugin) are:

\[
\begin{aligned}
\frac{dU}{dt} &= -k_{\text{add}} \, U \, S \\
\frac{dS}{dt} &= -k_{\text{add}} \, U \, S - k_{\text{deg}} \, S \\
\frac{dM}{dt} &= k_{\text{add}} \, U \, S
\end{aligned}
\]

Initial conditions use the registered concentrations:

- \(U(0) = \text{ntp\_conc}\)
- \(S(0) = \text{dms\_conc}\) (defaults to 15.64 mM unless supplied)
- \(M(0) = 0\)

A scaling factor (`S_factor`) can rescale the DMS trace before fitting. The solver (`scipy.integrate.solve_ivp`) integrates the system, and the optimizer (lmfit) adjusts \(k_{\text{add}}\) (and optionally \(k_{\text{deg}}\)) to minimize residuals between simulated and observed curves.

### Configuration Example

```yaml
run:
  label: ntp_adduct
  output_dir: outputs

nmr_add_kinetics:
  reaction_ids: [21]
  plugin: ode_lsq_ntp_add
  plugin_options:
    k_add_init: 0.003     # Optional initial guesses
    k_deg_init: 0.001
  trace_roles:
    peak_trace: peak_trace
    dms_trace: dms_trace
```

### Outputs

- JSON artifact summarizing fitted trajectories and parameters.
- Database rows in `nmr_fit_runs` (`plugin = ode_lsq_ntp_add`) and `nmr_fit_params` (entries for `k_value` ← \(k_{\text{add}}\), `k_deg`, R², etc.).
- Diagnostics such as solver status and scaling factors stored in `nmr_fit_params` as text or numeric columns.

---

## Interpreting Results

1. **Inspect the JSON artifact** for fitted curves and metadata (species labels, concentrations).
2. **Query the database**:
   - `SELECT * FROM nmr_fit_runs WHERE plugin='lmfit_deg'` for degradation runs.
   - `SELECT * FROM nmr_fit_params WHERE param_name='k_value'` to compare rate constants across experiments.
3. **Cross-reference reaction metadata** in `nmr_reactions` to verify buffer, substrate, and concentration settings.

Combine these NMR-derived rates with probe time-course fits or temperature-gradient analyses to understand how probe chemistry, adduct formation, and reaction conditions interact across the full pipeline. The CLI keeps every step reproducible—from raw integrals to stored rate constants—so you can iterate on models without losing track of provenance.
