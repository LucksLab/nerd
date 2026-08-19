# `nerd run nmr_kinetic_fit`

`nmr_kinetic_fit` is the single public workflow for fitting NMR degradation
and adduction kinetics after reactions and traces have been registered with
`nmr_create`.

```bash
nerd run nmr_kinetic_fit PATH/TO/config.yaml --db PATH/TO/nerd.sqlite
```

## Configuration

Select exactly one fit family with `fit_type`.

```yaml
run:
  label: nmr_fit
  output_dir: results

nmr_kinetic_fit:
  fit_type: degradation  # or adduction
  reaction_ids: [1, 2]
  search_roots:
    - ../trace-data
```

For degradation, the default plugin is `lmfit_deg` and each reaction needs a
registered `decay_trace`. For adduction, the default plugin is
`ode_lsq_ntp_add` and each reaction needs registered `peak_trace` and
`dms_trace` inputs. Both fit families accept their existing `plugin`,
`plugin_options`, `fit_params`, trace-role, species, and substrate settings
inside the unified block.

Artifacts are written beneath
`<output_dir>/<label>/nmr_kinetic_fit/`, and fit records are stored in
`nmr_fit_runs` and `nmr_fit_params`.
