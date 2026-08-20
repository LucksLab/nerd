"""nerd sample-input helper: a local webapp for building nerd 'create' configs.

See design-spec.md in the project (or docs/sample_input_helper.md, once
written) for the full design. This package is intentionally decoupled from
nerd's core pipeline logic — it only authors config files (CSV/YAML) in the
layout nerd's own `create` task expects, plus (in a later iteration) can
invoke that task in-process.
"""
