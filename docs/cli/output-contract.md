# Task output and JSON contract

`nerd run`, `nerd task show`, `nerd task list`, `nerd task wait`, and task
lifecycle commands support `--json`. In JSON mode, standard output contains one
JSON document and no progress text. File logs remain complete; interactive
progress and diagnostics use standard error when enabled.

## Task summary schema (v1.0)

Every run or single-task lifecycle result uses these top-level fields, in this
order: `schema_version`, `status`, `workflow`, `task_id`, `source_task_id`,
`label`, `plugin`, `engine`, `version`, `started_at`, `ended_at`,
`duration_seconds`, `timings`, `counts`, `metrics`, `warnings`, `failures`,
`artifacts`, `log_path`, `database_path`, and `next_actions`.

- Times are ISO-8601 strings; elapsed values are seconds measured with a
  monotonic clock for a live run or derived from persisted scheduler timestamps.
- `counts` contains integer aggregates. The common keys are `attempted`,
  `succeeded`, `failed`, and `skipped`; workflow-specific evidence-backed keys
  are included alongside them.
- `metrics` contains JSON-native scalar, list, or object values. It never
  contains exception objects or database rows.
- `warnings` and `failures` contain objects with `code`, `message`, optional
  `item`, and JSON-native `details`.
- `artifacts` contain `kind`, `path`, optional `label`, and optional `exists`.
- A cached result has `status: "cached"` and identifies the completed source in
  `source_task_id` when known.

Fields may be `null` when the selected engine, scheduler, or historical record
does not expose the value. Within schema version 1.x, field meanings are stable;
new optional workflow metrics may be added. A future incompatible change will
increment the major schema version.

`task list --json` returns an object with `schema_version` and a `tasks` array of
compact task records rather than one task summary.

## Examples

```bash
nerd run mut_count configs/mut_count.yaml --json > summary.json
nerd run mut_count configs/mut_count.yaml --detach --profile quest --json
nerd task show 42 --db results/nerd.sqlite --json
nerd task wait 42 --collect --db results/nerd.sqlite --json
```

Human output is now a concise completion summary for synchronous scientific
runs. Existing durable-task human fields remain available for scripts written
against the Phase 1/2 interface. Use `--json` for a versioned machine contract.

Known evidence gaps are reported conservatively. In particular, the current
derived-sample upsert helper does not distinguish create from update, so the
create workflow reports `derived_samples_processed` rather than guessing.
