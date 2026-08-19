# NERD CLI UX audit — `dev` after phases 1/2

## Executive assessment

Phase 1/2 delivered a credible durable scheduler and a safer ShapeMapper container path: remote collection is explicit, provenance is recorded, ShapeMapper failures are no longer masked, and mutation-count output is scientifically validated. The surrounding CLI still behaves like an internal API.

Highest priorities:

- **P0:** project/database context is ambiguous and inspection can silently create an empty database.
- **P0:** documented `probe_timecourse` is rejected; the enum accepts `probe_tc_kinetics`.
- **P0:** synchronous output-consumption failures do not receive the scheduler path's durable failure transition.
- **P1:** scientific workflows, scheduler operations, and image maintenance are mixed at the top level.
- **P1:** task/plugin output lacks success rates, failures, QC summaries, timings, and next actions.
- **P1:** path resolution depends on the invocation directory, while YAML carries project, infrastructure, and scientific configuration together.

## 1. User-facing UX and command discoverability

### P0

1. **Documentation and executable help disagree.** `RunStep` exposes `probe_tc_kinetics`, while the registry also has `probe_timecourse`; the enum prevents the documented form from reaching the registry. See [`nerd/cli.py:34`](nerd/cli.py#L34), [`nerd/pipeline/tasks/__init__.py:34`](nerd/pipeline/tasks/__init__.py#L34), and [`README.md:41`](README.md#L41).

2. **Read-only discovery creates misleading state.** `_scheduler_connection()` falls back to `Path("nerd.sqlite")`, connects, and initializes the schema. See [`nerd/cli.py:143`](nerd/cli.py#L143). Running `nerd ls` in a clean temporary directory created a 270 KB database and said “No tasks found.” It should instead report that no project/database was found.

3. **Global option placement is brittle.** `nerd status 1 --db PATH` is rejected; only `nerd --db PATH status 1` works. Project/database selectors should be accepted consistently on lifecycle commands.

### P1/P2

- **P1:** Help exposes internal identifiers such as `mut_count` and `nmr_deg_kinetics` without explaining prerequisites or products.
- **P1:** `ls` says “List available runs” but displays scheduler tasks and attempts. Rename it to `task list`; see [`nerd/cli.py:342`](nerd/cli.py#L342).
- **P1:** Add `config validate`, `config show --resolved`, plugin discovery, and structured output.
- **P2:** Add `--version`, stable aliases, `--no-color`, and completion metadata.
- Existing CLI tests check that selected words occur in help, not that documented commands work or outside-project behavior is safe. See [`tests/test_cli.py:76`](tests/test_cli.py#L76).

Compared with **gh**, NERD lacks noun-based grouping and consistent JSON output; compared with **uv**, it lacks explicit project discovery; compared with **Docker**, common and advanced commands are not separated; compared with **Claude Code**, persisted context and human/JSON/streaming modes are absent; compared with **Ollama**, the verb vocabulary and timing output are less compact and predictable.

## 2. Underlying command/group structure and mental model

NERD has three distinct concepts currently flattened together:

1. a **scientific workflow**: import, count, fit, exclude;
2. a durable **task execution**: submit, inspect, log, cancel, collect, retry;
3. **infrastructure**: executor profiles and container images.

Merge `run` and `submit` into one user model:

```text
nerd run WORKFLOW [CONFIG] [--detach] [--profile NAME]
```

Keep `submit` temporarily as a hidden compatibility alias. `--detach` matches the familiar Docker/Claude background model.

### Target hierarchy

```text
nerd
├── init [PATH]
├── run WORKFLOW [CONFIG]
│   ├── sample-import
│   ├── sample-exclude
│   ├── mutation-count
│   ├── nmr-import
│   ├── nmr-fit-degradation
│   ├── nmr-fit-adduction
│   ├── timecourse-fit
│   └── temperature-fit
├── task
│   ├── list
│   ├── show ID
│   ├── logs ID [--follow]
│   ├── wait ID [--collect]
│   ├── collect ID
│   ├── cancel ID
│   └── retry ID
├── config
│   ├── init WORKFLOW
│   ├── validate [FILE]
│   └── show [FILE] [--resolved]
├── plugin
│   ├── list
│   ├── info NAME
│   └── doctor NAME
├── image
│   ├── inspect NAME
│   └── prepare NAME
└── db
    ├── path
    └── info
```

The first help screen should emphasize `init`, `run`, `task`, and `config`; plugin, image, and raw database operations belong under advanced commands.

### P0 lifecycle issue

Synchronous execution calls `consume_outputs()` and only afterward marks the task complete, without recording consumption exceptions as failures. See [`nerd/pipeline/tasks/base.py:286`](nerd/pipeline/tasks/base.py#L286). The scheduler path correctly records `validation_failed` and task failure. See [`nerd/scheduler/service.py:201`](nerd/scheduler/service.py#L201). Both paths should use one lifecycle service so status and retry semantics cannot diverge.

## 3. Commands/options to prune, rename, merge, or hide

| Priority | Current | Target |
|---|---|---|
| P1 | `run` + `submit` | `run [--detach]`; hidden `submit` alias |
| P1 | `ls` | `task list`; hidden `ls` alias |
| P1 | `status` | `task show`; reserve top-level `status` for project summary |
| P1 | `logs`, `cancel`, `collect`, `retry` | Move below `task`; add `logs --follow`, `wait --collect` |
| P1 | `doctor CONFIG` | `plugin doctor shapemapper`; general `nerd doctor` checks the whole project |
| P2 | `prepare-image` | `image prepare shapemapper`; hide from common help |
| P1 | `create` | `sample-import` |
| P1 | `nmr_create` | `nmr-import` |
| P1 | `drop` | `sample-exclude` |
| P0 | implicit drop replacement | Require `--replace-selection`: current code resets all unlisted samples to `to_drop=0`; see [`drop.py:134`](nerd/pipeline/tasks/drop.py#L134) |
| P2 | `force_run` + `overwrite` | CLI `--force` and one canonical config key |
| P2 | `run.backend`, `run.executor`, `run.executor_profile` | Canonical `executor`; older names become migration aliases; see [`profiles.py:28`](nerd/scheduler/profiles.py#L28) |
| P1 | `ode_fit` | Hide as unavailable: it returns skipped rounds while the enclosing task may appear successful; see [`ode_fit.py:16`](nerd/pipeline/plugins/timecourse/ode_fit.py#L16) |
| P2 | `spats` | Hide until implemented and registered; see [`spats.py:1`](nerd/pipeline/plugins/mutcount/spats.py#L1) and [`mutcount/__init__.py:9`](nerd/pipeline/plugins/mutcount/__init__.py#L9) |

Accept friendly hyphenated names and old underscore names immediately, but display only canonical names in help.

## 4. Logging/output for every plugin and task

The phase-2 mutation-count summary is the best current model: it reports profiles and ingested runs and rejects incomplete validation. See [`mut_count.py:1020`](nerd/pipeline/tasks/mut_count.py#L1020). Generalize it into a shared result contract.

Every execution should finish with status, elapsed time, task ID, label, plugin/engine version, attempted/succeeded/failed/skipped counts, success rate, scientific QC, grouped warnings, output/log/database paths, and a next action. `--json` should expose the same stable fields.

| Task/plugin family | Required summary |
|---|---|
| Sample import | Constructs, buffers, sequencing runs, samples, derived samples: created/updated/unchanged/failed; unresolved references |
| NMR import | Reactions imported; traces expected/found/missing; unresolved paths; duplicates updated |
| `lmfit_deg` | Fits attempted/succeeded/failed; success rate; R² range/median; chi-square; convergence/bound warnings |
| `ode_lsq_ntp_add` | Same, split by species/substrate; missing trace pairs and integration failures |
| ShapeMapper | Samples requested/completed; profiles found; runs and nucleotide values ingested; read-depth range; mismatches; failed names |
| Timecourse baseline | Reaction groups, nucleotides, rounds attempted/succeeded/skipped; fit rate per round; outliers applied/unmatched; convergence failures |
| Timecourse R integration | Same plus R version/runtime, process failures, and artifacts |
| Timecourse ODE | Fail preflight as unavailable; never complete successfully with all rounds skipped |
| Arrhenius Python/R | Series attempted/fitted/failed; points included/excluded; fit-quality distribution; invalid rates; runtime |
| Bayesian Arrhenius | Same plus divergences, R-hat/ESS warnings, and sampling duration |
| Two-state melt | Series attempted/fitted; retained temperature points; Tm/bound warnings; fit quality |
| Sample exclude | Requested/found/changed/already set/missing/reset; prominent replacement warning |
| Scheduler | Task/scheduler IDs; transition; queue/run/collection durations; executor; valid next actions |
| Container readiness | Checks passed/total; host, architecture, runtime, immutable digest/SIF; remediation per failure |

### Revised output pattern

```text
✓ mutation-count completed in 2m 14s

Task       42 · experiment-7 · ShapeMapper 2.x
Samples    48/50 succeeded (96.0%)
Profiles   48 found · 2 missing
Database   7,384 nucleotide values inserted
Warnings   2 samples failed scientific validation
Output     /…/mutation-count/…/artifacts
Log        /…/command.log

Failed: sample_17, sample_42
Next: nerd task logs 42
      nerd task retry 42 --only-failed
```

Long local commands currently redirect all child output to `command.log`, so users see only start and exit messages. See [`local.py:41`](nerd/pipeline/runners/local.py#L41). Stream concise progress while keeping the complete log. Pure-Python tasks should not print “Command completed successfully.” Timecourse and temperature-fit persist results without aggregates; see [`timecourse.py:261`](nerd/pipeline/tasks/timecourse.py#L261) and [`tempgrad_fit.py:124`](nerd/pipeline/tasks/tempgrad_fit.py#L124). NMR tasks similarly lack aggregate fit quality; see [`nmr_deg_kinetics.py:89`](nerd/pipeline/tasks/nmr_deg_kinetics.py#L89).

Add `--quiet`, `--verbose`, `--json`, `--no-color`, and `--log-file` consistently. When JSON is requested, human progress belongs on stderr and JSON on stdout.

## 5. Input/config simplification and predictable outside-project behavior

### P0: deterministic project discovery

Use one precedence everywhere:

```text
--project / --db
→ nearest .nerd/project.toml in cwd or parent
→ NERD_PROJECT / NERD_DB
→ config file's parent
→ actionable error
```

Inspection must never create a database. Use an error such as:

```text
No NERD project found.
Run `nerd init`, pass `--project PATH`, or set NERD_PROJECT.
```

### P0: config-relative paths

Resolve every relative path against the config file’s parent and normalize it once. Currently:

- `run.output_dir` is resolved from `cwd` in [`nerd/cli.py:102`](nerd/cli.py#L102).
- Scheduler submission independently resolves it from `cwd` in [`service.py:135`](nerd/scheduler/service.py#L135).
- CSV discovery searches output-dependent roots and `Path.cwd()` in [`create.py:200`](nerd/pipeline/tasks/create.py#L200).
- `nt_info` uniquely assumes `<output>/<label>/configs` in [`create.py:161`](nerd/pipeline/tasks/create.py#L161).
- `run` defaults its database around `.` while the task base defaults outputs to `nerd_output`; compare [`cli.py:102`](nerd/cli.py#L102) and [`base.py:73`](nerd/pipeline/tasks/base.py#L73).

Persist a fully resolved config snapshot so collection never depends on the original working directory.

### P1: reduce YAML burden

- Add `nerd init` and `nerd config init WORKFLOW`.
- Put stable database, output-root, and executor settings in the project file.
- Limit per-run configs to scientific selection and model parameters.
- Convert the 198- and 427-line NMR demo YAML files into reaction CSV/TSV sheets plus a small reusable fit config.
- Support common direct forms:

```text
nerd run mutation-count --samples samples.csv
nerd run timecourse-fit --reaction-group 12 --engine baseline
nerd run nmr-fit-degradation --reaction 31 --reaction 32
```

- Support repeatable `--set key=value` for occasional overrides.
- Add schema-backed validation, unknown-key suggestions, and defaults. `load_config()` currently only performs `yaml.safe_load()`. See [`nerd/utils/config.py:10`](nerd/utils/config.py#L10).
- Add `config show --resolved` to display the database, outputs, samples, engine, executor, and image before mutation/submission.

### P2

Add `config migrate FILE` and one-release-cycle warnings for renamed keys and commands. Preserve compatibility, but document only the canonical schema.

## Recommended delivery order

1. **P0:** deterministic project/database discovery, documented aliases, unified failure transitions, safe sample-exclusion semantics.
2. **P1:** `task` grouping, friendly workflow names, `--detach`, and hidden compatibility aliases.
3. **P1:** shared result contract, task/plugin summaries, timings, progress, `--json`, and next actions.
4. **P1:** schema validation, config-relative paths, project file, `init`, and resolved-config inspection.
5. **P2:** migrations, completion metadata, `logs --follow`, `task wait`, plugin capability listing, and advanced help grouping.

## Validation performed and limits

- Inspected the actual dirty `dev` worktree containing phase 1/2 changes; implementation code was not modified.
- Ran top-level and relevant subcommand help from the repository’s available Conda environment.
- Reproduced outside-project database creation and position-sensitive `--db` behavior in `/private/tmp`.
- Ran `tests/test_cli.py`, `tests/test_scheduler.py`, and `tests/test_containers.py`: **27 passed in 3.07 seconds**.
- Compared installed Docker, Claude Code, and Ollama help and current official documentation for gh, uv, Docker, Claude Code, and Ollama.

Limits: external scientific binaries, remote SSH/Slurm hosts, and a released ShapeMapper image were unavailable for end-to-end execution. Large-dataset performance and numerical fit quality were not assessed. Existing uncommitted phase work was treated as the baseline and left untouched.
