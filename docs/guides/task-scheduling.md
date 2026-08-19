# Durable task scheduling

NERD can prepare a scientific task on the controller, submit its external
command without waiting, and reconcile it in a later invocation. The
controller is the only process that writes the NERD SQLite database. Local and
remote workers write command output and small status files in their work
directory; they never open the controller database.

The existing `nerd run STEP CONFIG` command remains available for synchronous
workflows. Use `nerd submit` for a durable asynchronous attempt.

## Executor profiles

Profiles are named under `executors`. Select a default with `run.executor`, or
override it with `nerd submit --profile NAME`.

```yaml
run:
  label: mutcount_batch
  output_dir: results
  executor: quest

executors:
  workstation:
    type: local

  quest_login:
    type: slurm
    partition: short
    account: my_allocation
    resources:
      cpus: 8
      memory: 32G
      time: "02:00:00"

  quest:
    type: ssh_slurm
    host: quest                 # OpenSSH Host alias is recommended
    remote_base_dir: /projects/my_lab/nerd-runs
    resources:
      partition: short
      account: my_allocation
      cpus: 8
      memory: 32G
      time: "02:00:00"
    preamble:
      - module load python
    stage_out:
      - artifacts/**
```

The supported types are:

- `local`: a detached process on the controller.
- `slurm`: `sbatch` on the current machine, suitable for a Quest login node.
- `ssh_slurm`: stage files and invoke `sbatch` through SSH from a Mac or other
  controller.

SSH profiles use the user's OpenSSH configuration and `ssh-agent`. NERD rejects
passwords and private-key settings in its YAML and never persists credentials.
Configure host aliases, jump hosts, and identity selection in `~/.ssh/config`.

Legacy `run.backend` configuration is still translated to a profile where
possible, but named profiles are recommended for new runs.

## Lifecycle commands

Submission prints the durable controller task ID and scheduler ID:

```bash
nerd submit mut_count configs/mut_count.yaml --profile quest
```

Later commands need the same controller database. If `--db` is omitted they
use `./nerd.sqlite`; passing it explicitly is safest:

```bash
nerd --db results/nerd.sqlite status 42
nerd --db results/nerd.sqlite logs 42 --tail 200
nerd --db results/nerd.sqlite cancel 42
nerd --db results/nerd.sqlite collect 42
nerd --db results/nerd.sqlite retry 42
nerd --db results/nerd.sqlite ls
```

`status` first asks Slurm (or the local executor) and then records the observed
state in SQLite. There is no required daemon. An SSH or controller disconnect
does not stop a Slurm job; run `status` after reconnecting.

`collect` is deliberately separate from scheduler completion. A successful
Slurm exit moves the task to `awaiting_collection`. Collection stages remote
files back when needed, invokes the task's existing output consumer, and only
then marks the scientific task `completed`. Missing or invalid output becomes
`validation_failed`, even when Slurm reported success.

`retry` is explicit and attempt-aware. It creates the next `try_index` for the
same task and is accepted only after a failed, cancelled, or validation-failed
attempt.

## Durable states and records

Task states summarize the scientific lifecycle: `pending`, `submitted`,
`running`, `awaiting_collection`, `collecting`, `cancel_requested`,
`completed`, `failed`, and `cancelled`.

Attempt states retain scheduler detail, including `submitting`, `queued`,
`running`, `scheduler_completed`, `scheduler_failed`, `submission_failed`,
`cancel_requested`, `cancelled`, `collecting`, `completed`,
`validation_failed`, and `unknown`.

SQLite stores every attempt's profile, executor type, scheduler job ID, local
and remote work directories, command, resources, log path, timestamps, exit
information, and errors. A submitted configuration snapshot is kept in the run
directory so later collection and retry do not depend on the original YAML
remaining unchanged. `core_state_transitions` provides an append-only state
history. Existing scientific provenance continues to use `core_tasks` and
`core_task_attempts`.

## Phase 1 boundaries

This foundation schedules the external command produced by an existing NERD
task. Controller-side preparation and scientific output import remain short,
explicit CLI operations. Phase 1 does not add a daemon, container integration,
or changes to PRIME normalization, kobs, or dG analyses. Automatic dependency
graphs and policy-driven retry/backoff are intentionally deferred.

Phase 2 adds the ShapeMapper container execution path described in
[Containerized ShapeMapper](shapemapper-containers.md). The runtime integration
is operational and its default is pinned to the published immutable GHCR image.
