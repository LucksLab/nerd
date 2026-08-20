# Projects and analysis configuration

NERD separates stable project context from scientific analysis choices:

- `.nerd/project.toml` says **where and how the NERD project operates**.
- YAML plus referenced CSV/TSV sheets says **what scientific analysis runs**.

The project file contains only the canonical project ID, database path, output
directory, optional default executor, named executor profiles, and stable
container settings. Samples, reaction selections, fit parameters, outliers,
run labels, and credentials do not belong there. Ordinary commands never
rewrite this file.

## Initialize a project

Project names use uppercase three-letter researcher initials and exact numeric
groups: `XXX.00.00.000`.

```bash
nerd init EKC.07.00.000
cd EKC.07.00.000
```

When the directory basename matches the pattern, `--name` is inferred. For an
existing directory with a different basename, be explicit:

```bash
nerd init analysis-root --name EKC.07.00.000 --existing
```

`--existing` permits adding NERD assets to a non-empty directory but never
permits overwriting an existing project. Initialization creates only:

- `.nerd/project.toml`;
- the configured output directory; and
- the initialized controller database.

Use `--json` for a machine-readable initialization result.

## Project file schema

The generated file is intentionally small:

```toml
[project]
name = "EKC.07.00.000"
default_executor = "local"

[paths]
database = ".nerd/nerd.sqlite"
output = "outputs"

[executors.local]
type = "local"
```

To browse and analyze FASTQs already stored on Quest or another HPC, define a
named SSH/Slurm profile before opening the sample helper. Each profile becomes
a separate FASTQ-location choice:

```toml
[executors.quest]
type = "ssh_slurm"
host = "quest"  # an OpenSSH Host alias in ~/.ssh/config
remote_base_dir = "/scratch/abc123/nerd-runs"

[executors.other_cluster]
type = "ssh_slurm"
host = "other-cluster"
remote_base_dir = "/work/abc123/nerd-runs"
```

Test each alias first with `ssh quest hostname` (substitute the alias). The
sample helper does not store passwords or key paths; authentication remains in
OpenSSH and `ssh-agent`. Restart/reconnect the helper after changing
`project.toml` so the new location appears in the dropdown.

Paths in `project.toml` are resolved relative to the project root. Executor
profiles may use `local`, `slurm`, or `ssh_slurm`. SSH credentials and private
keys are rejected; use OpenSSH configuration and `ssh-agent`. Stable container
runtime/cache options may be placed in `[containers]`, but scientific tool
choices that affect a run should stay in YAML so provenance remains explicit.

`--project` accepts either the project root or the direct
`.nerd/project.toml` path.

## Discovery and precedence

Database context resolves in this order:

1. explicit `--db` or `--project`;
2. the nearest `.nerd/project.toml` from the current directory or its parents;
3. `NERD_PROJECT`, then `NERD_DB`;
4. config-file context, when the command accepts a config;
5. otherwise, an actionable error.

Legacy parent `nerd.sqlite` discovery remains available before environment
fallbacks for compatibility with projects that have not migrated. Read-only
commands such as `task list`, `task show`, `task logs`, `db path`, `config
validate`, and `config show` do not create databases or output directories.

Because discovery starts at the current directory, synchronous runs, detached
runs, task inspection, logs, and collection all reach the same controller
database from nested directories.

## YAML paths and merge behavior

Relative paths in analysis YAML are resolved against the directory containing
that YAML file. Explicit CLI paths keep their normal caller-supplied meaning.

Configuration resolves as:

1. stable defaults from `project.toml`;
2. analysis YAML overrides;
3. explicit CLI overrides such as `--profile`.

The inherited keys are `run.output_dir`, `run.executor`, named `executors`, and
stable `containers` settings. YAML cannot redefine project identity. A YAML
executor profile with the same name overrides that project default, and an
explicit `--profile` selects the final executor.

Project identity, root, and database location are excluded from the scientific
config hash, so moving a project does not cause arbitrary cache misses. The
selected effective executor profile and container runtime settings remain in
the hash/provenance inputs. Detached
submission writes a fully resolved YAML snapshot into the run directory, so
retry and collection do not depend on the original working directory.

## Validate, inspect, and generate configs

```bash
nerd config validate configs/create.yaml
nerd config show configs/create.yaml --resolved
nerd config show configs/create.yaml --resolved --json
nerd config init mut_count --output configs/mut_count.yaml
```

Validation checks the root shape, known sections and keys, workflow selection,
run label, executor references, bundled plugin/engine names, and direct input
paths where appropriate. Unknown keys include a likely correction when one is
available. Resolved output reports project ID/root, database, output directory,
config base, workflow and run label, executor, plugin/engine, resolved inputs,
authored values, and inherited defaults. Secret-shaped values are redacted.

`nerd config init create` also produces a companion CSV because repeated sample
records are clearer in a sheet than as hundreds of YAML entries. Other starter
configs keep scientific selections and model parameters in YAML. Templates do
not invent credentials or cluster allocations.

## Build sample inputs in the web helper

Install the optional web UI dependencies, then launch the helper anywhere
inside an initialized project:

```bash
pip install -e ".[webui]"
nerd webui serve
```

Use `nerd webui serve --project PATH` when launching outside the project. The
helper reads the database and output directory from `.nerd/project.toml`, opens
already connected, and writes generated create configs under `configs/`.
Those configs inherit project output and executor defaults rather than copying
them into YAML. The displayed `nerd run create` commands retain the selected
project and any explicit database override.

Draft sample-sheet state is stored in `.nerd/sample-draft.json`. Existing
`.nerd_sample_draft.json` files are still loaded so an in-progress legacy draft
is not lost during migration. A folder without `project.toml` can still be
opened explicitly as a legacy project; it continues to use its existing
`nerd.sqlite` context.

## Migrating an existing project

Existing configs remain supported. To adopt project discovery without changing
scientific YAML:

1. run `nerd init PATH --name EKC.07.00.000 --existing`;
2. if preserving an existing database, edit `paths.database` once to point to
   it relative to the project root;
3. optionally move stable output/executor defaults from YAML into
   `project.toml`; and
4. run `nerd config show FILE --resolved` to verify the result.

Explicit `--db`, config-driven `run.output_dir`, and legacy `run.backend` remain
compatible. Direct scientific input flags are intentionally deferred: the
current run interface consumes a validated YAML file, and adding broad flag
parsing would duplicate task-specific normalization.
