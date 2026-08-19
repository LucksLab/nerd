# Containerized ShapeMapper

NERD can run the existing ShapeMapper mutation-count plugin through Apptainer
or Singularity on the executor selected by the Phase 1 scheduler. Image
building and publication are separate from this integration.

## Current image status

The packaged default is the immutable private GHCR image published for
pre-release validation:

```text
ghcr.io/edr-choi/nerd-shapemapper2@sha256:192732b14071da3f0e23979b2018d071ca2731639178b5f31c032eedbfbdea7e
```

No image configuration is required to select this default. An explicit image
override remains available:

```yaml
mut_count:
  plugin: shapemapper
  tool:
    execution: container
    container:
      image: ghcr.io/edr-choi/nerd-shapemapper2@sha256:192732b14071da3f0e23979b2018d071ca2731639178b5f31c032eedbfbdea7e
```

The digest is mandatory for downloaded images. It determines the SIF cache
filename, so changing a mutable tag cannot silently replace cached software.
The current package is private, so authenticate the selected Apptainer or
Singularity runtime to GHCR on the execution host before `prepare-image`.
For SSH-Slurm, that means authenticating on Quest, not on the Mac controller.
Do not place a GitHub token in the NERD YAML. NERD reports registry
authentication failures as a distinct preparation error and does not store
credentials. Once the package becomes public, no registry credentials should
be required.

An administrator-installed SIF remains an alternative:

```yaml
mut_count:
  plugin: shapemapper
  tool:
    execution: container
    container:
      sif: /projects/my_lab/software/nerd-shapemapper.sif
```

The SIF must be readable on the execution host. NERD records its SHA-256 and
runs the configured ShapeMapper version smoke test before submission.

## Doctor, prepare, submit

Run all three commands with the same configuration and executor profile:

```bash
nerd doctor configs/mut_count.yaml --profile quest
nerd prepare-image configs/mut_count.yaml --profile quest
nerd submit mut_count configs/mut_count.yaml --profile quest
```

`doctor` checks the selected execution environment without contacting GHCR. It
reports scheduler availability where applicable, host architecture, exact
Apptainer/Singularity version, cache writability, immutable image metadata, and
an installed SIF if configured.

`prepare-image` pulls `docker://<reference>@sha256:<digest>` only after all
metadata is ready. It writes a process-specific temporary file, coordinates
through a no-clobber lock, verifies a nonempty result, and atomically renames
the completed SIF into the shared cache. A cache hit skips the pull. It then
runs `shapemapper --version` inside the SIF.

`submit` repeats preparation/readiness so a stale doctor result cannot launch
an unready task. The durable attempt stores the fully quoted command and
container provenance. Scheduler completion remains separate from scientific
validation: `collect` must find and ingest all expected ShapeMapper profiles
before the task becomes complete.

## Executor behavior

For a `local` profile, NERD discovers and runs the container runtime on the
controller. This path is intended for a Linux workstation with Apptainer or
Singularity.

For a `slurm` profile, discovery and preparation run in the current cluster
login environment, including the profile's `preamble`. The submitted job uses
that same preamble, which is the appropriate place for a runtime module:

```yaml
executors:
  quest_login:
    type: slurm
    preamble:
      - module load singularityce
    container_runtime: singularity
    container_cache_dir: /scratch/my_user/nerd-container-cache
```

For `ssh_slurm`, every scheduler, architecture, runtime, cache, preparation,
and smoke-test command runs through OpenSSH on the remote cluster. A Mac
controller needs only OpenSSH and rsync; it does not need Docker, Apptainer, or
Singularity. FASTQs are staged by default:

```yaml
executors:
  quest:
    type: ssh_slurm
    host: quest
    remote_base_dir: /scratch/my_user/nerd-runs
    preamble:
      - module load singularityce
    container_runtime: singularity
    container_cache_dir: /scratch/my_user/nerd-container-cache
```

Set `shared_filesystem: true` only when the absolute input paths seen by the
controller are also valid on the remote cluster. NERD then binds those paths
directly instead of staging FASTQs. In both modes, the run/work directory,
inputs, temporary files, and output directories retain the same paths inside
the container. Bind arguments and scientific arguments are shell-quoted.

Native/custom plugins remain supported. A configured ShapeMapper binary also
keeps the legacy native path:

```yaml
tool:
  execution: native
  bin: /projects/my_lab/ShapeMapper/shapemapper
  version: "2.3"
```

## Provenance and collection

The controller writes `.nerd-container-provenance.json` in the run directory
and a `core_container_provenance` SQLite row. They include the requested OCI
reference and digest, SIF path/checksum, runtime command/version, ShapeMapper
smoke-test version, architecture, exact scientific command, timestamps, and
executor/host identity. The serialized job specification carries the same
metadata for attempt inspection.

Remote workers write only logs, output files, and Phase 1 job markers. They do
not open the controller's SQLite database. `nerd collect` transfers configured
outputs and performs database ingestion on the controller.

## Private-image operational checklist

For the current private image:

1. Authenticate the execution host's container runtime to GHCR without adding
   credentials to NERD configuration.
2. On an Apptainer host, run `doctor`, `prepare-image`, and a minimal real
   ShapeMapper dataset. Confirm the reported OCI digest, SIF SHA-256, runtime
   version, and ShapeMapper version.
3. Repeat on Quest with its supported Singularity module and a scratch cache.
4. Run the same Quest test from a Mac with the `ssh_slurm` profile, first with
   staged FASTQs and then, if applicable, with a verified shared filesystem.
5. Disconnect after submission, reconnect, run `status`, then `collect`.
   Confirm the worker never created or modified SQLite.
6. Force a scheduler-success/missing-output case and confirm collection ends in
   `validation_failed`, not `completed`.
7. Launch two preparations for the same digest and confirm both select one
   complete cached SIF and leave no temporary or lock file.
8. Archive the final doctor output, SIF checksum, job log, provenance JSON, and
   controller database record with the image release evidence.
