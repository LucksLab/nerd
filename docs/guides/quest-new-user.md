# Running NERD on Quest from a Mac

This guide is for a Mac user who already has access to Northwestern's Quest
computing system. It assumes you are using the Lucks Lab Quest allocation:

```text
account: b1044
partition: buyin
```

You will run NERD from your Mac. NERD will send the time-consuming ShapeMapper
work to Quest, check its progress, and bring the results back to your Mac.

You do not need to install ShapeMapper, Docker, or Singularity on your Mac.

## Before you begin

You need:

- Your Northwestern NetID and password.
- A Mac with the Terminal application.
- Your experiment metadata and FASTQ files on the Mac.
- Python 3.8 or newer on the Mac.

Open **Terminal** from `Applications > Utilities > Terminal` for the commands
in this guide.

## 1. Confirm that you can log into Quest

Replace `abc123` with your lowercase Northwestern NetID:

```bash
ssh abc123@login.quest.northwestern.edu
```

The first connection may ask whether you trust the Quest computer. Compare the
message with Northwestern's current
[Quest login instructions](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/login/login-quest.html),
then enter `yes`. Enter your Northwestern password when prompted.

The password will not appear while you type. This is normal.

If the login succeeds, you will see a new command prompt on Quest. Return to
your Mac by entering:

```bash
exit
```

## 2. Create a short SSH alias

NERD connects to Quest several times while checking and collecting a job. A
short SSH alias makes that connection simpler.

On your Mac, open the SSH configuration file:

```bash
mkdir -p ~/.ssh
nano ~/.ssh/config
```

Add the following block. Replace `abc123` with your lowercase NetID:

```sshconfig
Host quest
    HostName login.quest.northwestern.edu
    User abc123
    Port 22
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ControlMaster auto
    ControlPath ~/.ssh/nerd-quest-%C
    ControlPersist 15m
```

In `nano`, save by pressing `Control-O`, press `Return`, and exit with
`Control-X`.

Protect the configuration file:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/config
```

Test the new alias:

```bash
ssh quest
```

Enter your Northwestern password if prompted, then enter `exit` to return to
your Mac.

The `ControlMaster` settings let several NERD commands reuse the same login for
15 minutes. You may be asked for your password again later. Do not save your
Northwestern password in this file or in a NERD configuration file.

### Use the local SSH profile in NERD

The block beginning with `Host quest` is a local SSH profile. Its profile name
is `quest`:

```sshconfig
Host quest
    HostName login.quest.northwestern.edu
    User abc123
```

NERD uses that profile name in the `host` field of an `ssh_slurm` executor:

```yaml
executors:
  quest:
    type: ssh_slurm
    host: quest
```

These two uses of `quest` connect the files:

```text
NERD host: quest
        -> local ~/.ssh/config profile: Host quest
        -> abc123@login.quest.northwestern.edu
```

NERD passes the profile name to your Mac's normal `ssh` command. This means
that the hostname, NetID, port, connection reuse, and any institution-approved
SSH settings stay in `~/.ssh/config`. NERD does not need to know or store your
password.

Always test a local SSH profile before using it in NERD:

```bash
ssh quest 'hostname'
```

The command should print the name of a Quest login node and return to your Mac.
You can also ask SSH to show the settings it resolved for the profile:

```bash
ssh -G quest | head
```

If you want separate profiles, give each one a different local name. For
example, this profile keeps NERD-specific connection reuse settings separate
from an existing `quest` profile:

```sshconfig
Host quest-nerd
    HostName login.quest.northwestern.edu
    User abc123
    Port 22
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ControlMaster auto
    ControlPath ~/.ssh/nerd-quest-%C
    ControlPersist 15m
```

The matching NERD executor would use:

```yaml
executors:
  quest:
    type: ssh_slurm
    host: quest-nerd
```

The executor name under `executors` can be anything meaningful, such as
`quest`, `quest_test`, or `quest_large`. The `host` value must exactly match a
`Host` name in `~/.ssh/config`:

```yaml
run:
  executor: quest_test

executors:
  quest_test:
    type: ssh_slurm
    host: quest-nerd
```

Do not add `password`, `private_key`, `identity_file`, or `key_file` to the NERD
YAML. NERD intentionally rejects credential paths in executor profiles. If
Northwestern later provides an approved key-based login method, configure it
in `~/.ssh/config`, not in NERD.

## 3. Prepare your Quest folders

NERD needs one folder for jobs and one for the ShapeMapper software package.
Create both with this command on your Mac:

```bash
ssh quest 'mkdir -p /scratch/$USER/nerd-runs /scratch/$USER/nerd-container-cache'
```

Quest's `/scratch` area is intended for temporary working files. It is not
backed up, and files are normally removed after 30 days without use. NERD will
collect the important results back to your Mac.

## 4. Check that Quest provides Singularity

Singularity is the program Quest uses to open the packaged ShapeMapper
software. Check it once:

```bash
ssh quest
module spider singularity
module load singularityce/4.3.1-gcc-8.5.0
singularity --version
exit
```

If the `module load` command reports that this exact version is unavailable,
copy the current version shown by `module spider singularity`. You will use
that name in the NERD configuration later.

## 5. Install NERD on your Mac

NERD is currently installed from its GitHub repository. It uses an ordinary
Python environment; `uv` is not required.

Check Python:

```bash
python3 --version
```

If this reports Python 3.8 or newer, install NERD:

```bash
git clone https://github.com/LucksLab/nerd.git
cd nerd
git switch dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
nerd --help
```

If `python3` or `git` is not found, ask a lab member for help installing it
before continuing.

Whenever you open a new Terminal window, reactivate NERD:

```bash
cd /path/to/nerd
source .venv/bin/activate
```

Replace `/path/to/nerd` with the folder created by `git clone`.

## 6. Create an experiment folder

Keep experiment files outside the NERD source-code folder:

```bash
mkdir -p ~/nerd-work/my-experiment/configs
cd ~/nerd-work/my-experiment
```

Your FASTQ files must be readable from the Mac. NERD will copy them to Quest
when the job is submitted.

Prepare the sample sheet and sample-creation configuration described in the
[sample organization guide](sample-organization.md). In `configs/create.yaml`,
use the same experiment name and results folder that will appear in the Quest
configuration:

```yaml
run:
  label: my_experiment
  output_dir: ./results
```

Then register the experiment:

```bash
nerd run create configs/create.yaml
```

The command finishes with a completion summary naming the task ID, what was
registered, and where the log and database live. See
[Reading NERD output](#reading-nerd-output) if the summary is unfamiliar.

This creates `./results/nerd.sqlite`. It stores sample information and the
history of the analysis. Keep this file with the experiment. Later NERD
commands must use this same database.

## 7. Create the Quest analysis configuration

Create `configs/mut_count_quest.yaml` with the following starting template.
Replace:

- `abc123` with your lowercase NetID.
- `my_experiment` with a short experiment name.
- `<reaction-group>` with the reaction group registered in the previous step.
- The Singularity module name if Quest showed a different current version.

```yaml
run:
  label: my_experiment
  output_dir: ./results
  executor: quest
  threads: 8
  mem_gb: 32
  time: "04:00:00"

executors:
  quest:
    type: ssh_slurm
    host: quest
    remote_base_dir: /scratch/abc123/nerd-runs
    container_runtime: singularity
    container_cache_dir: /scratch/abc123/nerd-container-cache
    preamble:
      - module load singularityce/4.3.1-gcc-8.5.0
    resources:
      account: b1044
      partition: buyin
      cpus: 8
      memory: 32G
      time: "04:00:00"

mut_count:
  plugin: shapemapper
  reaction_group: "<reaction-group>"
  tool:
    execution: container
  params:
    dms_mode: true
    amplicon: true
    n_proc: 8
    output_N7: false
    per_read_histograms: true
```

The words `account`, `partition`, `cpus`, `memory`, and `time` tell Quest which
lab allocation to use and approximately how much computing power the analysis
needs. The starting values above are reasonable for an initial test and can be
adjusted later.

Do not add your Quest password or a GitHub token to this file.

## 8. Ask NERD to check the setup

Run:

```bash
nerd plugin doctor shapemapper configs/mut_count_quest.yaml --profile quest
```

This checks that your Mac can reach Quest and that Quest can find its job
system, Singularity, and the container-cache folder. It does not start an
analysis or download the ShapeMapper package.

Each check prints on its own line, prefixed `OK` or `NOT READY`, and the last
line reads `ready: yes` or `ready: no`:

```text
execution_host: quest
executor: quest (ssh_slurm)
OK  ssh_prerequisites: OpenSSH and rsync are available
OK  scheduler: sbatch, squeue, sacct, and scancel are available
OK  architecture: x86_64 (supported: x86_64)
OK  runtime: singularity: singularity version 4.3.1
OK  cache: /scratch/abc123/nerd-container-cache is writable
OK  image: immutable image identity configured: ghcr.io/edr-choi/nerd-shapemapper2@sha256:192732b1...
ready: yes
```

If any line reports `NOT READY`, read its message and consult the
troubleshooting section below. The command exits non-zero when a check fails,
and accepts `--json` if you want the result in a script.

## 9. Prepare the ShapeMapper package

ShapeMapper is distributed as a prebuilt package called a container. NERD
downloads it once to Quest and reuses it for later analyses.

The current test package is private. Ask the NERD maintainer whether it has
become public. If it is public, skip directly to `image prepare shapemapper` below.

If it is still private, the maintainer must first give your GitHub account read
access. Create a GitHub personal access token with only the `read:packages`
permission, then log in interactively on Quest:

```bash
ssh quest
module load singularityce/4.3.1-gcc-8.5.0
singularity registry login --username <github-username> docker://ghcr.io
exit
```

Paste the token only when Singularity asks for a password or token. Never put
the token in the NERD configuration, a script, or GitHub.

Now prepare ShapeMapper:

```bash
nerd image prepare shapemapper configs/mut_count_quest.yaml --profile quest
```

NERD downloads the exact tested image, checks it, and runs
`shapemapper --version`. The command prints the location and checksum of the
prepared file. Running it a second time should reuse the existing copy instead
of downloading it again.

## 10. Submit the analysis

Run:

```bash
nerd run mut_count configs/mut_count_quest.yaml --detach --profile quest
```

NERD prepares the inputs, copies the necessary files to Quest, submits the job,
and returns without waiting for the analysis to finish.

NERD prints one field per line. The two identifiers to note are:

- `task_id`: NERD's number for the analysis.
- `scheduler_id`: Quest's Slurm job ID.

The surrounding lines — `task_state`, `attempt`, `attempt_state`, `executor`,
and `executor_type` — describe where the job is and which profile submitted it.

Write down the task ID. Adding `--json` prints the same information as a single
JSON document, which is easier to capture in a script.

You may close Terminal after submission; the Quest job will continue.

## 11. Check progress and logs

Use the task ID printed during submission:

```bash
nerd task show <task-id> --db ./results/nerd.sqlite
nerd task logs <task-id> --db ./results/nerd.sqlite --tail 200
```

Common states are:

- `submitted` or `queued`: Quest is waiting to start the job.
- `running`: ShapeMapper is running.
- `awaiting_collection`: Quest finished and the results are ready to collect.
- `failed`: the job stopped with an error.

List all NERD tasks with:

```bash
nerd task list --db ./results/nerd.sqlite
```

## 12. Collect the results

When the state is `awaiting_collection`, run:

```bash
nerd task collect <task-id> --db ./results/nerd.sqlite
```

NERD copies the outputs from Quest, checks that the expected ShapeMapper files
are present, and adds the results to `nerd.sqlite`.

The task becomes `completed` only after these checks pass. If Quest finished
but an expected result is missing, NERD uses `validation_failed` instead of
incorrectly claiming success.

Important records include:

- ShapeMapper output files under the experiment's results folder.
- `command.log`, which records messages from the remote job.
- `.nerd-container-provenance.json`, which records the exact ShapeMapper
  package and Singularity version used.
- `nerd.sqlite`, which stores the experiment and analysis history.

## 13. Cancel or retry a job

Cancel the current attempt:

```bash
nerd task cancel <task-id> --db ./results/nerd.sqlite
```

Retry a failed, cancelled, or validation-failed task:

```bash
nerd task retry <task-id> --db ./results/nerd.sqlite
```

NERD keeps the earlier attempt and its logs so it is still possible to see what
happened.

## Reading NERD output

### Completion summaries

Commands that run a workflow on your Mac — `nerd run create`, and any `nerd run`
without `--detach` — end with a short completion summary instead of stopping
after the last progress line:

```text
create: completed (task 1, my_experiment)
counts: attempted=42, buffers_created=2, buffers_failed=0, buffers_unchanged=0,
buffers_updated=0, constructs_created=3, constructs_failed=0,
constructs_unchanged=0, constructs_updated=0, derived_samples_processed=12,
failed=0, samples_created=24, samples_failed=0, samples_unchanged=0,
samples_updated=0, sequencing_runs_created=1, sequencing_runs_failed=0,
sequencing_runs_unchanged=0, sequencing_runs_updated=0, skipped=0,
succeeded=42, unresolved_references=0
elapsed: 3.412s
log: ./results/run_logs/2026-08-19T10-14-02__cfg-a1b2c3d.log
database: /Users/you/nerd-work/my-experiment/results/nerd.sqlite
artifacts: ./results/my_experiment/create/create___cfg-a1b2c3d, ./results/my_experiment/create/create___cfg-a1b2c3d/command.log, ./results/my_experiment/create/create___cfg-a1b2c3d/created_objects.json, ./results/my_experiment/create/create___cfg-a1b2c3d/created_objects.log
next: nerd task show 1
```

`counts` and `artifacts` are each printed as one long line; they are wrapped
above only to fit this page.

Reading the summary:

- The first line gives the workflow, its final status, the NERD task ID, and the
  `run.label` from the configuration file.
- `counts` reports only numbers NERD can confirm from the run itself, sorted by
  name. `attempted`, `succeeded`, `failed`, and `skipped` always appear; the
  remaining keys depend on the workflow.
- `elapsed` is the wall-clock time for the run.
- `log` is the run log for this invocation, written under `run_logs` inside the
  configured `output_dir`.
- `database` is the absolute path of the `nerd.sqlite` NERD selected.
- `artifacts` lists the run directory first, then `command.log`, then any files
  the workflow produced.
- `next` suggests the command to run afterwards.

A `tool:` line appears only for workflows that drive an external program. A
`mut_count` run prints `tool: shapemapper 2.3`; `create` is pure NERD, so the
line is absent. Workflows that report extra measurements add a `metrics:` line,
and any problems appear as `warning [code]: ...` or `failure [code]: ...` lines.

Statuses you may see:

- `completed`: everything succeeded.
- `cached`: an identical configuration already finished, so nothing was rerun.
  The counts show `skipped=1`, and the `next:` line points at the earlier task
  that already produced the result.
- `partial_success`: some items succeeded and some failed.
- `failed`: the workflow did not finish.

### Progress messages and the summary go to different places

Progress and diagnostic messages are written to standard error. The completion
summary is written to standard output. Redirecting standard output therefore
saves the summary without the progress text:

```bash
nerd run create configs/create.yaml > create_summary.txt
```

### Output flags

These are global flags, so they come *before* the subcommand:

```bash
nerd --quiet run create configs/create.yaml
```

| Flag | Effect |
| --- | --- |
| `-v`, `--verbose` | Show detailed DEBUG progress messages. |
| `-q`, `--quiet` | Show only warnings and errors. The summary is still printed. |
| `--no-color` | Plain text without color. Useful when saving a transcript. |
| `--log-file PATH` | Write the log to `PATH` instead of the automatic run log. |

The file log is written at the same level as the console, so `--quiet` also
reduces what is recorded in it. Use `--verbose` when you want a full DEBUG
record on disk. Note that `--log-file` replaces the automatic
`run_logs/<timestamp>__cfg-<hash>.log` file rather than adding to it, and the
summary's `log:` line names whichever file was used.

### Choosing the database

Every command needs to know which `nerd.sqlite` to use. NERD resolves it in this
order:

1. `--db PATH`.
2. `--project DIR`, which means `DIR/nerd.sqlite` (or `DIR/.nerd/nerd.sqlite`).
3. An existing `nerd.sqlite` — or `.nerd/nerd.sqlite` — found in the current
   directory or any parent directory.
4. The `NERD_DB` or `NERD_PROJECT` environment variable.
5. For `nerd run`, the `run.output_dir` in the configuration file.

`--db` and `--project` work either globally or on the individual command, so
both of these are valid:

```bash
nerd --project ./results task list
nerd task list --project ./results
```

Because this guide keeps the database in `./results`, `--project ./results` is a
shorter equivalent of `--db ./results/nerd.sqlite`. Setting it once per session
avoids repeating either flag:

```bash
export NERD_PROJECT=~/nerd-work/my-experiment/results
```

To confirm which database a command would use, and what it contains:

```bash
nerd db path --project ./results
nerd db info --project ./results
```

Both refuse to create a database; they fail if none exists at the resolved
location.

### Machine-readable output

Add `--json` to get one JSON document on standard output and nothing else. This
works with `nerd run`, `nerd task show`, `nerd task list`, `nerd task wait`, and
the `collect`, `cancel`, and `retry` commands, as well as `nerd plugin doctor`
and `nerd image inspect` / `nerd image prepare`:

```bash
nerd run mut_count configs/mut_count_quest.yaml --detach --profile quest --json
nerd task show <task-id> --db ./results/nerd.sqlite --json
nerd task list --db ./results/nerd.sqlite --json
```

`nerd task logs` has no `--json`; it prints the log text as-is.

The fields are stable within schema version 1.x. See the
[output and JSON contract](../cli/output-contract.md) for the full schema.

### Exit codes

A synchronous `nerd run` exits with:

- `0` when the workflow completed or was cached.
- `2` on partial success.
- `1` on failure.

A detached `nerd run ... --detach` exits `0` once the job is submitted; it does
not wait for the result. `nerd plugin doctor` and `nerd image inspect` exit `1`
when the environment is not ready. Passing `--profile` without `--detach` is
rejected with exit code `2`.

This makes NERD usable inside a shell script without parsing its text output.

### Older command names

Earlier notes may use the flat command names. They still work, but each one
prints a deprecation notice on standard error and will be removed:

| Old | Use instead |
| --- | --- |
| `nerd submit WORKFLOW CONFIG` | `nerd run WORKFLOW CONFIG --detach` |
| `nerd ls` | `nerd task list` |
| `nerd status <task-id>` | `nerd task show <task-id>` |
| `nerd logs <task-id>` | `nerd task logs <task-id>` |
| `nerd cancel` / `collect` / `retry` | `nerd task cancel` / `collect` / `retry` |
| `nerd doctor CONFIG` | `nerd plugin doctor shapemapper CONFIG` |
| `nerd prepare-image CONFIG` | `nerd image prepare shapemapper CONFIG` |

## Troubleshooting

### `ssh quest` does not connect

First retry the complete address:

```bash
ssh <netid>@login.quest.northwestern.edu
```

Check that the NetID is lowercase and that the `Host quest` block was saved in
`~/.ssh/config`. For account or password problems, contact Northwestern Quest
support at `quest-help@northwestern.edu`.

### NERD asks for the Quest password repeatedly

Confirm that the `ControlMaster`, `ControlPath`, and `ControlPersist` lines are
present in `~/.ssh/config`. A new password prompt after the 15-minute reuse
period is expected.

### Doctor cannot find Singularity

Log into Quest and run:

```bash
module spider singularity
```

Copy the exact available module name into the `preamble` line of the YAML file.

### Quest rejects the job account or queue

Confirm that the YAML contains exactly:

```yaml
account: b1044
partition: buyin
```

If those values are present, ask a lab member to confirm that your Quest user
has access to the Lucks Lab allocation.

### Image preparation says `unauthorized` or `denied`

The current package may still be private. Confirm that:

- The maintainer gave your GitHub account read access.
- Your token has the `read:packages` permission.
- You ran `singularity registry login` on Quest, not on your Mac.

### The job finishes but collection fails

Run:

```bash
nerd task logs <task-id> --db ./results/nerd.sqlite --tail 200
```

Look near the end for a missing FASTQ, memory error, ShapeMapper error, or
missing output. Send the task ID, Slurm job ID, and log to a lab member when
asking for help.

### A file disappeared from Quest scratch

Scratch files are temporary and normally expire after 30 days without use.
Your collected results on the Mac and the local `nerd.sqlite` database are the
important copies to preserve.
