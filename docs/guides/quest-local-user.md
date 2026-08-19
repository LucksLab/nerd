# Running NERD while logged into Quest

This guide is for someone who:

- Is already logged into Quest with SSH.
- Has FASTQ files stored in a Quest directory.
- Wants to install and run NERD directly on Quest.
- Uses the Lucks Lab Quest allocation:

```text
account: b1044
partition: buyin
```

NERD runs on a Quest login node, but the time-consuming ShapeMapper analysis
runs as a scheduled Quest job. Do not run ShapeMapper directly on a login node.

## How this workflow works

```text
Quest login node
    -> NERD and nerd.sqlite
    -> Quest job queue
    -> Quest compute node
    -> Singularity and ShapeMapper
    -> results stored on Quest
```

Because the FASTQ files are already on Quest, NERD can use them in place. It
does not need to copy them from another computer.

## 1. Log into Quest

From your computer, replace `abc123` with your lowercase Northwestern NetID:

```bash
ssh abc123@login.quest.northwestern.edu
```

Enter your Northwestern password when prompted. The password does not appear
while you type.

All remaining commands in this guide run on Quest unless stated otherwise.

If you instead want NERD to stay on your Mac and send work to Quest, follow
[Quest from a Mac](quest-new-user.md). That guide explains how a local
`~/.ssh/config` profile becomes the `host` used by a NERD `ssh_slurm` executor.

## 2. Check your FASTQ location

Find the directory containing the FASTQ files:

```bash
pwd
ls -lh /path/to/fastq-folder
```

Replace `/path/to/fastq-folder` with the real Quest path. Use complete paths
that begin with `/projects`, `/scratch`, or `/home` in the sample sheet. For
example:

```text
/projects/b1044/my-data/experiment-01/fastq
```

The files must be readable from Quest compute nodes. Files under `/projects`,
`/scratch`, and `/home` are normally visible from both login and compute nodes.

## 3. Choose where NERD will store its files

For an initial test, create a personal working directory under Quest scratch.
Replace `abc123` with your NetID:

```bash
mkdir -p /scratch/abc123/nerd-work/my-experiment/configs
mkdir -p /scratch/abc123/nerd-container-cache
cd /scratch/abc123/nerd-work/my-experiment
```

Quest scratch is fast but temporary. It is not backed up, and files are
normally removed after 30 days without use. For long-term storage, use a lab
folder under `/projects/b1044` if one has been assigned to you.

## 4. Create a Python environment for NERD

Quest's built-in Python is too old for NERD, so load Quest's supported Mamba
module and create a separate Python environment:

```bash
module purge
module load mamba/24.3.0
mamba create --prefix "$HOME/.conda/envs/nerd" -c conda-forge python=3.11 pip --yes
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/nerd"
python --version
```

The final command should report Python 3.11.

You only create the environment once. After a future login, reactivate it with:

```bash
module purge
module load mamba/24.3.0
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/nerd"
```

## 5. Install NERD on Quest

Store the NERD source code in your home directory, where it will not be removed
with scratch files:

```bash
mkdir -p "$HOME/software"
cd "$HOME/software"
git clone https://github.com/LucksLab/nerd.git
cd nerd
git switch dev
python -m pip install --upgrade pip
python -m pip install -e .
nerd --help
```

If the final command displays the NERD command list, installation succeeded.

To update NERD later:

```bash
cd "$HOME/software/nerd"
git switch dev
git pull --ff-only
python -m pip install -e .
```

## 6. Check the Quest Singularity module

Singularity opens the packaged ShapeMapper software on Quest:

```bash
module spider singularity
module load singularityce/4.3.1-gcc-8.5.0
singularity --version
```

If that exact module version is unavailable, copy the current name displayed by
`module spider singularity`. Use the same name in the NERD configuration below.

## 7. Prepare the experiment information

Return to the experiment directory:

```bash
cd /scratch/abc123/nerd-work/my-experiment
```

Replace `abc123` with your NetID.

Prepare the sample sheet and `configs/create.yaml` described in the
[sample organization guide](sample-organization.md). FASTQ directories in the
sample sheet should use their complete Quest paths.

Use the same experiment name and results folder in every configuration:

```yaml
run:
  label: my_experiment
  output_dir: /scratch/abc123/nerd-work/my-experiment/results
```

Register the samples and experiment information:

```bash
nerd run create configs/create.yaml
```

This creates:

```text
/scratch/abc123/nerd-work/my-experiment/results/nerd.sqlite
```

The database stores the experiment information and analysis history.

## 8. Create the ShapeMapper configuration

Create `configs/mut_count_quest.yaml` with this template. Replace:

- `abc123` with your NetID.
- `my_experiment` with your experiment name.
- `<reaction-group>` with the reaction group registered in the previous step.
- The Singularity module if Quest showed a different current version.

```yaml
run:
  label: my_experiment
  output_dir: /scratch/abc123/nerd-work/my-experiment/results
  executor: quest_login
  threads: 8
  mem_gb: 32
  time: "04:00:00"

executors:
  quest_login:
    type: slurm
    shared_filesystem: true
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

`shared_filesystem: true` tells NERD that the FASTQs are already on Quest and
are visible to its compute nodes. NERD uses the existing files rather than
making another copy.

## 9. Check the setup

Run:

```bash
nerd plugin doctor shapemapper configs/mut_count_quest.yaml --profile quest_login
```

NERD checks that Quest can find its job commands, Singularity, the container
folder, and the correct ShapeMapper package information. It does not submit an
analysis.

Correct any failed checks before continuing.

## 10. Prepare the ShapeMapper package

NERD downloads the packaged ShapeMapper software once and reuses it for later
jobs.

The current test image may still be private. Ask the NERD maintainer whether it
is public. If it is public, skip the login command below.

For a private image, the maintainer must give your GitHub account read access.
Create a GitHub personal access token with only `read:packages`, then run:

```bash
module load singularityce/4.3.1-gcc-8.5.0
singularity registry login --username <github-username> docker://ghcr.io
```

Paste the token only when Singularity asks for it. Do not put the token in a
NERD configuration or script.

Prepare and test ShapeMapper:

```bash
nerd image prepare shapemapper configs/mut_count_quest.yaml --profile quest_login
```

NERD downloads the exact tested image, stores it in your container cache, and
runs `shapemapper --version`. Repeating this command should reuse the existing
file.

## 11. Submit the analysis

Submit the job to Quest:

```bash
nerd run mut_count configs/mut_count_quest.yaml --detach --profile quest_login
```

NERD prints:

- A **task ID**, used by NERD.
- A **Slurm job ID**, used by Quest.

Write down the task ID. The command returns after submission; it does not wait
for ShapeMapper to finish. You may log out of Quest, and the job will continue:

```bash
exit
```

## 12. Return later and check progress

Log into Quest again, activate the environment, and return to the experiment:

```bash
ssh abc123@login.quest.northwestern.edu
module purge
module load mamba/24.3.0
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/nerd"
cd /scratch/abc123/nerd-work/my-experiment
```

Check the task:

```bash
nerd task show <task-id> --db ./results/nerd.sqlite
nerd task logs <task-id> --db ./results/nerd.sqlite --tail 200
```

Common states are:

- `submitted` or `queued`: the job is waiting to start.
- `running`: ShapeMapper is running.
- `awaiting_collection`: ShapeMapper finished and NERD needs to check the
  results.
- `failed`: the job stopped with an error.

List every recorded task with:

```bash
nerd task list --db ./results/nerd.sqlite
```

## 13. Collect and check the results

When the task reports `awaiting_collection`, run:

```bash
nerd task collect <task-id> --db ./results/nerd.sqlite
```

Because NERD and the job are both using Quest storage, collection does not need
to transfer the results to another computer. NERD checks the expected
ShapeMapper output files and adds the results to `nerd.sqlite`.

The task becomes `completed` only after those checks pass. If an expected file
is missing, NERD reports `validation_failed` instead.

Important files include:

- ShapeMapper outputs under the experiment's results directory.
- `command.log`, containing messages from the Quest job.
- `.nerd-container-provenance.json`, recording the exact ShapeMapper package
  and Singularity version.
- `nerd.sqlite`, containing experiment information and analysis history.

Copy important final results out of `/scratch` to the appropriate lab project
directory or another backed-up location.

## 14. Cancel or retry

Cancel a submitted or running attempt:

```bash
nerd task cancel <task-id> --db ./results/nerd.sqlite
```

Retry a failed, cancelled, or validation-failed task:

```bash
nerd task retry <task-id> --db ./results/nerd.sqlite
```

NERD keeps the earlier attempt and its logs.

## Troubleshooting

### `nerd` is not found after logging in again

Reactivate the Python environment:

```bash
module purge
module load mamba/24.3.0
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/nerd"
```

### Doctor cannot find Singularity

Run:

```bash
module spider singularity
```

Copy the exact available module name into the executor's `preamble` line.

### Quest rejects the job account or queue

Confirm that the configuration contains:

```yaml
account: b1044
partition: buyin
```

If those values are present, ask a lab member to confirm your access to the
Lucks Lab allocation.

### NERD cannot find a FASTQ file

- Use complete Quest paths in the sample sheet.
- Confirm each path with `ls -lh /complete/path/to/file.fastq.gz`.
- Confirm the file is readable by your Quest user.
- Keep `shared_filesystem: true` in the local Slurm profile.

### Image preparation reports `unauthorized` or `denied`

- Confirm the maintainer granted your GitHub account read access.
- Use a token with `read:packages` only.
- Run `singularity registry login` on Quest.

### The Quest job finishes but collection fails

Read the end of the job log:

```bash
nerd task logs <task-id> --db ./results/nerd.sqlite --tail 200
```

Look for a missing file, memory error, or ShapeMapper error. Send the task ID,
Slurm job ID, and log to a lab member when asking for help.
