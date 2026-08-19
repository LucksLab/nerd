import json
from pathlib import Path
import subprocess

import pytest

from nerd.containers import (
    ContainerError,
    ContainerSpec,
    HostCommands,
    RuntimeInfo,
    SHAPEMAPPER_DEFAULT_IMAGE,
    SHAPEMAPPER_IMAGE_PLACEHOLDER,
    inspect_container,
    prepare_container,
    render_container_exec,
    shapemapper_container_spec,
)
from nerd.db import api as db_api
from nerd.pipeline.plugins.mutcount.shapemapper import ShapeMapperPlugin
from nerd.pipeline.tasks.mut_count import MutCountTask
from nerd.scheduler import store
from nerd.scheduler.profiles import ExecutorProfile
from nerd.scheduler.service import submit_task


DIGEST = "sha256:" + "a" * 64


class FakeHost:
    def __init__(self, *, runtime=True, scheduler=True, arch="x86_64", sif=False,
                 cache=True, preparation_error=None):
        self.runtime = runtime
        self.scheduler = scheduler
        self.arch = arch
        self.sif = sif
        self.cache = cache
        self.preparation_error = preparation_error
        self.calls = []

    def run(self, command, timeout=60):
        self.calls.append((command, timeout))
        if "sbatch squeue sacct scancel" in command:
            return subprocess.CompletedProcess([], 0 if self.scheduler else 1, "", "")
        if command == "uname -m":
            return subprocess.CompletedProcess([], 0, self.arch + "\n", "")
        if "--version" in command and "command -v" in command:
            if self.runtime:
                return subprocess.CompletedProcess([], 0, "apptainer version 1.3.6\n", "")
            return subprocess.CompletedProcess([], 1, "", "not found")
        if command.startswith("mkdir -p"):
            return subprocess.CompletedProcess([], 0 if self.cache else 1, "", "permission denied")
        if command.startswith("test -r"):
            return subprocess.CompletedProcess([], 0 if self.sif else 1,
                                               "b" * 64 + "  /opt/tool.sif\n" if self.sif else "", "")
        if " pull " in command:
            if self.preparation_error:
                return subprocess.CompletedProcess([], 1, "", self.preparation_error)
            return subprocess.CompletedProcess([], 0, "c" * 64 + "  cached.sif\n", "")
        if " exec " in command:
            return subprocess.CompletedProcess([], 0, "ShapeMapper v2.3\n", "")
        raise AssertionError("unexpected host command: %s" % command)


def _spec():
    return ContainerSpec("ghcr.io/luckslab/nerd-shapemapper:v2.3", "shapemapper", DIGEST)


def test_shapemapper_default_metadata_uses_published_immutable_identity():
    spec = shapemapper_container_spec({})
    assert spec.oci_reference == SHAPEMAPPER_DEFAULT_IMAGE
    assert spec.digest == "sha256:192732b14071da3f0e23979b2018d071ca2731639178b5f31c032eedbfbdea7e"
    spec.validate()


def test_invalid_or_mutable_image_metadata_is_rejected():
    with pytest.raises(ContainerError, match="64 hexadecimal"):
        shapemapper_container_spec({"container": {
            "image": "ghcr.io/luckslab/nerd-shapemapper:v2.3", "digest": "sha256:nope"
        }}).validate()
    with pytest.raises(ContainerError, match="mutable"):
        shapemapper_container_spec({"container": {
            "image": "ghcr.io/luckslab/nerd-shapemapper:v2.3"
        }}).validate()


def test_doctor_reports_placeholder_and_missing_runtime_without_network():
    fake = FakeHost(runtime=False)
    tool = {"container": {"image": SHAPEMAPPER_IMAGE_PLACEHOLDER}}
    result = inspect_container(shapemapper_container_spec(tool), ExecutorProfile("local", "local"), tool, fake)
    assert not result.ready
    assert {c["name"] for c in result.checks if not c["ok"]} == {"runtime", "image"}
    assert not any(" pull " in command for command, _ in fake.calls)


def test_submit_rejects_placeholder_before_creating_task_or_run_directory(tmp_path):
    output = tmp_path / "output"
    config = tmp_path / "config.yaml"
    config.write_text(
        "run:\n  label: pending-image\n  output_dir: %s\n  executor: linux\n"
        "executors:\n  linux:\n    type: local\n"
        "mut_count:\n  plugin: shapemapper\n  samples: [sample1]\n"
        "  tool:\n    container:\n      image: 'ghcr.io/luckslab/nerd-shapemapper:<release-tag>'\n" % output
    )
    conn = db_api.connect(output / "nerd.sqlite")
    db_api.init_schema(conn)
    with pytest.raises(ContainerError, match="not configured/ready"):
        submit_task(conn, "mut_count", config)
    assert conn.execute("SELECT count(*) FROM core_tasks").fetchone()[0] == 0
    assert not (output / "pending-image").exists()
    conn.close()


def test_local_linux_like_preinstalled_sif_is_ready_and_smoke_tested():
    fake = FakeHost(sif=True)
    tool = {"container": {"sif": "/opt/tool.sif"}}
    ready = prepare_container(shapemapper_container_spec(tool),
                              ExecutorProfile("linux", "local"), tool, fake)
    assert ready.sif_path == "/opt/tool.sif"
    assert ready.sif_checksum == "b" * 64
    assert ready.smoke_test_output == "ShapeMapper v2.3"
    assert not any(" pull " in command for command, _ in fake.calls)


def test_profile_level_preinstalled_sif_allows_placeholder_configuration():
    inputs = {"plugin": "shapemapper", "tool": {}}
    profile = ExecutorProfile("quest", "ssh_slurm", {
        "host": "quest", "remote_base_dir": "/scratch/u/nerd",
        "container_sif": "/projects/software/tool.sif",
    })
    MutCountTask().validate_execution_config(inputs, profile)


def test_quest_local_slurm_uses_preamble_runtime_override_and_scheduler_check():
    fake = FakeHost()
    profile = ExecutorProfile("quest-login", "slurm", {
        "preamble": ["module load singularityce"], "container_runtime": "singularity"
    })
    result = inspect_container(_spec(), profile, {}, fake)
    assert result.ready
    assert result.runtime.command == "singularity"
    assert any("sbatch squeue sacct scancel" in command for command, _ in fake.calls)
    assert any("command -v singularity" in command for command, _ in fake.calls)


def test_ssh_host_commands_put_all_container_checks_behind_openssh(monkeypatch):
    profile = ExecutorProfile("quest", "ssh_slurm", {
        "host": "quest", "remote_base_dir": "/scratch/u/nerd",
        "ssh_options": "-o BatchMode=yes", "preamble": "module load apptainer",
    })
    runner = HostCommands(profile)
    calls = []

    def fake_local(args, timeout=60):
        calls.append(list(args))
        remote = args[-1]
        if "uname -m" in remote:
            out = "x86_64\n"
        elif "--version" in remote:
            out = "apptainer version 1.3.6\n"
        else:
            out = ""
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(runner, "_local", fake_local)
    result = inspect_container(_spec(), profile, {}, runner)
    assert result.ready
    assert calls and all(call[0] == "ssh" for call in calls)
    assert all("quest" in call for call in calls)
    assert any("module load apptainer" in call[-1] for call in calls)


def test_ssh_prerequisites_fail_before_remote_commands(monkeypatch):
    monkeypatch.setattr("nerd.containers.shutil.which", lambda name: None)
    profile = ExecutorProfile("quest", "ssh_slurm", {
        "host": "quest", "remote_base_dir": "/scratch/u/nerd",
    })
    result = inspect_container(_spec(), profile, {})
    assert not result.ready
    check = next(c for c in result.checks if c["name"] == "ssh_prerequisites")
    assert "ssh" in check["message"] and "rsync" in check["message"]


def test_unsupported_architecture_and_missing_scheduler_are_clear():
    result = inspect_container(_spec(), ExecutorProfile("quest", "slurm"), {},
                               FakeHost(scheduler=False, arch="aarch64"))
    failures = {c["name"] for c in result.checks if not c["ok"]}
    assert failures == {"scheduler", "architecture"}


def test_inaccessible_cache_directory_is_not_ready():
    result = inspect_container(_spec(), ExecutorProfile("linux", "local"),
                               {"container": {"cache_dir": "/read-only/cache"}},
                               FakeHost(cache=False))
    assert not result.ready
    assert next(c for c in result.checks if c["name"] == "cache")["ok"] is False


def test_immutable_cache_preparation_uses_temp_lock_and_atomic_rename():
    fake = FakeHost()
    result = prepare_container(_spec(), ExecutorProfile("linux", "local"), {}, fake)
    prep = next(command for command, _ in fake.calls if " pull " in command)
    assert "sha256-%s.sif" % ("a" * 64) in prep
    assert '.tmp.$$' in prep
    assert 'set -o noclobber' in prep
    assert 'mv "$tmp" "$target"' in prep
    assert prep.index('if test -s "$target"') < prep.index(" pull ")
    assert result.sif_checksum == "c" * 64


def test_preparation_failure_has_registry_authentication_boundary():
    fake = FakeHost(preparation_error="unauthorized: authentication required")
    with pytest.raises(ContainerError, match="Registry authentication failed"):
        prepare_container(_spec(), ExecutorProfile("linux", "local"), {}, fake)


def test_container_command_quotes_arguments_and_maps_binds():
    command = render_container_exec(
        RuntimeInfo("apptainer", "1.3.6"), "/cache/image.sif",
        ["shapemapper", "--R1", "/data/a read.fastq", "--name", "x; touch nope"],
        ["/work dir", "/data", "/data"], "/work dir",
    )
    assert command.count("--bind") == 2
    assert "'/data/a read.fastq'" in command
    assert "'x; touch nope'" in command
    assert "--cleanenv" in command
    assert "TMPDIR=/work dir/.nerd-tmp" in command


def test_shapemapper_remote_command_uses_remote_workdir_and_relative_staged_paths():
    plugin = ShapeMapperPlugin(container_execution={
        "runtime": {"command": "singularity", "version": "4.3.1"},
        "sif_path": "/scratch/cache/tool.sif",
        "workdir": "/scratch/runs/attempt-1",
        "executable": "shapemapper",
    })
    command = plugin.command(
        sample_name="sample-1", r1_path=Path("artifacts/sample-1/r1.fastq"),
        r2_path=Path("artifacts/sample-1/r2.fastq"),
        fasta_path=Path("artifacts/sample-1/target.fa"),
        out_dir=Path("artifacts/sample-1"), options={"amplicon": True},
    )
    assert "--pwd /scratch/runs/attempt-1" in command
    assert "--bind /scratch/runs/attempt-1:/scratch/runs/attempt-1" in command
    assert "artifacts/sample-1/r1.fastq" in command
    assert command.startswith("singularity exec")


def test_container_provenance_is_persisted_on_controller(tmp_path):
    conn = db_api.connect(tmp_path / "nerd.sqlite")
    db_api.init_schema(conn)
    task_id = db_api.begin_task(conn, "mut_count", "sample", 1, "quest", str(tmp_path),
                                "run", "cache", tool="shapemapper")
    payload = {
        "oci_reference": "ghcr.io/luckslab/nerd-shapemapper:v2.3",
        "oci_digest": DIGEST, "sif_path": "/cache/tool.sif", "sif_checksum": "b" * 64,
        "runtime": {"command": "apptainer", "version": "apptainer 1.3.6"},
        "shapemapper_version": "ShapeMapper v2.3", "command": "apptainer exec ...",
        "execution_host": "quest", "executor_profile": "quest", "executor_type": "ssh_slurm",
        "recorded_at": "2026-08-18T00:00:00+00:00",
    }
    store.record_container_provenance(conn, task_id, payload)
    row = conn.execute("SELECT * FROM core_container_provenance WHERE task_id=?", (task_id,)).fetchone()
    assert row["runtime_version"] == "apptainer 1.3.6"
    assert json.loads(row["provenance_json"])["oci_digest"] == DIGEST
    conn.close()
