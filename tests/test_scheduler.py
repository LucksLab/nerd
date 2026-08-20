import sqlite3
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from nerd.db import api as db_api
from nerd.pipeline.tasks import TASK_REGISTRY
from nerd.pipeline.tasks.base import Task, TaskContext, TaskScope, WorkUnit
from nerd.scheduler import store
from nerd.scheduler.executors import LocalProcessExecutor, SlurmExecutor, SSHSlurmExecutor
from nerd.scheduler.models import AttemptState, JobSpec
from nerd.scheduler.profiles import ExecutorProfile, load_executor_profile
from nerd.scheduler.service import (
    cancel_task, collect_task, reconcile, retry_task, submit_task, watch_task,
)


def _db(path: Path) -> sqlite3.Connection:
    conn = db_api.connect(path)
    db_api.init_schema(conn)
    return conn


def test_named_profiles_and_ssh_credentials_are_explicit():
    cfg = {
        "run": {"label": "x"},
        "executors": {
            "laptop": {"type": "local"},
            "quest": {
                "type": "ssh_slurm",
                "host": "quest",
                "remote_base_dir": "/projects/example/nerd",
            },
        },
    }
    assert load_executor_profile(cfg, "laptop").executor_type == "local"
    assert load_executor_profile(cfg, "quest").options["host"] == "quest"
    cfg["executors"]["quest"]["password"] = "do-not-accept"
    with pytest.raises(ValueError, match="OpenSSH config"):
        load_executor_profile(cfg, "quest")


def test_scheduler_schema_records_attempt_and_transition_history(tmp_path):
    conn = _db(tmp_path / "nerd.sqlite")
    task_id = db_api.begin_task(
        conn, "example", "global", None, "local", str(tmp_path), "label", "key"
    )
    spec = JobSpec("true", tmp_path / "work")
    row = store.create_attempt(
        conn, task_id, "local", "local", spec, tmp_path / "config.yaml"
    )
    store.transition_attempt(conn, row["scheduler_attempt_id"], AttemptState.SUBMITTING)
    updated = store.get_attempt(conn, row["scheduler_attempt_id"])
    assert updated["try_index"] == 1
    assert updated["scheduler_state"] == "submitting"
    history = conn.execute(
        "SELECT to_state FROM core_state_transitions WHERE entity_kind='attempt' ORDER BY id"
    ).fetchall()
    assert [item[0] for item in history] == ["pending", "submitting"]
    conn.close()


def test_local_process_submit_is_detached_and_reconciles_result(tmp_path):
    profile = ExecutorProfile("local", "local")
    executor = LocalProcessExecutor(profile)
    spec = JobSpec("sleep 0.2; echo finished", tmp_path)
    started = time.monotonic()
    handle = executor.submit(spec)
    assert time.monotonic() - started < 0.15
    deadline = time.monotonic() + 5
    status = executor.status(handle, spec)
    while status.state not in {AttemptState.SCHEDULER_COMPLETED, AttemptState.SCHEDULER_FAILED}:
        assert time.monotonic() < deadline
        time.sleep(0.05)
        status = executor.status(handle, spec)
    assert status.state == AttemptState.SCHEDULER_COMPLETED
    assert status.exit_code == 0
    assert "finished" in executor.logs(handle, spec)


def test_slurm_submit_is_parsable_and_never_waits(tmp_path, monkeypatch):
    executor = SlurmExecutor(ExecutorProfile("quest-login", "slurm"))
    calls = []

    def fake_run(args, timeout=30):
        calls.append(list(args))
        return Mock(returncode=0, stdout="12345;cluster\n", stderr="")

    monkeypatch.setattr(executor, "_run", fake_run)
    handle = executor.submit(JobSpec("echo ok", tmp_path, resources={"cpus": 4, "memory": 8}))
    assert handle.scheduler_id == "12345"
    assert calls[0][0] == "sbatch"
    assert "--parsable" in calls[0]
    assert "--wait" not in calls[0]
    assert calls[0][calls[0].index("--mem") + 1] == "8G"


def test_ssh_slurm_submits_through_openssh_without_wait(tmp_path, monkeypatch):
    profile = ExecutorProfile(
        "quest", "ssh_slurm",
        {"host": "quest", "remote_base_dir": "/scratch/test", "ssh_options": "-o BatchMode=yes"},
    )
    executor = SSHSlurmExecutor(profile)
    local_calls = []
    remote_calls = []

    def fake_run(args, timeout=30):
        local_calls.append(list(args))
        return Mock(returncode=0, stdout="", stderr="")

    def fake_remote(args):
        remote_calls.append(list(args))
        if args and args[0] == "sbatch":
            return Mock(returncode=0, stdout="6789\n", stderr="")
        return Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(executor, "_run", fake_run)
    monkeypatch.setattr(executor, "_remote", fake_remote)
    spec = JobSpec("echo ok", tmp_path, remote_workdir="/scratch/test/run")
    handle = executor.submit(spec)
    assert handle.scheduler_id == "6789"
    sbatch = next(call for call in remote_calls if call and call[0] == "sbatch")
    assert "--parsable" in sbatch
    assert "--wait" not in sbatch
    assert executor._ssh_base() == ["ssh", "-o", "BatchMode=yes"]
    assert any(call[0] == "rsync" for call in local_calls)


class _AsyncFileTask(Task):
    name = "_scheduler_test"
    scope_kind = "global"

    def prepare(self, cfg):
        return {}, {}

    def resolve_scope(self, ctx, inputs):
        return TaskScope("global")

    def command(self, ctx, inputs, params):
        return "printf 'scientific output\\n' > result.txt"

    def consume_outputs(self, ctx, inputs, params, run_dir, task_id=None):
        value = (run_dir / "result.txt").read_text().strip()
        if value != "scientific output":
            raise ValueError("invalid output")
        (run_dir / "validated.txt").write_text(value)


class _SlowTask(_AsyncFileTask):
    name = "_scheduler_slow_test"

    def command(self, ctx, inputs, params):
        return "sleep 30"


class _MissingOutputTask(_AsyncFileTask):
    name = "_scheduler_missing_output_test"

    def command(self, ctx, inputs, params):
        return "true"


class _FanoutTask(_AsyncFileTask):
    name = "_scheduler_fanout_test"

    def plan_work_units(self, ctx, cfg, inputs, params):
        units = []
        for index in range(2):
            unit_cfg = dict(cfg)
            unit_cfg[self.name] = {"unit": index}
            units.append(WorkUnit(
                key="unit-%s" % index, label="unit %s" % index,
                config=unit_cfg, scope_kind="unit", scope_id=index,
            ))
        return units

    def prepare(self, cfg):
        return dict(cfg.get(self.name) or {}), {}


def test_scientific_task_is_only_completed_after_collection(tmp_path, monkeypatch):
    monkeypatch.setitem(TASK_REGISTRY, _AsyncFileTask.name, _AsyncFileTask)
    output = tmp_path / "output"
    config = tmp_path / "config.yaml"
    config.write_text(
        "run:\n  label: async\n  output_dir: output\n  executor: laptop\n"
        "executors:\n  laptop:\n    type: local\n"
    )
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()
    monkeypatch.chdir(invocation_dir)
    conn = _db(output / "nerd.sqlite")
    submitted = submit_task(conn, _AsyncFileTask.name, config)
    snapshot = Path(submitted["config_path"])
    assert str(output.resolve()) in snapshot.read_text()
    collection_dir = tmp_path / "collection"
    collection_dir.mkdir()
    monkeypatch.chdir(collection_dir)
    task_id = submitted["task_id"]
    assert submitted["task_state"] == "submitted"
    deadline = time.monotonic() + 5
    row = reconcile(conn, task_id)
    while row["scheduler_state"] not in {"scheduler_completed", "scheduler_failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.05)
        row = reconcile(conn, task_id)
    assert row["task_state"] == "awaiting_collection"
    collected = collect_task(conn, task_id)
    assert collected["scheduler_state"] == "completed"
    assert collected["task_state"] == "completed"
    assert (store.job_spec(collected).workdir / "validated.txt").is_file()
    conn.close()


def test_scheduler_success_with_missing_scientific_output_fails_validation(tmp_path, monkeypatch):
    monkeypatch.setitem(TASK_REGISTRY, _MissingOutputTask.name, _MissingOutputTask)
    output = tmp_path / "output"
    config = tmp_path / "config.yaml"
    config.write_text(
        "run:\n  label: missing\n  output_dir: %s\n  executor: laptop\n"
        "executors:\n  laptop:\n    type: local\n" % output
    )
    conn = _db(output / "nerd.sqlite")
    submitted = submit_task(conn, _MissingOutputTask.name, config)
    deadline = time.monotonic() + 5
    row = reconcile(conn, submitted["task_id"])
    while row["scheduler_state"] != "scheduler_completed":
        assert time.monotonic() < deadline
        time.sleep(0.05)
        row = reconcile(conn, submitted["task_id"])
    with pytest.raises(FileNotFoundError):
        collect_task(conn, submitted["task_id"])
    failed = store.latest_attempt_for_task(conn, submitted["task_id"])
    assert failed["scheduler_state"] == "validation_failed"
    assert failed["task_state"] == "failed"
    conn.close()


def test_cancel_and_retry_are_attempt_aware(tmp_path, monkeypatch):
    monkeypatch.setitem(TASK_REGISTRY, _SlowTask.name, _SlowTask)
    output = tmp_path / "output"
    config = tmp_path / "config.yaml"
    config.write_text(
        "run:\n  label: retry\n  output_dir: %s\n  executor: laptop\n"
        "executors:\n  laptop:\n    type: local\n" % output
    )
    conn = _db(output / "nerd.sqlite")
    first = submit_task(conn, _SlowTask.name, config)
    cancelled = cancel_task(conn, first["task_id"])
    assert cancelled["scheduler_state"] == "cancelled"
    assert cancelled["task_state"] == "cancelled"
    second = retry_task(conn, first["task_id"])
    assert second["try_index"] == 2
    assert second["task_state"] == "submitted"
    cancel_task(conn, first["task_id"])
    conn.close()


def test_wait_for_task_can_collect_when_scheduler_finishes(monkeypatch):
    from nerd.scheduler import service

    states = iter([
        {"task_state": "running"},
        {"task_state": "awaiting_collection"},
    ])
    sleeps = []
    collected = {"task_state": "completed", "scheduler_state": "completed"}

    monkeypatch.setattr(service, "reconcile", lambda conn, task_id: next(states))
    monkeypatch.setattr(service, "collect_task", lambda conn, task_id: collected)
    monkeypatch.setattr(service.time, "sleep", lambda interval: sleeps.append(interval))

    assert service.wait_for_task(object(), 17, collect=True, poll_interval=0.25) is collected
    assert sleeps == [0.25]


def test_fanout_parent_watches_and_collects_independent_units(tmp_path, monkeypatch):
    monkeypatch.setitem(TASK_REGISTRY, _FanoutTask.name, _FanoutTask)
    output = tmp_path / "output"
    config = tmp_path / "config.yaml"
    config.write_text(
        "run:\n  label: fanout\n  output_dir: %s\n  executor: laptop\n"
        "executors:\n  laptop:\n    type: local\n%s:\n  enabled: true\n"
        % (output, _FanoutTask.name)
    )
    conn = _db(output / "nerd.sqlite")
    submitted = submit_task(conn, _FanoutTask.name, config)
    assert submitted["is_parent"] is True
    assert submitted["total_units"] == 2
    assert len(store.child_task_ids(conn, submitted["task_id"])) == 2

    finished = watch_task(
        conn, submitted["task_id"], collect=True, poll_interval=0.02
    )
    assert finished["task_state"] == "completed"
    assert finished["counts"] == {"completed": 2}
    conn.close()


def test_reconcile_failed_job_syncs_diagnostics_locally(tmp_path, monkeypatch):
    from nerd.scheduler import service
    from nerd.scheduler.models import JobHandle, JobStatus

    conn = _db(tmp_path / "nerd.sqlite")
    config = tmp_path / "config.yaml"
    config.write_text(
        "run:\n  label: failed\n  executor: laptop\n"
        "executors:\n  laptop:\n    type: local\n"
    )
    task_id = db_api.begin_task(
        conn, "example", "global", None, "laptop", str(tmp_path), "failed", "key"
    )
    spec = JobSpec("false", tmp_path / "work")
    attempt = store.create_attempt(conn, task_id, "laptop", "local", spec, config)
    store.transition_attempt(
        conn, attempt["scheduler_attempt_id"], AttemptState.QUEUED,
        handle=JobHandle("123"),
    )

    class FailedExecutor:
        def status(self, handle, received_spec):
            return JobStatus(
                AttemptState.SCHEDULER_FAILED, exit_code=9, message="FAILED"
            )

        def collect_diagnostics(self, handle, received_spec):
            received_spec.workdir.mkdir(parents=True, exist_ok=True)
            (received_spec.workdir / "command.log").write_text("shape failed\n")

    monkeypatch.setattr(service, "executor_for", lambda profile: FailedExecutor())
    failed = reconcile(conn, task_id)
    assert failed["task_state"] == "failed"
    assert failed["exit_code"] == 9
    assert (spec.workdir / "command.log").read_text() == "shape failed\n"
    payload = __import__("json").loads((spec.workdir / "failure.json").read_text())
    assert payload["slurm_state"] == "FAILED"
    conn.close()


def test_mut_count_plans_one_unit_per_reaction_group_without_duplicate_samples(
    tmp_path, monkeypatch
):
    from nerd.pipeline.tasks.mut_count import MutCountTask

    task = MutCountTask()
    groups = {
        "first": (11, "first", ["sample-a", "sample-b"]),
        "second": (12, "second", ["sample-c"]),
    }
    monkeypatch.setattr(task, "_fetch_reaction_group_info", lambda ctx, value: groups[value])
    cfg = {
        "run": {"label": "fanout", "output_dir": str(tmp_path)},
        "mut_count": {
            "plugin": "shapemapper", "reaction_groups": ["first", "second"],
            "samples": ["sample-a", "standalone"],
        },
    }
    ctx = TaskContext(
        db=object(), backend="ssh_slurm", workdir=tmp_path, threads=1,
        mem_gb=1, time="00:10:00", label="fanout", output_dir=str(tmp_path),
        executor_profile="quest",
    )
    prepared, params = task.prepare(cfg)
    units = task.plan_work_units(ctx, cfg, prepared, params)
    assert [unit.key for unit in units] == ["rg-11", "rg-12", "ungrouped"]
    assert units[0].config["mut_count"]["reaction_group"] == 11
    assert units[1].config["mut_count"]["reaction_group"] == 12
    assert units[2].config["mut_count"]["samples"] == ["standalone"]
    assert "reaction_groups" not in units[2].config["mut_count"]
