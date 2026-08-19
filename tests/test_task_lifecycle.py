"""Characterize CLI lifecycle surfaces and current synchronous task transitions."""

from pathlib import Path

import pytest

from nerd.cli import app
from nerd.db import api as db_api
from nerd.pipeline.tasks.base import Task
from typer.main import get_command


LIFECYCLE_COMMANDS = {"submit", "status", "logs", "cancel", "collect", "retry", "ls"}


def test_current_lifecycle_command_inventory_is_available():
    assert LIFECYCLE_COMMANDS <= set(get_command(app).commands)


def test_submit_uses_explicit_controller_database_and_executor_profile(
    cli_runner, tmp_path, monkeypatch
):
    from nerd.scheduler import service

    selected_db = tmp_path / "controller.sqlite"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    observed = {}

    def fake_submit_task(conn, step, received_config_path, profile):
        observed["db"] = Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve()
        observed["step"] = step
        observed["config_path"] = Path(received_config_path)
        observed["profile"] = profile
        return {
            "task_id": 41,
            "task_name": step,
            "task_state": "submitted",
            "try_index": 1,
            "scheduler_state": "queued",
            "executor_profile": profile,
            "scheduler_id": "fake-41",
            "exit_code": None,
            "error": None,
        }

    monkeypatch.setattr(service, "submit_task", fake_submit_task)
    result = cli_runner.invoke(
        app,
        [
            "--db",
            str(selected_db),
            "submit",
            "create",
            str(config_path),
            "--profile",
            "quest",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == {
        "db": selected_db.resolve(),
        "step": "create",
        "config_path": config_path.resolve(),
        "profile": "quest",
    }


class ConsumeFailureTask(Task):
    name = "_phase0_consume_failure"
    scope_kind = "global"

    def prepare(self, cfg):
        return {}, {}

    def command(self, ctx, inputs, params):
        return None

    def consume_outputs(self, ctx, inputs, params, run_dir, task_id=None):
        raise ValueError("representative output validation failure")


def test_synchronous_output_failure_is_durably_failed(tmp_path):
    db_path = tmp_path / "nerd.sqlite"
    conn = db_api.connect(db_path)
    db_api.init_schema(conn)
    cfg = {
        "run": {
            "label": "consume-failure",
            "output_dir": str(tmp_path / "output"),
            "backend": "local",
        }
    }

    try:
        with pytest.raises(ValueError, match="representative output validation failure"):
            ConsumeFailureTask().exec(conn, cfg)
        row = conn.execute(
            "SELECT state, ended_at, message FROM core_tasks WHERE task_name=?",
            (ConsumeFailureTask.name,),
        ).fetchone()
    finally:
        conn.close()

    assert row["state"] == "failed"
    assert row["ended_at"] is not None
    assert row["message"] == (
        "Output validation failed: representative output validation failure"
    )
