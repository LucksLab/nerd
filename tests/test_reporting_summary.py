"""Semantic tests for the Phase 3 output contract."""

import json

from nerd.cli import app
from nerd.reporting.summary import (
    ArtifactReference,
    TaskIssue,
    TaskSummary,
    render_human,
    render_json,
)


EXPECTED_KEYS = [
    "schema_version", "status", "workflow", "task_id", "source_task_id",
    "label", "plugin", "engine", "version", "started_at", "ended_at",
    "duration_seconds", "timings", "counts", "metrics", "warnings",
    "failures", "artifacts", "log_path", "database_path", "next_actions",
]


def test_summary_schema_is_json_native_and_stable(tmp_path):
    summary = TaskSummary(
        status="completed", workflow="mut_count", task_id=7,
        counts={"succeeded": 2, "attempted": 2},
        metrics={"rate": 1.0},
        warnings=[TaskIssue("minor", "A warning")],
        artifacts=[ArtifactReference("output", str(tmp_path))],
    )
    payload = summary.to_dict()
    assert list(payload) == EXPECTED_KEYS
    assert payload["schema_version"] == "1.0"
    assert list(payload["counts"]) == ["attempted", "succeeded"]
    assert json.loads(render_json(summary)) == payload


def test_human_renderer_covers_success_warning_failure_and_cached():
    for status in ("completed", "cached", "skipped"):
        assert status in render_human(TaskSummary(status=status, workflow="create"))
    text = render_human(TaskSummary(
        status="partial_success", workflow="drop",
        warnings=[TaskIssue("missing", "one missing")],
        failures=[TaskIssue("update", "one failed")],
    ))
    assert "warning [missing]" in text
    assert "failure [update]" in text


def test_task_list_json_is_machine_readable(cli_runner, tmp_path):
    from nerd.db import api as db_api

    db_path = tmp_path / "nerd.sqlite"
    conn = db_api.connect(db_path)
    db_api.init_schema(conn)
    conn.close()
    result = cli_runner.invoke(app, ["task", "list", "--db", str(db_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {"schema_version": "1.0", "tasks": []}


def test_run_json_stdout_contains_no_progress(cli_runner, tmp_path, monkeypatch):
    from nerd.pipeline.tasks import TASK_REGISTRY

    config = tmp_path / "config.yaml"
    config.write_text("run:\n  label: json-test\n  output_dir: out\n")

    class JsonTask:
        def exec(self, conn, cfg, verbose=False):
            return TaskSummary(status="completed", workflow="create")

    monkeypatch.setitem(TASK_REGISTRY, "create", JsonTask)
    result = cli_runner.invoke(app, ["run", "create", str(config), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["workflow"] == "create"
    assert "Database schema" not in result.stdout


def test_json_failure_and_partial_success_exit_codes(cli_runner, tmp_path, monkeypatch):
    from nerd.pipeline.tasks import TASK_REGISTRY

    config = tmp_path / "config.yaml"
    config.write_text("run:\n  label: exit-test\n  output_dir: out\n")

    class PartialTask:
        def exec(self, conn, cfg, verbose=False):
            return TaskSummary(status="partial_success", workflow="create")

    monkeypatch.setitem(TASK_REGISTRY, "create", PartialTask)
    partial = cli_runner.invoke(app, ["run", "create", str(config), "--json"])
    assert partial.exit_code == 2
    assert json.loads(partial.stdout)["status"] == "partial_success"

    class FailedTask:
        def exec(self, conn, cfg, verbose=False):
            raise ValueError("invalid scientific configuration")

    monkeypatch.setitem(TASK_REGISTRY, "create", FailedTask)
    failed = cli_runner.invoke(app, ["run", "create", str(config), "--json"])
    assert failed.exit_code == 1
    payload = json.loads(failed.stdout)
    assert payload["status"] == "failed"
    assert payload["failures"][0]["message"] == "invalid scientific configuration"
