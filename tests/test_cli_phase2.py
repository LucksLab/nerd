"""Phase 2 command hierarchy and compatibility routing."""

from pathlib import Path

from typer.main import get_command

from nerd.cli import app
from nerd.db import api as db_api


def _row(step="create", task_id=41):
    return {
        "task_id": task_id,
        "task_name": step,
        "task_state": "submitted",
        "try_index": 1,
        "scheduler_attempt_id": 7,
        "scheduler_state": "queued",
        "executor_profile": "quest",
        "executor_type": "ssh_slurm",
        "scheduler_id": "fake-41",
        "exit_code": None,
        "error": None,
    }


def _initialized(path):
    conn = db_api.connect(path)
    db_api.init_schema(conn)
    conn.close()


def test_phase2_group_inventory_and_help(cli_runner):
    root = get_command(app)
    task = root.commands["task"]

    assert set(task.commands) == {
        "list", "show", "logs", "wait", "watch", "collect", "cancel", "retry"
    }
    assert set(root.commands["image"].commands) == {"inspect", "prepare"}
    assert set(root.commands["plugin"].commands["doctor"].commands) == {"shapemapper"}

    help_result = cli_runner.invoke(app, ["task", "--help"])
    assert help_result.exit_code == 0
    for name in task.commands:
        assert name in help_result.output


def test_run_detach_uses_durable_submission_and_phase1_context(
    cli_runner, tmp_path, monkeypatch
):
    from nerd.scheduler import service

    selected_db = tmp_path / "state" / "controller.sqlite"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run:\n  output_dir: ignored\n")
    observed = {}

    def fake_submit(conn, step, received_config, profile):
        observed["db"] = Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve()
        observed["step"] = step
        observed["config"] = Path(received_config)
        observed["profile"] = profile
        return _row(step)

    monkeypatch.setattr(service, "submit_task", fake_submit)
    result = cli_runner.invoke(app, [
        "run", "create", str(config_path), "--detach", "--profile", "quest",
        "--db", str(selected_db),
    ])

    assert result.exit_code == 0, result.output
    assert observed == {
        "db": selected_db.resolve(),
        "step": "create",
        "config": config_path.resolve(),
        "profile": "quest",
    }
    assert "task_id: 41" in result.output
    assert "attempt_id: 7" in result.output
    assert "scheduler_id: fake-41" in result.output


def test_task_subcommands_share_explicit_database_context(
    cli_runner, tmp_path, monkeypatch
):
    from nerd.scheduler import service

    selected_db = tmp_path / "controller.sqlite"
    _initialized(selected_db)
    observed = []

    def fake_reconcile(conn, task_id):
        observed.append(Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve())
        return _row(task_id=task_id)

    monkeypatch.setattr(service, "reconcile", fake_reconcile)
    result = cli_runner.invoke(
        app, ["task", "show", "41", "--db", str(selected_db)]
    )

    assert result.exit_code == 0, result.output
    assert observed == [selected_db.resolve()]
    assert "next_actions:" in result.output


def test_hidden_lifecycle_wrapper_routes_to_shared_service(
    cli_runner, tmp_path, monkeypatch
):
    from nerd.scheduler import service

    selected_db = tmp_path / "controller.sqlite"
    _initialized(selected_db)
    calls = []

    def fake_cancel(conn, task_id):
        calls.append(task_id)
        return _row(task_id=task_id)

    monkeypatch.setattr(service, "cancel_task", fake_cancel)
    grouped = cli_runner.invoke(app, ["task", "cancel", "41", "--db", str(selected_db)])
    legacy = cli_runner.invoke(app, ["cancel", "42", "--db", str(selected_db)])

    assert grouped.exit_code == legacy.exit_code == 0
    assert calls == [41, 42]
    assert "Deprecated:" not in grouped.output
    assert "Deprecated:" in legacy.output
    assert "nerd task cancel" in legacy.output


def test_task_list_filters_are_forwarded_without_schema_changes(
    cli_runner, tmp_path, monkeypatch
):
    from nerd.scheduler import store

    selected_db = tmp_path / "controller.sqlite"
    _initialized(selected_db)
    observed = {}

    def fake_list(conn, label=None, state=None, task_name=None, limit=50):
        observed.update(label=label, state=state, task_name=task_name, limit=limit)
        return []

    monkeypatch.setattr(store, "list_tasks", fake_list)
    result = cli_runner.invoke(app, [
        "task", "list", "--db", str(selected_db), "--label", "batch-a",
        "--state", "failed", "--workflow", "mut_count", "--limit", "3",
    ])

    assert result.exit_code == 0, result.output
    assert observed == {
        "label": "batch-a", "state": "failed", "task_name": "mut_count", "limit": 3
    }
