"""Characterize database/context selection before the CLI context redesign."""


from nerd.cli import app
from nerd.db import api as db_api
from nerd.project import ProjectContext


def _initialized(path):
    conn = db_api.connect(path)
    db_api.init_schema(conn)
    conn.close()


def test_explicit_database_selection_is_used_for_task_listing(
    cli_runner, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    selected_db = tmp_path / "state" / "selected.sqlite"
    _initialized(selected_db)

    result = cli_runner.invoke(app, ["--db", str(selected_db), "task", "list"])

    assert result.exit_code == 0, result.output
    assert "No tasks found" in result.output
    assert selected_db.is_file()
    assert not (tmp_path / "nerd.sqlite").exists()


def test_listing_without_context_does_not_create_database_in_cwd(
    cli_runner, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(app, ["task", "list"])

    assert result.exit_code == 1
    assert "No NERD database context found" in result.output
    assert not (tmp_path / "nerd.sqlite").exists()


def test_database_selector_is_natural_on_lifecycle_command(
    cli_runner, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    selected_db = tmp_path / "selected.sqlite"
    _initialized(selected_db)

    result = cli_runner.invoke(app, ["task", "show", "1", "--db", str(selected_db)])

    assert result.exit_code == 1
    assert "Task 1 has no asynchronous attempts" in result.output
    assert selected_db.exists()
    assert not (tmp_path / "nerd.sqlite").exists()


def test_listing_discovers_existing_database_in_parent(cli_runner, tmp_path, monkeypatch):
    selected_db = tmp_path / "nerd.sqlite"
    _initialized(selected_db)
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    result = cli_runner.invoke(app, ["task", "list"])

    assert result.exit_code == 0, result.output
    assert "No tasks found" in result.output


def test_context_precedence_is_explicit_discovered_environment_then_config(tmp_path):
    discovered = tmp_path / "nerd.sqlite"
    discovered.touch()
    env_db = tmp_path / "environment.sqlite"
    explicit_db = tmp_path / "explicit.sqlite"
    cfg_output = tmp_path / "config-output"
    cfg = {"run": {"output_dir": str(cfg_output)}}

    assert ProjectContext(db=explicit_db).resolve_database(
        config=cfg, environ={"NERD_DB": str(env_db)}, cwd=tmp_path
    ) == explicit_db
    assert ProjectContext().resolve_database(
        config=cfg, environ={"NERD_DB": str(env_db)}, cwd=tmp_path
    ) == discovered
    discovered.unlink()
    assert ProjectContext().resolve_database(
        config=cfg, environ={"NERD_DB": str(env_db)}, cwd=tmp_path
    ) == env_db
    assert ProjectContext().resolve_database(
        config=cfg, environ={}, cwd=tmp_path
    ) == cfg_output / "nerd.sqlite"
