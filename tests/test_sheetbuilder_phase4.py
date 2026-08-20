"""Integration coverage between the sample-input helper and Phase 4 projects."""

from __future__ import annotations

from pathlib import Path

import yaml

from nerd.cli import app
from nerd.configuration import resolve_config, validate_config
from nerd.project import ProjectContext, load_project
from nerd.sheetbuilder.catalog import EntityCatalog
from nerd.sheetbuilder.export import export
from nerd.sheetbuilder.model import Sheet
from nerd.sheetbuilder.session import Session


def _init_project(cli_runner, root: Path, *, database: str = ".nerd/nerd.sqlite",
                  output: str = "outputs"):
    result = cli_runner.invoke(app, [
        "init", str(root), "--name", "EKC.07.00.000",
        "--database", database, "--output-dir", output,
    ])
    assert result.exit_code == 0, result.output
    return load_project(root)


def test_session_uses_phase4_database_output_and_draft_location(cli_runner, tmp_path):
    root = tmp_path / "project"
    project = _init_project(
        cli_runner, root, database="state/controller.sqlite", output="results"
    )

    session = Session()
    info = session.connect(str(root), None, "import-01")

    assert session.project_config == project
    assert session.project_dir == root.resolve()
    assert session.db_path == (root / "state" / "controller.sqlite").resolve()
    assert session.output_dir == (root / "results").resolve()
    assert session.draft_path == (root / ".nerd" / "sample-draft.json").resolve()
    assert info["project_id"] == "EKC.07.00.000"
    assert not (root / "nerd.sqlite").exists()


def test_session_accepts_project_toml_and_project_relative_db_override(cli_runner, tmp_path):
    root = tmp_path / "project"
    project = _init_project(cli_runner, root)
    session = Session()

    session.connect(str(project.source_path), "scratch/helper.sqlite", "import-01")

    assert session.project_dir == root.resolve()
    assert session.project_config == project
    assert session.db_path == (root / "scratch" / "helper.sqlite").resolve()
    assert session.db_path.is_file()


def test_legacy_session_keeps_legacy_database_and_draft_locations(tmp_path):
    root = tmp_path / "legacy"
    session = Session()

    session.connect(str(root), None, "import-01")

    assert session.project_config is None
    assert session.db_path == (root / "nerd.sqlite").resolve()
    assert session.output_dir == root.resolve()
    assert session.draft_path == (root / ".nerd_sample_draft.json").resolve()


def test_phase4_export_inherits_project_defaults_and_quotes_commands(cli_runner, tmp_path):
    root = tmp_path / "project with spaces"
    project = _init_project(cli_runner, root, output="analysis outputs")

    result = export(
        Sheet(), EntityCatalog(), str(root), "sample-import",
        db_path=str(project.database), project_config=project,
    )
    config_path = root / "configs" / "create_probing_samples.yaml"
    authored = yaml.safe_load(config_path.read_text())
    resolved, selected_project = resolve_config(
        config_path, context=ProjectContext(project=root)
    )

    assert authored["run"] == {"label": "sample-import"}
    assert resolved["run"]["output_dir"] == str(project.output_dir)
    assert resolved["run"]["executor"] == project.default_executor
    assert selected_project == project
    assert validate_config(resolved, workflow="create") == "create"
    assert "'" in result["commands"][-1]
    assert "--project" in result["commands"][-1]
    assert "--db" not in result["commands"][-1]


def test_phase4_export_preserves_explicit_database_override(cli_runner, tmp_path):
    root = tmp_path / "project"
    project = _init_project(cli_runner, root)
    override = root / "scratch" / "helper.sqlite"

    result = export(
        Sheet(), EntityCatalog(), str(root), "sample-import",
        db_path=str(override), project_config=project,
    )

    command = result["commands"][-1]
    assert "--project" in command
    assert "--db" in command
    assert str(override) in command


def test_legacy_export_remains_self_contained_and_quotes_database(tmp_path):
    root = tmp_path / "legacy project"
    root.mkdir()
    database = root / "legacy database.sqlite"

    result = export(
        Sheet(), EntityCatalog(), str(root), "sample-import", db_path=str(database)
    )
    authored = yaml.safe_load(
        (root / "configs" / "create_probing_samples.yaml").read_text()
    )

    assert authored["run"]["output_dir"] == ".."
    assert "--db" in result["commands"][-1]
    assert "'" in result["commands"][-1]


def test_webui_serve_uses_global_phase4_project_context(
    cli_runner, tmp_path, monkeypatch,
):
    root = tmp_path / "project"
    project = _init_project(cli_runner, root)
    observed = {}

    def fake_run(application, **kwargs):
        observed["application"] = application
        observed.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = cli_runner.invoke(app, [
        "--project", str(root), "webui", "serve", "--no-open-browser",
    ])

    assert result.exit_code == 0, result.output
    assert "database: %s" % project.database in result.output
    assert "output_directory: %s" % project.output_dir in result.output
    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 8420

    from nerd.webui.app import session
    assert session.project_config == project
    assert session.db_path == project.database


def test_webui_serve_discovers_phase4_project_from_nested_directory(
    cli_runner, tmp_path, monkeypatch,
):
    root = tmp_path / "project"
    project = _init_project(cli_runner, root)
    nested = root / "samples" / "batch-01"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.setattr("uvicorn.run", lambda application, **kwargs: None)

    result = cli_runner.invoke(app, ["webui", "serve", "--no-open-browser"])

    assert result.exit_code == 0, result.output
    assert "project: %s" % root.resolve() in result.output
    assert "database: %s" % project.database in result.output
