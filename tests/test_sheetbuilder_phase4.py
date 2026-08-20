"""Integration coverage between the sample-input helper and Phase 4 projects."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import BackgroundTasks, HTTPException

from nerd.cli import app
from nerd.configuration import resolve_config, validate_config
from nerd.project import ProjectContext, load_project
from nerd.sheetbuilder.catalog import EntityCatalog
from nerd.sheetbuilder.export import default_nt_rows, export, validate_primer_annotations
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


def _stub_uvicorn_server(monkeypatch):
    observed = {}

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            observed["application"] = config.app
            observed["host"] = config.host
            observed["port"] = config.port

        def run(self):
            observed["ran"] = True

    monkeypatch.setattr("uvicorn.Server", FakeServer)
    return observed


def test_construct_primer_annotations_reject_all_target_labels():
    rows = [
        {"site": 1, "base": "A", "base_region": "1"},
        {"site": 2, "base": "C", "base_region": "1"},
    ]

    with pytest.raises(ValueError, match="cannot consist entirely of 1s"):
        validate_primer_annotations(rows)


def test_construct_primer_annotations_allow_target_and_rt_primer_without_five_prime_primer():
    rows = [
        {"site": 1, "base": "A", "base_region": "1"},
        {"site": 2, "base": "C", "base_region": "2"},
    ]

    validate_primer_annotations(rows)


def test_default_nt_rows_detects_lower_upper_lower_construct_regions():
    sequence = "ggcacctcataacataacTAAGGCAGATCTGAGCCTGGGAGCTCTCTGCCAATCCactaacctcactcacaatc"

    rows = default_nt_rows(sequence)

    first_target = sequence.index("T")
    first_rt_primer = first_target + len("TAAGGCAGATCTGAGCCTGGGAGCTCTCTGCCAATCC")
    assert [row["base_region"] for row in rows[:first_target]] == ["0"] * first_target
    assert [row["base_region"] for row in rows[first_target:first_rt_primer]] == ["1"] * (
        first_rt_primer - first_target
    )
    assert [row["base_region"] for row in rows[first_rt_primer:]] == ["2"] * (
        len(sequence) - first_rt_primer
    )


def test_default_nt_rows_detects_target_and_rt_primer_without_five_prime_primer():
    rows = default_nt_rows("ACGUacgu")

    assert [row["base_region"] for row in rows] == ["1"] * 4 + ["2"] * 4


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


@pytest.mark.parametrize("mode", ["create", "edit", "view", "analyze"])
def test_webui_modes_use_global_phase4_project_context(
    cli_runner, tmp_path, monkeypatch, mode,
):
    root = tmp_path / "project"
    project = _init_project(cli_runner, root)
    observed = _stub_uvicorn_server(monkeypatch)
    result = cli_runner.invoke(app, [
        "--project", str(root), "webui", mode, "--no-open-browser",
    ])

    assert result.exit_code == 0, result.output
    assert "database: %s" % project.database in result.output
    assert "output_directory: %s" % project.output_dir in result.output
    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 8420
    assert observed["ran"] is True

    from nerd.webui.app import session
    import nerd.webui.app as webui_module
    assert session.project_config == project
    assert session.db_path == project.database
    assert webui_module._state(save=False)["mode"] == mode
    if mode in {"view", "analyze"}:
        assert session.conn.execute("PRAGMA query_only").fetchone()[0] == 1


def test_webui_create_discovers_phase4_project_from_nested_directory(
    cli_runner, tmp_path, monkeypatch,
):
    root = tmp_path / "project"
    project = _init_project(cli_runner, root)
    nested = root / "samples" / "batch-01"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    _stub_uvicorn_server(monkeypatch)

    result = cli_runner.invoke(app, ["webui", "create", "--no-open-browser"])

    assert result.exit_code == 0, result.output
    assert "project: %s" % root.resolve() in result.output
    assert "database: %s" % project.database in result.output


def test_webui_serve_command_is_removed(cli_runner):
    result = cli_runner.invoke(app, ["webui", "serve"])

    assert result.exit_code != 0


def test_webui_construct_region_edit_preserves_ids(cli_runner, tmp_path):
    import nerd.webui.app as webui_module

    root = tmp_path / "project"
    project = _init_project(cli_runner, root)
    webui_module.set_mode("edit")
    webui_module.session.connect(str(root), None, "sample_import")
    conn = webui_module.session.conn
    with conn:
        cursor = conn.execute(
            "INSERT INTO meta_constructs (family, name, version, sequence, disp_name) "
            "VALUES (?, ?, ?, ?, ?)",
            ("switch", "WT", "v1", "ACGU", "switch_WT_v1"),
        )
        construct_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO meta_nucleotides (construct_id, site, base, base_region) "
            "VALUES (?, ?, ?, ?)",
            [
                (construct_id, 1, "A", "0"),
                (construct_id, 2, "C", "0"),
                (construct_id, 3, "G", "1"),
                (construct_id, 4, "U", "1"),
            ],
        )
    before = conn.execute(
        "SELECT id, site FROM meta_nucleotides WHERE construct_id = ? ORDER BY site",
        (construct_id,),
    ).fetchall()

    request = webui_module.BaseRegionUpdate(
        construct_id=construct_id,
        rows=[
            {"site": 1, "base_region": "0"},
            {"site": 2, "base_region": "1"},
            {"site": 3, "base_region": "1"},
            {"site": 4, "base_region": "2"},
        ],
    )
    result = webui_module.update_construct_base_regions(request)
    after = conn.execute(
        "SELECT id, site, base_region FROM meta_nucleotides "
        "WHERE construct_id = ? ORDER BY site",
        (construct_id,),
    ).fetchall()

    assert result["changed"] == 2
    assert [(row["id"], row["site"]) for row in before] == [
        (row["id"], row["site"]) for row in after
    ]
    assert [row["base_region"] for row in after] == ["0", "1", "1", "2"]
    assert (root / ".nerd" / "maintenance.jsonl").is_file()


def test_webui_construct_region_edit_rejects_invalid_annotation_atomically(
    cli_runner, tmp_path,
):
    import nerd.webui.app as webui_module

    root = tmp_path / "project"
    _init_project(cli_runner, root)
    webui_module.set_mode("edit")
    webui_module.session.connect(str(root), None, "sample_import")
    conn = webui_module.session.conn
    with conn:
        cursor = conn.execute(
            "INSERT INTO meta_constructs (family, name, version, sequence, disp_name) "
            "VALUES (?, ?, ?, ?, ?)",
            ("switch", "WT", "v1", "AC", "switch_WT_v1"),
        )
        construct_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO meta_nucleotides (construct_id, site, base, base_region) "
            "VALUES (?, ?, ?, ?)",
            [(construct_id, 1, "A", "1"), (construct_id, 2, "C", "2")],
        )

    request = webui_module.BaseRegionUpdate(
        construct_id=construct_id,
        rows=[
            {"site": 1, "base_region": "1"},
            {"site": 2, "base_region": "1"},
        ],
    )
    with pytest.raises(HTTPException, match="cannot consist entirely of 1s"):
        webui_module.update_construct_base_regions(request)

    saved = conn.execute(
        "SELECT base_region FROM meta_nucleotides WHERE construct_id = ? ORDER BY site",
        (construct_id,),
    ).fetchall()
    assert [row["base_region"] for row in saved] == ["1", "2"]


def test_webui_shutdown_stops_registered_server_after_response():
    import nerd.webui.app as webui_module

    server = SimpleNamespace(should_exit=False)
    tasks = BackgroundTasks()
    webui_module.set_server(server)
    try:
        assert webui_module.shutdown_server(tasks) == {"ok": True}
        assert server.should_exit is False
        asyncio.run(tasks())
        assert server.should_exit is True
    finally:
        webui_module.set_server(None)
