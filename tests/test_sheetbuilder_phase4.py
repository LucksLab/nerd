"""Integration coverage between the sample-input helper and Phase 4 projects."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import BackgroundTasks, HTTPException

from nerd.cli import WEBUI_DEFAULT_PORTS, app
from nerd.configuration import resolve_config, validate_config
from nerd.project import ProjectContext, load_project
from nerd.sheetbuilder import fillers
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

    monkeypatch.setattr("nerd.cli._ensure_webui_port_available", lambda *args: None)

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


def test_autofill_down_continues_numeric_series_and_repeating_values():
    sheet = Sheet()
    for temperature, probe in [(10, "DMS"), (20, "NMIA"), ("", ""), ("", ""), ("", "")]:
        sheet.add_row({"temperature": temperature, "probe": probe})

    result = fillers.autofill_down(
        sheet,
        ["temperature", "probe"],
        [sheet.rows[0].uid, sheet.rows[1].uid],
    )

    assert [row.get("temperature") for row in sheet.rows] == [10, 20, 30, 40, 50]
    assert [row.get("probe") for row in sheet.rows] == ["DMS", "NMIA", "DMS", "NMIA", "DMS"]
    assert result == {
        "written": 6,
        "rows_filled": 3,
        "columns": ["temperature", "probe"],
    }


def test_autofill_down_requires_contiguous_two_row_source():
    sheet = Sheet()
    for value in (1, 2, 3):
        sheet.add_row({"replicate": value})

    with pytest.raises(ValueError, match="contiguous"):
        fillers.autofill_down(
            sheet,
            ["replicate"],
            [sheet.rows[0].uid, sheet.rows[2].uid],
        )


def test_autofill_down_avoids_floating_point_artifacts():
    sheet = Sheet()
    for value in ("0.1", "0.2", "", ""):
        sheet.add_row({"reaction_time": value})

    fillers.autofill_down(
        sheet,
        ["reaction_time"],
        [sheet.rows[0].uid, sheet.rows[1].uid],
    )

    assert [row.get("reaction_time") for row in sheet.rows] == ["0.1", "0.2", 0.3, 0.4]


def test_pattern_split_construct_fills_display_name_and_creation_context():
    sheet = Sheet()
    row = sheet.add_row({"sample_name": "HIV_A27C_tp1_p"})

    result = fillers.pattern_fill(
        sheet,
        "[construct_family]_[construct_name]_[tp_num]_[treated]",
        Session().registry,
    )

    assert row.get("construct") == "HIV_A27C"
    assert row.get("treated") == 1
    assert row.context["construct_family"] == "HIV"
    assert row.context["construct_name"] == "A27C"
    assert row.context["tp_num"] == 1
    assert result["columns_written"] == ["construct", "treated"]


def test_pattern_explicit_construct_takes_precedence_over_split_construct():
    sheet = Sheet()
    row = sheet.add_row({"sample_name": "existing_HIV_A27C"})

    fillers.pattern_fill(
        sheet,
        "[construct]_[construct_family]_[construct_name]",
        Session().registry,
    )

    assert row.get("construct") == "existing"
    assert row.context["construct_family"] == "HIV"
    assert row.context["construct_name"] == "A27C"


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
    assert observed["port"] == WEBUI_DEFAULT_PORTS[mode]
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


def test_webui_explicit_port_override_is_preserved(cli_runner, tmp_path, monkeypatch):
    root = tmp_path / "project"
    _init_project(cli_runner, root)
    observed = _stub_uvicorn_server(monkeypatch)

    result = cli_runner.invoke(app, [
        "--project", str(root), "webui", "edit", "--port", "9123",
        "--no-open-browser",
    ])

    assert result.exit_code == 0, result.output
    assert observed["port"] == 9123


def test_webui_refuses_an_occupied_port_before_starting(
    cli_runner, tmp_path, monkeypatch,
):
    root = tmp_path / "project"
    _init_project(cli_runner, root)

    class BusySocket:
        def bind(self, address):
            raise OSError("address already in use")

        def close(self):
            pass

    monkeypatch.setattr(
        "nerd.cli.socket.socket", lambda *args, **kwargs: BusySocket()
    )
    port = 9124
    result = cli_runner.invoke(app, [
        "--project", str(root), "webui", "analyze", "--port", str(port),
        "--no-open-browser",
    ])

    assert result.exit_code == 1
    assert "Port %s is already in use" % port in result.output


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


def test_webui_edit_can_correct_probe_sample_creation_fields(cli_runner, tmp_path):
    import nerd.webui.app as webui_module

    root = tmp_path / "project"
    _init_project(cli_runner, root)
    webui_module.set_mode("edit")
    webui_module.session.connect(str(root), None, "sample_import")
    conn = webui_module.session.conn
    with conn:
        construct_id = conn.execute(
            "INSERT INTO meta_constructs (family, name, version, sequence, disp_name) "
            "VALUES ('switch', 'WT', 'v1', 'AC', 'switch_WT')"
        ).lastrowid
        buffer_id = conn.execute(
            "INSERT INTO meta_buffers (name, pH, composition, disp_name) "
            "VALUES ('fold', 7.0, 'salt', 'folding')"
        ).lastrowid
        seqrun_id = conn.execute(
            "INSERT INTO sequencing_runs (run_name, date, sequencer, run_manager) "
            "VALUES ('run-1', '2026-08-21', 'NovaSeq', 'EKC')"
        ).lastrowid
        sample_id = conn.execute(
            "INSERT INTO sequencing_samples "
            "(seqrun_id, sample_name, fq_source, fq_dir, r1_file, r2_file) "
            "VALUES (?, 'sample_q', 'local', '/reads', 'r1.fastq.gz', 'r2.fastq.gz')",
            (seqrun_id,),
        ).lastrowid
        conn.execute(
            "INSERT INTO probe_reaction_groups (rg_id, rg_label) VALUES (1, 'group-1')"
        )
        reaction_id = conn.execute(
            "INSERT INTO probe_reactions "
            "(rg_id, s_id, construct_id, buffer_id, temperature, replicate, "
            "reaction_time, probe_concentration, probe, rt_protocol, done_by, treated) "
            "VALUES (1, ?, ?, ?, 25, 1, 30, 0.01, 'DMS', 'MRT', 'EKC', 'q')",
            (sample_id, construct_id, buffer_id),
        ).lastrowid

    catalog = webui_module.database_entities()
    record = catalog["entities"]["probe_sample"][0]
    assert record["sample_name"] == "sample_q"
    assert record["treated"] == "q"
    assert catalog["reaction_groups"] == [{"rg_id": 1, "rg_label": "group-1"}]

    request = webui_module.ProbeSampleUpdate(**{
        **record,
        "sample_name": "sample_corrected",
        "treated": 1,
        "reaction_time": 45,
    })
    result = webui_module.update_probe_sample(request)

    assert result["record"]["sample_name"] == "sample_corrected"
    assert result["record"]["treated"] == 1
    assert result["record"]["reaction_time"] == 45
    assert result["record"]["sample_id"] == sample_id
    assert result["record"]["reaction_id"] == reaction_id
    assert conn.execute(
        "SELECT sample_name FROM sequencing_samples WHERE id = ?", (sample_id,)
    ).fetchone()[0] == "sample_corrected"
    assert conn.execute(
        "SELECT treated FROM probe_reactions WHERE id = ?", (reaction_id,)
    ).fetchone()[0] == 1


def test_webui_probe_sample_edit_rejects_invalid_treated_atomically(cli_runner, tmp_path):
    import nerd.webui.app as webui_module

    root = tmp_path / "project"
    _init_project(cli_runner, root)
    webui_module.set_mode("edit")
    webui_module.session.connect(str(root), None, "sample_import")
    conn = webui_module.session.conn
    with conn:
        construct_id = conn.execute(
            "INSERT INTO meta_constructs (family, name, version, sequence, disp_name) "
            "VALUES ('switch', 'WT', 'v1', 'AC', 'switch_WT')"
        ).lastrowid
        buffer_id = conn.execute(
            "INSERT INTO meta_buffers (name, pH, composition, disp_name) "
            "VALUES ('fold', 7.0, 'salt', 'folding')"
        ).lastrowid
        seqrun_id = conn.execute(
            "INSERT INTO sequencing_runs (run_name, date, sequencer, run_manager) "
            "VALUES ('run-1', '2026-08-21', 'NovaSeq', 'EKC')"
        ).lastrowid
        sample_id = conn.execute(
            "INSERT INTO sequencing_samples "
            "(seqrun_id, sample_name, fq_source, fq_dir, r1_file, r2_file) "
            "VALUES (?, 'sample', 'local', '/reads', 'r1.fastq.gz', 'r2.fastq.gz')",
            (seqrun_id,),
        ).lastrowid
        conn.execute("INSERT INTO probe_reaction_groups (rg_id, rg_label) VALUES (1, 'group-1')")
        conn.execute(
            "INSERT INTO probe_reactions "
            "(rg_id, s_id, construct_id, buffer_id, temperature, replicate, "
            "reaction_time, probe_concentration, probe, rt_protocol, done_by, treated) "
            "VALUES (1, ?, ?, ?, 25, 1, 30, 0.01, 'DMS', 'MRT', 'EKC', 1)",
            (sample_id, construct_id, buffer_id),
        )

    record = webui_module.database_entities()["entities"]["probe_sample"][0]
    request = webui_module.ProbeSampleUpdate(**{**record, "sample_name": "changed", "treated": 9})
    with pytest.raises(HTTPException, match="treated must be"):
        webui_module.update_probe_sample(request)
    assert conn.execute(
        "SELECT sample_name FROM sequencing_samples WHERE id = ?", (sample_id,)
    ).fetchone()[0] == "sample"


def test_webui_probe_sample_spreadsheet_bulk_save_is_atomic(cli_runner, tmp_path):
    import nerd.webui.app as webui_module

    root = tmp_path / "project"
    _init_project(cli_runner, root)
    webui_module.set_mode("edit")
    webui_module.session.connect(str(root), None, "sample_import")
    conn = webui_module.session.conn
    with conn:
        construct_id = conn.execute(
            "INSERT INTO meta_constructs (family, name, version, sequence, disp_name) "
            "VALUES ('switch', 'WT', 'v1', 'AC', 'switch_WT')"
        ).lastrowid
        buffer_id = conn.execute(
            "INSERT INTO meta_buffers (name, pH, composition, disp_name) "
            "VALUES ('fold', 7.0, 'salt', 'folding')"
        ).lastrowid
        seqrun_id = conn.execute(
            "INSERT INTO sequencing_runs (run_name, date, sequencer, run_manager) "
            "VALUES ('run-1', '2026-08-21', 'NovaSeq', 'EKC')"
        ).lastrowid
        conn.execute("INSERT INTO probe_reaction_groups (rg_id, rg_label) VALUES (1, 'group-1')")
        for index in (1, 2):
            sample_id = conn.execute(
                "INSERT INTO sequencing_samples "
                "(seqrun_id, sample_name, fq_source, fq_dir, r1_file, r2_file) "
                "VALUES (?, ?, 'local', '/reads', ?, ?)",
                (seqrun_id, "sample-%s" % index, "r%s-1.fastq.gz" % index,
                 "r%s-2.fastq.gz" % index),
            ).lastrowid
            conn.execute(
                "INSERT INTO probe_reactions "
                "(rg_id, s_id, construct_id, buffer_id, temperature, replicate, "
                "reaction_time, probe_concentration, probe, rt_protocol, done_by, treated) "
                "VALUES (1, ?, ?, ?, 25, ?, 30, 0.01, 'DMS', 'MRT', 'EKC', 1)",
                (sample_id, construct_id, buffer_id, index),
            )

    records = webui_module.database_entities()["entities"]["probe_sample"]
    requests = [
        webui_module.ProbeSampleUpdate(**{
            **records[0], "sample_name": "changed-1", "reaction_time": 45,
        }),
        webui_module.ProbeSampleUpdate(**{
            **records[1], "sample_name": "changed-2", "treated": 9,
        }),
    ]
    with pytest.raises(HTTPException, match="treated must be"):
        webui_module.update_probe_samples_bulk(
            webui_module.ProbeSampleBulkUpdate(records=requests)
        )
    assert [row[0] for row in conn.execute(
        "SELECT sample_name FROM sequencing_samples ORDER BY id"
    ).fetchall()] == ["sample-1", "sample-2"]

    requests[1].treated = 0
    result = webui_module.update_probe_samples_bulk(
        webui_module.ProbeSampleBulkUpdate(records=requests)
    )
    assert result["changed_records"] == 2
    assert [row[0] for row in conn.execute(
        "SELECT sample_name FROM sequencing_samples ORDER BY id"
    ).fetchall()] == ["changed-1", "changed-2"]


def test_webui_create_table_applies_pasted_cells_as_one_batch(cli_runner, tmp_path):
    import nerd.webui.app as webui_module

    root = tmp_path / "project"
    _init_project(cli_runner, root)
    webui_module.set_mode("create")
    webui_module.session.connect(str(root), None, "sample_import")
    first = webui_module.session.sheet.add_row({"sample_name": "sample-1"})
    second = webui_module.session.sheet.add_row({"sample_name": "sample-2"})

    result = webui_module.edit_cells(webui_module.CellEdits(edits=[
        {"uid": first.uid, "column": "probe", "value": "DMS"},
        {"uid": first.uid, "column": "treated", "value": 1},
        {"uid": second.uid, "column": "probe", "value": "DMS"},
        {"uid": second.uid, "column": "treated", "value": 0},
    ]))

    assert result["edited"] == 4
    assert first.get("probe") == second.get("probe") == "DMS"
    assert first.get("treated") == 1
    assert second.get("treated") == 0
    assert first.origin("probe") == "manual"


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
