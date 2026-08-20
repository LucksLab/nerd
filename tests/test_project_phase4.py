"""Phase 4 project initialization, discovery, and config UX."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nerd.cli import RunStep, app
from nerd.configuration import resolve_config
from nerd.db import api as db_api
from nerd.project import (
    ContextResolutionError, ProjectConfigError, ProjectContext, load_project,
    validate_project_name,
)
from nerd.utils.hashing import config_hash


@pytest.mark.parametrize("name", ["ekc.07.00.000", "EK.07.00.000", "EKCC.07.00.000",
                                  "EKC.7.00.000", "EKC.07.0.000", "EKC.07.00.00"])
def test_project_identifier_rejects_noncanonical_names(name):
    with pytest.raises(ProjectConfigError, match="EKC.07.00.000"):
        validate_project_name(name)


def test_project_identifier_accepts_canonical_name():
    assert validate_project_name("EKC.07.00.000") == "EKC.07.00.000"


def test_init_infers_matching_basename_and_refuses_overwrite(cli_runner, tmp_path):
    root = tmp_path / "EKC.07.00.000"
    first = cli_runner.invoke(app, ["init", str(root), "--json"])
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    assert payload["project_id"] == "EKC.07.00.000"
    assert Path(payload["database"]).is_file()
    assert Path(payload["output_directory"]).is_dir()
    before = (root / ".nerd" / "project.toml").read_text()

    second = cli_runner.invoke(app, ["init", str(root), "--existing"])
    assert second.exit_code == 1
    assert "refusing to overwrite" in second.output.lower()
    assert (root / ".nerd" / "project.toml").read_text() == before


def test_init_nonmatching_name_requires_option_and_existing_is_explicit(cli_runner, tmp_path):
    root = tmp_path / "research"
    root.mkdir()
    (root / "notes.txt").write_text("preserve me")
    missing = cli_runner.invoke(app, ["init", str(root)])
    assert missing.exit_code == 1
    assert "--name EKC.07.00.000" in missing.output
    safe = cli_runner.invoke(app, ["init", str(root), "--name", "EKC.07.00.000"])
    assert safe.exit_code == 1
    assert "--existing" in safe.output
    created = cli_runner.invoke(app, [
        "init", str(root), "--name", "EKC.07.00.000", "--existing"
    ])
    assert created.exit_code == 0, created.output
    assert (root / "notes.txt").read_text() == "preserve me"


def test_project_paths_are_root_relative_and_direct_toml_is_supported(cli_runner, tmp_path):
    root = tmp_path / "project"
    result = cli_runner.invoke(app, [
        "init", str(root), "--name", "EKC.07.00.000",
        "--database", "state/controller.sqlite", "--output-dir", "results",
    ])
    assert result.exit_code == 0, result.output
    project_file = root / ".nerd" / "project.toml"
    cfg = load_project(project_file)
    assert cfg.database == (root / "state" / "controller.sqlite").resolve()
    assert cfg.output_dir == (root / "results").resolve()
    assert ProjectContext(project=project_file).resolve_database() == cfg.database


def test_nested_discovery_beats_environment_and_explicit_beats_discovery(cli_runner, tmp_path, monkeypatch):
    discovered = tmp_path / "discovered"
    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"
    for root, name in ((discovered, "EKC.07.00.000"), (explicit, "ABC.01.02.003"),
                       (environment, "XYZ.98.76.543")):
        result = cli_runner.invoke(app, ["init", str(root), "--name", name])
        assert result.exit_code == 0, result.output
    nested = discovered / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.setenv("NERD_PROJECT", str(environment))
    assert ProjectContext().resolve_database() == (discovered / ".nerd" / "nerd.sqlite").resolve()
    assert ProjectContext(project=explicit).resolve_database() == (explicit / ".nerd" / "nerd.sqlite").resolve()


def test_malformed_nearest_project_is_actionable(tmp_path, monkeypatch):
    root = tmp_path / "bad"
    nested = root / "nested"
    (root / ".nerd").mkdir(parents=True)
    nested.mkdir()
    (root / ".nerd" / "project.toml").write_text("[project\nname = 'EKC.07.00.000'")
    monkeypatch.chdir(nested)
    with pytest.raises(ContextResolutionError, match="Malformed TOML"):
        ProjectContext().resolve_database()


def test_yaml_paths_are_config_relative_while_project_paths_are_root_relative(cli_runner, tmp_path, monkeypatch):
    root = tmp_path / "root"
    initialized = cli_runner.invoke(app, ["init", str(root), "--name", "EKC.07.00.000"])
    assert initialized.exit_code == 0, initialized.output
    config_dir = root / "configs"
    config_dir.mkdir()
    samples = config_dir / "samples.csv"
    samples.write_text("sample_name\n")
    source = config_dir / "create.yaml"
    source.write_text("run:\n  label: import-01\ncreate:\n  samples: samples.csv\n")
    nested = root / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    cfg, project = resolve_config(source)
    assert cfg["create"]["samples"] == str(samples.resolve())
    assert cfg["run"]["output_dir"] == str((root / "outputs").resolve())
    assert project.database == (root / ".nerd" / "nerd.sqlite").resolve()


def test_validate_and_show_are_read_only_and_redact_secrets(cli_runner, tmp_path, monkeypatch):
    root = tmp_path / "uninitialized"
    root.mkdir()
    config = root / "fit.yaml"
    config.write_text(
        "run:\n  label: fit-01\n"
        "executors:\n  local:\n    type: local\n    token: never-print-this\n"
        "tempgrad_fit:\n  mode: arrhenius\n  engine: arrhenius_python\n  data_source: nmr\n"
    )
    monkeypatch.chdir(root)
    validated = cli_runner.invoke(app, ["config", "validate", str(config)])
    assert validated.exit_code == 0, validated.output
    shown = cli_runner.invoke(app, ["config", "show", str(config), "--resolved", "--json"])
    assert shown.exit_code == 0, shown.output
    assert "never-print-this" not in shown.output
    assert "<redacted>" in shown.output
    assert sorted(item.name for item in root.iterdir()) == ["fit.yaml"]


def test_unknown_keys_suggest_correction_and_bad_executor_fails(cli_runner, tmp_path):
    typo = tmp_path / "typo.yaml"
    typo.write_text("run:\n  label: x\nmut_count:\n  plguin: shapemapper\n")
    result = cli_runner.invoke(app, ["config", "validate", str(typo)])
    assert result.exit_code == 1
    assert "Did you mean 'plugin'" in result.output
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "run:\n  label: x\n  executor: missing\n"
        "drop:\n  sample_names: [EKC.07.00.000_sample_001]\n"
    )
    result = cli_runner.invoke(app, ["config", "validate", str(bad)])
    assert result.exit_code == 1
    assert "not defined" in result.output


@pytest.mark.parametrize("workflow", list(RunStep))
def test_config_templates_parse_and_validate(cli_runner, tmp_path, workflow):
    output = tmp_path / (workflow.value + ".yaml")
    generated = cli_runner.invoke(app, [
        "config", "init", workflow.value, "--output", str(output)
    ])
    assert generated.exit_code == 0, generated.output
    assert isinstance(yaml.safe_load(output.read_text()), dict)
    validated = cli_runner.invoke(app, [
        "config", "validate", str(output), "--workflow", workflow.value
    ])
    assert validated.exit_code == 0, validated.output


def test_nested_task_list_uses_initialized_database_without_creating_another(cli_runner, tmp_path, monkeypatch):
    root = tmp_path / "project"
    initialized = cli_runner.invoke(app, ["init", str(root), "--name", "EKC.07.00.000"])
    assert initialized.exit_code == 0, initialized.output
    nested = root / "nested" / "deeper"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    listed = cli_runner.invoke(app, ["task", "list"])
    assert listed.exit_code == 0, listed.output
    assert "No tasks found" in listed.output
    assert not (nested / "nerd.sqlite").exists()


def test_sync_and_detached_runs_share_discovered_project_database(cli_runner, tmp_path, monkeypatch):
    from nerd.pipeline.tasks import TASK_REGISTRY
    from nerd.scheduler import service

    root = tmp_path / "project"
    initialized = cli_runner.invoke(app, ["init", str(root), "--name", "EKC.07.00.000"])
    assert initialized.exit_code == 0, initialized.output
    config = root / "run.yaml"
    config.write_text("run:\n  label: analysis-01\ncreate: {}\n")
    nested = root / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    observed = []

    class RecordingTask:
        def exec(self, conn, cfg, verbose=False):
            observed.append(Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve())

    def fake_submit(conn, step, config_path, profile, resolved_config=None):
        observed.append(Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve())
        return {
            "task_id": 1, "task_name": step, "task_state": "submitted", "try_index": 1,
            "scheduler_attempt_id": 1, "scheduler_state": "queued",
            "executor_profile": "local", "executor_type": "local", "scheduler_id": "1",
            "exit_code": None, "error": None,
        }

    monkeypatch.setitem(TASK_REGISTRY, "create", RecordingTask)
    synchronous = cli_runner.invoke(app, ["run", "create", str(config)])
    assert synchronous.exit_code == 0, synchronous.output
    monkeypatch.setattr(service, "submit_task", fake_submit)
    detached = cli_runner.invoke(app, ["run", "create", str(config), "--detach"])
    assert detached.exit_code == 0, detached.output
    assert observed == [
        (root / ".nerd" / "nerd.sqlite").resolve(),
        (root / ".nerd" / "nerd.sqlite").resolve(),
    ]


def test_project_executor_rejects_credentials_and_unknown_reference(tmp_path):
    marker = tmp_path / ".nerd" / "project.toml"
    marker.parent.mkdir()
    marker.write_text(
        "[project]\nname = 'EKC.07.00.000'\ndefault_executor = 'quest'\n"
        "[paths]\ndatabase = '.nerd/nerd.sqlite'\noutput = 'outputs'\n"
        "[executors.quest]\ntype = 'ssh_slurm'\nhost = 'quest'\npassword = 'nope'\n"
    )
    with pytest.raises(ProjectConfigError, match="must not contain credentials"):
        load_project(marker)
    marker.write_text(
        "[project]\nname = 'EKC.07.00.000'\ndefault_executor = 'missing'\n"
        "[paths]\ndatabase = '.nerd/nerd.sqlite'\noutput = 'outputs'\n"
    )
    with pytest.raises(ProjectConfigError, match="not defined"):
        load_project(marker)


def test_project_identity_and_location_do_not_change_scientific_config_hash(cli_runner, tmp_path):
    roots = [tmp_path / "one", tmp_path / "two"]
    names = ["EKC.07.00.000", "ABC.01.02.003"]
    for root, name in zip(roots, names):
        initialized = cli_runner.invoke(app, ["init", str(root), "--name", name])
        assert initialized.exit_code == 0, initialized.output
    source = tmp_path / "fit.yaml"
    source.write_text(
        "run:\n  label: fit-01\n"
        "tempgrad_fit:\n  mode: arrhenius\n  data_source: nmr\n"
    )
    first, _ = resolve_config(source, context=ProjectContext(project=roots[0]))
    second, _ = resolve_config(source, context=ProjectContext(project=roots[1]))
    assert first["run"]["output_dir"] != second["run"]["output_dir"]
    assert config_hash(first, length=64) == config_hash(second, length=64)


def test_analysis_yaml_cannot_redefine_project_identity(cli_runner, tmp_path):
    root = tmp_path / "root"
    initialized = cli_runner.invoke(app, ["init", str(root), "--name", "EKC.07.00.000"])
    assert initialized.exit_code == 0, initialized.output
    source = root / "bad.yaml"
    source.write_text(
        "project_name: ABC.01.02.003\nrun:\n  label: fit-01\n"
        "tempgrad_fit:\n  mode: arrhenius\n"
    )
    result = cli_runner.invoke(app, ["config", "validate", str(source), "--project", str(root)])
    assert result.exit_code == 1
    assert "must not redefine project identity" in result.output
