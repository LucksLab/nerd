"""Characterize current config/path resolution before project-aware resolution."""

from pathlib import Path

from nerd.cli import _scheduler_connection, app
from nerd.pipeline.tasks import TASK_REGISTRY
from nerd.utils.config import load_config
from nerd.utils.hashing import config_hash


def test_load_config_resolves_paths_relative_to_source(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "run:\n  label: relative\n  output_dir: relative-output\n"
        "create:\n  source: data/samples.csv\n"
    )

    loaded = load_config(config_path)

    assert loaded.source_path == config_path.resolve()
    assert loaded.base_dir == tmp_path.resolve()
    assert loaded["run"]["output_dir"] == str((tmp_path / "relative-output").resolve())
    assert loaded["create"]["source"] == str((tmp_path / "data/samples.csv").resolve())
    assert config_hash(loaded) == config_hash(
        {
            "run": {"label": "relative", "output_dir": "relative-output"},
            "create": {"source": "data/samples.csv"},
        }
    )


def test_run_output_and_default_database_are_resolved_from_config(
    cli_runner, tmp_path, monkeypatch
):
    invocation_dir = tmp_path / "invocation"
    config_dir = tmp_path / "configs"
    invocation_dir.mkdir()
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "run:\n  label: relative\n  output_dir: relative-output\n"
    )
    observed = {}

    class RecordingTask:
        def exec(self, conn, cfg, verbose=False):
            observed["db"] = Path(
                conn.execute("PRAGMA database_list").fetchone()[2]
            ).resolve()
            observed["output_dir"] = cfg["run"]["output_dir"]

    monkeypatch.setitem(TASK_REGISTRY, "create", RecordingTask)
    monkeypatch.chdir(invocation_dir)

    result = cli_runner.invoke(app, ["run", "create", str(config_path)])

    config_output = config_dir / "relative-output"
    assert result.exit_code == 0, result.output
    assert observed == {
        "db": (config_output / "nerd.sqlite").resolve(),
        "output_dir": str(config_output.resolve()),
    }
    assert (config_output / "run_logs").is_dir()
    assert (config_output / "nerd.sqlite").exists()
    assert not (invocation_dir / "relative-output").exists()


def test_scheduler_database_from_config_is_resolved_from_config(
    tmp_path, monkeypatch
):
    invocation_dir = tmp_path / "invocation"
    config_dir = tmp_path / "configs"
    invocation_dir.mkdir()
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text("run:\n  output_dir: scheduler-output\n")
    monkeypatch.chdir(invocation_dir)
    conn = _scheduler_connection(config_path)
    try:
        selected_db = Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve()
    finally:
        conn.close()

    assert selected_db == (config_dir / "scheduler-output" / "nerd.sqlite").resolve()
    assert (config_dir / "scheduler-output" / "nerd.sqlite").exists()
    assert not (invocation_dir / "scheduler-output").exists()
