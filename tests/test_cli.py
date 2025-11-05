import sqlite3
from pathlib import Path

import yaml

from nerd.cli import app


def _write_create_config(base_dir: Path) -> Path:
    """
    Create a minimal yet valid create-task configuration and return its path.
    This mirrors the structure that CreateTask.prepare expects on disk.
    """
    output_dir = base_dir / "outputs"
    label = "cli_run"
    label_dir = output_dir / label
    fq_dir = label_dir / "fastqs"

    fq_dir.mkdir(parents=True, exist_ok=True)
    (fq_dir / "R1.fastq.gz").write_text("placeholder\n")
    (fq_dir / "R2.fastq.gz").write_text("placeholder\n")

    cfg = {
        "run": {
            "label": label,
            "output_dir": str(output_dir),
            "backend": "local",
        },
        "create": {
            "construct": {
                "family": "TestFamily",
                "name": "TestConstruct",
                "version": 1,
                "sequence": "ACGT",
                "disp_name": "test_construct",
            },
            "buffer": {
                "name": "TestBuffer",
                "pH": 7.0,
                "composition": "Mock buffer",
                "disp_name": "test_buffer",
            },
            "sequencing_run": {
                "run_name": "RUN001",
                "date": "20240101",
                "sequencer": "Illumina_MiSeq",
                "run_manager": "Tester",
            },
            "samples": [
                {
                    "sample_name": "sample_1",
                    "fq_dir": "fastqs",
                    "r1_file": "R1.fastq.gz",
                    "r2_file": "R2.fastq.gz",
                    "reaction_group": "rg_1",
                    "temperature": 42,
                    "replicate": 1,
                    "reaction_time": 20,
                    "probe": "dms",
                    "probe_concentration": 0.01,
                    "rt_protocol": "standard",
                    "treated": 1,
                    "construct": "test_construct",
                    "buffer": "test_buffer",
                    "done_by": "Tester",
                }
            ],
        },
    }

    cfg_path = base_dir / "create_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_cli_help_lists_commands(cli_runner):
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout
    assert "run   Execute a specific step" in result.stdout
    assert "ls    List available runs" in result.stdout


def test_cli_run_create_populates_database(cli_runner, tmp_path):
    cfg_path = _write_create_config(tmp_path)
    db_path = tmp_path / "nerd.sqlite"

    result = cli_runner.invoke(app, ["--db", str(db_path), "run", "create", str(cfg_path)])
    assert result.exit_code == 0, result.stdout

    assert db_path.is_file()

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM meta_constructs").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM meta_buffers").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM sequencing_samples").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM probe_reaction_groups").fetchone()[0] == 1
    finally:
        conn.close()

    run_logs = cfg_path.parent / "outputs" / "run_logs"
    assert run_logs.is_dir()


def test_cli_run_unknown_step_errors(cli_runner, tmp_path):
    cfg_path = _write_create_config(tmp_path)
    result = cli_runner.invoke(app, ["run", "nonexistent_step", str(cfg_path)])
    assert result.exit_code != 0
    message = (result.stderr or result.stdout).lower()
    assert "invalid value" in message
