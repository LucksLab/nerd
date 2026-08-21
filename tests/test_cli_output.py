"""Semantic CLI help/output assertions that avoid Rich rendering snapshots."""

import pytest
from typer.main import get_command

from nerd.cli import _show_scheduler_row, app


TOP_LEVEL_COMMANDS = {
    "run",
    "init",
    "config",
    "task",
    "plugin",
    "image",
    "db",
    "submit",
    "status",
    "logs",
    "cancel",
    "collect",
    "retry",
    "doctor",
    "prepare-image",
    "ls",
    "webui",
}

PUBLIC_COMMANDS = {"run", "init", "config", "task", "plugin", "image", "db", "webui"}


def test_top_level_help_and_command_inventory(cli_runner):
    root = get_command(app)
    result = cli_runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert set(root.commands) == TOP_LEVEL_COMMANDS
    normalized_help = " ".join(result.output.split())
    assert "NERD: A toolkit for quantitative analysis of RNA reactivity, energetics, and kinetics" in normalized_help
    for command_name in PUBLIC_COMMANDS:
        assert command_name in normalized_help
    for hidden_wrapper in TOP_LEVEL_COMMANDS - PUBLIC_COMMANDS:
        assert "│ %s " % hidden_wrapper not in result.output


@pytest.mark.parametrize(
    ("command_name", "semantic_help"),
    [
        ("run", "scientific workflow synchronously"),
        ("task", "manage durable tasks"),
        ("plugin", "plugin maintenance"),
        ("image", "immutable tool images"),
        ("db", "project database"),
    ],
)
def test_relevant_subcommand_help_remains_available(
    cli_runner, command_name, semantic_help
):
    result = cli_runner.invoke(app, [command_name, "--help"])

    assert result.exit_code == 0
    normalized_help = " ".join(result.output.split())
    assert "Usage" in normalized_help
    assert semantic_help in normalized_help


def test_bare_run_guides_instead_of_erroring(cli_runner):
    from nerd.cli import RunStep

    result = cli_runner.invoke(app, ["run"])

    assert result.exit_code == 0, result.output
    normalized_help = " ".join(result.output.split())
    assert "Error" not in result.output
    assert "scientific workflow synchronously" in normalized_help
    for step in RunStep:
        assert step.value in normalized_help


def test_incomplete_run_guides_with_nonzero_exit(cli_runner):
    result = cli_runner.invoke(app, ["run", "create"])

    assert result.exit_code == 2
    assert "Error" not in result.output
    assert "Usage" in " ".join(result.output.split())


def test_scheduler_row_output_exposes_durable_identifiers(cli_runner, tmp_path, monkeypatch):
    from nerd.scheduler import service

    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")

    def fake_submit_task(conn, step, received_config_path, profile):
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
            str(tmp_path / "controller.sqlite"),
            "submit",
            "create",
            str(config_path),
            "--profile",
            "quest",
        ],
    )

    assert result.exit_code == 0, result.output
    normalized = " ".join(result.output.split())
    for semantic_field in (
        "Task 41 · create",
        "Status submitted",
        "Executor quest · job fake-41",
        "Attempt 1",
        "Next",
        "nerd task logs 41",
        "nerd task watch 41",
    ):
        assert semantic_field in normalized
    assert "Attempt ID" not in normalized


def test_remote_task_output_hides_internal_paths_until_requested(capsys):
    row = {
        "task_id": 45,
        "task_name": "mut_count",
        "task_state": "running",
        "try_index": 3,
        "scheduler_attempt_id": 42,
        "scheduler_state": "running",
        "executor_profile": "quest",
        "executor_type": "ssh_slurm",
        "scheduler_id": "3675949",
        "exit_code": None,
        "error": None,
        "task_message": "RUNNING",
        "unit_label": "HIV_U4_1_HIV_U4",
        "output_dir": "/local/output",
        "remote_workdir": "/scratch/remote-run",
        "log_path": "/local/output/command.log",
    }

    _show_scheduler_row(row, no_color=True)
    concise = " ".join(capsys.readouterr().out.split())
    assert "Task 45 · mut_count" in concise
    assert "quest · Slurm job 3675949" in concise
    assert "Message RUNNING" not in concise
    assert "Attempt ID" not in concise
    assert "/local/output" not in concise
    assert "/scratch/remote-run" not in concise

    _show_scheduler_row(row, detailed=True, no_color=True)
    detailed = " ".join(capsys.readouterr().out.split())
    assert "Attempt ID 42" in detailed
    assert "Remote work /scratch/remote-run" in detailed
    assert "Log /local/output/command.log" in detailed


def test_remote_batch_table_omits_per_unit_log_paths(capsys):
    row = {
        "is_parent": True,
        "task_id": 36,
        "task_name": "mut_count",
        "task_state": "running",
        "counts": {"completed": 1, "running": 1},
        "total_units": 2,
        "output_dir": "/local/output",
        "scheduler_state": "batch",
        "children": [
            {
                "unit_label": "HIV_A27C_1",
                "task_state": "completed",
                "task_id": 37,
                "scheduler_id": "3673074",
                "log_path": "/local/output/rg-1/command.log",
            },
            {
                "unit_label": "HIV_U4_1_HIV_U4",
                "task_state": "running",
                "task_id": 45,
                "scheduler_id": "3673228",
                "log_path": "/local/output/rg-9/command.log",
            },
        ],
    }

    _show_scheduler_row(row, no_color=True)
    output = " ".join(capsys.readouterr().out.split())
    assert "Progress 1/2 complete · 0 failed" in output
    assert "UNIT STATE TASK SCHEDULER JOB" in output
    assert "HIV_U4_1_HIV_U4 running 45 3673228" in output
    assert "LOG" not in output
    assert "command.log" not in output
