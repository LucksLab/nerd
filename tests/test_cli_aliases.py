"""Characterize current workflow names and command-line option aliases."""

from typer.main import get_command

from nerd.cli import RunStep, app
from nerd.pipeline.tasks import TASK_REGISTRY


SCIENTIFIC_WORKFLOWS = (
    "create",
    "mut_count",
    "nmr_create",
    "nmr_kinetic_fit",
    "drop",
    "probe_timecourse",
    "tempgrad_fit",
)


def _command_model():
    return get_command(app)


def _option_spellings(command_name: str):
    command = _command_model().commands[command_name]
    return {parameter.name: set(parameter.opts) for parameter in command.params}


def test_run_and_submit_expose_the_same_scientific_workflow_choices():
    root = _command_model()

    assert tuple(root.commands["run"].params[0].type.choices) == SCIENTIFIC_WORKFLOWS
    assert tuple(root.commands["submit"].params[0].type.choices) == SCIENTIFIC_WORKFLOWS
    assert tuple(step.value for step in RunStep) == SCIENTIFIC_WORKFLOWS


def test_removed_workflow_aliases_are_not_accepted_by_cli_or_registry(
    cli_runner, tmp_path
):
    assert "probe_timecourse" in TASK_REGISTRY
    assert "probe_tc_kinetics" not in TASK_REGISTRY
    assert "tc_free" not in TASK_REGISTRY
    assert "nmr_deg_kinetics" not in TASK_REGISTRY
    assert "nmr_add_kinetics" not in TASK_REGISTRY
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")

    for removed in (
        "probe_tc_kinetics", "tc_free", "nmr_deg_kinetics", "nmr_add_kinetics"
    ):
        result = cli_runner.invoke(app, ["run", removed, str(config_path)])
        assert result.exit_code != 0
        assert "Invalid value" in result.output


def test_current_short_option_aliases_are_characterized():
    root_options = {
        parameter.name: set(parameter.opts) for parameter in _command_model().params
    }

    assert root_options["verbose"] == {"--verbose", "-v"}
    assert _option_spellings("run")["profile"] == {"--profile", "-p"}
    assert _option_spellings("submit")["profile"] == {"--profile", "-p"}
    assert _option_spellings("logs")["tail"] == {"--tail", "-n"}
    assert _option_spellings("ls")["label"] == {"--label", "-l"}
    assert _option_spellings("doctor")["profile"] == {"--profile", "-p"}
    assert _option_spellings("prepare-image")["profile"] == {"--profile", "-p"}
