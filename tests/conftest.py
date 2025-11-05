import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Reusable Typer CLI runner with stderr merged into stdout for assertions."""
    return CliRunner()
