import json
import subprocess

import pytest

from nerd.scheduler.executors import _render_script, RESULT_FILE
from nerd.scheduler.models import JobSpec


@pytest.mark.parametrize('exit_code', [0, 7])
def test_large_command_runs_and_records_exit_status(tmp_path, exit_code):
    # A single argument this large exceeds Linux's per-argument limit.
    command = '# ' + ('x' * 200_000) + '\nprintf "%%s\\n" "$NERD_TEST_VALUE"\nexit %d' % exit_code
    spec = JobSpec(command, tmp_path, env={'NERD_TEST_VALUE': 'literal $value'})
    script = tmp_path / 'job.sh'
    script.write_text(_render_script(spec, str(tmp_path)))
    result = subprocess.run(['bash', str(script)], capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    assert result.stdout.strip() == 'literal $value'
    assert json.loads((tmp_path / RESULT_FILE).read_text())['exit_code'] == exit_code


def test_command_can_contain_heredoc_delimiter(tmp_path):
    command = "cat <<'END'\nNERD_COMMAND_EOF\nEND\nprintf '%s' '$(echo not-expanded)'"
    script = tmp_path / 'job.sh'
    script.write_text(_render_script(JobSpec(command, tmp_path), str(tmp_path)))
    result = subprocess.run(['bash', str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'NERD_COMMAND_EOF\n$(echo not-expanded)'
