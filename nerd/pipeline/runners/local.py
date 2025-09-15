# nerd/pipeline/runners/local.py
"""
A simple runner for executing commands on the local machine.
"""

from pathlib import Path
import os
import subprocess
from typing import Optional, Dict

from nerd.utils.logging import get_logger
from nerd.utils.paths import get_command_log_path
from .base import Runner


class LocalRunner(Runner):
    """
    Executes a shell command in a specified working directory and logs its output.
    """

    def run(
        self,
        command: str,
        workdir: Path,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> int:
        """
        Runs the command, captures its output, and returns the exit code.

        Args:
            command: The shell command to execute.
            workdir: The working directory for the command.
            env: Optional environment variables to set/override.
            timeout: Optional timeout in seconds.

        Returns:
            The integer exit code of the command.
        """
        log = get_logger(__name__)
        log_path = get_command_log_path(workdir)
        log.info("Executing command locally. Log file: %s", log_path)

        try:
            merged_env = os.environ.copy()
            if env:
                merged_env.update(env)

            with open(log_path, 'w') as log_file:
                process = subprocess.run(
                    command,
                    shell=True,
                    check=False,  # We handle the non-zero exit code manually
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=workdir,
                    text=True,
                    env=merged_env,
                    timeout=timeout,
                )

            log.info("Command finished with exit code: %d", process.returncode)
            return process.returncode

        except subprocess.TimeoutExpired:
            log.error("Command timed out after %s seconds", timeout)
            return 124  # common timeout code
        except Exception as e:
            log.exception("LocalRunner failed to execute command: %s", e)
            return -1  # Return a non-zero code to indicate failure
