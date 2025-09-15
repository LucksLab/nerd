"""
Abstract runner interface for executing shell commands.

Implementations should subclass `Runner` and provide a `run` method
that executes a shell command in a given working directory and returns
the process's exit code. Optional environment variables and timeouts
may be supported by implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict


class Runner(ABC):
    """
    Abstract base class for command runners.
    """

    @abstractmethod
    def run(
        self,
        command: str,
        workdir: Path,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> int:
        """
        Execute a shell command.

        Args:
            command: The shell command to execute.
            workdir: Working directory for the command.
            env: Optional environment variables to set/override.
            timeout: Optional timeout in seconds.

        Returns:
            Process exit code (0 indicates success).
        """
        raise NotImplementedError
