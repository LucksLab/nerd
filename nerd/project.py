"""Deterministic project and controller-database resolution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, Optional


class ContextResolutionError(ValueError):
    """Raised when a command has no unambiguous NERD database context."""


@dataclass(frozen=True)
class ProjectContext:
    """Invocation-scoped inputs used to select a NERD database."""

    db: Optional[Path] = None
    project: Optional[Path] = None

    def with_overrides(
        self, *, db: Optional[Path] = None, project: Optional[Path] = None
    ) -> "ProjectContext":
        return ProjectContext(db=db or self.db, project=project or self.project)

    def resolve_database(
        self,
        *,
        config: Optional[Mapping] = None,
        config_path: Optional[Path] = None,
        must_exist: bool = False,
        environ: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None,
    ) -> Path:
        """Resolve a database using one documented precedence order."""
        env = os.environ if environ is None else environ
        start = (cwd or Path.cwd()).expanduser().resolve()

        candidate: Optional[Path] = None
        if self.db is not None:
            candidate = self.db
        elif self.project is not None:
            candidate = _database_for_project(self.project)
        else:
            candidate = _discover_database(start)
            if candidate is None and env.get("NERD_DB"):
                candidate = Path(env["NERD_DB"])
            if candidate is None and env.get("NERD_PROJECT"):
                candidate = _database_for_project(Path(env["NERD_PROJECT"]))
            if candidate is None and config is not None:
                run = config.get("run", {}) or {}
                output = Path(str(run.get("output_dir", "."))).expanduser()
                if not output.is_absolute():
                    base = Path(config_path).resolve().parent if config_path else start
                    output = base / output
                candidate = output / "nerd.sqlite"

        if candidate is None:
            raise ContextResolutionError(
                "No NERD database context found. Pass --db PATH or --project DIR, "
                "set NERD_DB/NERD_PROJECT, or run the command from a directory "
                "containing an existing nerd.sqlite database."
            )

        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = start / resolved
        resolved = resolved.resolve()
        if must_exist and not resolved.is_file():
            raise ContextResolutionError(
                "NERD database does not exist at %s. Inspection commands never create "
                "a database; select an existing one with --db or --project." % resolved
            )
        return resolved


def _database_for_project(project: Path) -> Path:
    project = project.expanduser()
    if project.suffix in {".sqlite", ".sqlite3", ".db"}:
        return project
    hidden = project / ".nerd" / "nerd.sqlite"
    if hidden.is_file():
        return hidden
    return project / "nerd.sqlite"


def _discover_database(start: Path) -> Optional[Path]:
    for directory in (start, *start.parents):
        hidden = directory / ".nerd" / "nerd.sqlite"
        direct = directory / "nerd.sqlite"
        if hidden.is_file():
            return hidden
        if direct.is_file():
            return direct
    return None
