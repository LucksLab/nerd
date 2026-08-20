"""Typed NERD project configuration and deterministic context discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - exercised on supported Python 3.8-3.10
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_NAME_PATTERN = re.compile(r"^[A-Z]{3}\.[0-9]{2}\.[0-9]{2}\.[0-9]{3}$")
PROJECT_FILE = Path(".nerd/project.toml")
_CREDENTIAL_KEYS = {
    "password", "passphrase", "private_key", "identity_file", "key_file",
    "token", "secret", "api_key", "access_key", "secret_key",
}


class ContextResolutionError(ValueError):
    """Raised when a command has no valid, unambiguous project context."""


class ProjectConfigError(ValueError):
    """Raised when ``.nerd/project.toml`` is malformed or unsafe."""


def validate_project_name(name: str) -> str:
    """Return a canonical project identifier or raise an actionable error."""
    if not PROJECT_NAME_PATTERN.fullmatch(str(name)):
        raise ProjectConfigError(
            "Invalid NERD project name %r. Use uppercase initials and exactly "
            "2, 2, and 3 digits, for example EKC.07.00.000." % name
        )
    return str(name)


@dataclass(frozen=True)
class ProjectConfig:
    """Stable project context loaded from ``.nerd/project.toml``."""

    name: str
    root: Path
    source_path: Path
    database: Path
    output_dir: Path
    default_executor: Optional[str] = None
    executors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    containers: Dict[str, Any] = field(default_factory=dict)

    def analysis_defaults(self) -> Dict[str, Any]:
        run: Dict[str, Any] = {"output_dir": str(self.output_dir)}
        if self.default_executor:
            run["executor"] = self.default_executor
        result: Dict[str, Any] = {"run": run}
        if self.executors:
            result["executors"] = {key: dict(value) for key, value in self.executors.items()}
        if self.containers:
            result["containers"] = dict(self.containers)
        return result


def _resolve_from_root(value: Any, root: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProjectConfigError("project.toml %s must be a non-empty path string." % field_name)
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _reject_credentials(value: Any, location: str = "executors") -> None:
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        key_text = str(key).lower()
        here = "%s.%s" % (location, key)
        if key_text in _CREDENTIAL_KEYS or any(
            part in key_text for part in ("password", "secret", "private_key")
        ):
            raise ProjectConfigError(
                "%s must not contain credentials or private keys; use OpenSSH config, "
                "ssh-agent, or the relevant environment-based credential provider." % here
            )
        _reject_credentials(item, here)


def project_file_for(value: Path, *, require: bool = True) -> Path:
    """Normalize a project root, ``.nerd`` directory, or direct TOML path."""
    candidate = value.expanduser()
    if candidate.name == "project.toml":
        source = candidate
    elif candidate.name == ".nerd":
        source = candidate / "project.toml"
    else:
        source = candidate / PROJECT_FILE
    source = source.resolve()
    if require and not source.is_file():
        raise ContextResolutionError(
            "No .nerd/project.toml found at %s. Run 'nerd init PATH --name EKC.07.00.000' "
            "or pass --db PATH for a legacy project." % source
        )
    return source


def load_project(value: Path) -> ProjectConfig:
    source = project_file_for(value)
    try:
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ProjectConfigError("Malformed TOML in %s: %s" % (source, exc)) from exc
    except OSError as exc:
        raise ProjectConfigError("Could not read %s: %s" % (source, exc)) from exc
    unknown = set(raw) - {"project", "paths", "executors", "containers"}
    if unknown:
        raise ProjectConfigError(
            "%s has unknown top-level section(s): %s." % (source, ", ".join(sorted(unknown)))
        )
    project = raw.get("project") or {}
    paths = raw.get("paths") or {}
    executors = raw.get("executors") or {}
    containers = raw.get("containers") or {}
    for label, table in (("project", project), ("paths", paths),
                         ("executors", executors), ("containers", containers)):
        if not isinstance(table, dict):
            raise ProjectConfigError("%s [%s] must be a table." % (source, label))
    unknown_project = set(project) - {"name", "default_executor"}
    unknown_paths = set(paths) - {"database", "output"}
    if unknown_project or unknown_paths:
        bad = sorted(unknown_project | unknown_paths)
        raise ProjectConfigError("%s has unknown project setting(s): %s." % (source, ", ".join(bad)))
    name = validate_project_name(project.get("name", ""))
    root = source.parent.parent.resolve()
    database = _resolve_from_root(paths.get("database", ".nerd/nerd.sqlite"), root, "paths.database")
    output = _resolve_from_root(paths.get("output", "outputs"), root, "paths.output")
    normalized_executors: Dict[str, Dict[str, Any]] = {}
    for profile_name, profile in executors.items():
        if not isinstance(profile_name, str) or not profile_name:
            raise ProjectConfigError("Executor profile names must be non-empty strings.")
        if isinstance(profile, str):
            profile = {"type": profile}
        if not isinstance(profile, dict):
            raise ProjectConfigError("Executor profile %r must be a table or type string." % profile_name)
        normalized = dict(profile)
        for path_key in ("container_sif", "cache_dir", "script_path"):
            if isinstance(normalized.get(path_key), str):
                normalized[path_key] = str(_resolve_from_root(
                    normalized[path_key], root, "executors.%s.%s" % (profile_name, path_key)
                ))
        normalized_executors[profile_name] = normalized
    unknown_containers = set(containers) - {"cache_dir", "runtime", "runtime_dir"}
    if unknown_containers:
        raise ProjectConfigError(
            "%s has unsupported container setting(s): %s." % (
                source, ", ".join(sorted(unknown_containers))
            )
        )
    normalized_containers = dict(containers)
    for path_key in ("cache_dir", "runtime_dir"):
        if isinstance(normalized_containers.get(path_key), str):
            normalized_containers[path_key] = str(_resolve_from_root(
                normalized_containers[path_key], root, "containers.%s" % path_key
            ))
    _reject_credentials(normalized_containers, "containers")
    _reject_credentials(normalized_executors)
    default_executor = project.get("default_executor")
    if default_executor is not None:
        default_executor = str(default_executor)
        if default_executor not in normalized_executors and default_executor not in {
            "local", "process", "local_process", "slurm", "local_slurm",
            "ssh_slurm", "remote_slurm",
        }:
            raise ProjectConfigError(
                "Default executor %r is not defined in [executors]." % default_executor
            )
    if normalized_executors:
        try:
            from nerd.scheduler.profiles import load_executor_profile
            cfg = {"executors": normalized_executors}
            for profile_name in normalized_executors:
                load_executor_profile(cfg, profile_name)
        except ValueError as exc:
            raise ProjectConfigError(str(exc)) from exc
    return ProjectConfig(
        name=name, root=root, source_path=source, database=database,
        output_dir=output, default_executor=default_executor,
        executors=normalized_executors, containers=normalized_containers,
    )


def discover_project(start: Optional[Path] = None) -> Optional[ProjectConfig]:
    directory = (start or Path.cwd()).expanduser().resolve()
    if directory.is_file():
        directory = directory.parent
    for parent in (directory, *directory.parents):
        source = parent / PROJECT_FILE
        if source.exists():
            if not source.is_file():
                raise ContextResolutionError("Project marker %s is not a regular file." % source)
            try:
                return load_project(source)
            except ProjectConfigError as exc:
                raise ContextResolutionError(str(exc)) from exc
    return None


@dataclass(frozen=True)
class ProjectContext:
    """Invocation-scoped explicit selectors plus project discovery."""

    db: Optional[Path] = None
    project: Optional[Path] = None

    def with_overrides(
        self, *, db: Optional[Path] = None, project: Optional[Path] = None
    ) -> "ProjectContext":
        return ProjectContext(db=db or self.db, project=project or self.project)

    def resolve_project(
        self, *, environ: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None, required: bool = False,
    ) -> Optional[ProjectConfig]:
        env = os.environ if environ is None else environ
        try:
            if self.project is not None:
                return load_project(self.project)
            discovered = discover_project(cwd)
            if discovered is not None:
                return discovered
            if env.get("NERD_PROJECT"):
                return load_project(Path(env["NERD_PROJECT"]))
        except ProjectConfigError as exc:
            raise ContextResolutionError(str(exc)) from exc
        if required:
            raise ContextResolutionError(
                "No NERD project found. Run 'nerd init PATH --name EKC.07.00.000' or pass --project PATH."
            )
        return None

    def resolve_database(
        self, *, config: Optional[Mapping] = None,
        config_path: Optional[Path] = None, must_exist: bool = False,
        environ: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None,
    ) -> Path:
        """Resolve database: explicit, discovered project, environment, YAML."""
        env = os.environ if environ is None else environ
        start = (cwd or Path.cwd()).expanduser().resolve()
        candidate: Optional[Path] = None
        if self.db is not None:
            candidate = self.db
        elif self.project is not None:
            marker = project_file_for(self.project, require=False)
            try:
                candidate = load_project(marker).database if marker.exists() else _legacy_database_for_project(self.project)
            except ProjectConfigError as exc:
                raise ContextResolutionError(str(exc)) from exc
        else:
            project = discover_project(start)
            if project is not None:
                candidate = project.database
            else:
                candidate = _discover_legacy_database(start)
            if candidate is None and env.get("NERD_PROJECT"):
                env_project = Path(env["NERD_PROJECT"])
                marker = project_file_for(env_project, require=False)
                try:
                    candidate = load_project(marker).database if marker.exists() else _legacy_database_for_project(env_project)
                except ProjectConfigError as exc:
                    raise ContextResolutionError(str(exc)) from exc
            elif candidate is None and env.get("NERD_DB"):
                candidate = Path(env["NERD_DB"])
            if candidate is None and config is not None:
                run = config.get("run", {}) or {}
                output = Path(str(run.get("output_dir", "."))).expanduser()
                if not output.is_absolute():
                    base = Path(config_path).resolve().parent if config_path else start
                    output = base / output
                candidate = output / "nerd.sqlite"
        if candidate is None:
            raise ContextResolutionError(
                "No NERD database context found. Pass --db PATH or --project PATH, "
                "set NERD_PROJECT/NERD_DB, run 'nerd init PATH --name EKC.07.00.000', "
                "or use a config with run.output_dir."
            )
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = start / resolved
        resolved = resolved.resolve()
        if must_exist and not resolved.is_file():
            raise ContextResolutionError(
                "NERD database does not exist at %s. Read-only commands never initialize "
                "a database; select an existing one with --db or --project." % resolved
            )
        return resolved


def _legacy_database_for_project(project: Path) -> Path:
    project = project.expanduser()
    if project.name == "project.toml":
        project = project.parent.parent
    if project.suffix in {".sqlite", ".sqlite3", ".db"}:
        return project
    hidden = project / ".nerd" / "nerd.sqlite"
    return hidden if hidden.is_file() else project / "nerd.sqlite"


def _discover_legacy_database(start: Path) -> Optional[Path]:
    for directory in (start, *start.parents):
        hidden = directory / ".nerd" / "nerd.sqlite"
        direct = directory / "nerd.sqlite"
        if hidden.is_file():
            return hidden
        if direct.is_file():
            return direct
    return None


def render_project_toml(
    name: str, *, database: str = ".nerd/nerd.sqlite",
    output: str = "outputs", default_executor: Optional[str] = "local",
) -> str:
    """Deterministically render the small project file controlled by ``nerd init``."""
    validate_project_name(name)
    lines = ["[project]", 'name = "%s"' % name]
    if default_executor:
        lines.append('default_executor = "%s"' % default_executor)
    lines.extend([
        "", "[paths]", 'database = "%s"' % database.replace('"', '\\"'),
        'output = "%s"' % output.replace('"', '\\"'),
    ])
    if default_executor == "local":
        lines.extend(["", "[executors.local]", 'type = "local"'])
    return "\n".join(lines) + "\n"
