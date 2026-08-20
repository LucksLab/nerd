"""Explicit FASTQ storage sources and source-aware filesystem operations."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import shlex
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from nerd.scheduler.profiles import ExecutorProfile, load_executor_profile


LOCAL = "local"
SRA = "sra"
REMOTE_PREFIX = "remote_hpc:"


class FastqSourceError(ValueError):
    pass


def normalize_source(value: Any) -> str:
    """Return the canonical source string, defaulting legacy rows to local."""
    text = str(value or LOCAL).strip()
    lowered = text.lower()
    if lowered == LOCAL:
        return LOCAL
    if lowered == SRA:
        return SRA
    if lowered in {"remote_hpc", "remote-hpc"}:
        raise FastqSourceError("A remote HPC FASTQ source must include a configured alias.")
    for prefix in (REMOTE_PREFIX, "remote_hpc/", "remote-hpc:", "remote-hpc/"):
        if lowered.startswith(prefix):
            alias = text[len(prefix):].strip()
            if not alias:
                raise FastqSourceError("A remote HPC FASTQ source must include a configured alias.")
            return REMOTE_PREFIX + alias
    raise FastqSourceError(
        "Unknown FASTQ source %r; use 'local', 'remote_hpc:<alias>', or 'sra'." % text
    )


def remote_alias(value: Any) -> Optional[str]:
    source = normalize_source(value)
    return source[len(REMOTE_PREFIX):] if source.startswith(REMOTE_PREFIX) else None


def remote_profiles(executors: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Return configured SSH/Slurm profiles suitable for remote FASTQ access."""
    cfg = {"executors": dict(executors or {})}
    choices: List[Dict[str, str]] = []
    for name in executors or {}:
        try:
            profile = load_executor_profile(cfg, str(name))
        except ValueError:
            continue
        if profile.executor_type == "ssh_slurm" and profile.options.get("host"):
            choices.append({
                "value": REMOTE_PREFIX + str(name),
                "label": "Remote HPC — %s" % name,
                "alias": str(name),
                "host": str(profile.options["host"]),
            })
    return choices


def profile_for_source(source: Any, executors: Mapping[str, Any]) -> ExecutorProfile:
    alias = remote_alias(source)
    if alias is None:
        raise FastqSourceError("FASTQ source is not a remote HPC source.")
    try:
        profile = load_executor_profile({"executors": dict(executors or {})}, alias)
    except ValueError as exc:
        raise FastqSourceError(
            "Remote HPC alias %r is not configured. Add it under [executors.%s] "
            "in .nerd/project.toml before creating samples." % (alias, alias)
        ) from exc
    if profile.executor_type != "ssh_slurm" or not profile.options.get("host"):
        raise FastqSourceError(
            "Executor %r must be an ssh_slurm profile with a host to access remote FASTQs."
            % alias
        )
    return profile


def _ssh_base(profile: ExecutorProfile) -> Tuple[List[str], str]:
    host = str(profile.options.get("host") or "")
    user = profile.options.get("user")
    destination = "%s@%s" % (user, host) if user else host
    args = ["ssh"]
    if profile.options.get("port"):
        args.extend(["-p", str(profile.options["port"])])
    options = profile.options.get("ssh_options") or profile.options.get("options")
    if options:
        args.extend(shlex.split(str(options)))
    return args, destination


def _remote_run(profile: ExecutorProfile, command: Iterable[str], timeout: int = 30) -> subprocess.CompletedProcess:
    ssh, destination = _ssh_base(profile)
    try:
        return subprocess.run(
            ssh + [destination, shlex.join([str(item) for item in command])],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise FastqSourceError("OpenSSH is required to access remote HPC FASTQs.") from exc
    except subprocess.TimeoutExpired as exc:
        raise FastqSourceError("Timed out connecting to remote HPC alias %r." % profile.name) from exc


def list_remote_directory(directory: str, profile: ExecutorProfile) -> List[str]:
    """List regular files in a remote directory, returning their basenames."""
    result = _remote_run(profile, ["find", directory, "-maxdepth", "1", "-type", "f", "-print"])
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "remote listing failed").strip()
        raise FastqSourceError("Could not list %s via %s: %s" % (directory, profile.name, message))
    return sorted(PurePosixPath(line.strip()).name for line in result.stdout.splitlines() if line.strip())


def check_remote_fastqs(
    directory: str, filenames: Iterable[str], profile: ExecutorProfile,
    known_files: Optional[Set[str]] = None,
) -> Set[str]:
    """Verify one remote directory and all requested files in a single SSH call."""
    available = known_files if known_files is not None else set(list_remote_directory(directory, profile))
    missing = [str(name) for name in filenames if str(name) and str(name) not in available]
    if missing:
        paths = [str(PurePosixPath(directory) / name) for name in missing]
        raise FileNotFoundError(
            "Remote FASTQ file(s) not found via %s: %s" % (profile.name, ", ".join(paths))
        )
    return available


def local_fastq_paths(directory: str, r1_file: str, r2_file: str, label_dir: Path) -> Tuple[Path, Path]:
    base = Path(directory).expanduser()
    if not base.is_absolute():
        base = label_dir / base
    base = base.resolve()
    return (base / r1_file).resolve(), (base / r2_file).resolve()


def remote_fastq_paths(directory: str, r1_file: str, r2_file: str) -> Tuple[PurePosixPath, PurePosixPath]:
    base = PurePosixPath(directory)
    return base / r1_file, base / r2_file
