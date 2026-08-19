"""Named executor profile loading, including legacy run configuration shims."""

from __future__ import annotations

from dataclasses import dataclass, field
import shlex
from typing import Any, Dict, Optional


@dataclass
class ExecutorProfile:
    name: str
    executor_type: str
    options: Dict[str, Any] = field(default_factory=dict)


_TYPE_ALIASES = {
    "local": "local",
    "process": "local",
    "local_process": "local",
    "slurm": "slurm",
    "local_slurm": "slurm",
    "ssh_slurm": "ssh_slurm",
    "remote_slurm": "ssh_slurm",
}


def _legacy_profile(cfg: Dict[str, Any]) -> ExecutorProfile:
    run = cfg.get("run", {}) or {}
    backend = str(run.get("backend", "local")).lower()
    remote = run.get("remote") or run.get("slurm") or run.get("ssh") or {}
    slurm = remote.get("slurm_config") or run.get("slurm_config") or {}
    ssh = remote.get("ssh") or {}
    options = dict(slurm)
    if remote.get("remote_base_dir"):
        options["remote_base_dir"] = remote["remote_base_dir"]
    if remote.get("preamble"):
        options["preamble"] = remote["preamble"]
    if remote.get("stage_out"):
        options["stage_out"] = remote["stage_out"]
    options.update({key: value for key, value in ssh.items() if value not in (None, "")})
    if backend in {"slurm"} and not ssh.get("host"):
        executor_type = "slurm"
    elif backend in {"slurm", "remote_slurm", "ssh", "remote"}:
        executor_type = "ssh_slurm"
    else:
        executor_type = "local"
    return ExecutorProfile(name=backend or "local", executor_type=executor_type, options=options)


def load_executor_profile(
    cfg: Dict[str, Any], requested: Optional[str] = None
) -> ExecutorProfile:
    """Resolve a named profile, falling back to the existing ``run.backend`` shape."""
    run = cfg.get("run", {}) or {}
    name = requested or run.get("executor") or run.get("executor_profile")
    profiles = cfg.get("executors") or run.get("executors") or {}
    if name:
        if name not in profiles:
            if name in _TYPE_ALIASES:
                return ExecutorProfile(name=name, executor_type=_TYPE_ALIASES[name])
            raise ValueError("Executor profile '%s' is not defined." % name)
        raw = profiles[name] or {}
        if isinstance(raw, str):
            raw = {"type": raw}
        executor_type = _TYPE_ALIASES.get(str(raw.get("type", name)).lower())
        if executor_type is None:
            raise ValueError("Executor profile '%s' has unknown type '%s'." % (name, raw.get("type")))
        profile = ExecutorProfile(name=str(name), executor_type=executor_type,
                                  options={k: v for k, v in raw.items() if k != "type"})
        _validate_ssh_auth(profile)
        return profile
    profile = _legacy_profile(cfg)
    _validate_ssh_auth(profile)
    return profile


def _validate_ssh_auth(profile: ExecutorProfile) -> None:
    if profile.executor_type != "ssh_slurm":
        return
    forbidden = {"password", "private_key", "identity_file", "key_file"}
    supplied = forbidden.intersection(profile.options)
    options = str(profile.options.get("ssh_options") or profile.options.get("options") or "")
    tokens = shlex.split(options)
    if supplied or "-i" in tokens or any(token.lower().startswith("identityfile=") for token in tokens):
        raise ValueError(
            "SSH credentials must not be stored in a NERD profile; use OpenSSH config and ssh-agent."
        )


def profile_resources(profile: ExecutorProfile, run: Dict[str, Any]) -> Dict[str, Any]:
    """Combine task defaults and scheduler profile resource settings."""
    resources = dict(profile.options.get("resources") or {})
    for key in ("partition", "account", "qos", "constraint", "time"):
        if key in profile.options and key not in resources:
            resources[key] = profile.options[key]
    resources.setdefault("time", run.get("time", "02:00:00"))
    resources.setdefault("cpus", run.get("threads", 8))
    resources.setdefault("memory", run.get("mem_gb", 32))
    return resources
