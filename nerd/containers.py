"""Container metadata, execution-host preflight, and immutable SIF caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from nerd.scheduler.profiles import ExecutorProfile


SHAPEMAPPER_IMAGE_PLACEHOLDER = "ghcr.io/luckslab/nerd-shapemapper:<release-tag>"
SHAPEMAPPER_DEFAULT_IMAGE = (
    "ghcr.io/edr-choi/nerd-shapemapper2@"
    "sha256:192732b14071da3f0e23979b2018d071ca2731639178b5f31c032eedbfbdea7e"
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_CHECKSUM_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ContainerError(RuntimeError):
    """A user-actionable container readiness or preparation failure."""


@dataclass(frozen=True)
class ContainerSpec:
    """Small, plugin-neutral description of a containerized executable."""

    oci_reference: str
    executable: str
    digest: Optional[str] = None
    digest_policy: str = "required"
    supported_architectures: Sequence[str] = ("x86_64", "amd64")
    runtime_preferences: Sequence[str] = ("apptainer", "singularity")
    smoke_test: Sequence[str] = ("--version",)

    @property
    def configured(self) -> bool:
        return bool(self.oci_reference) and "<release-tag>" not in self.oci_reference

    @property
    def immutable_reference(self) -> Optional[str]:
        if not self.configured:
            return None
        if "@sha256:" in self.oci_reference:
            return self.oci_reference
        if self.digest and _DIGEST_RE.fullmatch(self.digest):
            return "%s@%s" % (self.oci_reference.split("@", 1)[0], self.digest)
        return None

    def validate(self, preinstalled_sif: Optional[str] = None) -> None:
        if preinstalled_sif:
            return
        if not self.configured:
            raise ContainerError(
                "Container image is not configured/ready. Replace %s with a concrete OCI "
                "release tag and sha256 digest, or configure an installed SIF."
                % (self.oci_reference or "the empty image reference")
            )
        if self.digest and not _DIGEST_RE.fullmatch(self.digest):
            raise ContainerError("Container digest must have the form sha256:<64 hexadecimal characters>.")
        if self.digest_policy == "required" and not self.immutable_reference:
            raise ContainerError(
                "Container image is mutable: configure tool.container.digest or use an @sha256 reference."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeInfo:
    command: str
    version: str


@dataclass
class ContainerReadiness:
    ready: bool
    profile: str
    executor_type: str
    execution_host: str
    architecture: Optional[str] = None
    runtime: Optional[RuntimeInfo] = None
    cache_dir: Optional[str] = None
    sif_path: Optional[str] = None
    sif_checksum: Optional[str] = None
    smoke_test_output: Optional[str] = None
    checks: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, message: str) -> None:
        self.checks.append({"name": name, "ok": ok, "message": message})
        if not ok:
            self.ready = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


class HostCommands:
    """Run inspection/preparation commands where an executor will execute."""

    def __init__(self, profile: ExecutorProfile):
        self.profile = profile

    def _local(self, args: Sequence[str], timeout: int = 60) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(list(args), check=False, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=timeout)
        except FileNotFoundError as exc:
            return subprocess.CompletedProcess(list(args), 127, "", "%s is unavailable" % exc.filename)

    def run(self, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
        preamble = self.profile.options.get("preamble") or []
        if isinstance(preamble, str):
            preamble = [preamble]
        shell_command = "\n".join([str(item) for item in preamble] + [command])
        if self.profile.executor_type == "ssh_slurm":
            host = self.profile.options.get("host")
            if not host:
                raise ContainerError("ssh_slurm executor requires a host.")
            destination = "%s@%s" % (self.profile.options["user"], host) \
                if self.profile.options.get("user") else str(host)
            args = ["ssh"]
            if self.profile.options.get("port"):
                args.extend(["-p", str(self.profile.options["port"])])
            options = self.profile.options.get("ssh_options") or self.profile.options.get("options")
            if options:
                args.extend(shlex.split(str(options)))
            args.extend([destination, "bash -lc %s" % shlex.quote(shell_command)])
            return self._local(args, timeout)
        return self._local(["bash", "-lc", shell_command], timeout)


def shapemapper_container_spec(tool_cfg: Optional[Dict[str, Any]] = None) -> ContainerSpec:
    tool = tool_cfg or {}
    raw = tool.get("container") or {}
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise ContainerError("mut_count.tool.container must be a mapping.")
    reference = str(raw.get("image") or raw.get("reference") or SHAPEMAPPER_DEFAULT_IMAGE)
    digest = raw.get("digest")
    if "@sha256:" in reference and not digest:
        digest = "sha256:" + reference.rsplit("@sha256:", 1)[1]
    return ContainerSpec(
        oci_reference=reference,
        digest=str(digest) if digest else None,
        digest_policy=str(raw.get("digest_policy", "required")),
        executable=str(raw.get("executable", "shapemapper")),
        supported_architectures=tuple(raw.get("architectures") or ("x86_64", "amd64")),
        runtime_preferences=tuple(raw.get("runtimes") or ("apptainer", "singularity")),
        smoke_test=tuple(raw.get("smoke_test") or ("--version",)),
    )


def container_requested(tool_cfg: Optional[Dict[str, Any]]) -> bool:
    """A custom binary stays native; otherwise ShapeMapper defaults to its container."""
    tool = tool_cfg or {}
    mode = str(tool.get("execution") or "").lower()
    if mode in {"native", "custom"}:
        return False
    if tool.get("container") is False:
        return False
    if mode == "container" or "container" in tool:
        return True
    return not bool(tool.get("bin"))


def _cache_dir(profile: ExecutorProfile, tool_cfg: Dict[str, Any]) -> str:
    container = tool_cfg.get("container") or {}
    if not isinstance(container, dict):
        container = {}
    value = container.get("cache_dir") or profile.options.get("container_cache_dir")
    if value:
        return str(value)
    if profile.executor_type == "ssh_slurm":
        base = profile.options.get("remote_base_dir")
        if not base:
            raise ContainerError("ssh_slurm profile requires remote_base_dir for its container cache.")
        return str(Path(str(base)).parent / ".nerd-container-cache")
    return str(Path.home() / ".cache" / "nerd" / "containers")


def _installed_sif(tool_cfg: Dict[str, Any], profile: ExecutorProfile) -> Optional[str]:
    container = tool_cfg.get("container") or {}
    if not isinstance(container, dict):
        return None
    value = container.get("sif") or profile.options.get("container_sif")
    return str(value) if value else None


def _runtime_candidates(spec: ContainerSpec, profile: ExecutorProfile) -> Sequence[str]:
    override = profile.options.get("container_runtime") or profile.options.get("runtime")
    return (str(override),) if override else spec.runtime_preferences


def _checksum(output: str) -> Optional[str]:
    for line in reversed((output or "").splitlines()):
        token = line.strip().split()[0] if line.strip() else ""
        if _CHECKSUM_RE.fullmatch(token):
            return token.lower()
    return None


def inspect_container(
    spec: ContainerSpec,
    profile: ExecutorProfile,
    tool_cfg: Optional[Dict[str, Any]] = None,
    runner: Optional[HostCommands] = None,
) -> ContainerReadiness:
    tool = tool_cfg or {}
    host = str(profile.options.get("host") or "local")
    result = ContainerReadiness(True, profile.name, profile.executor_type, host)
    commands = runner or HostCommands(profile)

    if profile.executor_type == "ssh_slurm" and runner is None:
        missing = [name for name in ("ssh", "rsync") if shutil.which(name) is None]
        result.add("ssh_prerequisites", not missing,
                   "OpenSSH and rsync are available" if not missing else
                   "Missing controller command(s): %s" % ", ".join(missing))
        if missing:
            try:
                spec.validate(_installed_sif(tool, profile))
            except ContainerError as exc:
                result.add("image", False, str(exc))
            return result

    if profile.executor_type in {"slurm", "ssh_slurm"}:
        scheduler = commands.run(
            "for command in sbatch squeue sacct scancel; do command -v \"$command\" >/dev/null 2>&1 || exit 1; done"
        )
        result.add("scheduler", scheduler.returncode == 0,
                   "sbatch, squeue, sacct, and scancel are available" if scheduler.returncode == 0 else
                   "One or more required Slurm commands are unavailable: sbatch, squeue, sacct, scancel")

    arch_cp = commands.run("uname -m")
    if arch_cp.returncode == 0:
        result.architecture = arch_cp.stdout.strip().splitlines()[-1]
        result.add("architecture", result.architecture in spec.supported_architectures,
                   "%s (supported: %s)" % (result.architecture, ", ".join(spec.supported_architectures)))
    else:
        result.add("architecture", False, (arch_cp.stderr or "could not inspect architecture").strip())

    runtime = None
    for candidate in _runtime_candidates(spec, profile):
        cp = commands.run("command -v %s >/dev/null 2>&1 && %s --version" %
                          (shlex.quote(candidate), shlex.quote(candidate)))
        if cp.returncode == 0:
            version = (cp.stdout or cp.stderr).strip().splitlines()[0]
            runtime = RuntimeInfo(candidate, version)
            break
    result.runtime = runtime
    result.add("runtime", runtime is not None,
               "%s: %s" % (runtime.command, runtime.version) if runtime else
               "Neither Apptainer nor Singularity is available in the execution environment")

    sif = _installed_sif(tool, profile)
    cache = _cache_dir(profile, tool)
    result.cache_dir = cache
    if sif:
        result.add("cache", True, "not required because a preinstalled SIF is configured")
    else:
        cache_cp = commands.run("mkdir -p %s && test -d %s && test -w %s" %
                                tuple(shlex.quote(cache) for _ in range(3)))
        result.add("cache", cache_cp.returncode == 0,
                   "%s is writable" % cache if cache_cp.returncode == 0 else "%s is not accessible and writable" % cache)

    try:
        spec.validate(sif)
    except ContainerError as exc:
        result.add("image", False, str(exc))
    else:
        result.add("image", True, "preinstalled SIF configured" if sif else
                   "immutable image identity configured: %s" % spec.immutable_reference)

    if sif:
        check = commands.run("test -r %s && sha256sum %s" % (shlex.quote(sif), shlex.quote(sif)))
        checksum = _checksum(check.stdout)
        if check.returncode == 0 and checksum:
            result.sif_path = sif
            result.sif_checksum = checksum
            result.add("sif", True, "%s (sha256:%s)" % (sif, result.sif_checksum))
        else:
            result.add("sif", False, "Configured SIF is not readable on the execution host: %s" % sif)
    return result


def _cache_path(spec: ContainerSpec, cache_dir: str) -> str:
    identity = spec.digest or (spec.immutable_reference or "").rsplit("@", 1)[-1]
    if not _DIGEST_RE.fullmatch(identity):
        raise ContainerError("An immutable sha256 digest is required to derive the SIF cache identity.")
    return str(Path(cache_dir) / ("sha256-%s.sif" % identity.split(":", 1)[1]))


def prepare_container(
    spec: ContainerSpec,
    profile: ExecutorProfile,
    tool_cfg: Optional[Dict[str, Any]] = None,
    runner: Optional[HostCommands] = None,
) -> ContainerReadiness:
    tool = tool_cfg or {}
    commands = runner or HostCommands(profile)
    readiness = inspect_container(spec, profile, tool, commands)
    if not readiness.ready:
        messages = [check["message"] for check in readiness.checks if not check["ok"]]
        raise ContainerError("; ".join(messages))
    assert readiness.runtime is not None
    if readiness.sif_path:
        target = readiness.sif_path
    else:
        immutable = spec.immutable_reference
        if not immutable:
            raise ContainerError("Container image is not configured with an immutable identity.")
        target = _cache_path(spec, str(readiness.cache_dir))
        # noclobber lock, unique temp, verified completion, and atomic rename protect shared caches.
        script = """set -eu
target={target}
if test -s "$target"; then sha256sum "$target"; exit 0; fi
lock="$target.lock"
if ( set -o noclobber; : > "$lock" ) 2>/dev/null; then
  trap 'rm -f "$tmp" "$lock"' EXIT HUP INT TERM
  tmp="$target.tmp.$$.sif"
  {runtime} pull "$tmp" {image}
  test -s "$tmp"
  chmod 0444 "$tmp"
  mv "$tmp" "$target"
  rm -f "$lock"
  trap - EXIT HUP INT TERM
else
  n=0
  while test -e "$lock" && test "$n" -lt 60; do sleep 1; n=$((n+1)); done
  test -s "$target" || {{ echo 'container cache preparation lock did not produce an image' >&2; exit 73; }}
fi
sha256sum "$target""".format(
            target=shlex.quote(target), runtime=shlex.quote(readiness.runtime.command),
            image=shlex.quote("docker://" + immutable),
        )
        cp = commands.run(script, timeout=1800)
        if cp.returncode != 0:
            detail = (cp.stderr or cp.stdout or "container image preparation failed").strip()
            if any(token in detail.lower() for token in ("unauthorized", "authentication", "denied")):
                raise ContainerError("Registry authentication failed while preparing the image: %s" % detail)
            raise ContainerError("Container image preparation failed: %s" % detail)
        checksum = _checksum(cp.stdout)
        if not checksum:
            raise ContainerError("Container image preparation did not return a valid SIF SHA-256 checksum.")
        readiness.sif_path = target
        readiness.sif_checksum = checksum
        readiness.add("prepared", True, "cached immutable SIF at %s" % target)
    smoke_args = [readiness.runtime.command, "exec", "--cleanenv", str(target),
                  spec.executable] + list(spec.smoke_test)
    smoke = commands.run(shlex.join(smoke_args), timeout=120)
    if smoke.returncode != 0:
        raise ContainerError("Container smoke test failed: %s" %
                             (smoke.stderr or smoke.stdout or "no output").strip())
    readiness.smoke_test_output = (smoke.stdout or smoke.stderr).strip()
    readiness.add("smoke_test", True, readiness.smoke_test_output or "container smoke test passed")
    return readiness


def render_container_exec(runtime: RuntimeInfo, sif_path: str, command: Sequence[str],
                          binds: Sequence[str], workdir: str) -> str:
    args = [runtime.command, "exec", "--cleanenv", "--env", "TMPDIR=%s/.nerd-tmp" % workdir]
    for bind in dict.fromkeys(str(item) for item in binds):
        args.extend(["--bind", "%s:%s" % (bind, bind)])
    args.extend(["--pwd", workdir, sif_path])
    args.extend(str(item) for item in command)
    return shlex.join(args)


def provenance_payload(spec: ContainerSpec, readiness: ContainerReadiness,
                       command: str, profile: ExecutorProfile) -> Dict[str, Any]:
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "oci_reference": spec.oci_reference,
        "oci_digest": spec.digest,
        "immutable_reference": spec.immutable_reference,
        "sif_path": readiness.sif_path,
        "sif_checksum": readiness.sif_checksum,
        "runtime": asdict(readiness.runtime) if readiness.runtime else None,
        "architecture": readiness.architecture,
        "executable": spec.executable,
        "smoke_test": list(spec.smoke_test),
        "shapemapper_version": readiness.smoke_test_output,
        "command": command,
        "executor_profile": profile.name,
        "executor_type": profile.executor_type,
        "execution_host": readiness.execution_host,
    }


def write_provenance(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
