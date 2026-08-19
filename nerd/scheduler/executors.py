"""Detached local, local Slurm, and SSH-to-Slurm executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
from typing import List, Optional, Sequence

from .models import AttemptState, JobHandle, JobSpec, JobStatus
from .profiles import ExecutorProfile


RESULT_FILE = ".nerd-job-result.json"
STARTED_FILE = ".nerd-job-started"
SCRIPT_FILE = ".nerd-job.sh"


class ExecutorError(RuntimeError):
    pass


class Executor(ABC):
    def __init__(self, profile: ExecutorProfile) -> None:
        self.profile = profile

    @abstractmethod
    def submit(self, spec: JobSpec) -> JobHandle:
        raise NotImplementedError

    @abstractmethod
    def status(self, handle: JobHandle, spec: JobSpec) -> JobStatus:
        raise NotImplementedError

    @abstractmethod
    def logs(self, handle: JobHandle, spec: JobSpec, tail: int = 100) -> str:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, handle: JobHandle, spec: JobSpec) -> None:
        raise NotImplementedError

    @abstractmethod
    def collect(self, handle: JobHandle, spec: JobSpec) -> JobStatus:
        raise NotImplementedError


def _tail(path: Path, count: int) -> str:
    if not path.exists():
        return ""
    if count <= 0:
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-count:])


def _result_status(path: Path) -> Optional[JobStatus]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        exit_code = int(data["exit_code"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return JobStatus(AttemptState.UNKNOWN, message="Job result file is unreadable.")
    state = AttemptState.SCHEDULER_COMPLETED if exit_code == 0 else AttemptState.SCHEDULER_FAILED
    return JobStatus(
        state,
        exit_code=exit_code,
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
    )


def _render_script(spec: JobSpec, workdir: str) -> str:
    result_path = str(Path(workdir) / RESULT_FILE)
    started_path = str(Path(workdir) / STARTED_FILE)
    lines = [
        "#!/usr/bin/env bash",
        "set +e",
        "cd %s" % shlex.quote(workdir),
    ]
    for key, value in spec.env.items():
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)) is None:
            raise ExecutorError("Invalid environment variable name: %s" % key)
        lines.append("export %s=%s" % (key, shlex.quote(str(value))))
    if spec.preamble:
        if isinstance(spec.preamble, list):
            lines.extend(str(item) for item in spec.preamble)
        else:
            lines.append(str(spec.preamble))
    lines.extend([
        "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "printf '%s\\n' \"$started_at\" > %s" % ("%s", shlex.quote(started_path)),
        "bash -lc %s" % shlex.quote(spec.command),
        "rc=$?",
        "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "tmp=%s.tmp.$$" % shlex.quote(result_path),
        "printf '{\"exit_code\":%s,\"started_at\":\"%s\",\"finished_at\":\"%s\"}\\n' "
        "\"$rc\" \"$started_at\" \"$finished_at\" > \"$tmp\"",
        "mv \"$tmp\" %s" % shlex.quote(result_path),
        "exit \"$rc\"",
    ])
    return "\n".join(lines) + "\n"


class LocalProcessExecutor(Executor):
    def submit(self, spec: JobSpec) -> JobHandle:
        spec.workdir.mkdir(parents=True, exist_ok=True)
        for marker in (RESULT_FILE, STARTED_FILE):
            try:
                (spec.workdir / marker).unlink()
            except FileNotFoundError:
                pass
        script = spec.workdir / SCRIPT_FILE
        script.write_text(_render_script(spec, str(spec.workdir)))
        script.chmod(0o700)
        log_path = spec.workdir / "command.log"
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                ["bash", str(script)],
                cwd=str(spec.workdir),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return JobHandle(str(process.pid), {"pid": process.pid})

    def status(self, handle: JobHandle, spec: JobSpec) -> JobStatus:
        result = _result_status(spec.workdir / RESULT_FILE)
        if result is not None:
            return result
        started = spec.workdir / STARTED_FILE
        try:
            os.kill(int(handle.scheduler_id), 0)
        except (ProcessLookupError, ValueError):
            return JobStatus(AttemptState.UNKNOWN, message="Process ended without a result file.")
        except PermissionError:
            pass
        return JobStatus(
            AttemptState.RUNNING if started.exists() else AttemptState.QUEUED,
            started_at=started.read_text().strip() if started.exists() else None,
        )

    def logs(self, handle: JobHandle, spec: JobSpec, tail: int = 100) -> str:
        return _tail(spec.workdir / "command.log", tail)

    def cancel(self, handle: JobHandle, spec: JobSpec) -> None:
        try:
            os.killpg(int(handle.scheduler_id), signal.SIGTERM)
        except ProcessLookupError:
            return

    def collect(self, handle: JobHandle, spec: JobSpec) -> JobStatus:
        return self.status(handle, spec)


_SLURM_STATE_MAP = {
    "PENDING": AttemptState.QUEUED,
    "CONFIGURING": AttemptState.QUEUED,
    "REQUEUED": AttemptState.QUEUED,
    "RUNNING": AttemptState.RUNNING,
    "COMPLETING": AttemptState.RUNNING,
    "COMPLETED": AttemptState.SCHEDULER_COMPLETED,
    "CANCELLED": AttemptState.CANCELLED,
    "FAILED": AttemptState.SCHEDULER_FAILED,
    "TIMEOUT": AttemptState.SCHEDULER_FAILED,
    "OUT_OF_MEMORY": AttemptState.SCHEDULER_FAILED,
    "NODE_FAIL": AttemptState.SCHEDULER_FAILED,
    "PREEMPTED": AttemptState.SCHEDULER_FAILED,
    "BOOT_FAIL": AttemptState.SCHEDULER_FAILED,
}


def _parse_exit_code(value: str) -> Optional[int]:
    token = (value or "").strip().split(":", 1)[0]
    try:
        return int(token)
    except ValueError:
        return None


def _slurm_status_from_output(output: str) -> JobStatus:
    line = next((item.strip() for item in output.splitlines() if item.strip()), "")
    if not line:
        return JobStatus(AttemptState.UNKNOWN, message="Job not found in squeue or sacct.")
    fields = line.split("|")
    raw_state = fields[0].split()[0].split("+")[0].upper()
    state = _SLURM_STATE_MAP.get(raw_state, AttemptState.UNKNOWN)
    exit_code = _parse_exit_code(fields[1]) if len(fields) > 1 else None
    started_at = (fields[2] or None) if len(fields) > 2 else None
    finished_at = (fields[3] or None) if len(fields) > 3 else None
    return JobStatus(state, exit_code=exit_code, message=raw_state,
                     started_at=started_at, finished_at=finished_at)


class SlurmExecutor(Executor):
    def _run(self, args: Sequence[str], timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(list(args), check=False, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout)

    def _script_workdir(self, spec: JobSpec) -> str:
        return str(spec.workdir)

    def _write_script(self, spec: JobSpec) -> Path:
        spec.workdir.mkdir(parents=True, exist_ok=True)
        script = spec.workdir / SCRIPT_FILE
        script.write_text(_render_script(spec, self._script_workdir(spec)))
        script.chmod(0o700)
        return script

    def _sbatch_args(self, spec: JobSpec, script: str) -> List[str]:
        workdir = self._script_workdir(spec)
        args = ["sbatch", "--parsable", "--chdir", workdir,
                "--output", str(Path(workdir) / "command.log"),
                "--error", str(Path(workdir) / "command.log")]
        mapping = {
            "partition": "--partition", "account": "--account", "qos": "--qos",
            "constraint": "--constraint", "time": "--time", "cpus": "--cpus-per-task",
            "memory": "--mem",
        }
        for key, flag in mapping.items():
            value = spec.resources.get(key)
            if value not in (None, ""):
                if key == "memory" and str(value).isdigit():
                    value = "%sG" % value
                args.extend([flag, str(value)])
        args.append(script)
        return args

    def submit(self, spec: JobSpec) -> JobHandle:
        script = self._write_script(spec)
        cp = self._run(self._sbatch_args(spec, str(script)))
        if cp.returncode != 0:
            raise ExecutorError((cp.stderr or cp.stdout or "sbatch failed").strip())
        job_id = (cp.stdout or "").strip().splitlines()[-1].split(";", 1)[0]
        if not job_id.isdigit():
            raise ExecutorError("Could not parse Slurm job ID from: %s" % cp.stdout.strip())
        return JobHandle(job_id)

    def status(self, handle: JobHandle, spec: JobSpec) -> JobStatus:
        queued = self._run(["squeue", "-h", "-j", handle.scheduler_id, "-o", "%T"])
        if queued.returncode == 0 and queued.stdout.strip():
            return _slurm_status_from_output(queued.stdout)
        accounting = self._run([
            "sacct", "-n", "-P", "-X", "-j", handle.scheduler_id,
            "--format=State,ExitCode,Start,End",
        ])
        if accounting.returncode != 0:
            return JobStatus(AttemptState.UNKNOWN, message=(accounting.stderr or "sacct failed").strip())
        return _slurm_status_from_output(accounting.stdout)

    def logs(self, handle: JobHandle, spec: JobSpec, tail: int = 100) -> str:
        return _tail(spec.workdir / "command.log", tail)

    def cancel(self, handle: JobHandle, spec: JobSpec) -> None:
        cp = self._run(["scancel", handle.scheduler_id])
        if cp.returncode != 0:
            raise ExecutorError((cp.stderr or "scancel failed").strip())

    def collect(self, handle: JobHandle, spec: JobSpec) -> JobStatus:
        return self.status(handle, spec)


class SSHSlurmExecutor(SlurmExecutor):
    def _destination(self) -> str:
        host = self.profile.options.get("host")
        if not host:
            raise ExecutorError("ssh_slurm executor requires a host.")
        user = self.profile.options.get("user")
        return "%s@%s" % (user, host) if user else str(host)

    def _ssh_base(self) -> List[str]:
        args = ["ssh"]
        if self.profile.options.get("port"):
            args.extend(["-p", str(self.profile.options["port"])])
        options = self.profile.options.get("ssh_options") or self.profile.options.get("options")
        if options:
            args.extend(shlex.split(str(options)))
        return args

    def _remote(self, args: Sequence[str]) -> subprocess.CompletedProcess:
        return self._run(self._ssh_base() + [self._destination(), shlex.join(list(args))])

    def _script_workdir(self, spec: JobSpec) -> str:
        if not spec.remote_workdir:
            raise ExecutorError("ssh_slurm job is missing a remote work directory.")
        return spec.remote_workdir

    def submit(self, spec: JobSpec) -> JobHandle:
        remote_dir = self._script_workdir(spec)
        prep = self._remote(["mkdir", "-p", remote_dir])
        if prep.returncode != 0:
            raise ExecutorError((prep.stderr or "Could not create remote work directory.").strip())
        for item in spec.stage_in:
            src = Path(item["src"])
            if not src.is_file():
                raise ExecutorError("Stage-in source does not exist: %s" % src)
            destination = str(Path(remote_dir) / item["dst"])
            parent = str(Path(destination).parent)
            made = self._remote(["mkdir", "-p", parent])
            if made.returncode != 0:
                raise ExecutorError((made.stderr or "Could not create remote stage-in directory.").strip())
            cp = self._run(["rsync", "-az", str(src), "%s:%s" % (self._destination(), destination)])
            if cp.returncode != 0:
                raise ExecutorError((cp.stderr or "rsync stage-in failed").strip())
        script = self._write_script(spec)
        upload = self._run(["rsync", "-az", str(script),
                            "%s:%s" % (self._destination(), str(Path(remote_dir) / SCRIPT_FILE))])
        if upload.returncode != 0:
            raise ExecutorError((upload.stderr or "Could not upload Slurm script.").strip())
        args = self._sbatch_args(spec, str(Path(remote_dir) / SCRIPT_FILE))
        cp = self._remote(args)
        if cp.returncode != 0:
            raise ExecutorError((cp.stderr or cp.stdout or "remote sbatch failed").strip())
        job_id = (cp.stdout or "").strip().splitlines()[-1].split(";", 1)[0]
        if not job_id.isdigit():
            raise ExecutorError("Could not parse remote Slurm job ID from: %s" % cp.stdout.strip())
        return JobHandle(job_id, {"host": self._destination()})

    def status(self, handle: JobHandle, spec: JobSpec) -> JobStatus:
        queued = self._remote(["squeue", "-h", "-j", handle.scheduler_id, "-o", "%T"])
        if queued.returncode == 0 and queued.stdout.strip():
            return _slurm_status_from_output(queued.stdout)
        accounting = self._remote([
            "sacct", "-n", "-P", "-X", "-j", handle.scheduler_id,
            "--format=State,ExitCode,Start,End",
        ])
        if accounting.returncode != 0:
            return JobStatus(AttemptState.UNKNOWN, message=(accounting.stderr or "remote sacct failed").strip())
        return _slurm_status_from_output(accounting.stdout)

    def logs(self, handle: JobHandle, spec: JobSpec, tail: int = 100) -> str:
        path = str(Path(self._script_workdir(spec)) / "command.log")
        cp = self._remote(["tail", "-n", str(max(0, tail)), path])
        if cp.returncode != 0:
            raise ExecutorError((cp.stderr or "Could not read remote log.").strip())
        return cp.stdout

    def cancel(self, handle: JobHandle, spec: JobSpec) -> None:
        cp = self._remote(["scancel", handle.scheduler_id])
        if cp.returncode != 0:
            raise ExecutorError((cp.stderr or "remote scancel failed").strip())

    def collect(self, handle: JobHandle, spec: JobSpec) -> JobStatus:
        status = self.status(handle, spec)
        remote_dir = self._script_workdir(spec)
        patterns = ["command.log", RESULT_FILE, STARTED_FILE] + list(spec.stage_out)
        spec.workdir.mkdir(parents=True, exist_ok=True)
        for pattern in dict.fromkeys(patterns):
            cp = self._run([
                "rsync", "-az", "--relative", "--prune-empty-dirs",
                "%s:%s/./%s" % (self._destination(), remote_dir, pattern),
                "%s/" % spec.workdir,
            ])
            if cp.returncode != 0 and pattern in {"command.log", RESULT_FILE}:
                raise ExecutorError((cp.stderr or "Could not collect required remote output.").strip())
        return status


def executor_for(profile: ExecutorProfile) -> Executor:
    if profile.executor_type == "local":
        return LocalProcessExecutor(profile)
    if profile.executor_type == "slurm":
        return SlurmExecutor(profile)
    if profile.executor_type == "ssh_slurm":
        return SSHSlurmExecutor(profile)
    raise ValueError("Unsupported executor type: %s" % profile.executor_type)
