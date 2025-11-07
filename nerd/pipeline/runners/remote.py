"""
Unified remote runner that supports two modes on the same cluster:

- mode = 'slurm' (aka remote_slurm): submit the rendered script via sbatch and wait
- mode = 'ssh'   (aka remote/login): execute the rendered script directly over SSH

Common features:
- Ensures remote working directory
- Stages input files (FASTQs, etc.) via rsync based on SLURM_STAGE_IN (JSON)
- Renders a script that cds into the remote workdir, applies an optional preamble,
  then runs the provided command; logs to command.log
- Stages out command.log and any configured patterns; preserves relative paths
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import os
import shutil
import subprocess

from nerd.utils.logging import get_logger
from nerd.utils.paths import get_command_log_path
from .base import Runner


class RemoteRunner(Runner):
    def __init__(self, mode: str = "ssh") -> None:
        self.mode = (mode or "ssh").lower()

    def run(
        self,
        command: str,
        workdir: Path,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> int:
        log = get_logger(__name__)
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        get_command_log_path(workdir)  # ensure parent exists

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        # SSH connection details
        host = merged_env.get("SLURM_REMOTE_HOST")
        user = merged_env.get("SLURM_REMOTE_USER")
        port = merged_env.get("SLURM_SSH_PORT")
        options = merged_env.get("SLURM_SSH_OPTIONS")
        if not host:
            log.error("RemoteRunner requires SLURM_REMOTE_HOST (host) in env.")
            return 127
        ssh_dest = f"{user+'@' if user else ''}{host}"
        ssh_base = ["ssh"]
        if port:
            ssh_base += ["-p", str(port)]
        if options:
            ssh_base += options.split()

        # Remote working directory
        remote_base = merged_env.get("SLURM_REMOTE_BASE_DIR")
        if remote_base:
            remote_dir = str(Path(remote_base) / workdir.name)
        else:
            remote_dir = str(workdir)

        # Ensure remote dir
        subprocess.run(
            ssh_base + [ssh_dest, f"mkdir -p '{remote_dir}'"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        log.info("Remote prep: host=%s, dir=%s, mode=%s", ssh_dest, remote_dir, self.mode)

        # Stage-in inputs (JSON env: [ {src, dst}, ... ])
        import json as _json
        stage_in = (merged_env or {}).get("SLURM_STAGE_IN")
        if stage_in:
            try:
                items = _json.loads(stage_in)
                staged_ok = 0
                for it in items:
                    src = str(it.get("src") or "")
                    dst_rel = str(it.get("dst") or "")
                    if not src or not dst_rel:
                        continue
                    src_path = Path(src)
                    if not src_path.exists():
                        log.error("Stage-in source missing: %s", src)
                        continue
                    if not src_path.is_file():
                        log.error("Stage-in source is not a regular file: %s", src)
                        continue
                    remote_path = str(Path(remote_dir) / dst_rel)
                    parent = str(Path(remote_path).parent)
                    subprocess.run(
                        ssh_base + [ssh_dest, f"mkdir -p '{parent}'"],
                        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                    )
                    cp = subprocess.run(
                        ["rsync", "-az", src, f"{ssh_dest}:{remote_path}"],
                        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                    )
                    if cp.returncode != 0:
                        log.error(
                            "RemoteRunner stage-in failed (rc=%s) for %s -> %s: %s",
                            cp.returncode, src, remote_path, (cp.stderr or "").strip(),
                        )
                    else:
                        # Log success with approximate size
                        try:
                            size = os.path.getsize(src)
                            human = f"{size/1048576:.1f} MB"
                        except Exception:
                            human = "?"
                        log.info("Stage-in: %s -> %s (%s)", src, remote_path, human)
                        staged_ok += 1
                if staged_ok:
                    log.info("Stage-in completed: %d file(s) uploaded", staged_ok)
            except Exception:
                pass

        # Render remote script and upload via heredoc
        script_text = self._render_script(command, chdir=remote_dir, preamble=merged_env.get("SLURM_PREAMBLE"))
        _DELIM = "NERD_REMOTE_EOF"
        heredoc = f"cat > '{remote_dir}/job.sbatch.sh' << '{_DELIM}'\n{script_text}{_DELIM}\n"
        subprocess.run(
            ssh_base + [ssh_dest, heredoc],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        subprocess.run(
            ssh_base + [ssh_dest, f"chmod +x '{remote_dir}/job.sbatch.sh'"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )

        # Execute: sbatch or direct ssh
        if self.mode in {"slurm", "remote_slurm"}:
            get_logger(__name__).info("Submitting remote Slurm job in %s", remote_dir)
            rc = self._submit_slurm(ssh_base, ssh_dest, remote_dir, merged_env, timeout)
        else:
            get_logger(__name__).info("Executing remote script via SSH in %s", remote_dir)
            rc = self._exec_ssh(ssh_base, ssh_dest, remote_dir, timeout)

        # Stage out logs and patterns; preserve relative paths
        self._stage_out(ssh_base, ssh_dest, remote_dir, workdir, merged_env)
        return rc

    def _render_script(self, command: str, chdir: str, preamble: Optional[str]) -> str:
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd \"{chdir}\"",
        ]
        # Ensure all output captured in command.log regardless of mode
        lines.append("exec > command.log 2>&1")
        # Try to initialize environment modules (Lmod or Environment Modules) if available,
        # so 'module load ...' in preamble works in non-interactive batch shells.
        lines += [
            "if ! command -v module >/dev/null 2>&1; then",
            "  if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; fi",
            "  if [ -f /usr/share/lmod/lmod/init/bash ]; then . /usr/share/lmod/lmod/init/bash; fi",
            "  if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi",
            "  if [ -f /usr/share/Modules/init/bash ]; then . /usr/share/Modules/init/bash; fi",
            "fi",
        ]
        if preamble:
            lines.append(str(preamble))
        lines.append(command)
        return "\n".join(lines) + "\n"

    def _submit_slurm(self, ssh_base: list, ssh_dest: str, remote_dir: str, env: Dict[str, str], timeout: Optional[int]) -> int:
        log = get_logger(__name__)
        sbatch_parts = [
            "sbatch",
            "--chdir", f"{remote_dir}",
            "--output", f"{remote_dir}/command.log",
            "--error", f"{remote_dir}/command.log",
            "--wait",
            "--parsable",
        ]
        if env.get("SLURM_PARTITION"):
            sbatch_parts += ["--partition", env["SLURM_PARTITION"]]
        if env.get("SLURM_ACCOUNT"):
            sbatch_parts += ["--account", env["SLURM_ACCOUNT"]]
        if env.get("SLURM_TIME"):
            sbatch_parts += ["--time", env["SLURM_TIME"]]
        if env.get("SLURM_CPUS"):
            sbatch_parts += ["--cpus-per-task", env["SLURM_CPUS"]]
        if env.get("SLURM_MEM"):
            sbatch_parts += ["--mem", env["SLURM_MEM"]]
        sbatch_parts.append(f"{remote_dir}/job.sbatch.sh")

        try:
            cp = subprocess.run(
                ssh_base + [ssh_dest, " ".join(sbatch_parts)],
                check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.error("remote sbatch wait timed out after %s seconds", timeout)
            return 124
        except Exception as e:
            log.exception("Failed to submit remote Slurm job: %s", e)
            return -1

        return self._final_exit_code(cp, env, ssh=(ssh_base, ssh_dest))

    def _exec_ssh(self, ssh_base: list, ssh_dest: str, remote_dir: str, timeout: Optional[int]) -> int:
        log = get_logger(__name__)
        try:
            # Run via a login shell so environment modules and profile.d init scripts are sourced
            cp = subprocess.run(
                ssh_base + [ssh_dest, f"bash -lc '{remote_dir}/job.sbatch.sh'"],
                check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.error("ssh remote execution timed out after %s seconds", timeout)
            return 124
        except Exception as e:
            log.exception("Failed to execute remote script over SSH: %s", e)
            return -1
        log.info("ssh returned code=%s", cp.returncode)
        return int(cp.returncode)

    def _stage_out(self, ssh_base: list, ssh_dest: str, remote_dir: str, workdir: Path, env: Dict[str, str]) -> None:
        log = get_logger(__name__)
        patterns = ["command.log"]
        stage_out = env.get("SLURM_STAGE_OUT")
        if stage_out:
            patterns += [p.strip() for p in stage_out.split(",") if p.strip()]
        for pat in patterns:
            try:
                log.info("Stage-out: attempting pattern '%s'", pat)
                cp = subprocess.run(
                    [
                        "rsync", "-az", "--relative", "--prune-empty-dirs",
                        f"{ssh_dest}:{remote_dir}/./{pat}",
                        f"{str(workdir)}/",
                    ],
                    check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                if cp.returncode != 0:
                    log.warning(
                        "Stage-out failed for pattern '%s' (rc=%s): %s",
                        pat,
                        cp.returncode,
                        (cp.stderr or "").strip(),
                    )
                else:
                    transferred = (cp.stdout or "").strip()
                    if transferred:
                        log.info("Stage-out succeeded for pattern '%s': %s", pat, transferred)
                    else:
                        log.info("Stage-out succeeded for pattern '%s'", pat)
                    if cp.stderr:
                        log.debug(
                            "Stage-out stderr for pattern '%s': %s",
                            pat,
                            cp.stderr.strip(),
                        )
            except Exception:
                log.exception("Stage-out raised unexpected exception for pattern '%s'", pat)
                pass

    def _final_exit_code(self, cp: subprocess.CompletedProcess, env: Dict[str, str], ssh: Optional[tuple] = None) -> int:
        """Parse job ID from sbatch output and try sacct/scontrol for true exit code."""
        log = get_logger(__name__)
        stdout = (cp.stdout or "").strip()
        log.info("sbatch returned code=%s, output='%s'", cp.returncode, stdout)

        job_id = None
        if stdout:
            last = stdout.splitlines()[-1].strip()
            job_id = last.split(";")[0]
            if not job_id.isdigit():
                job_id = None

        def run_cmd(args):
            return subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        if job_id:
            # Try sacct via SSH if provided
            if ssh is not None:
                ssh_base, ssh_dest = ssh
                try:
                    sacct = subprocess.run(
                        ssh_base + [ssh_dest, f"sacct -j {job_id} --format=ExitCode --noheader"],
                        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    )
                    out = (sacct.stdout or "").strip()
                    if out:
                        code_part = out.split()[0].split(":")[0]
                        if code_part.isdigit():
                            return int(code_part)
                except Exception:
                    pass
                # scontrol fallback via SSH
                try:
                    sctl = subprocess.run(
                        ssh_base + [ssh_dest, f"scontrol show job {job_id}"],
                        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    )
                    text_out = sctl.stdout or ""
                    for token in text_out.replace("\n", " ").split():
                        if token.startswith("ExitCode="):
                            val = token.split("=", 1)[1]
                            code_part = val.split(":")[0]
                            if code_part.isdigit():
                                return int(code_part)
                            break
                except Exception:
                    pass

        return cp.returncode
