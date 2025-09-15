"""
Slurm runner that submits a command via sbatch and waits for completion.

Notes:
- Writes logs to the task's `command.log` via sbatch --output/--error.
- Uses `--wait` to block until job finishes. Attempts to read final exit code
  via `sacct` or `scontrol` if available; otherwise falls back to sbatch's
  return code.
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


class SlurmRunner(Runner):
    """Submit commands to Slurm using `sbatch` and wait for completion."""

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
        log_path = get_command_log_path(workdir)

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        remote_host = (env or {}).get("SLURM_REMOTE_HOST")
        if remote_host:
            return self._run_remote(command, workdir, merged_env, timeout)
        else:
            return self._run_local(command, workdir, merged_env, timeout)

    def _run_local(self, command: str, workdir: Path, env: Dict[str, str], timeout: Optional[int]) -> int:
        log = get_logger(__name__)
        if shutil.which("sbatch") is None:
            log.error("sbatch not found in PATH. Cannot use local Slurm mode.")
            return 127

        log_path = get_command_log_path(workdir)
        script_path = workdir / "job.sbatch.sh"

        script_contents = self._render_script(command, chdir=str(workdir), preamble=env.get("SLURM_PREAMBLE"))
        script_path.write_text(script_contents)
        os.chmod(script_path, 0o750)

        sbatch_cmd = [
            "sbatch",
            "--chdir",
            str(workdir),
            "--output",
            str(log_path),
            "--error",
            str(log_path),
            "--wait",
            "--parsable",
        ]
        # Optional args
        if env.get("SLURM_PARTITION"):
            sbatch_cmd += ["--partition", env["SLURM_PARTITION"]]
        if env.get("SLURM_ACCOUNT"):
            sbatch_cmd += ["--account", env["SLURM_ACCOUNT"]]
        if env.get("SLURM_TIME"):
            sbatch_cmd += ["--time", env["SLURM_TIME"]]
        sbatch_cmd.append(str(script_path))

        log.info("Submitting local Slurm job: %s", " ".join(sbatch_cmd))
        try:
            cp = subprocess.run(
                sbatch_cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.error("sbatch wait timed out after %s seconds", timeout)
            return 124
        except Exception as e:
            log.exception("Failed to submit Slurm job: %s", e)
            return -1

        return self._final_exit_code(cp, env)

    def _run_remote(self, command: str, workdir: Path, env: Dict[str, str], timeout: Optional[int]) -> int:
        log = get_logger(__name__)
        host = env.get("SLURM_REMOTE_HOST")
        user = env.get("SLURM_REMOTE_USER")
        port = env.get("SLURM_SSH_PORT")
        options = env.get("SLURM_SSH_OPTIONS")
        ssh_dest = f"{user+'@' if user else ''}{host}"
        ssh_base = ["ssh"]
        if port:
            ssh_base += ["-p", str(port)]
        if options:
            ssh_base += options.split()

        remote_base = env.get("SLURM_REMOTE_BASE_DIR")
        if remote_base:
            remote_dir = str(Path(remote_base) / workdir.name)
        else:
            # Fallback: mirror local path (may not always be valid on remote)
            remote_dir = str(workdir)
        log_path_local = get_command_log_path(workdir)
        log_path_remote = str(Path(remote_dir) / "command.log")

        # Ensure remote dir
        mkdir_cmd = ssh_base + [ssh_dest, f"mkdir -p '{remote_dir}'"]
        subprocess.run(mkdir_cmd, check=False)

        # Write remote script via heredoc
        script = self._render_script(command, chdir=remote_dir, preamble=env.get("SLURM_PREAMBLE"))
        heredoc = f"cat > '{remote_dir}/job.sbatch.sh' << 'EOF'\n{script}EOF"
        subprocess.run(ssh_base + [ssh_dest, heredoc], check=False)
        subprocess.run(ssh_base + [ssh_dest, f"chmod +x '{remote_dir}/job.sbatch.sh'"], check=False)

        # Submit with sbatch
        sbatch_parts = [
            "sbatch",
            "--chdir",
            f"{remote_dir}",
            "--output",
            f"{remote_dir}/command.log",
            "--error",
            f"{remote_dir}/command.log",
            "--wait",
            "--parsable",
        ]
        if env.get("SLURM_PARTITION"):
            sbatch_parts += ["--partition", env["SLURM_PARTITION"]]
        if env.get("SLURM_ACCOUNT"):
            sbatch_parts += ["--account", env["SLURM_ACCOUNT"]]
        if env.get("SLURM_TIME"):
            sbatch_parts += ["--time", env["SLURM_TIME"]]
        sbatch_parts.append(f"{remote_dir}/job.sbatch.sh")

        try:
            cp = subprocess.run(
                ssh_base + [ssh_dest] + [" ".join(sbatch_parts)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.error("remote sbatch wait timed out after %s seconds", timeout)
            return 124
        except Exception as e:
            log.exception("Failed to submit remote Slurm job: %s", e)
            return -1

        rc = self._final_exit_code(cp, env, ssh=(ssh_base, ssh_dest))

        # Stage out command.log and any requested patterns
        stage_out = env.get("SLURM_STAGE_OUT")
        patterns = ["command.log"]
        if stage_out:
            patterns += [p.strip() for p in stage_out.split(",") if p.strip()]
        for pat in patterns:
            # Use rsync for directories/globs; fallback to scp-like via rsync
            try:
                subprocess.run(
                    [
                        "rsync",
                        "-avz",
                        f"{ssh_dest}:{remote_dir}/{pat}",
                        f"{str(workdir)}/",
                    ],
                    check=False,
                )
            except Exception:
                pass

        return rc

    def _render_script(self, command: str, chdir: str, preamble: Optional[str]) -> str:
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd \"{chdir}\""]
        if preamble:
            lines.append(str(preamble))
        lines.append(command)
        return "\n".join(lines) + "\n"

    def _final_exit_code(self, cp: subprocess.CompletedProcess, env: Dict[str, str], ssh: Optional[tuple] = None) -> int:
        log = get_logger(__name__)
        stdout = (cp.stdout or "").strip()
        log.info("sbatch returned code=%s, output='%s'", cp.returncode, stdout)

        # Parse job id
        job_id = None
        if stdout:
            first_field = stdout.splitlines()[-1].strip()
            job_id = first_field.split(";")[0]
            if not job_id.isdigit():
                job_id = None

        def run_cmd(args):
            return subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        if job_id:
            if ssh is None and shutil.which("sacct"):
                try:
                    sacct = run_cmd(["sacct", "-j", job_id, "--format=ExitCode", "--noheader"])
                    out = (sacct.stdout or "").strip()
                    if out:
                        code_part = out.split()[0].split(":")[0]
                        if code_part.isdigit():
                            return int(code_part)
                except Exception:
                    pass
            if ssh is not None:
                ssh_base, ssh_dest = ssh
                try:
                    sacct = subprocess.run(
                        ssh_base + [ssh_dest, f"sacct -j {job_id} --format=ExitCode --noheader"],
                        check=False,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    out = (sacct.stdout or "").strip()
                    if out:
                        code_part = out.split()[0].split(":")[0]
                        if code_part.isdigit():
                            return int(code_part)
                except Exception:
                    pass
            # scontrol fallback
            if ssh is None and shutil.which("scontrol"):
                try:
                    sctl = run_cmd(["scontrol", "show", "job", job_id])
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
            if ssh is not None:
                try:
                    sctl = subprocess.run(
                        ssh_base + [ssh_dest, f"scontrol show job {job_id}"],
                        check=False,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
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
