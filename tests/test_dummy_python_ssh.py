from pathlib import Path
import os, subprocess, sys

USER = os.environ.get("NERD_SSH_USER", "ekc5108")
HOST = os.environ.get("NERD_SSH_HOST", "login.quest.northwestern.edu")
LOCAL_DIR = Path(os.environ.get("NERD_LOCAL_DIR", "/tmp/nerd_slurm_test")).resolve()
REMOTE_DIR = os.environ.get("NERD_REMOTE_DIR", f"/scratch/{USER}/nerd/runner_test")

def sh(cmd, check=True, capture=False):
    print("$", cmd)
    cp = subprocess.run(cmd, shell=True, text=True, capture_output=capture)
    if check and cp.returncode != 0:
        sys.exit(cp.returncode)
    return cp.stdout.strip() if capture else cp.returncode

def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    (LOCAL_DIR / "input.txt").write_text("hello input\n")
    sh(f"ssh {USER}@{HOST} 'mkdir -p {REMOTE_DIR}'")
    sh(f"rsync -avz {LOCAL_DIR}/ {USER}@{HOST}:{REMOTE_DIR}/")

    job = f"""#!/usr/bin/env bash
set -euo pipefail
cd "{REMOTE_DIR}"
echo "Remote hostname: $(hostname)" > command.log
echo "Remote run ok" > out.txt
"""
    sh(f"ssh {USER}@{HOST} 'cat > {REMOTE_DIR}/job.sbatch.sh << \"EOF\"\n{job}EOF'")
    sh(f"ssh {USER}@{HOST} 'chmod +x {REMOTE_DIR}/job.sbatch.sh'")
    job_id = sh(
        f"ssh {USER}@{HOST} 'sbatch --partition=buyin --account=b1044 --time=00:01:00 "
        f"--chdir {REMOTE_DIR} --output {REMOTE_DIR}/command.log --error {REMOTE_DIR}/command.log "
        f"--parsable {REMOTE_DIR}/job.sbatch.sh'",
        capture=True,
    ).split(";", 1)[0]
    print(f"Submitted detached Slurm job {job_id}.")
    print(f"Reconcile later with: ssh {USER}@{HOST} 'sacct -j {job_id}'")

if __name__ == "__main__":
    main()
