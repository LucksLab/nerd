# nerd/pipeline/tasks/base.py
"""
Defines the abstract base class for tasks and the context for their execution.
"""

import abc
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from nerd.utils.logging import get_logger
from nerd.utils.paths import make_run_dir, update_latest_symlink, get_command_log_path
from nerd.utils.hashing import config_hash
from nerd.db import api as db_api
from nerd.pipeline.runners.local import LocalRunner


@dataclass
class TaskContext:
    """
    Provides execution context for a task, including database connections,
    configuration, and paths.
    """
    db: sqlite3.Connection
    backend: str
    workdir: Path
    threads: int
    mem_gb: int
    time: str
    label: str
    output_dir: str


class Task(abc.ABC):
    """
    An abstract base class for a runnable task in the pipeline.
    """
    name: str = "base_task"
    scope_kind: str = "sample"  # or 'rg' for reaction_group

    def exec(self, db_conn: sqlite3.Connection, cfg: Dict[str, Any], verbose: bool = False):
        """
        Orchestrates the full lifecycle of a task execution.
        """
        log = get_logger(__name__)
        output_dir = cfg.get("run", {}).get("output_dir", "nerd_output")
        label = cfg.get("run", {}).get("label")
        if not label:
            raise ValueError("Configuration must contain a 'run.label'.")
        
        label_path = Path(output_dir) / label  # Updated to use output_dir from config

        # 0. Compute hashes used for caching and paths
        cfg_hash_short = config_hash(cfg, length=7)
        cache_key_full = config_hash(cfg, length=64)

        # 0.5. Check for an existing completed task with same signature and skip if found
        existing = db_api.find_task_by_signature(db_conn, label, output_dir, cache_key_full)

        if existing is not None and str(existing["state"]).lower() == "completed":
            # Record a cached task to make the skip visible in DB, then return
            msg = f"Identical config (cfg={cfg_hash_short}) previously completed as task_id={existing['id']} — skipping."
            scope_val = self.scope_id(None)
            db_api.record_cached_task(
                db_conn, self.name, self.scope_kind, scope_val, cfg.get("run", {}).get("backend", "local"),
                output_dir, label, cache_key_full, msg
            )
            log.info("%s", msg)
            return

        # 1. Create a unique directory for this run.
        run_dir = make_run_dir(label_path, self.name, suffix=f"__cfg-{cfg_hash_short}")

        # The logger is already configured by the CLI, but we could add a file handler here.
        log.info("Executing task '%s' in run directory: %s", self.name, run_dir)

        # 2. Create the task context.
        ctx = TaskContext(
            db=db_conn,
            backend=cfg.get("run", {}).get("backend", "local"),
            workdir=run_dir,
            threads=cfg.get("run", {}).get("threads", 8),
            mem_gb=cfg.get("run", {}).get("mem_gb", 32),
            time=cfg.get("run", {}).get("time", "02:00:00"),
            label=label,
            output_dir=output_dir  # Added output_dir to context
        )

        # 3. Prepare inputs and parameters.
        inputs, params = self.prepare(cfg)
        
        # 4. Record the start of the task in the database.
        task_id = db_api.begin_task(
            ctx.db, self.name, self.scope_kind, self.scope_id(inputs),
            ctx.backend, ctx.output_dir, ctx.label, cache_key=cache_key_full
        )
        if task_id is None:
            log.error("Failed to begin task in database. Aborting.")
            raise SystemExit(1)

        # 5. Build the command to be executed.
        cmd = self.command(inputs, params)
        rc = 0

        # 6. Run the command using the appropriate runner.
        if cmd:
            log.debug("Executing command: %s", cmd)
            # In a more complex app, we would select the runner based on ctx.backend
            runner = LocalRunner()
            rc = runner.run(cmd, run_dir)
            
            # Record the attempt.
            db_api.attempt(ctx.db, task_id, 1, cmd, {}, get_command_log_path(run_dir))
        
        # 7. Check the result and update the task status.
        if rc != 0:
            log.error("Task command failed with exit code %d.", rc)
            db_api.finish_task(ctx.db, task_id, "failed", f"Command returned non-zero exit code: {rc}")
            raise SystemExit(rc)

        # 8. Consume the outputs of the task.
        log.info("Command completed successfully. Consuming outputs.")
        self.consume_outputs(ctx, inputs, params, run_dir)

        # 9. Mark the task as completed.
        db_api.finish_task(ctx.db, task_id, "completed")
        log.info("Task '%s' (ID: %d) completed successfully.", self.name, task_id)

        # 10. Update the 'latest' symlink to point to this run.
        update_latest_symlink(label_path, self.name, run_dir)

    def scope_id(self, inputs: Any) -> Optional[int]:
        """Determines the primary ID for the task's scope (e.g., a sample ID)."""
        return None

    @abc.abstractmethod
    def prepare(self, cfg: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Prepare inputs and parameters for the task from the configuration.
        
        Returns:
            A tuple of (inputs, params).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def command(self, inputs: Any, params: Any) -> Optional[str]:
        """
        Construct the shell command to be executed.
        
        Returns:
            A string command, or None if no command is needed.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def consume_outputs(self, ctx: TaskContext, inputs: Any, params: Any, run_dir: Path):
        """
        Process the outputs after the command has successfully run.
        """
        raise NotImplementedError
