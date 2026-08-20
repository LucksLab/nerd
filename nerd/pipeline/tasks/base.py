# nerd/pipeline/tasks/base.py
"""
Defines the abstract base class for tasks and the context for their execution.
"""

import abc
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List

from nerd.utils.logging import get_logger
from nerd.utils.paths import make_run_dir, update_latest_symlink, get_command_log_path
from nerd.utils.hashing import config_hash
from nerd.db import api as db_api
from nerd.pipeline.runners.local import LocalRunner
from nerd.reporting.summary import ArtifactReference, TaskSummary


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
    executor_profile: Optional[str] = None


@dataclass
class TaskScopeMember:
    """
    Associates a task run with a specific domain entity (sample, rg, etc.).
    """
    kind: str
    ref_id: Optional[int] = None
    label: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class TaskScope:
    """
    Describes the scope that a task execution applies to.
    """
    kind: str
    scope_id: Optional[int] = None
    label: Optional[str] = None
    members: List[TaskScopeMember] = field(default_factory=list)

    def with_member(self, member: TaskScopeMember) -> "TaskScope":
        self.members.append(member)
        return self


@dataclass
class WorkUnit:
    """One independently schedulable slice of a scientific task."""

    key: str
    label: str
    config: Dict[str, Any]
    scope_kind: Optional[str] = None
    scope_id: Optional[int] = None


class Task(abc.ABC):
    """
    An abstract base class for a runnable task in the pipeline.
    """
    name: str = "base_task"
    scope_kind: str = "sample"  # Default scope; tasks may override resolve_scope()

    def plan_work_units(
        self,
        ctx: TaskContext,
        cfg: Dict[str, Any],
        inputs: Any,
        params: Any,
    ) -> List[WorkUnit]:
        """Return independently schedulable units; workflows are single-unit by default."""
        return [WorkUnit("main", self.name, cfg)]

    def exec(self, db_conn: sqlite3.Connection, cfg: Dict[str, Any], verbose: bool = False):
        """
        Orchestrates the full lifecycle of a task execution.
        """
        log = get_logger(__name__)
        total_started = time.monotonic()
        started_at = datetime.now().isoformat()
        output_dir = cfg.get("run", {}).get("output_dir", "nerd_output")
        label = cfg.get("run", {}).get("label")
        if not label:
            raise ValueError("Configuration must contain a 'run.label'.")
        
        label_path = Path(output_dir) / label  # Updated to use output_dir from config

        # 0. Compute hashes used for caching and paths
        cfg_hash_short = config_hash(cfg, length=7)
        cache_key_full = config_hash(cfg, length=64)

        # 0.5. Check for an existing completed task with same signature and skip if found
        # Robustly detect prior completion even if a newer 'cached' entry exists
        task_block = cfg.get(self.name)
        force_rerun = False
        if isinstance(task_block, dict):
            force_rerun = bool(task_block.get("force_run")) or bool(task_block.get("overwrite"))
        existing = db_api.find_completed_task_by_signature(db_conn, label, output_dir, cache_key_full)
        cached_scope = self.resolve_scope(ctx=None, inputs=None)
        if existing is not None and not force_rerun:
            # Record a cached task to make the skip visible in DB, then return
            msg = f"Identical config (cfg={cfg_hash_short}) previously completed as task_id={existing['id']} — skipping."
            cached_task_id = db_api.record_cached_task(
                db_conn, self.name, cached_scope.kind, cached_scope.scope_id, cfg.get("run", {}).get("backend", "local"),
                output_dir, label, cache_key_full, msg,
                tool=self.task_tool(inputs=None), tool_version=self.task_tool_version(inputs=None)
            )
            log.info("%s", msg)
            ended_at = datetime.now().isoformat()
            return TaskSummary(
                status="cached", workflow=self.name, task_id=cached_task_id,
                source_task_id=int(existing["id"]), label=label,
                plugin=self.task_tool(inputs=None), version=self.task_tool_version(inputs=None),
                started_at=started_at, ended_at=ended_at,
                duration_seconds=round(time.monotonic() - total_started, 6),
                timings={"preparation_seconds": 0.0, "execution_seconds": 0.0,
                         "collection_seconds": 0.0},
                counts={"attempted": 0, "succeeded": 0, "failed": 0, "skipped": 1},
                database_path=self._database_path(db_conn),
                next_actions=["nerd task show %s" % existing["id"]],
            )

        # 1. Create a unique directory for this run.
        run_dir = make_run_dir(label_path, self.name, suffix=f"__cfg-{cfg_hash_short}")

        # The logger is already configured by the CLI, but we could add a file handler here.
        log.info("Executing task '%s' in run directory: %s", self.name, run_dir)

        # 2. Create the task context.
        run_cfg = cfg.get("run", {})
        # Prefer new slurm_config for resource keys (cpus, memory, time) if present
        remote_block = (run_cfg.get("remote") or {})
        slurm_config = (remote_block.get("slurm_config") or run_cfg.get("slurm_config") or {})

        def _pick_mem(val_run, val_slurm):
            v = val_run if val_run not in (None, "") else val_slurm
            if v in (None, ""):
                return 32
            try:
                return int(v)
            except Exception:
                # If string like '32G' was provided, strip suffix safely
                try:
                    return int(str(v).rstrip().rstrip('GgMm'))
                except Exception:
                    return 32

        threads_val = run_cfg.get("threads")
        mem_val = run_cfg.get("mem_gb")
        time_val = run_cfg.get("time")

        ctx = TaskContext(
            db=db_conn,
            backend=run_cfg.get("backend", "local"),
            workdir=run_dir,
            threads=int(threads_val if threads_val not in (None, "") else slurm_config.get("cpus", 8)),
            mem_gb=_pick_mem(mem_val, slurm_config.get("memory")),
            time=str(time_val if time_val not in (None, "") else slurm_config.get("time", "02:00:00")),
            label=label,
            output_dir=output_dir  # Added output_dir to context
        )

        # 3. Prepare inputs and parameters.
        prepare_started = time.monotonic()
        inputs, params = self.prepare(cfg)
        execution_provenance = None
        if hasattr(self, "prepare_execution"):
            from nerd.scheduler.profiles import load_executor_profile

            execution_profile = load_executor_profile(cfg)
            if hasattr(self, "validate_execution_config"):
                self.validate_execution_config(inputs, execution_profile)
            if execution_profile.executor_type != "local":
                raise RuntimeError(
                    "Containerized Slurm tasks must be launched with 'nerd submit'."
                )
            execution_provenance = self.prepare_execution(cfg, execution_profile, ctx, inputs)
        scope = self.resolve_scope(ctx, inputs)
        preparation_seconds = time.monotonic() - prepare_started
        
        # 4. Record the start of the task in the database.
        task_id = db_api.begin_task(
            ctx.db, self.name, scope.kind, scope.scope_id,
            ctx.backend, ctx.output_dir, ctx.label, cache_key=cache_key_full,
            tool=self.task_tool(inputs), tool_version=self.task_tool_version(inputs)
        )
        if task_id is None:
            log.error("Failed to begin task in database. Aborting.")
            raise SystemExit(1)
        if scope.members:
            try:
                db_api.record_task_scope_members(ctx.db, task_id, scope.members)
            except Exception:
                log.exception("Failed to record task scope members for task_id=%s", task_id)

        # 5. Build the command to be executed.
        cmd = self.command(ctx, inputs, params)
        if execution_provenance is not None:
            from nerd.containers import write_provenance
            from nerd.scheduler import store as scheduler_store

            execution_provenance["command"] = cmd
            write_provenance(run_dir / ".nerd-container-provenance.json", execution_provenance)
            scheduler_store.record_container_provenance(ctx.db, task_id, execution_provenance)
        rc = 0
        execution_started = time.monotonic()

        # 6. Run the command using the appropriate runner.
        if cmd:
            backend = str(ctx.backend).lower()
            log.info("Starting command (backend=%s) in %s", backend or "local", run_dir)
            log.debug("Executing command: %s", cmd)
            if backend in {"slurm", "remote_slurm"}:
                db_api.finish_task(
                    ctx.db,
                    task_id,
                    "failed",
                    "Blocking Slurm execution is no longer supported; use 'nerd submit'.",
                )
                raise RuntimeError(
                    "Slurm tasks must be launched with 'nerd submit' so their job ID and "
                    "attempt state survive controller disconnection."
                )
            # Select runner based on backend
            if backend in {"slurm", "remote_slurm", "ssh", "login", "remote_login", "remote"}:
                try:
                    from nerd.pipeline.runners.remote import RemoteRunner
                    mode = "slurm" if backend in {"slurm", "remote_slurm"} else "ssh"
                    runner = RemoteRunner(mode=mode)
                except Exception:
                    log.exception("Failed to load RemoteRunner; falling back to LocalRunner.")
                    runner = LocalRunner()
            else:
                runner = LocalRunner()

            # Build optional environment for runner
            env = dict(run_cfg.get("env", {}) or {})
            if backend in {"slurm", "remote_slurm", "ssh", "login", "remote_login", "remote"}:
                # Allow settings under 'remote' (recommended), fallback to legacy 'slurm' or 'ssh'
                remote_cfg = (run_cfg.get("remote") or {})
                slurm_cfg = (run_cfg.get("slurm") or run_cfg.get("ssh") or remote_cfg)
                slurm_conf = (slurm_cfg.get("slurm_config") or run_cfg.get("slurm_config") or {})
                # SSH settings
                ssh_cfg = slurm_cfg.get("ssh") or {}
                if ssh_cfg.get("host"):
                    env["SLURM_REMOTE_HOST"] = str(ssh_cfg.get("host"))
                if ssh_cfg.get("user"):
                    env["SLURM_REMOTE_USER"] = str(ssh_cfg.get("user"))
                if ssh_cfg.get("port"):
                    env["SLURM_SSH_PORT"] = str(ssh_cfg.get("port"))
                if ssh_cfg.get("options"):
                    env["SLURM_SSH_OPTIONS"] = str(ssh_cfg.get("options"))
                # Remote base dir for staging
                if slurm_cfg.get("remote_base_dir"):
                    env["SLURM_REMOTE_BASE_DIR"] = str(slurm_cfg.get("remote_base_dir"))
                if backend in {"slurm", "remote_slurm"}:
                    # sbatch resources
                    part = slurm_conf.get("partition") or slurm_cfg.get("partition")
                    acct = slurm_conf.get("account") or slurm_cfg.get("account")
                    time_s = slurm_conf.get("time") or slurm_cfg.get("time") or ctx.time
                    cpus = slurm_conf.get("cpus")
                    mem = slurm_conf.get("memory")
                    if part:
                        env["SLURM_PARTITION"] = str(part)
                    if acct:
                        env["SLURM_ACCOUNT"] = str(acct)
                    env["SLURM_TIME"] = str(time_s)
                    if cpus not in (None, ""):
                        env["SLURM_CPUS"] = str(cpus)
                    if mem not in (None, ""):
                        m = str(mem)
                        # If plain number, assume GB and add 'G'
                        if m.isdigit():
                            m = f"{m}G"
                        env["SLURM_MEM"] = m
                # Optional preamble and stage-out patterns
                preamble = slurm_cfg.get("preamble")
                if isinstance(preamble, list):
                    env["SLURM_PREAMBLE"] = "\n".join(str(x) for x in preamble)
                elif isinstance(preamble, str):
                    env["SLURM_PREAMBLE"] = preamble
                patterns = slurm_cfg.get("stage_out")
                combined: List[str] = []
                if isinstance(patterns, list):
                    combined.extend(str(x) for x in patterns)
                elif isinstance(patterns, str):
                    combined.extend([p.strip() for p in patterns.split(',') if p.strip()])
                extra = self.stage_out_patterns() or []
                if extra:
                    combined.extend(extra)
                if combined:
                    env["SLURM_STAGE_OUT"] = ",".join(combined)

            # Task-provided stage-in specification (e.g., local files to upload to remote)
            try:
                import json as _json
                if hasattr(self, "stage_in_pairs"):
                    pairs = self.stage_in_pairs()  # list of dicts {src, dst}
                    if pairs:
                        env["SLURM_STAGE_IN"] = _json.dumps(pairs)
            except Exception:
                pass

            rc = runner.run(cmd, run_dir, env=env or None)

            # Record the attempt.
            db_api.attempt(ctx.db, task_id, 1, cmd, {}, get_command_log_path(run_dir))
        execution_seconds = time.monotonic() - execution_started
        
        # 7. Check the result and update the task status.
        if rc != 0:
            log.error("Task command failed with exit code %d.", rc)
            db_api.finish_task(ctx.db, task_id, "failed", f"Command returned non-zero exit code: {rc}")
            raise SystemExit(rc)

        # 8. Consume the outputs of the task.
        log.info("Command completed successfully. Consuming outputs.")
        try:
            collection_started = time.monotonic()
            result = self.consume_outputs(ctx, inputs, params, run_dir, task_id=task_id)
            collection_seconds = time.monotonic() - collection_started
        except Exception as exc:
            db_api.finish_task(
                ctx.db,
                task_id,
                "failed",
                "Output validation failed: %s" % exc,
            )
            log.exception("Output validation failed for task_id=%s: %s", task_id, exc)
            raise

        # 9. Mark the task as completed.
        db_api.finish_task(ctx.db, task_id, "completed")
        log.info("Task '%s' (ID: %d) completed successfully.", self.name, task_id)

        # 10. Update the 'latest' symlink to point to this run.
        update_latest_symlink(label_path, self.name, run_dir)
        ended_at = datetime.now().isoformat()
        return self.build_summary(
            ctx, inputs, params, result, task_id=task_id, run_dir=run_dir,
            status="completed", started_at=started_at, ended_at=ended_at,
            timings={
                "preparation_seconds": round(preparation_seconds, 6),
                "execution_seconds": round(execution_seconds, 6),
                "collection_seconds": round(collection_seconds, 6),
            },
            duration_seconds=round(time.monotonic() - total_started, 6),
        )

    @staticmethod
    def _database_path(conn: sqlite3.Connection) -> Optional[str]:
        try:
            return str(Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve())
        except Exception:
            return None

    def build_summary(
        self, ctx: TaskContext, inputs: Any, params: Any, result: Any, *,
        task_id: int, run_dir: Path, status: str, started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        timings: Optional[Dict[str, Optional[float]]] = None,
        duration_seconds: Optional[float] = None,
    ) -> TaskSummary:
        """Build an evidence-backed summary from a task's optional result mapping."""
        details = dict(result) if isinstance(result, dict) else {}
        counts = dict(details.pop("counts", {}) or {})
        counts.setdefault("attempted", 1)
        counts.setdefault("succeeded", 1 if status == "completed" else 0)
        counts.setdefault("failed", 0)
        counts.setdefault("skipped", 0)
        if status == "completed" and counts.get("failed", 0) > 0:
            status = "partial_success" if counts.get("succeeded", 0) > 0 else "failed"
        command_log = get_command_log_path(run_dir)
        artifacts = [
            ArtifactReference("output_directory", str(run_dir), exists=run_dir.exists()),
            ArtifactReference("command_log", str(command_log), exists=command_log.exists()),
        ]
        for item in details.pop("artifacts", []) or []:
            artifacts.append(item if isinstance(item, ArtifactReference) else ArtifactReference(**item))
        plugin = details.pop("plugin", None) or self.task_tool(inputs)
        engine = details.pop("engine", None)
        version = details.pop("version", None) or self.task_tool_version(inputs)
        return TaskSummary(
            status=status, workflow=self.name, task_id=task_id, label=ctx.label,
            plugin=plugin, engine=engine, version=version,
            started_at=started_at, ended_at=ended_at,
            duration_seconds=duration_seconds, timings=timings or {}, counts=counts,
            metrics=dict(details.pop("metrics", {}) or {}),
            warnings=list(details.pop("warnings", []) or []),
            failures=list(details.pop("failures", []) or []), artifacts=artifacts,
            log_path=str(command_log),
            database_path=self._database_path(ctx.db),
            next_actions=["nerd task show %s" % task_id],
        )

    def scope_id(self, ctx: Optional[TaskContext], inputs: Any) -> Optional[int]:
        """Determines the primary ID for the task's scope (e.g., a sample ID)."""
        return None

    def resolve_scope(self, ctx: Optional[TaskContext], inputs: Any) -> TaskScope:
        """
        Compute the scope metadata for this task execution. Default implementation
        uses the class-level scope_kind and scope_id() result.
        """
        scope_kind = getattr(self, "scope_kind", "global") or "global"
        try:
            scope_value = self.scope_id(ctx, inputs)
        except Exception:
            scope_value = None
        return TaskScope(kind=scope_kind, scope_id=scope_value)

    def task_tool(self, inputs: Any) -> Optional[str]:
        return None

    def task_tool_version(self, inputs: Any) -> Optional[str]:
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
    def command(self, ctx: TaskContext, inputs: Any, params: Any) -> Optional[str]:
        """
        Construct the shell command to be executed.
        
        Returns:
            A string command, or None if no command is needed.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def consume_outputs(self, ctx: TaskContext, inputs: Any, params: Any, run_dir: Path, task_id: Optional[int] = None):
        """
        Process the outputs after the command has successfully run.
        """
        raise NotImplementedError

    def stage_out_patterns(self) -> Optional[List[str]]:
        """Optional additional stage-out patterns a task wants the runner to fetch."""
        return None
