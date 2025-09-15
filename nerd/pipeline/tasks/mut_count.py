# nerd/pipeline/tasks/mut_count.py
"""
Task for mutation counting using pluggable tools (e.g., shapemapper).
Parent samples only (no derived materialization yet).
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

from .base import Task, TaskContext
from nerd.utils.logging import get_logger
from nerd.pipeline.runners.local import LocalRunner

log = get_logger(__name__)


class MutCountTask(Task):
    """
    A task that invokes a mutation counting tool via a runner and imports
    resulting data. This initial version handles only parent samples specified
    by name in the config.
    """
    name = "mut_count"
    scope_kind = "sample"  # per-sample operations; aggregated under one task

    def prepare(self, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.name not in cfg:
            raise ValueError(f"Configuration must contain a '{self.name}' section.")

        mc = cfg[self.name]
        # Minimal required fields
        plugin = mc.get("plugin")
        if not plugin:
            raise ValueError("mut_count.plugin is required (e.g., 'shapemapper').")

        samples = mc.get("samples", [])
        if not samples:
            raise ValueError("mut_count.samples must list parent sample names for this initial implementation.")

        # Resolve parent samples from DB now so command() can run everything; no derived support yet
        from nerd.db import api as db_api  # local import
        # We cannot access ctx here, so defer fq roots to command() using run_dir; only fetch DB rows now.
        return mc, {}

    def command(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
        """
        Build a composite shell command that iterates over parent samples,
        creates per-sample output dirs, and invokes the tool (or placeholder).
        """
        plugin_name = inputs.get("plugin")
        tool_cfg = inputs.get("tool", {})
        tool_bin = tool_cfg.get("bin")  # optional; rely on PATH if None
        threads = inputs.get("params", {}).get("n_proc") or inputs.get("threads") or 1

        sample_names: List[str] = inputs.get("samples", [])
        if not sample_names:
            return None

        # Build a shell script string to be executed in run_dir
        # Resolve sample rows by name at runtime via sqlite3 CLI could be overkill; instead,
        # we simply generate per-sample commands that assume fq_dir and files are relative to label root.
        # Here we query DB at runtime is not trivial; instead, we will resolve in consume_outputs for now.
        # For a working skeleton, create dirs and write placeholders if no tool_bin.

        cmds: List[str] = []
        for name in sample_names:
            # Each sample gets its own subdir
            sample_dir_cmd = f"mkdir -p samples/{name}"
            if tool_bin:
                # Without DB access here, we cannot resolve actual FASTQ paths safely.
                # Emit a placeholder that records intent; real command will be refined with plugin later.
                cmd = (
                    f"echo '[mut_count] Would run {plugin_name} for {name} with {threads} threads using {tool_bin}' "
                    f"> samples/{name}/intent.txt"
                )
            else:
                # Simple echo test that produces an out.txt file for stage-out
                cmd = (
                    "bash -lc '"
                    f"echo Running mut_count echo test for {name} > samples/{name}/out.txt"
                    "'"
                )
            cmds.append(sample_dir_cmd)
            cmds.append(cmd)

        # Chain with && to stop on first failure
        return " && ".join(cmds)

    def _resolve_fastqs(self, ctx: TaskContext, row) -> Tuple[Path, Path]:
        label_dir = Path(ctx.output_dir) / ctx.label
        fq_dir = Path(row["fq_dir"]) if isinstance(row["fq_dir"], str) else Path(str(row["fq_dir"]))
        if not fq_dir.is_absolute():
            fq_dir = label_dir / fq_dir
        r1 = fq_dir / row["r1_file"]
        r2 = fq_dir / row["r2_file"]
        return r1, r2

    def consume_outputs(self, ctx: TaskContext, inputs: Dict[str, Any], params: Dict[str, Any], run_dir: Path):
        """
        Execute mutation counting for the selected parent samples.
        """
        # For now, consumption only logs; later, parse outputs from run_dir/samples/<name>
        sample_names: List[str] = inputs.get("samples", [])
        if not sample_names:
            log.warning("mut_count.consume_outputs called with no samples.")
            return
        log.info("mut_count completed for %d samples (command phase). Ready to parse outputs.", len(sample_names))
