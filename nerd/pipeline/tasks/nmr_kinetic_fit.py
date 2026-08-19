"""Unified public NMR kinetic-fitting task."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import Task, TaskContext, TaskScope
from .nmr_add_kinetics import NmrAddKineticsTask
from .nmr_deg_kinetics import NmrDegKineticsTask


class NmrKineticFitTask(Task):
    """Dispatch degradation or adduction fitting through one public workflow."""

    name = "nmr_kinetic_fit"
    scope_kind = "nmr_batch"
    _TASKS = {
        "degradation": NmrDegKineticsTask,
        "adduction": NmrAddKineticsTask,
    }

    def _delegate(self, fit_type: str) -> Task:
        task_class = self._TASKS.get(fit_type)
        if task_class is None:
            raise ValueError(
                "nmr_kinetic_fit.fit_type must be 'degradation' or 'adduction'."
            )
        return task_class()

    def prepare(self, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        block = cfg.get(self.name)
        if not isinstance(block, dict):
            raise ValueError("Configuration must contain an 'nmr_kinetic_fit' section.")
        fit_type = str(block.get("fit_type", "")).strip().lower()
        delegate = self._delegate(fit_type)
        delegated_cfg = dict(cfg)
        delegated_cfg[delegate.name] = {
            key: value for key, value in block.items() if key != "fit_type"
        }
        inputs, params = delegate.prepare(delegated_cfg)
        inputs["fit_type"] = fit_type
        return inputs, params

    def command(
        self, ctx: TaskContext, inputs: Dict[str, Any], params: Dict[str, Any]
    ) -> Optional[str]:
        return self._delegate(str(inputs["fit_type"])).command(ctx, inputs, params)

    def consume_outputs(
        self,
        ctx: TaskContext,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        run_dir: Path,
        task_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._delegate(str(inputs["fit_type"])).consume_outputs(
            ctx, inputs, params, run_dir, task_id=task_id
        )
        rows = ctx.db.execute(
            "SELECT id, plugin, status FROM nmr_fit_runs WHERE task_id=? ORDER BY id",
            (task_id,),
        ).fetchall() if task_id is not None else []
        completed = sum(1 for row in rows if row["status"] == "completed")
        failed = sum(1 for row in rows if row["status"] == "failed")
        fit_ids = [int(row["id"]) for row in rows]
        r2_values = []
        chisq_values = []
        if fit_ids:
            placeholders = ",".join("?" for _ in fit_ids)
            for row in ctx.db.execute(
                "SELECT param_name, param_numeric FROM nmr_fit_params "
                "WHERE fit_run_id IN (%s) AND param_name IN ('r2','chisq')" % placeholders,
                fit_ids,
            ).fetchall():
                if row["param_name"] == "r2" and row["param_numeric"] is not None:
                    r2_values.append(float(row["param_numeric"]))
                if row["param_name"] == "chisq" and row["param_numeric"] is not None:
                    chisq_values.append(float(row["param_numeric"]))
        attempted = len(rows)
        plugin = str(inputs.get("plugin") or (rows[0]["plugin"] if rows else "")) or None
        def _quality_summary(values):
            if not values:
                return None
            return {"count": len(values), "min": min(values),
                    "mean": sum(values) / len(values), "max": max(values)}
        return {
            "plugin": plugin,
            "counts": {"attempted": attempted, "succeeded": completed, "failed": failed,
                       "skipped": max(0, attempted - completed - failed),
                       "reactions_attempted": attempted, "reactions_succeeded": completed,
                       "reactions_failed": failed,
                       "reactions_skipped": max(0, attempted - completed - failed),
                       "convergence_failures": failed, "missing_trace_failures": 0},
            "metrics": {"fit_type": inputs["fit_type"],
                        "fit_success_rate": (completed / attempted if attempted else None),
                        "r2_summary": _quality_summary(r2_values),
                        "chisq_summary": _quality_summary(chisq_values)},
            "artifacts": [{"kind": "results_directory", "path": str(run_dir / "results")}],
        }

    def resolve_scope(
        self, ctx: Optional[TaskContext], inputs: Any
    ) -> TaskScope:
        if not isinstance(inputs, dict) or not inputs.get("fit_type"):
            return TaskScope(kind=self.scope_kind)
        return self._delegate(str(inputs["fit_type"])).resolve_scope(ctx, inputs)
