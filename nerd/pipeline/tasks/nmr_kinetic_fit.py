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
    ) -> None:
        self._delegate(str(inputs["fit_type"])).consume_outputs(
            ctx, inputs, params, run_dir, task_id=task_id
        )

    def resolve_scope(
        self, ctx: Optional[TaskContext], inputs: Any
    ) -> TaskScope:
        if not isinstance(inputs, dict) or not inputs.get("fit_type"):
            return TaskScope(kind=self.scope_kind)
        return self._delegate(str(inputs["fit_type"])).resolve_scope(ctx, inputs)
