from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from nerd.db import api as db_api

from ._nmr_common import (
    build_scope_members,
    fetch_reactions,
    prepare_reaction_inputs,
    run_fit_for_reaction,
    search_roots,
    write_result_artifact,
)
from .base import Task, TaskContext, TaskScope, TaskScopeMember


class NmrAddKineticsTask(Task):
    """
    Run adduction (kadd) fits for NMR reactions involving NTP reporters.
    """

    name = "nmr_add_kinetics"
    scope_kind = "nmr_batch"
    _DEFAULT_TRACE_ROLES = {
        "peak_trace": "peak_trace",
        "dms_trace": "dms_trace",
    }

    def prepare(self, cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if self.name not in cfg:
            raise ValueError(f"Configuration must contain a '{self.name}' section.")

        block = dict(cfg[self.name])
        block.setdefault("plugin", "ode_lsq_ntp_add")

        reaction_ids: List[int] = []
        ids_raw = block.get("reaction_ids") or block.get("reactions") or block.get("select")
        if ids_raw:
            if not isinstance(ids_raw, (list, tuple, set)):
                ids_raw = [ids_raw]
            for val in ids_raw:
                if val in (None, ""):
                    continue
                try:
                    reaction_ids.append(int(val))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid reaction id '{val}' in {self.name} config.") from exc
        block["reaction_ids"] = reaction_ids

        trace_roles = dict(self._DEFAULT_TRACE_ROLES)
        for key, value in (block.get("trace_roles") or {}).items():
            if key:
                trace_roles[str(key)] = str(value)
        block["trace_roles"] = trace_roles

        block["fit_params"] = dict(block.get("fit_params") or {})
        block["plugin_options"] = dict(block.get("plugin_options") or {})
        block["search_roots"] = list(block.get("search_roots") or [])
        block["species"] = block.get("species")
        block["model"] = block.get("model") or block.get("plugin")

        return block, {}

    def command(self, ctx: TaskContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
        return None

    def consume_outputs(
        self,
        ctx: TaskContext,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        run_dir: Path,
        task_id: Optional[int] = None,
    ):
        reactions = fetch_reactions(
            ctx.db,
            reaction_type="add",
            reaction_ids=inputs.get("reaction_ids"),
        )
        if not reactions:
            raise ValueError("No NMR adduction reactions matched the provided filters.")

        roots = search_roots(Path(ctx.output_dir) / ctx.label, inputs.get("search_roots", []))
        trace_map = db_api.fetch_nmr_trace_files(ctx.db, [row["id"] for row in reactions])
        plugin_name = str(inputs.get("plugin")).strip()
        plugin_options = inputs.get("plugin_options") or {}
        fit_params = inputs.get("fit_params") or {}
        model_name = str(inputs.get("model") or plugin_name or "nmr_add")

        for row in reactions:
            metadata = dict(row)
            metadata.setdefault("species", inputs.get("species") or row["substrate"] or "ntp_probe")
            metadata.setdefault("ntp_conc", row["substrate_conc"])
            metadata.setdefault("dms_conc", row["probe_conc"])

            prepared = prepare_reaction_inputs(
                row,
                trace_map.get(row["id"], {}),
                inputs["trace_roles"],
                roots,
                run_dir,
                metadata=metadata,
            )
            species = str(inputs.get("species") or metadata.get("species") or "ntp_probe")
            result = run_fit_for_reaction(
                ctx.db,
                prepared,
                task_id=task_id,
                plugin_name=plugin_name,
                plugin_options=plugin_options,
                fit_params=fit_params,
                species=species,
                model_name=model_name,
            )
            write_result_artifact(run_dir, prepared, result)

    def resolve_scope(self, ctx: Optional[TaskContext], inputs: Any) -> TaskScope:
        if ctx is None or inputs is None:
            return TaskScope(kind=self.scope_kind or "nmr_batch")

        reactions = fetch_reactions(
            ctx.db,
            reaction_type="add",
            reaction_ids=(inputs.get("reaction_ids") if isinstance(inputs, dict) else None),
        )
        members_dicts = build_scope_members(reactions)
        members = [
            TaskScopeMember(kind=m["kind"], ref_id=m["ref_id"], label=m["label"], extra=m["extra"])
            for m in members_dicts
        ]
        if len(members) == 1:
            member = members[0]
            return TaskScope(kind="nmr_reaction", scope_id=member.ref_id, label=member.label, members=members)
        scope = TaskScope(kind="nmr_batch", members=members)
        if members:
            scope.label = f"{len(members)} reactions"
        return scope
