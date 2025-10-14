"""
Task for ingesting NMR reaction metadata and trace registrations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from nerd.db import api as db_api
from nerd.utils.logging import get_logger

from ._nmr_common import search_roots
from .base import Task, TaskContext, TaskScope, TaskScopeMember

log = get_logger(__name__)


class NmrCreateTask(Task):
    """
    Ingest NMR reactions from configuration into the database.
    """

    name = "nmr_create"
    scope_kind = "nmr_batch"

    REQUIRED_FIELDS: tuple[str, ...] = (
        "reaction_type",
        "temperature",
        "replicate",
        "num_scans",
        "time_per_read",
        "total_kinetic_reads",
        "total_kinetic_time",
        "probe",
        "probe_conc",
        "probe_solvent",
        "substrate",
        "substrate_conc",
        "buffer",  # or buffer_id
        "nmr_machine",
        "kinetic_data_dir",
    )

    def prepare(self, cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if self.name not in cfg:
            raise ValueError(f"Configuration must contain a '{self.name}' section.")

        block = dict(cfg[self.name])
        reactions_raw = block.get("reactions")
        if not reactions_raw:
            raise ValueError("nmr_create requires a 'reactions' list.")
        if isinstance(reactions_raw, dict):
            reactions_raw = [reactions_raw]
        if not isinstance(reactions_raw, list):
            raise TypeError("nmr_create.reactions must be a list of reaction definitions.")

        reactions: List[Dict[str, Any]] = []
        for idx, entry in enumerate(reactions_raw, start=1):
            if not isinstance(entry, dict):
                raise TypeError(f"Reaction entry {idx} must be a mapping.")
            missing = [field for field in self.REQUIRED_FIELDS if entry.get(field) in (None, "")]
            if missing:
                raise ValueError(f"Reaction entry {idx} missing required field(s): {', '.join(missing)}")
            reactions.append(dict(entry))

        block["reactions"] = reactions
        block["search_roots"] = list(block.get("search_roots") or [])

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
        label_dir = Path(ctx.output_dir) / ctx.label
        roots = search_roots(label_dir, inputs.get("search_roots") or [])

        for reaction in inputs.get("reactions", []):
            record, trace_map = self._build_reaction_payload(ctx, reaction)
            reaction_id = db_api.upsert_nmr_reaction(ctx.db, record)
            if reaction_id is None:
                raise RuntimeError(f"Failed to insert/update NMR reaction '{record['kinetic_data_dir']}'.")

            for role, info in trace_map.items():
                path_value = info["path"]
                resolved = self._resolve_trace_path(path_value, roots)
                db_api.register_nmr_trace_file(
                    ctx.db,
                    nmr_reaction_id=reaction_id,
                    role=role,
                    path=str(path_value),
                    species=info.get("species"),
                    checksum=None,
                    task_id=task_id,
                )
                log.info("Registered trace '%s' for reaction_id=%s (%s)", role, reaction_id, resolved)

    def resolve_scope(self, ctx: Optional[TaskContext], inputs: Any) -> TaskScope:
        if ctx is None or not isinstance(inputs, dict):
            reactions = inputs.get("reactions") if isinstance(inputs, dict) else []
            label = f"{len(reactions or [])} reactions" if reactions else None
            scope = TaskScope(kind=self.scope_kind, label=label)
            return scope

        members: List[TaskScopeMember] = []
        for reaction in inputs.get("reactions", []):
            kinetic_dir = reaction.get("kinetic_data_dir")
            reaction_id = db_api.get_nmr_reaction_id_by_dir(ctx.db, kinetic_dir) if kinetic_dir else None
            members.append(
                TaskScopeMember(
                    kind="nmr_reaction",
                    ref_id=reaction_id,
                    label=kinetic_dir,
                    extra={"reaction_type": reaction.get("reaction_type")},
                )
            )

        if len(members) == 1 and members[0].ref_id is not None:
            member = members[0]
            return TaskScope(kind="nmr_reaction", scope_id=member.ref_id, label=member.label, members=members)

        scope = TaskScope(kind=self.scope_kind, members=members)
        if members:
            scope.label = f"{len(members)} reactions"
        return scope

    def _build_reaction_payload(
        self,
        ctx: TaskContext,
        reaction: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, Optional[str]]]]:
        buffer_id = reaction.get("buffer_id")
        if buffer_id in (None, ""):
            buffer_identifier = reaction.get("buffer")
            if isinstance(buffer_identifier, str):
                buffer_identifier = buffer_identifier.strip()
            buffer_id = db_api.get_buffer_id_by_name_or_disp(ctx.db, buffer_identifier)
        if buffer_id is None:
            available = ctx.db.execute("SELECT id, name, disp_name FROM meta_buffers ORDER BY id LIMIT 10").fetchall()
            log.error(
                "Available buffers (first 10): %s",
                [(row['id'], row['name'], row['disp_name']) for row in available],
            )
            raise ValueError(f"Buffer '{reaction.get('buffer')}' not found in database.")

        record = {
            "reaction_type": reaction["reaction_type"],
            "temperature": reaction["temperature"],
            "replicate": reaction["replicate"],
            "num_scans": reaction["num_scans"],
            "time_per_read": reaction["time_per_read"],
            "total_kinetic_reads": reaction["total_kinetic_reads"],
            "total_kinetic_time": reaction["total_kinetic_time"],
            "probe": reaction["probe"],
            "probe_conc": reaction["probe_conc"],
            "probe_solvent": reaction["probe_solvent"],
            "substrate": reaction["substrate"],
            "substrate_conc": reaction["substrate_conc"],
            "buffer_id": buffer_id,
            "nmr_machine": reaction["nmr_machine"],
            "kinetic_data_dir": reaction["kinetic_data_dir"],
            "mnova_analysis_dir": reaction.get("mnova_analysis_dir"),
            "raw_fid_dir": reaction.get("raw_fid_dir"),
        }

        trace_files = reaction.get("trace_files") or {}
        if not isinstance(trace_files, dict):
            raise TypeError("trace_files must be a mapping of role → path.")

        normalized: Dict[str, Dict[str, Optional[str]]] = {}
        for role, value in trace_files.items():
            role_str = str(role)
            species_val: Optional[str] = None
            if isinstance(value, dict):
                path_val = value.get("path")
                species_raw = value.get("species")
                if species_raw not in (None, ""):
                    species_val = str(species_raw).strip()
            else:
                path_val = value
            if path_val in (None, ""):
                raise ValueError(f"Trace role '{role_str}' is missing a path.")
            normalized[role_str] = {
                "path": str(path_val),
                "species": species_val,
            }

        return record, normalized

    @staticmethod
    def _resolve_trace_path(path_value: str, roots: List[Path]) -> Path:
        candidate = Path(path_value)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for root in roots:
            potential = root / candidate
            if potential.exists():
                return potential
        raise FileNotFoundError(f"Trace file '{path_value}' not found in search roots: {', '.join(str(r) for r in roots)}")
