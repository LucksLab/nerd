"""
Helpers shared by NMR kinetic fitting tasks.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from nerd.db import api as db_api
from nerd.pipeline.plugins.nmr_fit_kinetics import FitRequest, FitResult, load_nmr_fit_plugin
from nerd.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PreparedReaction:
    reaction_id: int
    row: sqlite3.Row
    files: Dict[str, Path]
    metadata: Dict[str, Any]


def fetch_reactions(
    ctx_db: sqlite3.Connection,
    *,
    reaction_type: Optional[str],
    reaction_ids: Iterable[int] | None,
) -> List[sqlite3.Row]:
    """Load NMR reactions filtered by type/id."""
    return db_api.fetch_nmr_reactions(
        ctx_db,
        reaction_type=reaction_type,
        reaction_ids=list(reaction_ids or []),
    )


def search_roots(base_dir: Path, extra_roots: Iterable[str]) -> List[Path]:
    """Compute search roots for locating trace files."""
    roots: List[Path] = [base_dir]
    for raw in extra_roots or []:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        roots.append(candidate)
    return roots


def prepare_reaction_inputs(
    reaction: sqlite3.Row,
    trace_records: Mapping[str, sqlite3.Row],
    trace_roles: Mapping[str, str],
    roots: Iterable[Path],
    run_dir: Path,
    metadata: Mapping[str, Any],
) -> PreparedReaction:
    """
    Copy trace files for a reaction into the run directory and build metadata.
    """
    files: Dict[str, Path] = {}
    trace_species: Dict[str, Any] = {}
    for plugin_role, db_role in trace_roles.items():
        record = trace_records.get(db_role)
        if record is None:
            raise ValueError(f"Missing trace role '{db_role}' for reaction_id={reaction['id']}.")
        resolved = _resolve_trace_path(str(record["path"]), roots)
        staged = _copy_into_run_dir(resolved, run_dir, int(reaction["id"]), plugin_role)
        files[plugin_role] = staged
        if "species" in record.keys():
            trace_species[plugin_role] = record["species"]

    return PreparedReaction(
        reaction_id=int(reaction["id"]),
        row=reaction,
        files=files,
        metadata={**dict(metadata), "trace_species": {**(metadata.get("trace_species") or {}), **trace_species}},
    )


def run_fit_for_reaction(
    ctx_db: sqlite3.Connection,
    prepared: PreparedReaction,
    *,
    task_id: Optional[int],
    plugin_name: str,
    plugin_options: Mapping[str, Any],
    fit_params: Mapping[str, Any],
    species: str,
    model_name: str,
) -> FitResult:
    """
    Execute the configured plugin for a reaction and persist the resulting rate.
    """
    plugin = load_nmr_fit_plugin(plugin_name, **dict(plugin_options))

    fit_run_id: Optional[int] = None
    if task_id is not None:
        fit_run_id = db_api.begin_nmr_fit_run(
            ctx_db,
            task_id=task_id,
            nmr_reaction_id=prepared.reaction_id,
            plugin=plugin_name,
            params={**plugin.options(), **dict(fit_params)},
            model=model_name,
            species=species,
        )

    request = FitRequest(
        reaction_id=prepared.reaction_id,
        files=prepared.files,
        metadata=prepared.metadata,
        params=fit_params,
    )

    try:
        result = plugin.fit(request)

        if fit_run_id is not None:
            entries = [
                {"param_name": "k_value", "param_numeric": float(result.k_value)},
            ]
            if result.k_error is not None:
                entries.append({"param_name": "k_error", "param_numeric": float(result.k_error)})
            if result.r2 is not None:
                entries.append({"param_name": "r2", "param_numeric": float(result.r2)})
            if result.chisq is not None:
                entries.append({"param_name": "chisq", "param_numeric": float(result.chisq)})
            if species:
                entries.append({"param_name": "species", "param_text": str(species)})
            if model_name:
                entries.append({"param_name": "model", "param_text": str(model_name)})
            diagnostics = getattr(result, "diagnostics", {}) or {}
            for key, value in diagnostics.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    entries.append({"param_name": str(key), "param_numeric": float(value)})
                else:
                    entries.append({"param_name": str(key), "param_text": str(value)})

            db_api.record_nmr_fit_params(ctx_db, fit_run_id=fit_run_id, entries=entries)
            db_api.finish_nmr_fit_run(ctx_db, fit_run_id, "completed")
        return result
    except Exception as exc:  # noqa: BLE001
        if fit_run_id is not None:
            db_api.finish_nmr_fit_run(ctx_db, fit_run_id, "failed", message=str(exc))
        raise


def write_result_artifact(run_dir: Path, prepared: PreparedReaction, result: FitResult) -> Path:
    """Persist a JSON artifact summarizing the fit."""
    out_dir = run_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "reaction_id": prepared.reaction_id,
        "files": {k: str(v) for k, v in prepared.files.items()},
        "metadata": prepared.metadata,
        "result": {
            "k_value": result.k_value,
            "k_error": result.k_error,
            "r2": result.r2,
            "chisq": result.chisq,
            "diagnostics": dict(result.diagnostics),
        },
    }
    target = out_dir / f"reaction_{prepared.reaction_id}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return target


def build_scope_members(reactions: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
    """Return TaskScopeMember-compatible dicts for the provided reactions."""
    members: List[Dict[str, Any]] = []
    for row in reactions:
        label = row["kinetic_data_dir"] if "kinetic_data_dir" in row.keys() else None
        members.append(
            {
                "kind": "nmr_reaction",
                "ref_id": int(row["id"]),
                "label": label,
                "extra": {"reaction_type": row["reaction_type"]},
            }
        )
    return members


def _resolve_trace_path(path_value: str, roots: Iterable[Path]) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in roots:
        potential = root / candidate
        if potential.exists():
            return potential
    raise FileNotFoundError(f"Trace file '{path_value}' not found in roots: {', '.join(str(r) for r in roots)}")


def _copy_into_run_dir(path: Path, run_dir: Path, reaction_id: int, role: str) -> Path:
    dest_dir = run_dir / "inputs" / f"reaction_{reaction_id}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{role}{path.suffix or '.csv'}"
    shutil.copy2(path, dest_path)
    return dest_path
