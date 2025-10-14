"""
Task for running probe timecourse fitting engines.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from nerd.db import api as db_api
from nerd.pipeline.plugins.timecourse import (
    NucleotideSeries,
    TimecourseRequest,
    load_timecourse_engine,
    ROUND_CONSTRAINED,
    ROUND_FREE,
    ROUND_GLOBAL,
)
from nerd.pipeline.tasks.base import Task, TaskContext, TaskScope, TaskScopeMember
from nerd.utils.logging import get_logger

log = get_logger(__name__)


class ProbeTimecourseTask(Task):
    """
    Execute a configured timecourse engine for one or more reaction groups.
    """

    name = "probe_timecourse"
    scope_kind = "rg_batch"

    _DEFAULT_ROUNDS: Tuple[str, ...] = (ROUND_FREE, ROUND_GLOBAL, ROUND_CONSTRAINED)

    def prepare(self, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.name not in cfg:
            raise ValueError(f"Configuration must contain a '{self.name}' section.")

        block = dict(cfg[self.name])
        block["engine"] = str(block.get("engine") or "python_baseline")

        rounds_raw = block.get("rounds")
        if not rounds_raw:
            rounds = list(self._DEFAULT_ROUNDS)
        elif isinstance(rounds_raw, (list, tuple)):
            rounds = [str(r) for r in rounds_raw if r is not None]
        else:
            rounds = [str(rounds_raw)]
        if not rounds:
            raise ValueError("At least one round must be specified for probe_timecourse.")
        block["rounds"] = rounds

        engine_options = block.get("engine_options")
        block["engine_options"] = dict(engine_options or {})

        global_metadata = block.get("global_metadata")
        block["global_metadata"] = dict(global_metadata or {})

        rg_inputs = block.get("rg_ids") or block.get("reaction_groups") or block.get("rg")
        if not rg_inputs:
            raise ValueError("probe_timecourse requires 'rg_ids' (list of reaction group IDs).")

        rg_ids: List[int] = []
        if isinstance(rg_inputs, (list, tuple, set)):
            sources = rg_inputs
        else:
            sources = [rg_inputs]
        for raw in sources:
            if raw in (None, ""):
                continue
            try:
                rg_ids.append(int(raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid rg_id '{raw}' in probe_timecourse config.") from exc
        if not rg_ids:
            raise ValueError("probe_timecourse config resolved to zero reaction group IDs.")
        block["rg_ids"] = rg_ids

        block["overwrite"] = bool(block.get("overwrite", False))
        block["valtype"] = str(block.get("valtype") or "fmod")

        fmod_run_id = block.get("fmod_run_id")
        block["fmod_run_id"] = int(fmod_run_id) if fmod_run_id not in (None, "") else None

        nt_ids = block.get("nt_ids")
        if nt_ids:
            if not isinstance(nt_ids, (list, tuple, set)):
                nt_ids = [nt_ids]
            normalized: List[int] = []
            for raw in nt_ids:
                if raw in (None, ""):
                    continue
                try:
                    normalized.append(int(raw))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid nt_id '{raw}' in probe_timecourse config.") from exc
            block["nt_ids"] = normalized
        else:
            block["nt_ids"] = []

        min_points = block.get("min_points")
        block["min_points"] = int(min_points) if min_points not in (None, "") else 3

        block["_scope_cache"] = None  # placeholder populated in resolve_scope
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
        engine_name = inputs["engine"]
        rounds = inputs["rounds"]
        overwrite = bool(inputs.get("overwrite", True))
        valtype = inputs.get("valtype")
        nt_filter = set(inputs.get("nt_ids") or [])
        min_points = int(inputs.get("min_points", 3))

        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        engine = load_timecourse_engine(engine_name, **dict(inputs.get("engine_options") or {}))

        for rg_id in inputs["rg_ids"]:
            try:
                series, rg_meta = self._load_rg_dataset(
                    ctx.db,
                    rg_id,
                    valtype=valtype,
                    min_points=min_points,
                    nt_filter=nt_filter,
                    preferred_fmod_run=inputs.get("fmod_run_id"),
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("Failed to load timecourse data for rg_id=%s: %s", rg_id, exc)
                continue

            if not series:
                log.warning("No usable nucleotides found for rg_id=%s; skipping engine run.", rg_id)
                continue

            global_metadata = dict(inputs.get("global_metadata") or {})
            global_metadata.update(rg_meta)

            request = TimecourseRequest(
                rg_id=rg_id,
                rounds=rounds,
                nucleotides=series,
                global_metadata=global_metadata,
                options=dict(inputs.get("engine_options") or {}),
            )
            result = engine.run(request)

            self._write_result_artifact(results_dir, rg_id, result)
            if task_id is not None:
                self._persist_result(
                    ctx.db,
                    result,
                    model=engine_name,
                    overwrite=overwrite,
                )

    def resolve_scope(self, ctx: Optional[TaskContext], inputs: Any) -> TaskScope:
        if ctx is None or inputs is None:
            return TaskScope(kind=self.scope_kind or "rg_batch")

        rg_ids: Sequence[int] = inputs.get("rg_ids") if isinstance(inputs, dict) else []
        members: List[TaskScopeMember] = []
        labels: Dict[int, Optional[str]] = {}
        for rg_id in rg_ids or []:
            row = ctx.db.execute(
                "SELECT rg_label FROM probe_reaction_groups WHERE rg_id = ?",
                (rg_id,),
            ).fetchone()
            label = row["rg_label"] if row else None
            labels[int(rg_id)] = label
            members.append(TaskScopeMember(kind="rg", ref_id=int(rg_id), label=label))

        if len(rg_ids) == 1:
            rg_id = int(rg_ids[0])
            return TaskScope(kind="rg", scope_id=rg_id, label=labels.get(rg_id), members=members)

        scope = TaskScope(kind="rg_batch", members=members)
        if members:
            scope.label = f"{len(members)} reaction groups"
        inputs["_scope_cache"] = {"labels": labels}
        return scope

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_result_artifact(self, results_dir: Path, rg_id: int, result) -> None:
        payload = asdict(result)
        target = results_dir / f"rg_{rg_id}.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _load_rg_dataset(
        self,
        conn,
        rg_id: int,
        *,
        valtype: str,
        min_points: int,
        nt_filter: Iterable[int],
        preferred_fmod_run: Optional[int],
    ) -> Tuple[List[NucleotideSeries], Dict[str, Any]]:
        nt_filter_set = {int(x) for x in nt_filter or []}
        fmod_run_id = self._resolve_fmod_run(conn, rg_id, valtype, preferred_fmod_run)

        sql = """
            SELECT
                fv.nt_id,
                fv.fmod_val,
                fv.fmod_run_id,
                pr.reaction_time,
                pr.treated,
                pr.temperature,
                mb.pH AS buffer_pH,
                mn.site,
                mn.base
            FROM probe_fmod_values fv
            JOIN probe_reactions pr ON pr.id = fv.rxn_id
            JOIN meta_buffers mb ON mb.id = pr.buffer_id
            JOIN meta_nucleotides mn ON mn.id = fv.nt_id
            WHERE pr.rg_id = :rg_id
              AND fv.fmod_val IS NOT NULL
              AND fv.valtype = :valtype
              AND (:fmod_run_id IS NULL OR fv.fmod_run_id = :fmod_run_id)
            ORDER BY fv.nt_id, pr.reaction_time, pr.id
        """
        rows = conn.execute(
            sql,
            {"rg_id": rg_id, "valtype": valtype, "fmod_run_id": fmod_run_id},
        ).fetchall()
        if not rows:
            raise ValueError(f"No timecourse data found for rg_id={rg_id}.")

        log.debug(
            "Loaded %d probe_fmod rows for rg_id=%s (valtype=%s, fmod_run_id=%s).",
            len(rows),
            rg_id,
            valtype,
            fmod_run_id,
        )

        series_map: Dict[int, Dict[str, Any]] = {}
        temperatures: List[float] = []
        pH_values: List[float] = []

        for row in rows:
            nt_id = int(row["nt_id"])
            if nt_filter_set and nt_id not in nt_filter_set:
                log.debug("Skipping nt_id=%s for rg_id=%s due to nt filter.", nt_id, rg_id)
                continue

            treated_flag = int(row["treated"])
            time_val = float(row["reaction_time"]) if treated_flag else 0.0
            fmod_val = float(row["fmod_val"])

            entry = series_map.setdefault(
                nt_id,
                {
                    "time": [],
                    "fmod": [],
                    "metadata": {
                        "site": int(row["site"]),
                        "base": str(row["base"]).upper(),
                        "fmod_run_id": row["fmod_run_id"],
                    },
                },
            )
            entry["time"].append(time_val)
            entry["fmod"].append(fmod_val)

            temp = row["temperature"]
            if temp is not None:
                temperatures.append(float(temp))
            ph_val = row["buffer_pH"]
            if ph_val is not None:
                pH_values.append(float(ph_val))

        if not series_map:
            raise ValueError(f"Timecourse data for rg_id={rg_id} did not match nt filter.")
        #print(series_map)
        series_list: List[NucleotideSeries] = []
        for nt_id, content in series_map.items():
            paired = sorted(zip(content["time"], content["fmod"]), key=lambda x: x[0])
            times_sorted = [p[0] for p in paired]
            fmods_sorted = [p[1] for p in paired]
            if len(times_sorted) < min_points:
                log.debug(
                    "Skipping nt_id=%s for rg_id=%s (points=%d < min_points=%d).",
                    nt_id,
                    rg_id,
                    len(times_sorted),
                    min_points,
                )
                continue
            series_list.append(
                NucleotideSeries(
                    nt_id=nt_id,
                    timepoints=times_sorted,
                    fmod_values=fmods_sorted,
                    metadata=content["metadata"],
                )
            )

        if not series_list:
            raise ValueError(f"No nucleotides met inclusion criteria for rg_id={rg_id}.")

        temperature = _unique_or_none(temperatures)
        buffer_pH = _unique_or_none(pH_values)

        metadata = {
            "rg_id": rg_id,
            "fmod_run_id": fmod_run_id,
        }
        if temperature is not None:
            metadata["temperature_c"] = temperature
        if buffer_pH is not None:
            metadata["buffer_pH"] = buffer_pH

        rg_label_row = conn.execute(
            "SELECT rg_label FROM probe_reaction_groups WHERE rg_id = ?",
            (rg_id,),
        ).fetchone()
        if rg_label_row:
            metadata["rg_label"] = rg_label_row["rg_label"]

        return series_list, metadata

    def _resolve_fmod_run(self, conn, rg_id: int, valtype: str, preferred: Optional[int]) -> Optional[int]:
        if preferred not in (None, ""):
            return int(preferred)
        # Default: use all runs for the reaction group so timepoints aggregate correctly.
        return None

    def _persist_result(
        self,
        conn,
        result,
        *,
        model: str,
        overwrite: bool,
    ) -> None:
        rg_id = int(result.metadata.get("rg_id")) if result.metadata.get("rg_id") is not None else None

        for round_result in result.rounds:
            round_id = round_result.round_id
            if overwrite:
                db_api.delete_probe_tc_fit_runs(conn, fit_kind=round_id, rg_id=rg_id, nt_id=None)

            # Global parameters (nt_id = None)
            if round_result.global_params or round_result.qc_metrics:
                fit_run_id = db_api.begin_probe_tc_fit_run(
                    conn,
                    fit_kind=round_id,
                    rg_id=rg_id,
                    nt_id=None,
                    model=model,
                )
                if fit_run_id is not None:
                    entries = [
                        {"param_name": "status", "param_text": round_result.status},
                        {"param_name": "engine", "param_text": result.engine},
                        {"param_name": "engine_version", "param_text": result.engine_version},
                    ]
                    if round_result.notes:
                        entries.append({"param_name": "note", "param_text": str(round_result.notes)})
                    entries.extend(_entries_from_mapping(round_result.global_params))
                    entries.extend(_entries_from_mapping(round_result.qc_metrics, prefix="qc:"))
                    db_api.record_probe_tc_fit_params(conn, fit_run_id=fit_run_id, entries=entries)

            # Per-nucleotide fits
            for per_nt in round_result.per_nt:
                nt_id = per_nt.nt_id
                if overwrite:
                    db_api.delete_probe_tc_fit_runs(conn, fit_kind=round_id, rg_id=rg_id, nt_id=nt_id)

                fit_run_id = db_api.begin_probe_tc_fit_run(
                    conn,
                    fit_kind=round_id,
                    rg_id=rg_id,
                    nt_id=nt_id,
                    model=model,
                )
                if fit_run_id is None:
                    continue
                entries = [
                    {"param_name": "status", "param_text": round_result.status},
                    {"param_name": "engine", "param_text": result.engine},
                    {"param_name": "engine_version", "param_text": result.engine_version},
                ]
                if round_result.notes:
                    entries.append({"param_name": "note", "param_text": str(round_result.notes)})
                entries.extend(_entries_from_mapping(per_nt.params))
                entries.extend(_entries_from_mapping(per_nt.diagnostics, prefix="diag:"))
                db_api.record_probe_tc_fit_params(conn, fit_run_id=fit_run_id, entries=entries)


def _entries_from_mapping(data: Mapping[str, Any], *, prefix: str = "") -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not data:
        return entries
    for key, value in data.items():
        if value is None:
            continue
        name = f"{prefix}{key}"
        if isinstance(value, (int, float)) and math.isfinite(value):
            entries.append({"param_name": name, "param_numeric": float(value)})
        elif isinstance(value, (dict, list, tuple)):
            entries.append({"param_name": name, "param_text": json.dumps(value, sort_keys=True)})
        else:
            entries.append({"param_name": name, "param_text": str(value)})
    return entries


def _unique_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    unique_vals = {round(v, 6) for v in values if v is not None}
    if len(unique_vals) == 1:
        return next(iter(unique_vals))
    log.debug("Multiple unique values encountered where one was expected: %s", unique_vals)
    return None
