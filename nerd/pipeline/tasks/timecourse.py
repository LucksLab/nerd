"""
Task for running probe timecourse fitting engines.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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

        valtype_cfg = block.get("valtype")
        if isinstance(valtype_cfg, (list, tuple, set)):
            valtypes = [str(v) for v in valtype_cfg if v not in (None, "")]
        elif valtype_cfg in (None, ""):
            valtypes = ["modrate"]
        else:
            valtypes = [str(valtype_cfg)]
        if not valtypes:
            raise ValueError("probe_timecourse requires at least one valtype.")
        block["valtypes"] = valtypes
        block["valtype"] = valtypes[0] if len(valtypes) == 1 else list(valtypes)

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

        include_dropped = block.get("include_dropped_samples")
        if include_dropped in (None, ""):
            block["include_dropped_samples"] = False
        else:
            block["include_dropped_samples"] = bool(include_dropped)

        done_by_cfg = block.get("done_by")
        if done_by_cfg in (None, "", False):
            block["done_by"] = []
        elif isinstance(done_by_cfg, (list, tuple, set)):
            done_by_norm = []
            for raw in done_by_cfg:
                if raw in (None, ""):
                    continue
                done_by_norm.append(str(raw).strip())
            block["done_by"] = [val for val in done_by_norm if val]
        else:
            done_by_val = str(done_by_cfg).strip()
            block["done_by"] = [done_by_val] if done_by_val else []

        rt_cfg = block.get("rt_protocol") or block.get("RT_protocol")
        if rt_cfg in (None, "", False):
            block["rt_protocol"] = []
        elif isinstance(rt_cfg, (list, tuple, set)):
            rt_norm: List[str] = []
            for item in rt_cfg:
                if item in (None, ""):
                    continue
                rt_norm.append(str(item).strip().lower())
            block["rt_protocol"] = [val for val in rt_norm if val]
        else:
            block["rt_protocol"] = [str(rt_cfg).strip().lower()]

        outliers_cfg = block.get("outliers") or []
        parsed_outliers: List[Dict[str, Any]] = []
        if isinstance(outliers_cfg, str):
            outliers_cfg = [outliers_cfg]
        if isinstance(outliers_cfg, (list, tuple, set)):
            for raw in outliers_cfg:
                if raw in (None, ""):
                    continue
                parts = str(raw).split(":")
                if len(parts) != 4:
                    raise ValueError(
                        f"Invalid outlier specification '{raw}'. Expected format 'rg_id:sample_name:site_base:valtype'."
                    )
                rg_part, sample_name, site_base, valtype = parts
                try:
                    rg_value = int(rg_part)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid rg_id '{rg_part}' in outlier spec '{raw}'.") from exc
                if "_" not in site_base:
                    raise ValueError(
                        f"Invalid site_base '{site_base}' in outlier spec '{raw}'. Expected '<site>_<base>'."
                    )
                site_str, base_str = site_base.split("_", 1)
                try:
                    site_value = int(site_str)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid site '{site_str}' in outlier spec '{raw}'.") from exc
                base_value = base_str.strip().upper()
                if base_value == "":
                    raise ValueError(f"Empty base in outlier spec '{raw}'.")
                parsed_outliers.append(
                    {
                        "rg_id": rg_value,
                        "sample_name": sample_name,
                        "site": site_value,
                        "base": base_value,
                        "valtype": str(valtype).strip(),
                        "original": str(raw),
                    }
                )
        else:
            raise TypeError("outliers config must be a list of 'rg_id:sample:site_base:valtype' strings.")
        block["outliers"] = parsed_outliers

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
        valtypes = list(inputs.get("valtypes") or [])

        if not valtypes:
            valtypes = ["modrate"]
        nt_filter = set(inputs.get("nt_ids") or [])
        min_points = int(inputs.get("min_points", 3))
        done_by_filter = list(inputs.get("done_by") or [])
        rt_filter = list(inputs.get("rt_protocol") or [])
        include_dropped = bool(inputs.get("include_dropped_samples"))
        outlier_specs: List[Dict[str, Any]] = list(inputs.get("outliers") or [])

        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        engine_options_template = dict(inputs.get("engine_options") or {})
        self._ensure_outlier_column(ctx.db)
        outlier_report = self._apply_outlier_flags(
            ctx.db,
            outlier_specs,
            target_rg_ids=list(inputs.get("rg_ids") or []),
        )
        if hasattr(ctx.db, "commit"):
            try:
                ctx.db.commit()
            except sqlite3.Error:  # noqa: BLE001
                log.exception("Failed to commit outlier flag updates prior to fitting.")

        if outlier_report["applied"]:
            log.info(
                "Marked %d fmod values as outliers across %d reaction group(s).",
                outlier_report["applied"],
                len(outlier_report["rg_ids"]),
            )
        if outlier_report["missing"]:
            log.warning(
                "Outlier specifications did not match any rows: %s",
                ", ".join(outlier_report["missing"][:10]) + (
                    "..." if len(outlier_report["missing"]) > 10 else ""
                ),
            )

        engine = load_timecourse_engine(engine_name, **dict(engine_options_template))

        arrhenius_target = engine_options_template.get("initialize_kdeg_arrhenius")
        arrhenius_fit = None
        if arrhenius_target not in (None, ""):
            arrhenius_fit = self._fetch_arrhenius_fit(ctx.db, str(arrhenius_target))

        for rg_id in inputs["rg_ids"]:
            global_metadata = dict(inputs.get("global_metadata") or {})
            global_metadata["rg_id"] = rg_id
            global_metadata["valtypes"] = list(valtypes)

            all_series: List[NucleotideSeries] = []
            last_rg_meta: Dict[str, Any] = {}

            for valtype in valtypes:
                try:
                    series, rg_meta = self._load_rg_dataset(
                        ctx.db,
                        rg_id,
                        valtype=valtype,
                        min_points=min_points,
                        nt_filter=nt_filter,
                        preferred_fmod_run=inputs.get("fmod_run_id"),
                        done_by=done_by_filter,
                        rt_protocols=rt_filter,
                        include_dropped=include_dropped,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("Failed to load timecourse data for rg_id=%s valtype=%s: %s", rg_id, valtype, exc)
                    continue
            
                if not series:
                    log.warning("No usable nucleotides found for rg_id=%s valtype=%s; skipping engine run.", rg_id, valtype)
                    continue
                all_series.extend(series)
                last_rg_meta = dict(rg_meta or {})

            if not all_series:
                log.warning("No series loaded for rg_id=%s across valtypes %s; skipping.", rg_id, valtypes)
                continue

            selected_site_bases = sorted(
                {
                    f"{series.metadata.get('site')}_{series.metadata.get('base')}"
                    for series in all_series
                    if series.metadata.get("site") is not None and series.metadata.get("base")
                }
            )
            if selected_site_bases:
                global_metadata["selected_site_bases"] = selected_site_bases

            # Merge rg-specific metadata (temperature, pH, label) into global_metadata
            for key in ("temperature_c", "buffer_pH", "rg_label", "fmod_run_id", "done_by", "rt_protocol"):
                if key in last_rg_meta and key not in global_metadata:
                    global_metadata[key] = last_rg_meta[key]
            if done_by_filter and "done_by" not in global_metadata:
                global_metadata["done_by"] = list(done_by_filter)
            if rt_filter and "rt_protocol" not in global_metadata:
                global_metadata["rt_protocol"] = list(rt_filter)
            if outlier_report["rg_counts"]:
                count_for_rg = outlier_report["rg_counts"].get(rg_id)
                if count_for_rg:
                    global_metadata["outliers_applied"] = count_for_rg
            specs_for_rg = outlier_report["rg_specs"].get(rg_id, [])
            if specs_for_rg:
                global_metadata["outlier_specs"] = specs_for_rg

            options_for_rg = dict(engine_options_template)
            if arrhenius_fit is not None:
                temperature_c = self._resolve_rg_temperature(ctx.db, rg_id, last_rg_meta.get("temperature_c") if last_rg_meta else None)
                log_kdeg_init, kdeg_init = self._arrhenius_initial_kdeg(arrhenius_fit, temperature_c, rg_id)
                options_for_rg["initial_log_kdeg"] = log_kdeg_init
                options_for_rg["initial_kdeg"] = kdeg_init
                global_metadata["initial_log_kdeg"] = log_kdeg_init
                global_metadata["initial_kdeg"] = kdeg_init
                global_metadata["initial_kdeg_source"] = f"arrhenius:{arrhenius_fit['target_label']}"

            request = TimecourseRequest(
                rg_id=rg_id,
                rounds=rounds,
                nucleotides=all_series,
                global_metadata=global_metadata,
                options=options_for_rg,
            )
            result = engine.run(request)
            if result.rounds:
                normalized_requested = {str(r).strip().lower() for r in (inputs.get("rounds") or [])}
                filtered_rounds = [
                    round_result
                    for round_result in result.rounds
                    if str(round_result.round_id).strip().lower() in normalized_requested
                ]
                if len(filtered_rounds) != len(result.rounds):
                    result = TimecourseResult(
                        engine=result.engine,
                        engine_version=result.engine_version,
                        metadata=dict(result.metadata or {}),
                        rounds=tuple(filtered_rounds),
                        artifacts=dict(result.artifacts or {}),
                    )

            result_meta: Dict[str, Any] = dict(result.metadata or {})
            result_meta.setdefault("rg_id", rg_id)
            for meta_key in ("selected_site_bases", "outliers_applied", "outlier_specs", "done_by", "rt_protocol"):
                if meta_key in global_metadata and meta_key not in result_meta:
                    result_meta[meta_key] = global_metadata[meta_key]
            result_meta["valtype"] = valtype
            result_meta.pop("valtype", None)
            result.metadata = result_meta

            self._write_result_artifact(results_dir, rg_id, result)
            if task_id is not None:
                self._persist_result(
                    ctx.db,
                    result,
                    model=engine_name,
                    overwrite=overwrite,
                )

    def _fetch_arrhenius_fit(self, conn, target_label: str) -> Dict[str, Any]:
        rows = conn.execute(
            "SELECT id FROM tempgrad_fit_runs WHERE target_label = ? ORDER BY id DESC",
            (target_label,),
        ).fetchall()
        if not rows:
            raise ValueError(f"No tempgrad_fit_runs found for target_label='{target_label}'.")
        if len(rows) > 1:
            log.info(
                "Multiple tempgrad_fit_runs found for target_label=%s; using the most recent (id=%s).",
                target_label,
                rows[0]["id"],
            )
        fit_run_id = int(rows[0]["id"])

        params_rows = conn.execute(
            "SELECT param_name, param_numeric, param_text FROM tempgrad_fit_params WHERE fit_run_id = ?",
            (fit_run_id,),
        ).fetchall()
        slope = self._extract_numeric_param(params_rows, "slope")
        intercept = self._extract_numeric_param(params_rows, "intercept")
        if slope is None or intercept is None:
            raise ValueError(
                f"Arrhenius fit run {fit_run_id} (target_label='{target_label}') is missing slope or intercept parameters."
            )
        return {
            "fit_run_id": fit_run_id,
            "slope": slope,
            "intercept": intercept,
            "target_label": target_label,
        }

    @staticmethod
    def _extract_numeric_param(rows, param_name: str) -> Optional[float]:
        for row in rows:
            if row["param_name"] != param_name:
                continue
            if row["param_numeric"] is not None:
                return float(row["param_numeric"])
            if row["param_text"] not in (None, ""):
                try:
                    return float(row["param_text"])
                except (TypeError, ValueError):
                    return None
        return None

    def _resolve_rg_temperature(self, conn, rg_id: int, metadata_value: Optional[float]) -> float:
        if metadata_value not in (None, ""):
            return float(metadata_value)

        rows = conn.execute(
            "SELECT DISTINCT temperature FROM probe_reactions WHERE rg_id = ? AND temperature IS NOT NULL",
            (rg_id,),
        ).fetchall()
        if not rows:
            raise ValueError(f"No reaction temperature recorded for rg_id={rg_id}.")

        temps = {round(float(row["temperature"]), 6) for row in rows if row["temperature"] is not None}
        if len(temps) > 1:
            raise ValueError(f"Multiple reaction temperatures found for rg_id={rg_id}: {sorted(temps)}.")
        return float(next(iter(temps)))

    def _arrhenius_initial_kdeg(
        self,
        arrhenius_fit: Mapping[str, Any],
        temperature_c: float,
        rg_id: int,
    ) -> Tuple[float, float]:
        slope = float(arrhenius_fit["slope"])
        intercept = float(arrhenius_fit["intercept"])
        temp_k = float(temperature_c) + 273.15
        if temp_k <= 0:
            raise ValueError(f"Invalid absolute temperature encountered for rg_id={rg_id}: {temperature_c} °C.")
        log_kdeg = slope * (1.0 / temp_k) + intercept
        if not math.isfinite(log_kdeg):
            raise ValueError(
                f"Non-finite log(kdeg) computed for rg_id={rg_id} using Arrhenius parameters (slope={slope}, intercept={intercept})."
            )
        kdeg = math.exp(log_kdeg)
        if not math.isfinite(kdeg) or kdeg <= 0:
            raise ValueError(
                f"Non-positive kdeg computed for rg_id={rg_id} using Arrhenius parameters (slope={slope}, intercept={intercept})."
            )
        return log_kdeg, kdeg

    def resolve_scope(self, ctx: Optional[TaskContext], inputs: Any) -> TaskScope:
        if ctx is None or inputs is None:
            return TaskScope(kind=self.scope_kind or "rg_batch")

        rg_ids: Sequence[int] = list(inputs.get("rg_ids") or []) if isinstance(inputs, dict) else []
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
        done_by: Optional[Iterable[str]] = None,
        rt_protocols: Optional[Iterable[str]] = None,
        include_dropped: bool = False,
    ) -> Tuple[List[NucleotideSeries], Dict[str, Any]]:
        nt_filter_set = {int(x) for x in nt_filter or []}
        if valtype in (None, ""):
            raise ValueError("A valtype must be provided to load timecourse data.")
        normalized_valtype = str(valtype)

        fmod_run_id = self._resolve_fmod_run(conn, rg_id, [normalized_valtype], preferred_fmod_run)

        done_by_values = [
            str(item).strip().lower()
            for item in (done_by or [])
            if item not in (None, "")
        ]
        done_by_values = [value for value in done_by_values if value]

        done_by_clause = ""
        if done_by_values:
            placeholders = ", ".join(f":done_by_{idx}" for idx, _ in enumerate(done_by_values))
            done_by_clause = f" AND LOWER(pr.done_by) IN ({placeholders})"

        rt_values = [
            str(item).strip().lower()
            for item in (rt_protocols or [])
            if item not in (None, "")
        ]
        rt_values = [value for value in rt_values if value]

        rt_clause = ""
        if rt_values:
            placeholders = ", ".join(f":rt_protocol_{idx}" for idx, _ in enumerate(rt_values))
            rt_clause = f" AND LOWER(pr.rt_protocol) IN ({placeholders})"

        sql = f"""
            SELECT
                fv.nt_id,
                fv.fmod_val,
                fv.fmod_run_id,
                fv.valtype AS valtype,
                pr.reaction_time,
                pr.treated,
                pr.temperature,
                pr.done_by,
                pr.rt_protocol,
                fv.outlier,
                mb.pH AS buffer_pH,
                mn.site,
                mn.base
            FROM probe_fmod_values fv
            JOIN probe_reactions pr ON pr.id = fv.rxn_id
            JOIN meta_buffers mb ON mb.id = pr.buffer_id
            JOIN meta_nucleotides mn ON mn.id = fv.nt_id
            JOIN sequencing_samples ss ON ss.id = pr.s_id
            WHERE pr.rg_id = :rg_id
              AND fv.fmod_val IS NOT NULL
              AND fv.valtype = :valtype
              AND (:fmod_run_id IS NULL OR fv.fmod_run_id = :fmod_run_id)
              {"AND ss.to_drop = 0" if not include_dropped else ""}
              AND fv.outlier = 0
              {done_by_clause}
              {rt_clause}
            ORDER BY fv.nt_id, pr.reaction_time, pr.id
        """
        params: Dict[str, Any] = {"rg_id": rg_id, "fmod_run_id": fmod_run_id, "valtype": normalized_valtype}
        for idx, value in enumerate(done_by_values):
            params[f"done_by_{idx}"] = value
        for idx, value in enumerate(rt_values):
            params[f"rt_protocol_{idx}"] = value

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            if done_by_values:
                raise ValueError(f"No timecourse data found for rg_id={rg_id} with done_by={done_by_values}.")
            if rt_values:
                raise ValueError(f"No timecourse data found for rg_id={rg_id} with rt_protocol={rt_values}.")
            raise ValueError(f"No timecourse data found for rg_id={rg_id}.")

        log.debug(
            "Loaded %d probe_fmod rows for rg_id=%s (valtype=%s, fmod_run_id=%s).",
            len(rows),
            rg_id,
            normalized_valtype,
            fmod_run_id,
        )

        series_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
        temperatures: List[float] = []
        pH_values: List[float] = []
        done_by_seen: List[str] = []
        rt_seen: List[str] = []

        for row in rows:
            nt_id = int(row["nt_id"])
            if nt_filter_set and nt_id not in nt_filter_set:
                log.debug("Skipping nt_id=%s for rg_id=%s due to nt filter.", nt_id, rg_id)
                continue

            treated_flag = int(row["treated"])
            time_val = float(row["reaction_time"]) if treated_flag else 0.0
            fmod_val = float(row["fmod_val"])

            # Rows are filtered to a single valtype in the SQL, so we can use the requested valtype
            valtype_row = normalized_valtype
            fmod_run_id = row["fmod_run_id"] if "fmod_run_id" in row.keys() else None

            key = (nt_id, valtype_row)
            entry = series_map.setdefault(
                key,
                {
                    "time": [],
                    "fmod": [],
                    "metadata": {
                        "site": int(row["site"]),
                        "base": str(row["base"]).upper(),
                        "valtype": valtype_row,
                        "fmod_run_ids": set(),
                    },
                },
            )
            entry["time"].append(time_val)
            entry["fmod"].append(fmod_val)
            if fmod_run_id not in (None, ""):
                entry["metadata"]["fmod_run_ids"].add(int(fmod_run_id))

            temp = row["temperature"]
            if temp is not None:
                temperatures.append(float(temp))
            ph_val = row["buffer_pH"]
            if ph_val is not None:
                pH_values.append(float(ph_val))
            done_by_value = row["done_by"]
            if done_by_value not in (None, ""):
                done_by_seen.append(str(done_by_value))
            rt_value = row["rt_protocol"] if "rt_protocol" in row.keys() else None
            if rt_value not in (None, ""):
                rt_seen.append(str(rt_value))

        if not series_map:
            raise ValueError(f"Timecourse data for rg_id={rg_id} did not match nt filter.")
        #print(series_map)
        series_list: List[NucleotideSeries] = []
        for (nt_id, valtype_row), content in series_map.items():
            paired = sorted(zip(content["time"], content["fmod"]), key=lambda x: x[0])
            times_sorted = [p[0] for p in paired]
            fmods_sorted = [p[1] for p in paired]
            if len(times_sorted) < min_points:
                log.debug(
                    "Skipping nt_id=%s valtype=%s for rg_id=%s (points=%d < min_points=%d).",
                    nt_id,
                    valtype_row,
                    rg_id,
                    len(times_sorted),
                    min_points,
                )
                continue
            metadata = dict(content["metadata"])
            run_ids = metadata.pop("fmod_run_ids", set())
            if run_ids:
                sorted_ids = sorted(run_ids)
                metadata["fmod_run_ids"] = sorted_ids
                if len(sorted_ids) == 1:
                    metadata["fmod_run_id"] = sorted_ids[0]
            metadata["valtype"] = valtype_row

            series_list.append(
                NucleotideSeries(
                    nt_id=nt_id,
                    timepoints=times_sorted,
                    fmod_values=fmods_sorted,
                    metadata=metadata,
                )
            )

        if not series_list:
            raise ValueError(f"No nucleotides met inclusion criteria for rg_id={rg_id}.")

        temperature = _unique_or_none(temperatures)
        buffer_pH = _unique_or_none(pH_values)

        metadata = {
            "rg_id": rg_id,
            "fmod_run_id": fmod_run_id,
            "valtype": normalized_valtype,
        }
        if temperature is not None:
            metadata["temperature_c"] = temperature
        if buffer_pH is not None:
            metadata["buffer_pH"] = buffer_pH
        if done_by_seen:
            metadata["done_by"] = sorted({value.strip() for value in done_by_seen if value})
        if rt_seen:
            metadata["rt_protocol"] = sorted({value.strip() for value in rt_seen if value})

        rg_label_row = conn.execute(
            "SELECT rg_label FROM probe_reaction_groups WHERE rg_id = ?",
            (rg_id,),
        ).fetchone()
        if rg_label_row:
            metadata["rg_label"] = rg_label_row["rg_label"]

        return series_list, metadata

    def _ensure_outlier_column(self, conn) -> None:
        try:
            info = conn.execute("PRAGMA table_info(probe_fmod_values)").fetchall()
        except sqlite3.Error:
            log.exception("Failed to inspect probe_fmod_values schema for outlier column.")
            return
        if any(len(row) > 1 and row[1] == "outlier" for row in info):
            return
        try:
            with conn:
                conn.execute("ALTER TABLE probe_fmod_values ADD COLUMN outlier INTEGER NOT NULL DEFAULT 0")
            log.info("Added 'outlier' column to probe_fmod_values table.")
        except sqlite3.Error as exc:
            # Column may already exist if added concurrently; ignore duplicate errors.
            if "duplicate column name" in str(exc).lower():
                log.debug("'outlier' column already present in probe_fmod_values.")
            else:
                log.exception("Failed to add 'outlier' column to probe_fmod_values: %s", exc)

    def _apply_outlier_flags(
        self,
        conn,
        specs: List[Dict[str, Any]],
        *,
        target_rg_ids: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        report = {
            "applied": 0,
            "missing": [],
            "rg_ids": set(),
            "specs": [spec.get("original", "") for spec in specs],
            "rg_specs": {},
            "rg_counts": {},
        }

        reset_rg_ids: Set[int] = set()
        if target_rg_ids:
            reset_rg_ids.update(int(rg_id) for rg_id in target_rg_ids)
        reset_rg_ids.update(int(spec["rg_id"]) for spec in specs if spec.get("rg_id") is not None)
        ordered_reset_rg_ids: Tuple[int, ...] = tuple(sorted(reset_rg_ids))
        if not ordered_reset_rg_ids:
            return report

        try:
            with conn:
                placeholders = ",".join("?" for _ in ordered_reset_rg_ids)
                conn.execute(
                    f"UPDATE probe_fmod_values SET outlier = 0 WHERE rxn_id IN (SELECT id FROM probe_reactions WHERE rg_id IN ({placeholders}))",
                    ordered_reset_rg_ids,
                )

                for spec in specs:
                    rg_id = spec["rg_id"]
                    report["rg_ids"].add(rg_id)
                    cursor = conn.execute(
                        """
                        UPDATE probe_fmod_values
                        SET outlier = 1
                        WHERE id IN (
                            SELECT fv.id
                            FROM probe_fmod_values fv
                            JOIN probe_reactions pr ON pr.id = fv.rxn_id
                            JOIN sequencing_samples ss ON ss.id = pr.s_id
                            JOIN meta_nucleotides mn ON mn.id = fv.nt_id
                            WHERE pr.rg_id = ?
                              AND ss.sample_name = ?
                              AND mn.site = ?
                              AND UPPER(mn.base) = ?
                              AND fv.valtype = ?
                        )
                        """,
                        (
                            rg_id,
                            spec["sample_name"],
                            spec["site"],
                            spec["base"],
                            spec["valtype"],
                        ),
                    )
                    affected = cursor.rowcount if cursor is not None else 0
                    if affected:
                        report["applied"] += affected
                        report["rg_specs"].setdefault(rg_id, []).append(spec.get("original", ""))
                        report["rg_counts"][rg_id] = report["rg_counts"].get(rg_id, 0) + affected
                    else:
                        report["missing"].append(spec.get("original", ""))
        except sqlite3.Error:
            log.exception("Failed to apply outlier flags to probe_fmod_values.")

        report["rg_ids"] = sorted(report["rg_ids"])
        return report

    def _resolve_fmod_run(
        self,
        conn,
        rg_id: int,
        valtypes: Sequence[str],
        preferred: Optional[int],
    ) -> Optional[int]:
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
        metadata_valtype = result.metadata.get("valtype")
        metadata_valtypes = result.metadata.get("valtypes")
        valtype_param = str(metadata_valtype) if metadata_valtype not in (None, "") else None

        for round_result in result.rounds:
            round_id = round_result.round_id
            if overwrite:
                db_api.delete_probe_tc_fit_runs(
                    conn,
                    fit_kind=round_id,
                    rg_id=rg_id,
                    nt_id=None,
                )

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
                    if valtype_param is not None:
                        entries.append({"param_name": "valtype", "param_text": valtype_param})
                    if metadata_valtypes and isinstance(metadata_valtypes, (list, tuple)):
                        entries.append(
                            {
                                "param_name": "valtypes",
                                "param_text": json.dumps(list(metadata_valtypes), sort_keys=True),
                            }
                        )
                    selected_sites = result.metadata.get("selected_site_bases")
                    if selected_sites:
                        entries.append(
                            {
                                "param_name": "selected_site_bases",
                                "param_text": json.dumps(list(selected_sites), sort_keys=True),
                            }
                        )
                    outliers_applied = result.metadata.get("outliers_applied")
                    if outliers_applied not in (None, ""):
                        try:
                            entries.append({"param_name": "outliers_applied", "param_numeric": float(outliers_applied)})
                        except (TypeError, ValueError):
                            entries.append({"param_name": "outliers_applied", "param_text": str(outliers_applied)})
                    outlier_specs = result.metadata.get("outlier_specs")
                    if outlier_specs:
                        entries.append(
                            {
                                "param_name": "outlier_specs",
                                "param_text": json.dumps(list(outlier_specs), sort_keys=True),
                            }
                        )
                    if round_result.notes:
                        entries.append({"param_name": "note", "param_text": str(round_result.notes)})
                    entries.extend(_entries_from_mapping(round_result.global_params))
                    entries.extend(_entries_from_mapping(round_result.qc_metrics, prefix="qc:"))
                    db_api.record_probe_tc_fit_params(conn, fit_run_id=fit_run_id, entries=entries)

            # Per-nucleotide fits
            for per_nt in round_result.per_nt:
                nt_id = per_nt.nt_id
                per_nt_valtype = per_nt.valtype if per_nt.valtype not in (None, "") else None
                
                if overwrite:
                    db_api.delete_probe_tc_fit_runs(
                        conn,
                        fit_kind=round_id,
                        rg_id=rg_id,
                        nt_id=nt_id,
                        valtype=per_nt_valtype,
                    )

                fit_run_id = db_api.begin_probe_tc_fit_run(
                    conn,
                    fit_kind=round_id,
                    rg_id=rg_id,
                    nt_id=nt_id,
                    model=model,
                    valtype=per_nt_valtype,
                )
                if fit_run_id is None:
                    continue
                entries = [
                    {"param_name": "status", "param_text": round_result.status},
                    {"param_name": "engine", "param_text": result.engine},
                    {"param_name": "engine_version", "param_text": result.engine_version},
                ]
                # Note: valtype is already inserted by begin_probe_tc_fit_run if provided
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
