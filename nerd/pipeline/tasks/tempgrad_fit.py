"""
Task for temperature-gradient fitting (Arrhenius and melt models).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import sqlite3

from nerd.db import api as db_api
from nerd.pipeline.plugins.tempgrad import (
    TempgradRequest,
    TempgradSeries,
    load_tempgrad_engine,
)
from .base import Task, TaskContext, TaskScope
from nerd.utils.logging import get_logger

log = get_logger(__name__)


class TempgradFitTask(Task):
    """
    Execute Arrhenius or two-state melt fits using pluggable engines.
    """

    name = "tempgrad_fit"
    scope_kind = "global"

    _DEFAULT_ENGINE = {
        "arrhenius": "arrhenius_python",
        "two_state_melt": "two_state_melt",
    }

    def prepare(self, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.name not in cfg:
            raise ValueError(f"Configuration must contain a '{self.name}' section.")

        block = dict(cfg[self.name])
        mode = str(block.get("mode") or "").strip().lower()
        if mode not in {"arrhenius", "two_state_melt"}:
            raise ValueError("tempgrad_fit.mode must be one of: 'arrhenius', 'two_state_melt'.")
        block["mode"] = mode

        engine = str(block.get("engine") or "").strip().lower()
        if not engine:
            engine = self._DEFAULT_ENGINE[mode]
        block["engine"] = engine

        block["series"] = list(block.get("series") or [])
        block["filters"] = dict(block.get("filters") or {})
        block["engine_options"] = dict(block.get("engine_options") or {})
        block["metadata"] = dict(block.get("metadata") or {})
        block["overwrite"] = bool(block.get("overwrite", False))
        if block["overwrite"]:
            block["force_run"] = True
        else:
            block["force_run"] = bool(block.get("force_run", False))

        # Allow a simple boolean toggle to select probe timecourse vs. NMR sources.
        use_probe_tc = block.get("use_probe_tc")
        if use_probe_tc is not None:
            if isinstance(use_probe_tc, str):
                norm_flag = use_probe_tc.strip().lower()
                flag_value = norm_flag in {"1", "true", "yes", "on"}
            else:
                flag_value = bool(use_probe_tc)
            block["data_source"] = "probe_tc" if flag_value else "nmr"
        else:
            block["data_source"] = str(
                block.get("data_source") or block["filters"].get("data_source") or "manual"
            )

        model_value = block.get("model")
        if model_value is None:
            model_value = block["engine_options"].get("model")
        model_str = str(model_value or "").strip().lower()
        block["model"] = model_str or None
        if block["model"]:
            block["engine_options"]["model"] = block["model"]
            block["metadata"].setdefault("model", block["model"])

        fit_name = block.get("fit_name")
        if fit_name is not None:
            fit_name_str = str(fit_name).strip()
            block["fit_name"] = fit_name_str or None
        else:
            block["fit_name"] = None
        if block["fit_name"]:
            block["metadata"].setdefault("fit_name", block["fit_name"])

        group_by_cfg = block.get("group_by")
        if not group_by_cfg and "group_by" in block["engine_options"]:
            group_by_cfg = block["engine_options"].pop("group_by")
        block["group_by"] = self._normalize_group_by(group_by_cfg)

        outliers_cfg = block.get("outliers")
        if outliers_cfg is None:
            block["outliers"] = []
        elif isinstance(outliers_cfg, (list, tuple, set)):
            block["outliers"] = list(outliers_cfg)
        else:
            block["outliers"] = [outliers_cfg]

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
        series_list = self._build_series(ctx, inputs)
        if not series_list:
            raise ValueError("No input series available for tempgrad fitting.")

        engine = load_tempgrad_engine(inputs["engine"], **dict(inputs.get("engine_options") or {}))

        request = TempgradRequest(
            mode=inputs["mode"],
            series=series_list,
            options=dict(inputs.get("engine_options") or {}),
            metadata=dict(inputs.get("metadata") or {}),
        )

        result = engine.run(request)

        self._write_artifact(run_dir, result)

        overwrite = bool(inputs.get("overwrite", False))
        for series, series_result in zip(series_list, result.series_results):
            scope_kind = str(series.metadata.get("scope_kind") or inputs.get("scope_kind") or self.scope_kind or "global")
            scope_id = self._safe_int(series.metadata.get("scope_id") or inputs.get("scope_id"))
            tg_id = self._safe_int(series.metadata.get("tg_id"))
            nt_id = self._safe_int(series.metadata.get("nt_id"))

            if overwrite:
                db_api.delete_tempgrad_fit_runs(
                    ctx.db,
                    fit_kind=request.mode,
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                    tg_id=tg_id,
                    nt_id=nt_id,
                )

            fit_run_id = db_api.begin_tempgrad_fit_run(
                ctx.db,
                fit_kind=request.mode,
                task_id=task_id,
                scope_kind=scope_kind,
                scope_id=scope_id,
                data_source=str(inputs.get("data_source")),
                target_label=series.series_id,
                rg_id=self._safe_int(series.metadata.get("rg_id")),
                tg_id=tg_id,
                nt_id=nt_id,
            )
            if fit_run_id is None:
                continue

            entries = []
            entries.extend(self._entries_from_mapping(series_result.params))
            entries.extend(self._entries_from_mapping(series_result.diagnostics, prefix="diag:"))
            entries.extend(self._entries_from_mapping(series.metadata or {}, prefix="meta:"))
            entries.extend(self._entries_from_mapping(result.metadata or {}, prefix="request:"))
            db_api.record_tempgrad_fit_params(ctx.db, fit_run_id=fit_run_id, entries=entries)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_series(self, ctx: TaskContext, inputs: Dict[str, Any]) -> List[TempgradSeries]:
        mode = inputs["mode"]
        series_cfg = inputs.get("series") or []

        series_list: List[TempgradSeries] = []
        series_list.extend(self._series_from_config(series_cfg, mode, inputs))

        if not series_list:
            if mode == "arrhenius":
                data_source = str(inputs.get("data_source") or "nmr").lower()
                if data_source == "probe_tc":
                    series_list.extend(self._series_from_db_probe_tc(ctx, inputs))
                else:
                    series_list.extend(self._series_from_db_arrhenius(ctx, inputs))
            elif mode == "two_state_melt":
                series_list.extend(self._series_from_db_melt(ctx, inputs))

        return series_list

    def _series_from_config(
        self,
        series_cfg: Sequence[Mapping[str, Any]],
        mode: str,
        inputs: Mapping[str, Any],
    ) -> List[TempgradSeries]:
        series_list: List[TempgradSeries] = []
        for idx, entry in enumerate(series_cfg):
            if not isinstance(entry, Mapping):
                continue
            series_id = str(entry.get("series_id") or entry.get("label") or f"series_{idx}")
            temperatures = entry.get("temperatures") or entry.get("temperature_c") or entry.get("temp_c")
            responses = entry.get("rates") or entry.get("k_values") or entry.get("responses")
            if temperatures is None or responses is None:
                log.warning("Series config '%s' is missing temperatures or responses; skipping.", series_id)
                continue
            temp_vals = [float(v) for v in temperatures]
            resp_vals = [float(v) for v in responses]
            if len(temp_vals) != len(resp_vals):
                log.warning("Series '%s' has mismatched temperature/response lengths; skipping.", series_id)
                continue
            weights = None
            errors_key = "rate_errors" if mode == "arrhenius" else "response_errors"
            errors = entry.get(errors_key) or entry.get("errors")
            if errors:
                try:
                    errs = np.asarray(errors, dtype=float)
                    if errs.shape == (len(temp_vals),):
                        with np.errstate(divide="ignore"):
                            if mode == "arrhenius":
                                # Convert to ln(k) uncertainty: sigma_y = error / rate
                                sigma_y = errs / np.asarray(resp_vals, dtype=float)
                                weights = np.where(sigma_y > 0, 1.0 / sigma_y, 0.0)
                            else:
                                weights = np.where(errs > 0, 1.0 / errs, 0.0)
                    else:
                        log.warning("Error array for series '%s' has unexpected shape; ignoring.", series_id)
                except Exception:
                    log.warning("Failed to parse errors for series '%s'; ignoring.", series_id)
            metadata = dict(entry.get("metadata") or {})
            metadata.setdefault("temperature_unit", entry.get("temperature_unit") or inputs.get("temperature_unit") or "c")
            for key in ("scope_kind", "scope_id", "tg_id", "nt_id", "rg_id"):
                if key not in metadata and key in entry:
                    metadata[key] = entry[key]
            series_list.append(
                TempgradSeries(
                    series_id=series_id,
                    x_values=temp_vals,
                    y_values=resp_vals,
                    weights=weights,
                    metadata=metadata,
                )
            )
        return series_list

    def _series_from_db_arrhenius(self, ctx: TaskContext, inputs: Dict[str, Any]) -> List[TempgradSeries]:
        filters = dict(inputs.get("filters") or {})
        data_source = str(inputs.get("data_source") or filters.get("data_source") or "nmr").lower()
        if data_source != "nmr":
            log.info("Arrhenius data_source '%s' not supported yet; falling back to manual series.", data_source)
            return []

        reaction_type = filters.get("reaction_type")
        substrate = filters.get("substrate") or filters.get("species")
        buffer_name = filters.get("buffer") or filters.get("buffer_name")
        plugin = filters.get("plugin")
        model = filters.get("model")

        buffer_id = None
        buffer_disp = None
        if buffer_name:
            row = ctx.db.execute(
                """
                SELECT id, disp_name FROM meta_buffers
                WHERE name = ? OR disp_name = ?
                ORDER BY id LIMIT 1
                """,
                (buffer_name, buffer_name),
            ).fetchone()
            if row:
                buffer_id = int(row["id"] if hasattr(row, "keys") else row[0])
                buffer_disp = str(row["disp_name"] if hasattr(row, "keys") else row[1])
            else:
                log.warning("Buffer '%s' not found; continuing without buffer filter.", buffer_name)

        params = []
        clauses = ["fr.status = 'completed'"]
        if reaction_type:
            clauses.append("nr.reaction_type = ?")
            params.append(str(reaction_type))
        if substrate:
            clauses.append("(fr.species = ? OR nr.substrate = ?)")
            params.extend([str(substrate), str(substrate)])
        if buffer_id is not None:
            clauses.append("nr.buffer_id = ?")
            params.append(buffer_id)
        if plugin:
            clauses.append("fr.plugin = ?")
            params.append(str(plugin))
        if model:
            clauses.append("fr.model = ?")
            params.append(str(model))

        sql = f"""
            SELECT
                nr.id AS reaction_id,
                nr.temperature,
                nr.buffer_id,
                fr.id AS fit_run_id,
                fr.plugin,
                fr.model,
                fr.species,
                kv.param_numeric AS k_value,
                ke.param_numeric AS k_error
            FROM nmr_fit_runs fr
            JOIN nmr_reactions nr ON nr.id = fr.nmr_reaction_id
            JOIN nmr_fit_params kv ON kv.fit_run_id = fr.id AND kv.param_name = 'k_value'
            LEFT JOIN nmr_fit_params ke ON ke.fit_run_id = fr.id AND ke.param_name = 'k_error'
            WHERE { ' AND '.join(clauses) }
            ORDER BY nr.temperature, fr.id
        """

        rows = ctx.db.execute(sql, params).fetchall()
        if not rows:
            log.warning("No NMR fits matched filters %s; manual series required.", filters)
            return []

        temperatures: List[float] = []
        rates: List[float] = []
        weights: List[float] = []
        for row in rows:
            if hasattr(row, "keys"):
                temp = float(row["temperature"])
                k_value = float(row["k_value"])
                k_error = row["k_error"]
                spec = row["species"]
            else:
                temp = float(row[1])
                k_value = float(row[7])
                k_error = row[8]
                spec = row[6]
            if k_value <= 0:
                continue
            temperatures.append(temp)
            rates.append(k_value)
            if k_error is not None and k_error > 0 and k_value > 0:
                sigma_y = float(k_error) / k_value
                weights.append(1.0 / sigma_y if sigma_y > 0 else 0.0)
            else:
                weights.append(math.nan)

        if not temperatures:
            return []

        weights_array = None
        if any(math.isfinite(w) and w > 0 for w in weights):
            weights_array = [w if math.isfinite(w) and w > 0 else 0.0 for w in weights]

        metadata: Dict[str, Any] = {
            "temperature_unit": "c",
            "data_source": "nmr",
            "reaction_type": reaction_type,
            "substrate": substrate,
            "plugin": plugin,
            "model": model,
        }
        if buffer_id is not None:
            metadata["buffer_id"] = buffer_id
            metadata["buffer_name"] = buffer_name
            metadata["buffer_disp_name"] = buffer_disp
        if substrate:
            metadata.setdefault("species", substrate)

        series_id_parts = ["nmr"]
        if reaction_type:
            series_id_parts.append(reaction_type)
        if substrate:
            series_id_parts.append(str(substrate))
        if buffer_name:
            series_id_parts.append(str(buffer_name))
        series_id = ":".join(part for part in series_id_parts if part)

        return [
            TempgradSeries(
                series_id=series_id or "nmr_arrhenius",
                x_values=temperatures,
                y_values=rates,
                weights=weights_array,
                metadata=metadata,
            )
        ]

    def _series_from_db_melt(self, ctx: TaskContext, inputs: Dict[str, Any]) -> List[TempgradSeries]:
        filters = dict(inputs.get("filters") or {})
        options = dict(inputs.get("engine_options") or {})
        group_fields = tuple(self._normalize_group_by(inputs.get("group_by")))
        if not group_fields:
            group_fields = tuple(self._normalize_group_by(options.get("group_by")))

        model = str(inputs.get("model") or options.get("model") or "free_kadd").strip().lower()

        outlier_rg_ids, outlier_nt_ids, outlier_pairs = self._parse_outliers(
            inputs.get("outliers") or options.get("outliers")
        )
        outlier_summary = self._summarize_outliers(outlier_rg_ids, outlier_nt_ids, outlier_pairs)
        fit_name = str(inputs.get("fit_name") or "").strip() or None

        fit_kind = str(filters.get("fit_kind") or "round3_constrained").strip()

        min_points = int(filters.get("min_points") or options.get("min_points") or 3)

        construct_ids = None
        if "construct_id" in filters:
            construct_ids = self._coerce_int_list(filters.get("construct_id"))
        elif "construct" in filters:
            construct_ids = self._resolve_construct_ids(ctx.db, filters["construct"])

        buffer_ids = None
        if "buffer_id" in filters:
            buffer_ids = self._coerce_int_list(filters.get("buffer_id"))
        elif "buffer" in filters:
            buffer_ids = self._resolve_buffer_ids(ctx.db, filters["buffer"])

        nt_ids = self._coerce_int_list(filters.get("nt_id")) if "nt_id" in filters else None
        bases = None
        if "base" in filters:
            value = filters["base"]
            if isinstance(value, (list, tuple, set)):
                bases = [str(v).upper() for v in value if v not in (None, "")]
            else:
                bases = [str(value).upper()]

        probe_filter = filters.get("probe")
        probe_conc_filter = filters.get("probe_conc") or filters.get("probe_concentration")
        rt_protocol_filter = filters.get("rt_protocol")
        rg_label_filter = filters.get("rg_label") or filters.get("reaction_group")
        valtype_filter = filters.get("valtype")

        temperature_min = filters.get("temperature_min") or filters.get("temperature_ge")
        temperature_max = filters.get("temperature_max") or filters.get("temperature_le")

        where: List[str] = ["r.fit_kind = ?"]
        params: List[Any] = [fit_kind]

        if construct_ids:
            placeholders = ",".join("?" for _ in construct_ids)
            where.append(f"prd.construct_id IN ({placeholders})")
            params.extend(construct_ids)
        if buffer_ids:
            placeholders = ",".join("?" for _ in buffer_ids)
            where.append(f"prd.buffer_id IN ({placeholders})")
            params.extend(buffer_ids)
        if nt_ids:
            placeholders = ",".join("?" for _ in nt_ids)
            where.append(f"r.nt_id IN ({placeholders})")
            params.extend(nt_ids)
        if bases:
            placeholders = ",".join("?" for _ in bases)
            where.append(f"UPPER(mn.base) IN ({placeholders})")
            params.extend(bases)
        if probe_filter:
            values = probe_filter if isinstance(probe_filter, (list, tuple, set)) else [probe_filter]
            values = [str(v).strip() for v in values if v not in (None, "")]
            if values:
                placeholders = ",".join("?" for _ in values)
                where.append(f"prd.probe IN ({placeholders})")
                params.extend(values)
        if probe_conc_filter not in (None, ""):
            try:
                conc_val = float(probe_conc_filter)
                where.append("prd.probe_conc = ?")
                params.append(conc_val)
            except (TypeError, ValueError):
                log.warning("Invalid probe_concentration filter '%s'; ignoring.", probe_conc_filter)
        if rt_protocol_filter:
            values = rt_protocol_filter if isinstance(rt_protocol_filter, (list, tuple, set)) else [rt_protocol_filter]
            values = [str(v).strip() for v in values if v not in (None, "")]
            if values:
                placeholders = ",".join("?" for _ in values)
                where.append(f"prd.rt_protocol IN ({placeholders})")
                params.extend(values)
        if rg_label_filter:
            values = rg_label_filter if isinstance(rg_label_filter, (list, tuple, set)) else [rg_label_filter]
            values = [str(v).strip() for v in values if v not in (None, "")]
            if values:
                placeholders = ",".join("?" for _ in values)
                where.append(f"rg.rg_label IN ({placeholders})")
                params.extend(values)
        if valtype_filter not in (None, ""):
            values = valtype_filter if isinstance(valtype_filter, (list, tuple, set)) else [valtype_filter]
            values = [str(v).strip() for v in values if v not in (None, "")]
            if values:
                placeholders = ",".join("?" for _ in values)
                where.append(f"r.valtype IN ({placeholders})")
                params.extend(values)
        if temperature_min not in (None, ""):
            try:
                tmin = float(temperature_min)
                where.append("prd.temperature >= ?")
                params.append(tmin)
            except (TypeError, ValueError):
                log.warning("Invalid temperature_min filter '%s'; ignoring.", temperature_min)
        if temperature_max not in (None, ""):
            try:
                tmax = float(temperature_max)
                where.append("prd.temperature <= ?")
                params.append(tmax)
            except (TypeError, ValueError):
                log.warning("Invalid temperature_max filter '%s'; ignoring.", temperature_max)

        sql = """
            SELECT
                r.id AS fit_run_id,
                r.rg_id,
                r.nt_id,
                r.valtype,
                rg.rg_label,
                prd.temperature,
                prd.probe,
                prd.probe_conc,
                prd.rt_protocol,
                prd.buffer_id,
                prd.construct_id,
                mc.name AS construct_name,
                mc.disp_name AS construct_disp_name,
                mc.version AS construct_version,
                mb.name AS buffer_name,
                mb.disp_name AS buffer_disp_name,
                mn.site AS nt_site,
                mn.base AS nt_base,
                k.param_numeric AS kobs,
                lk.param_numeric AS log_kobs,
                le.param_numeric AS log_kobs_err,
                kd.param_numeric AS log_kdeg
            FROM probe_tc_fit_runs r
            JOIN probe_tc_fit_params k ON k.fit_run_id = r.id AND k.param_name = 'kobs'
            JOIN probe_tc_fit_params lk ON lk.fit_run_id = r.id AND lk.param_name = 'log_kobs'
            LEFT JOIN probe_tc_fit_params le ON le.fit_run_id = r.id AND le.param_name = 'log_kobs_err'
            LEFT JOIN probe_tc_fit_params kd ON kd.fit_run_id = r.id AND kd.param_name = 'log_kdeg'
            JOIN (
                SELECT
                    rg_id,
                    MAX(temperature) AS temperature,
                    MAX(probe) AS probe,
                    MAX(probe_concentration) AS probe_conc,
                    MAX(rt_protocol) AS rt_protocol,
                    MAX(buffer_id) AS buffer_id,
                    MAX(construct_id) AS construct_id
                FROM probe_reactions
                WHERE treated = 1
                GROUP BY rg_id
            ) prd ON prd.rg_id = r.rg_id
            JOIN probe_reaction_groups rg ON rg.rg_id = r.rg_id
            LEFT JOIN meta_constructs mc ON mc.id = prd.construct_id
            LEFT JOIN meta_buffers mb ON mb.id = prd.buffer_id
            LEFT JOIN meta_nucleotides mn ON mn.id = r.nt_id
        """
        if where:
            sql += " WHERE " + " AND ".join(where)

        rows = ctx.db.execute(sql, params).fetchall()
        if not rows:
            log.info("No probe timecourse fits matched the requested filters.")
            return []

        series_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        for row in rows:
            rg_id_val = self._safe_int(row["rg_id"])
            nt_id_val = self._safe_int(row["nt_id"])
            if (rg_id_val is not None and rg_id_val in outlier_rg_ids) or (
                nt_id_val is not None and nt_id_val in outlier_nt_ids
            ) or (
                rg_id_val is not None
                and nt_id_val is not None
                and (rg_id_val, nt_id_val) in outlier_pairs
            ):
                continue

            log_kobs = row["log_kobs"]
            if log_kobs in (None, ""):
                continue
            try:
                log_kobs = float(log_kobs)
            except (TypeError, ValueError):
                continue

            temperature = row["temperature"]
            if temperature in (None, ""):
                continue
            try:
                temperature = float(temperature)
            except (TypeError, ValueError):
                continue

            log_err = row["log_kobs_err"]
            if log_err in (None, ""):
                log_err = None
            else:
                try:
                    log_err = float(log_err)
                except (TypeError, ValueError):
                    log_err = None

            group_components = []
            row_keys = set(row.keys()) if hasattr(row, "keys") else set()
            for field in group_fields:
                name = str(field).strip().lower()
                if name == "construct":
                    group_components.append(row["construct_disp_name"] or row["construct_name"] or row["construct_id"])
                elif name == "buffer":
                    group_components.append(row["buffer_disp_name"] or row["buffer_name"] or row["buffer_id"])
                elif name == "base":
                    group_components.append(row["nt_base"])
                elif name == "probe":
                    group_components.append(row["probe"])
                elif name == "valtype":
                    group_components.append(row["valtype"])
                else:
                    value = row[field] if field in row_keys else None
                    group_components.append(value)
            group_key = tuple(group_components) if group_components else tuple()
            group_label = "|".join([str(comp) for comp in group_components if comp not in (None, "")]) or "default"

            series_key = group_key + (row["nt_id"],)

            entry = series_map.setdefault(
                series_key,
                {
                    "construct_id": row["construct_id"],
                    "construct_name": row["construct_name"],
                    "construct_disp": row["construct_disp_name"],
                    "construct_version": row["construct_version"],
                    "buffer_id": row["buffer_id"],
                    "buffer_name": row["buffer_name"],
                    "buffer_disp": row["buffer_disp_name"],
                    "probe": row["probe"],
                    "probe_conc": row["probe_conc"],
                    "rt_protocol": row["rt_protocol"],
                    "nt_id": row["nt_id"],
                    "nt_site": row["nt_site"],
                    "nt_base": row["nt_base"],
                    "valtype": row["valtype"],
                    "temps": [],
                    "log_kobs": [],
                    "log_errs": [],
                    "log_kdeg_values": [],
                    "kobs_values": [],
                    "point_rg_ids": [],
                    "point_nt_ids": [],
                    "rg_ids": set(),
                    "group_key": group_key,
                    "group_label": group_label,
                },
            )

            entry["temps"].append(temperature)
            entry["log_kobs"].append(log_kobs)
            entry["log_errs"].append(log_err)
            log_kdeg = row["log_kdeg"]
            if log_kdeg in (None, ""):
                log_kdeg_val = None
            else:
                try:
                    log_kdeg_val = float(log_kdeg)
                except (TypeError, ValueError):
                    log_kdeg_val = None
            kobs_val = row["kobs"]
            if kobs_val in (None, ""):
                kobs_numeric = math.exp(log_kobs)
            else:
                try:
                    kobs_numeric = float(kobs_val)
                except (TypeError, ValueError):
                    kobs_numeric = math.exp(log_kobs)
            entry["log_kdeg_values"].append(log_kdeg_val)
            entry["kobs_values"].append(kobs_numeric)
            entry["point_rg_ids"].append(rg_id_val if rg_id_val is not None else row["rg_id"])
            entry["point_nt_ids"].append(nt_id_val if nt_id_val is not None else row["nt_id"])
            if rg_id_val is not None:
                entry["rg_ids"].add(rg_id_val)
            elif row["rg_id"] not in (None, ""):
                entry["rg_ids"].add(row["rg_id"])

        series_list: List[TempgradSeries] = []

        for entry in series_map.values():
            temps = entry["temps"]
            log_values = entry["log_kobs"]
            if len(temps) < min_points:
                continue

            combined = sorted(
                zip(
                    temps,
                    log_values,
                    entry["log_errs"],
                    entry["log_kdeg_values"],
                    entry["kobs_values"],
                    entry["point_rg_ids"],
                    entry["point_nt_ids"],
                ),
                key=lambda x: x[0],
                reverse=True,
            )
            sorted_temps: List[float] = []
            sorted_log: List[float] = []
            sorted_point_records: List[Dict[str, Any]] = []
            for temp, log_val, log_err_val, log_kdeg_val, kobs_val, rg_point, nt_point in combined:
                temp_f = float(temp)
                log_f = float(log_val)
                sorted_temps.append(temp_f)
                sorted_log.append(log_f)
                if log_err_val is None:
                    err_val = None
                else:
                    try:
                        err_val = float(log_err_val)
                    except (TypeError, ValueError):
                        err_val = None
                if log_kdeg_val is None:
                    log_kdeg_out = None
                else:
                    log_kdeg_out = float(log_kdeg_val)
                log_kobs2_val = log_f
                if log_kdeg_out is not None and math.isfinite(log_kdeg_out):
                    log_kobs2_val = log_f + log_kdeg_out
                if kobs_val is None or not math.isfinite(kobs_val) or kobs_val <= 0:
                    kobs2_val = math.exp(log_kobs2_val)
                else:
                    kobs2_val = float(kobs_val)
                sorted_point_records.append(
                    {
                        "temperature_c": temp_f,
                        "kobs2": kobs2_val,
                        "log_kobs2": float(log_kobs2_val),
                        "log_kobs2_err": err_val,
                        "log_kdeg": log_kdeg_out,
                        "rg_id": rg_point,
                        "nt_id": nt_point,
                    }
                )

            construct_label = entry["construct_disp"] or entry["construct_name"] or entry["construct_id"]
            nt_id = entry["nt_id"]
            series_id = f"{construct_label}|nt{nt_id}"

            unique_temps: List[float] = []
            for temp in sorted_temps:
                if not unique_temps or temp != unique_temps[-1]:
                    unique_temps.append(temp)

            metadata: Dict[str, Any] = {
                "source": "probe_tc",
                "construct_id": entry["construct_id"],
                "construct_name": entry["construct_name"],
                "construct_disp_name": entry["construct_disp"],
                "construct_version": entry["construct_version"],
                "buffer_id": entry["buffer_id"],
                "buffer_name": entry["buffer_name"],
                "buffer_disp_name": entry["buffer_disp"],
                "probe": entry["probe"],
                "probe_conc": entry["probe_conc"],
                "rt_protocol": entry["rt_protocol"],
                "valtype": entry["valtype"],
                "nt_id": nt_id,
                "nt_site": entry["nt_site"],
                "nt_base": entry["nt_base"],
                "rg_ids": sorted(entry["rg_ids"]),
                "temperatures_used_c": unique_temps,
                "src_kobs_data": sorted_point_records,
                "group_key": list(entry["group_key"]),
                "group_label": entry["group_label"],
                "model": model,
            }
            if outlier_summary:
                metadata["outliers_removed"] = outlier_summary
            if fit_name:
                metadata["fit_seed_ab"] = None

            series_list.append(
                TempgradSeries(
                    series_id=series_id,
                    x_values=sorted_temps,
                    y_values=sorted_log,
                    weights=None,
                    metadata=metadata,
                )
            )

        return series_list

    def _series_from_db_probe_tc(self, ctx: TaskContext, inputs: Dict[str, Any]) -> List[TempgradSeries]:
        filters = dict(inputs.get("filters") or {})
        options = dict(inputs.get("engine_options") or {})
        group_fields = tuple(self._normalize_group_by(inputs.get("group_by")))
        if not group_fields:
            group_fields = tuple(self._normalize_group_by(options.get("group_by")))
        outlier_rg_ids, outlier_nt_ids, outlier_pairs = self._parse_outliers(
            inputs.get("outliers") or options.get("outliers")
        )
        outlier_summary = self._summarize_outliers(outlier_rg_ids, outlier_nt_ids, outlier_pairs)
        fit_name = str(inputs.get("fit_name") or "").strip() or None

        fit_kind = str(filters.get("fit_kind") or "round3_constrained").strip()
        weighted = bool(options.get("weighted", False))

        threshold = (
            filters.get("melt_threshold_c")
            or options.get("melt_threshold_c")
            or options.get("temperature_threshold_c")
        )
        threshold_c: Optional[float] = None
        if threshold not in (None, ""):
            try:
                threshold_c = float(threshold)
            except (TypeError, ValueError):
                log.warning("Invalid melt_threshold_c value '%s'; ignoring.", threshold)

        min_points = int(filters.get("min_points") or options.get("min_points") or 2)

        construct_ids = None
        if "construct_id" in filters:
            construct_ids = self._coerce_int_list(filters.get("construct_id"))
        elif "construct" in filters:
            construct_ids = self._resolve_construct_ids(ctx.db, filters["construct"])

        buffer_ids = None
        if "buffer_id" in filters:
            buffer_ids = self._coerce_int_list(filters.get("buffer_id"))
        elif "buffer" in filters:
            buffer_ids = self._resolve_buffer_ids(ctx.db, filters["buffer"])

        nt_ids = self._coerce_int_list(filters.get("nt_id")) if "nt_id" in filters else None
        bases = None
        if "base" in filters:
            value = filters["base"]
            if isinstance(value, (list, tuple, set)):
                bases = [str(v).upper() for v in value if v not in (None, "")]
            else:
                bases = [str(value).upper()]

        probe_filter = filters.get("probe")
        probe_conc_filter = filters.get("probe_conc") or filters.get("probe_concentration")
        rt_protocol_filter = filters.get("rt_protocol")
        rg_label_filter = filters.get("rg_label") or filters.get("reaction_group")
        valtype_filter = filters.get("valtype")
        if valtype_filter in (None, ""):
            valtype_values = ["modrate"]
        else:
            if isinstance(valtype_filter, (list, tuple, set)):
                valtype_values = [str(v).strip() for v in valtype_filter if v not in (None, "")]
            else:
                valtype_values = [str(valtype_filter).strip()]
            if not valtype_values:
                valtype_values = ["modrate"]

        temperature_min = filters.get("temperature_min") or filters.get("temperature_ge")
        temperature_max = filters.get("temperature_max") or filters.get("temperature_le")

        where: List[str] = ["r.fit_kind = ?"]
        params: List[Any] = [fit_kind]

        if construct_ids:
            placeholders = ",".join("?" for _ in construct_ids)
            where.append(f"prd.construct_id IN ({placeholders})")
            params.extend(construct_ids)
        if buffer_ids:
            placeholders = ",".join("?" for _ in buffer_ids)
            where.append(f"prd.buffer_id IN ({placeholders})")
            params.extend(buffer_ids)
        if valtype_values:
            placeholders = ",".join("?" for _ in valtype_values)
            where.append(f"r.valtype IN ({placeholders})")
            params.extend(valtype_values)
        if nt_ids:
            placeholders = ",".join("?" for _ in nt_ids)
            where.append(f"r.nt_id IN ({placeholders})")
            params.extend(nt_ids)
        if bases:
            placeholders = ",".join("?" for _ in bases)
            where.append(f"UPPER(mn.base) IN ({placeholders})")
            params.extend(bases)
        if probe_filter:
            values = probe_filter if isinstance(probe_filter, (list, tuple, set)) else [probe_filter]
            values = [str(v).strip() for v in values if v not in (None, "")]
            if values:
                placeholders = ",".join("?" for _ in values)
                where.append(f"prd.probe IN ({placeholders})")
                params.extend(values)
        if probe_conc_filter not in (None, ""):
            try:
                conc_val = float(probe_conc_filter)
                where.append("prd.probe_conc = ?")
                params.append(conc_val)
            except (TypeError, ValueError):
                log.warning("Invalid probe_concentration filter '%s'; ignoring.", probe_conc_filter)
        if rt_protocol_filter:
            values = rt_protocol_filter if isinstance(rt_protocol_filter, (list, tuple, set)) else [rt_protocol_filter]
            values = [str(v).strip() for v in values if v not in (None, "")]
            if values:
                placeholders = ",".join("?" for _ in values)
                where.append(f"prd.rt_protocol IN ({placeholders})")
                params.extend(values)
        if rg_label_filter:
            values = rg_label_filter if isinstance(rg_label_filter, (list, tuple, set)) else [rg_label_filter]
            values = [str(v).strip() for v in values if v not in (None, "")]
            if values:
                placeholders = ",".join("?" for _ in values)
                where.append(f"rg.rg_label IN ({placeholders})")
                params.extend(values)
        if temperature_min not in (None, ""):
            try:
                tmin = float(temperature_min)
                where.append("prd.temperature >= ?")
                params.append(tmin)
            except (TypeError, ValueError):
                log.warning("Invalid temperature_min filter '%s'; ignoring.", temperature_min)
        if temperature_max not in (None, ""):
            try:
                tmax = float(temperature_max)
                where.append("prd.temperature <= ?")
                params.append(tmax)
            except (TypeError, ValueError):
                log.warning("Invalid temperature_max filter '%s'; ignoring.", temperature_max)

        sql = """
            SELECT
                r.id AS fit_run_id,
                r.rg_id,
                r.nt_id,
                r.valtype,
                rg.rg_label,
                prd.temperature,
                prd.replicate,
                prd.done_by,
                prd.probe,
                prd.probe_conc,
                prd.rt_protocol,
                prd.buffer_id,
                prd.construct_id,
                mc.name AS construct_name,
                mc.disp_name AS construct_disp_name,
                mc.version AS construct_version,
                mb.name AS buffer_name,
                mb.disp_name AS buffer_disp_name,
                mn.site AS nt_site,
                mn.base AS nt_base,
                k.param_numeric AS kobs,
                le.param_numeric AS log_kobs_err,
                lk.param_numeric AS log_kobs,
                kd.param_numeric AS log_kdeg
            FROM probe_tc_fit_runs r
            JOIN probe_tc_fit_params k ON k.fit_run_id = r.id AND k.param_name = 'kobs'
            LEFT JOIN probe_tc_fit_params le ON le.fit_run_id = r.id AND le.param_name = 'log_kobs_err'
            LEFT JOIN probe_tc_fit_params lk ON lk.fit_run_id = r.id AND lk.param_name = 'log_kobs'
            LEFT JOIN probe_tc_fit_params kd ON kd.fit_run_id = r.id AND kd.param_name = 'log_kdeg'
            JOIN (
                SELECT
                    rg_id,
                    MAX(temperature) AS temperature,
                    MAX(replicate) AS replicate,
                    MAX(done_by) AS done_by,
                    MAX(probe) AS probe,
                    MAX(probe_concentration) AS probe_conc,
                    MAX(rt_protocol) AS rt_protocol,
                    MAX(buffer_id) AS buffer_id,
                    MAX(construct_id) AS construct_id
                FROM probe_reactions
                WHERE treated = 1
                GROUP BY rg_id
            ) prd ON prd.rg_id = r.rg_id
            JOIN probe_reaction_groups rg ON rg.rg_id = r.rg_id
            LEFT JOIN meta_constructs mc ON mc.id = prd.construct_id
            LEFT JOIN meta_buffers mb ON mb.id = prd.buffer_id
            LEFT JOIN meta_nucleotides mn ON mn.id = r.nt_id
        """
        if where:
            sql += " WHERE " + " AND ".join(where)

        rows = ctx.db.execute(sql, params).fetchall()
        if not rows:
            log.info("No probe timecourse fits matched the requested filters.")
            return []

        series_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        for row in rows:
            rg_id_val = self._safe_int(row["rg_id"])
            nt_id_val = self._safe_int(row["nt_id"])
            if (rg_id_val is not None and rg_id_val in outlier_rg_ids) or (
                nt_id_val is not None and nt_id_val in outlier_nt_ids
            ) or (
                rg_id_val is not None
                and nt_id_val is not None
                and (rg_id_val, nt_id_val) in outlier_pairs
            ):
                continue

            temperature = row["temperature"]
            if temperature is None:
                continue
            temperature = float(temperature)
            if threshold_c is not None and temperature <= threshold_c:
                continue

            kobs = row["kobs"]
            if kobs in (None, ""):
                continue
            kobs = float(kobs)
            if kobs <= 0:
                continue

            log_kobs = row["log_kobs"]
            log_kdeg = row["log_kdeg"]
            if log_kobs in (None, "") or log_kdeg in (None, ""):
                continue
            log_kobs = float(log_kobs)
            log_kdeg = float(log_kdeg)
            log_kobs_transformed = log_kobs + log_kdeg

            log_err = row["log_kobs_err"]
            if log_err not in (None, ""):
                try:
                    log_err = float(log_err)
                except (TypeError, ValueError):
                    log_err = None
            else:
                log_err = None

            group_key = self._build_group_key(row, group_fields)
            if group_fields:
                series_key = ("group",) + group_key
            else:
                series_key = group_key + (row["nt_id"],)

            entry = series_map.setdefault(
                series_key,
                {
                    "construct_id": row["construct_id"],
                    "construct_name": row["construct_name"],
                    "construct_disp": row["construct_disp_name"],
                    "construct_version": row["construct_version"],
                    "buffer_id": row["buffer_id"],
                    "buffer_name": row["buffer_name"],
                    "buffer_disp": row["buffer_disp_name"],
                    "probe": row["probe"],
                    "probe_conc": row["probe_conc"],
                    "rt_protocol": row["rt_protocol"],
                    "nt_id": row["nt_id"],
                    "nt_site": row["nt_site"],
                    "nt_base": row["nt_base"],
                    "valtype": row["valtype"],
                    "temps": [],
                    "rates": [],
                    "log_errs": [],
                    "rg_ids": set(),
                    "replicates": set(),
                    "done_by": set(),
                    "nt_ids": set(),
                    "nt_sites": set(),
                    "nt_bases": set(),
                    "construct_ids_set": set(),
                    "construct_names_set": set(),
                    "construct_disps_set": set(),
                    "valtypes": set(),
                    "log_kobs_raw": [],
                    "log_kdeg_values": [],
                    "log_kobs_transformed": [],
                    "point_metadata": [],
                    "group_values": {},
                    "group_labels": {},
                    "fit_name": fit_name,
                },
            )

            entry["temps"].append(float(temperature))
            entry["rates"].append(float(log_kobs_transformed))
            entry["log_errs"].append(log_err)
            entry["log_kobs_raw"].append(log_kobs)
            entry["log_kdeg_values"].append(log_kdeg)
            entry["log_kobs_transformed"].append(log_kobs_transformed)
            if rg_id_val is not None:
                entry["rg_ids"].add(rg_id_val)
            elif row["rg_id"] not in (None, ""):
                entry["rg_ids"].add(row["rg_id"])
            replicate_val = row["replicate"]
            try:
                replicate_val = int(replicate_val)
            except (TypeError, ValueError, OverflowError):
                replicate_val = None
            if replicate_val is not None:
                entry["replicates"].add(replicate_val)
            done_by_val = row["done_by"]
            if done_by_val not in (None, ""):
                entry["done_by"].add(str(done_by_val))
            entry["point_metadata"].append(
                {
                    "rg_id": rg_id_val if rg_id_val is not None else row["rg_id"],
                    "nt_id": nt_id_val if nt_id_val is not None else row["nt_id"],
                }
            )
            if nt_id_val is not None:
                entry["nt_ids"].add(nt_id_val)
            elif row["nt_id"] not in (None, ""):
                entry["nt_ids"].add(row["nt_id"])
            if row["nt_site"] not in (None, ""):
                entry["nt_sites"].add(row["nt_site"])
            if row["nt_base"] not in (None, ""):
                entry["nt_bases"].add(str(row["nt_base"]))
            if row["construct_id"] not in (None, ""):
                entry["construct_ids_set"].add(row["construct_id"])
            if row["construct_name"] not in (None, ""):
                entry["construct_names_set"].add(str(row["construct_name"]))
            if row["construct_disp_name"] not in (None, ""):
                entry["construct_disps_set"].add(str(row["construct_disp_name"]))
            if row["valtype"] not in (None, ""):
                entry["valtypes"].add(str(row["valtype"]))

            if group_fields:
                for idx, field in enumerate(group_fields):
                    value = group_key[idx] if idx < len(group_key) else None
                    entry["group_values"].setdefault(field, value)
                    label = self._group_label(field, value, row)
                    if label is not None:
                        entry["group_labels"].setdefault(field, label)

        series_list: List[TempgradSeries] = []

        for entry in series_map.values():
            temps = entry["temps"]
            rates = entry["rates"]
            if len(temps) < min_points:
                continue

            combined = sorted(
                zip(
                    temps,
                    rates,
                    entry["log_errs"],
                    entry["log_kobs_raw"],
                    entry["log_kdeg_values"],
                    entry["log_kobs_transformed"],
                    entry["point_metadata"],
                ),
                key=lambda x: x[0],
            )
            sorted_temps: List[float] = []
            sorted_log_rates: List[float] = []
            sorted_log_errs: List[Optional[float]] = []
            sorted_point_records: List[Dict[str, Any]] = []
            for temp, log_rate, log_err, log_kobs_raw, log_kdeg_val, log_kobs2_val, meta in combined:
                sorted_temps.append(float(temp))
                sorted_log_rates.append(float(log_rate))
                err_val = float(log_err) if log_err is not None else None
                sorted_log_errs.append(err_val)
                try:
                    kobs2_calc = math.exp(log_kobs2_val)
                except (TypeError, ValueError, OverflowError):
                    kobs2_value = float("nan")
                else:
                    kobs2_value = float(kobs2_calc) if math.isfinite(kobs2_calc) else float("nan")
                sorted_point_records.append(
                    {
                        "temperature_c": float(temp),
                        "kobs2": kobs2_value,
                        "log_kobs2": float(log_kobs2_val),
                        "log_kobs2_err": err_val,
                        "log_kdeg": float(log_kdeg_val),
                        "rg_id": meta["rg_id"],
                        "nt_id": meta["nt_id"],
                    }
                )

            weights = None
            if weighted:
                sigma_values: List[float] = []
                has_valid_sigma = False
                for log_err in sorted_log_errs:
                    if log_err is None or not math.isfinite(log_err) or log_err <= 0:
                        sigma_values.append(float("nan"))
                    else:
                        sigma_values.append(float(log_err))
                        has_valid_sigma = True
                if has_valid_sigma:
                    weights = sigma_values

            construct_label = entry["construct_disp"] or entry["construct_name"] or entry["construct_id"]
            nt_ids_sorted = sorted(entry["nt_ids"])
            nt_sites_sorted = sorted(entry["nt_sites"])
            nt_bases_sorted = sorted(entry["nt_bases"])
            valtypes_sorted = sorted(entry["valtypes"])
            construct_ids_sorted = sorted(entry["construct_ids_set"])
            construct_names_sorted = sorted(entry["construct_names_set"])
            construct_disps_sorted = sorted(entry["construct_disps_set"])
            construct_labels = construct_disps_sorted or construct_names_sorted or [
                str(v) for v in construct_ids_sorted if v not in (None, "")
            ]

            if group_fields:
                label_parts: List[str] = []
                for field in group_fields:
                    label = entry["group_labels"].get(field)
                    if label is None:
                        value = entry["group_values"].get(field)
                        label = value if value not in (None, "") else field
                    label_parts.append(str(label))
                series_id = "|".join(str(part) for part in label_parts if part not in (None, ""))
                if not series_id:
                    series_id = f"{construct_label}|group"
            else:
                nt_id = entry["nt_id"]
                series_id = f"{construct_label}|nt{nt_id}"

            unique_temperatures = sorted({rec["temperature_c"] for rec in sorted_point_records})

            metadata: Dict[str, Any] = {
                "source": "probe_tc",
                "construct_id": entry["construct_id"],
                "construct_name": entry["construct_name"],
                "construct_disp_name": entry["construct_disp"],
                "construct_version": entry["construct_version"],
                "buffer_id": entry["buffer_id"],
                "buffer_name": entry["buffer_name"],
                "buffer_disp_name": entry["buffer_disp"],
                "probe": entry["probe"],
                "probe_conc": entry["probe_conc"],
                "rt_protocol": entry["rt_protocol"],
                "valtype": entry["valtype"],
                "fit_name": fit_name,
                "rg_ids": sorted(entry["rg_ids"]),
                "replicates": sorted(entry["replicates"]),
                "done_by": sorted(entry["done_by"]),
                "temperature_threshold_c": threshold_c,
                "temperatures_used_c": unique_temperatures,
                "src_kobs_data": sorted_point_records,
            }
            if outlier_summary:
                metadata["outliers_removed"] = outlier_summary
            if construct_labels:
                metadata["construct"] = construct_labels
                metadata["constructs_included"] = construct_labels
            if construct_ids_sorted:
                metadata["construct_ids"] = construct_ids_sorted
                if len(construct_ids_sorted) == 1:
                    metadata["construct_id"] = construct_ids_sorted[0]
                else:
                    metadata.pop("construct_id", None)
            else:
                metadata.pop("construct_id", None)
            if construct_names_sorted:
                metadata["construct_names"] = construct_names_sorted
                if len(construct_names_sorted) == 1:
                    metadata["construct_name"] = construct_names_sorted[0]
                else:
                    metadata.pop("construct_name", None)
            else:
                metadata.pop("construct_name", None)
            if construct_disps_sorted:
                metadata["construct_disp_names"] = construct_disps_sorted
                if len(construct_disps_sorted) == 1:
                    metadata["construct_disp_name"] = construct_disps_sorted[0]
                else:
                    metadata.pop("construct_disp_name", None)
            else:
                metadata.pop("construct_disp_name", None)
            if group_fields:
                metadata["group_by"] = list(group_fields)
                metadata["group_values"] = dict(entry["group_values"])
                if entry["group_labels"]:
                    metadata["group_labels"] = dict(entry["group_labels"])

            if nt_ids_sorted:
                metadata["nt_ids"] = nt_ids_sorted
                if len(nt_ids_sorted) == 1:
                    metadata["nt_id"] = nt_ids_sorted[0]

            if nt_sites_sorted:
                if len(nt_sites_sorted) == 1:
                    metadata["nt_site"] = nt_sites_sorted[0]
                metadata["nt_sites"] = nt_sites_sorted

            if nt_bases_sorted:
                if len(nt_bases_sorted) == 1:
                    metadata["nt_base"] = nt_bases_sorted[0]
                metadata["nt_bases"] = nt_bases_sorted

            if valtypes_sorted:
                metadata["valtypes"] = valtypes_sorted
                if len(valtypes_sorted) == 1:
                    metadata["valtype"] = valtypes_sorted[0]

            series_list.append(
                TempgradSeries(
                    series_id=series_id,
                    x_values=sorted_temps,
                    y_values=sorted_log_rates,
                    weights=weights,
                    metadata=metadata,
                )
            )

        return series_list

    def _write_artifact(self, run_dir: Path, result) -> None:
        out_dir = run_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = asdict(result)
        target = out_dir / "tempgrad_result.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @staticmethod
    def _entries_from_mapping(data: Mapping[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
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

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int_list(value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        if not isinstance(value, (list, tuple, set)):
            value = [value]
        ints: List[int] = []
        for item in value:
            if item in (None, ""):
                continue
            try:
                ints.append(int(item))
            except (TypeError, ValueError):
                continue
        return ints or None

    def _parse_outliers(self, value: Any) -> Tuple[Set[int], Set[int], Set[Tuple[int, int]]]:
        rg_ids: Set[int] = set()
        nt_ids: Set[int] = set()
        pairs: Set[Tuple[int, int]] = set()
        if not value:
            return rg_ids, nt_ids, pairs

        if isinstance(value, Mapping):
            items = [value]
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [value]

        for item in items:
            if item is None:
                continue
            if isinstance(item, Mapping):
                for key, raw in item.items():
                    self._apply_outlier_token(key, raw, rg_ids, nt_ids, pairs)
            else:
                text = str(item)
                if ":" in text:
                    key, raw = text.split(":", 1)
                    self._apply_outlier_token(key, raw, rg_ids, nt_ids, pairs)
        return rg_ids, nt_ids, pairs

    def _apply_outlier_token(
        self,
        key: Any,
        raw: Any,
        rg_ids: Set[int],
        nt_ids: Set[int],
        pairs: Set[Tuple[int, int]],
    ) -> None:
        field = str(key or "").strip().lower()
        if not field:
            return
        if field in {"rg_id", "rg", "reaction_group"}:
            for value in self._iter_outlier_numbers(raw):
                if value is not None:
                    rg_ids.add(value)
            return
        if field in {"nt_id", "nt", "nucleotide"}:
            for value in self._iter_outlier_numbers(raw):
                if value is not None:
                    nt_ids.add(value)
            return
        if field in {"rg_nt_id", "rg_nt", "rgnt", "rgntid"}:
            for rg_val, nt_val in self._iter_outlier_pairs(raw):
                if rg_val is not None and nt_val is not None:
                    pairs.add((rg_val, nt_val))
            return
        # Fallback: if the key encodes both identifiers (e.g., "20:418"), try to parse as pair.
        for rg_val, nt_val in self._iter_outlier_pairs({field: raw}):
            if rg_val is not None and nt_val is not None:
                pairs.add((rg_val, nt_val))

    def _iter_outlier_numbers(self, raw: Any) -> Iterable[Optional[int]]:
        if raw is None:
            return
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                yield from self._iter_outlier_numbers(item)
            return
        text = str(raw).strip()
        if not text:
            return
        # Normalize separators to whitespace
        tokens = re.split(r"[,\s]+", text)
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            yield self._safe_int(token)

    def _iter_outlier_pairs(self, raw: Any) -> Iterable[Tuple[Optional[int], Optional[int]]]:
        if raw is None:
            return
        if isinstance(raw, Mapping):
            rg_val = self._safe_int(raw.get("rg") or raw.get("rg_id") or raw.get("rgid"))
            nt_val = self._safe_int(raw.get("nt") or raw.get("nt_id") or raw.get("ntid"))
            yield (rg_val, nt_val)
            return
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                yield from self._iter_outlier_pairs(item)
            return
        text = str(raw).strip()
        if not text:
            return
        # Try common separators (colon, comma, semicolon, slash, whitespace)
        for pattern in (r":", r",", r";", r"/"):
            if re.search(pattern, text):
                parts = [p.strip() for p in re.split(pattern, text) if p.strip()]
                if len(parts) >= 2:
                    yield (self._safe_int(parts[0]), self._safe_int(parts[1]))
                    return
        parts = [p.strip() for p in text.split() if p.strip()]
        if len(parts) >= 2:
            yield (self._safe_int(parts[0]), self._safe_int(parts[1]))

    @staticmethod
    def _summarize_outliers(
        rg_ids: Set[int],
        nt_ids: Set[int],
        pairs: Set[Tuple[int, int]],
    ) -> Optional[Dict[str, Any]]:
        if not (rg_ids or nt_ids or pairs):
            return None
        summary: Dict[str, Any] = {}
        if rg_ids:
            summary["rg_ids"] = sorted(rg_ids)
        if nt_ids:
            summary["nt_ids"] = sorted(nt_ids)
        if pairs:
            summary["rg_nt_ids"] = [f"{rg}:{nt}" for rg, nt in sorted(pairs)]
        return summary

    @staticmethod
    def _resolve_construct_ids(conn: sqlite3.Connection, value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        values = value if isinstance(value, (list, tuple, set)) else [value]
        values = [str(v).strip() for v in values if v not in (None, "")]
        if not values:
            return None
        placeholders = ",".join("?" for _ in values)
        sql = f"""
            SELECT id
            FROM meta_constructs
            WHERE lower(disp_name) IN ({placeholders})
               OR lower(name) IN ({placeholders})
        """
        params = [v.lower() for v in values] + [v.lower() for v in values]
        rows = conn.execute(sql, params).fetchall()
        ids = [int(row[0]) for row in rows]
        return ids or None

    @staticmethod
    def _normalize_group_by(value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            value = [value]
        result: List[str] = []
        for item in value:
            if item in (None, ""):
                continue
            result.append(str(item).strip().lower())
        return result

    def _build_group_key(self, row: Mapping[str, Any], fields: Sequence[str]) -> Tuple[Any, ...]:
        if not fields:
            return (
                row["construct_id"],
                row["buffer_id"],
                row["probe"],
                row["probe_conc"],
                row["rt_protocol"],
            )
        key_parts: List[Any] = []
        for field in fields:
            key_parts.append(self._group_value(field, row))
        return tuple(key_parts)

    def _group_value(self, field: str, row: Mapping[str, Any]) -> Any:
        name = str(field).strip().lower()
        if name in {"construct", "construct_id"}:
            return self._row_get(row, "construct_id")
        if name in {"construct_name", "construct_label"}:
            return self._row_get(row, "construct_disp_name") or self._row_get(row, "construct_name")
        if name in {"buffer", "buffer_id"}:
            return self._row_get(row, "buffer_id")
        if name in {"buffer_name", "buffer_label"}:
            return self._row_get(row, "buffer_disp_name") or self._row_get(row, "buffer_name")
        if name in {"probe"}:
            return self._row_get(row, "probe")
        if name in {"probe_conc", "probe_concentration"}:
            return self._row_get(row, "probe_conc")
        if name in {"rt_protocol", "protocol"}:
            return self._row_get(row, "rt_protocol")
        if name in {"base", "nt_base"}:
            return self._row_get(row, "nt_base")
        if name in {"valtype"}:
            return self._row_get(row, "valtype")
        if name in {"nt", "nt_id"}:
            return self._row_get(row, "nt_id")
        if name in {"rg", "rg_label"}:
            return self._row_get(row, "rg_label")
        return self._row_get(row, field)

    def _group_label(self, field: str, value: Any, row: Mapping[str, Any]) -> Optional[str]:
        name = str(field).strip().lower()
        if value in (None, ""):
            return None
        if name in {"construct", "construct_id", "construct_name", "construct_label"}:
            label = self._row_get(row, "construct_disp_name") or self._row_get(row, "construct_name") or value
            return str(label)
        if name in {"buffer", "buffer_id", "buffer_name", "buffer_label"}:
            label = self._row_get(row, "buffer_disp_name") or self._row_get(row, "buffer_name") or value
            return str(label)
        if name in {"probe"}:
            return str(self._row_get(row, "probe") or value)
        if name in {"probe_conc", "probe_concentration"}:
            try:
                return f"{float(value):g}"
            except (TypeError, ValueError):
                return str(value)
        if name in {"rt_protocol", "protocol"}:
            return str(self._row_get(row, "rt_protocol") or value)
        if name in {"base", "nt_base"}:
            return str(self._row_get(row, "nt_base") or value)
        if name in {"valtype"}:
            return str(self._row_get(row, "valtype") or value)
        if name in {"nt", "nt_id"}:
            return f"nt{value}"
        if name in {"rg", "rg_label"}:
            return str(self._row_get(row, "rg_label") or value)
        return str(value)

    @staticmethod
    def _row_get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
        if row is None:
            return default
        if isinstance(row, Mapping):
            return row.get(key, default)
        if hasattr(row, "keys"):
            try:
                if key in row.keys():
                    return row[key]
            except Exception:  # pragma: no cover - defensive
                return default
            return default
        try:
            return row[key]
        except Exception:
            return default

    @staticmethod
    def _resolve_buffer_ids(conn: sqlite3.Connection, value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        values = value if isinstance(value, (list, tuple, set)) else [value]
        values = [str(v).strip() for v in values if v not in (None, "")]
        if not values:
            return None
        placeholders = ",".join("?" for _ in values)
        sql = f"""
            SELECT id
            FROM meta_buffers
            WHERE lower(name) IN ({placeholders})
               OR lower(disp_name) IN ({placeholders})
        """
        params = [v.lower() for v in values] + [v.lower() for v in values]
        rows = conn.execute(sql, params).fetchall()
        ids = [int(row[0]) for row in rows]
        return ids or None


# numpy is optional at runtime; import only after definitions to keep annotations simple.
try:
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover - numpy should be present
    raise RuntimeError("TempgradFitTask requires numpy to be installed.") from exc
