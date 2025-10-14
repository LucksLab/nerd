"""
Task for temperature-gradient fitting (Arrhenius and melt models).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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

        block["data_source"] = str(block.get("data_source") or block["filters"].get("data_source") or "manual")

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
        log.info("No database loader implemented for two_state_melt; supply 'series' entries in config.")
        return []

    def _series_from_db_probe_tc(self, ctx: TaskContext, inputs: Dict[str, Any]) -> List[TempgradSeries]:
        filters = dict(inputs.get("filters") or {})
        options = dict(inputs.get("engine_options") or {})

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
                le.param_numeric AS log_kobs_err
            FROM probe_tc_fit_runs r
            JOIN probe_tc_fit_params k ON k.fit_run_id = r.id AND k.param_name = 'kobs'
            LEFT JOIN probe_tc_fit_params le ON le.fit_run_id = r.id AND le.param_name = 'log_kobs_err'
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
            temperature = row["temperature"]
            if temperature is None:
                continue
            temperature = float(temperature)
            if threshold_c is not None and temperature < threshold_c:
                continue

            kobs = row["kobs"]
            if kobs in (None, ""):
                continue
            kobs = float(kobs)
            if kobs <= 0:
                continue

            log_err = row["log_kobs_err"]
            if log_err not in (None, ""):
                try:
                    log_err = float(log_err)
                except (TypeError, ValueError):
                    log_err = None
            else:
                log_err = None

            group_key = (
                row["construct_id"],
                row["buffer_id"],
                row["probe"],
                row["probe_conc"],
                row["rt_protocol"],
            )
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
                    "temps": [],
                    "rates": [],
                    "log_errs": [],
                    "rg_ids": set(),
                    "replicates": set(),
                    "done_by": set(),
                },
            )

            entry["temps"].append(float(temperature))
            entry["rates"].append(kobs)
            entry["log_errs"].append(log_err)
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

        series_list: List[TempgradSeries] = []

        for entry in series_map.values():
            temps = entry["temps"]
            rates = entry["rates"]
            if len(temps) < min_points:
                continue

            combined = sorted(zip(temps, rates, entry["log_errs"]), key=lambda x: x[0])
            sorted_temps = [c[0] for c in combined]
            sorted_rates = [c[1] for c in combined]
            sorted_log_errs = [c[2] for c in combined]

            weights = None
            if weighted:
                weight_values = []
                for log_err in sorted_log_errs:
                    if log_err is None or log_err <= 0:
                        weight_values.append(0.0)
                    else:
                        weight_values.append(1.0 / float(log_err))
                if any(w > 0 for w in weight_values):
                    weights = weight_values

            construct_label = entry["construct_disp"] or entry["construct_name"] or entry["construct_id"]
            nt_id = entry["nt_id"]
            series_id = f"{construct_label}|nt{nt_id}"

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
                "nt_id": nt_id,
                "nt_site": entry["nt_site"],
                "nt_base": entry["nt_base"],
                "rg_ids": sorted(entry["rg_ids"]),
                "replicates": sorted(entry["replicates"]),
                "done_by": sorted(entry["done_by"]),
                "temperature_threshold_c": threshold_c,
                "temperatures_used_c": sorted_temps,
                "kobs_values": sorted_rates,
                "log_kobs_err": sorted_log_errs,
            }

            series_list.append(
                TempgradSeries(
                    series_id=series_id,
                    x_values=sorted_temps,
                    y_values=sorted_rates,
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
