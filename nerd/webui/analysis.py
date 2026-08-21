"""Read-only query helpers for the Web UI analysis workspace."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence


def _csv_values(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    return sorted({item for item in str(value).split(",") if item})


def _dict_rows(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def analysis_catalog(conn: Any) -> Dict[str, Any]:
    """Return experiment-aware selectors for modification runs and timecourses."""
    fmod_runs = _dict_rows(conn.execute(
        """
        SELECT
            fr.id AS fmod_run_id,
            fr.software_name,
            fr.software_version,
            fr.run_datetime,
            s.id AS sample_id,
            s.sample_name,
            pr.rg_id,
            pr.reaction_time,
            pr.treated,
            pr.temperature,
            pr.replicate,
            pr.probe,
            pr.probe_concentration,
            pr.rt_protocol,
            pr.done_by,
            mc.disp_name AS construct_name,
            COALESCE(mb.disp_name, mb.name) AS buffer_name,
            GROUP_CONCAT(DISTINCT fv.valtype) AS valtypes
        FROM probe_fmod_runs fr
        JOIN sequencing_samples s ON s.id = fr.s_id
        JOIN probe_fmod_values fv ON fv.fmod_run_id = fr.id
        JOIN probe_reactions pr ON pr.id = fv.rxn_id
        JOIN meta_constructs mc ON mc.id = pr.construct_id
        JOIN meta_buffers mb ON mb.id = pr.buffer_id
        GROUP BY fr.id
        ORDER BY fr.run_datetime DESC, fr.id DESC
        """
    ).fetchall())
    for run in fmod_runs:
        run["valtypes"] = _csv_values(run.get("valtypes"))

    reaction_groups = _dict_rows(conn.execute(
        """
        SELECT
            prg.rg_id,
            prg.rg_label,
            GROUP_CONCAT(DISTINCT mc.disp_name) AS construct_name,
            GROUP_CONCAT(DISTINCT COALESCE(mb.disp_name, mb.name)) AS buffer_name,
            GROUP_CONCAT(DISTINCT pr.temperature) AS temperature,
            GROUP_CONCAT(DISTINCT pr.replicate) AS replicate,
            GROUP_CONCAT(DISTINCT pr.probe) AS probe,
            GROUP_CONCAT(DISTINCT pr.probe_concentration) AS probe_concentration,
            GROUP_CONCAT(DISTINCT pr.rt_protocol) AS rt_protocol,
            COUNT(DISTINCT pr.s_id) AS sample_count,
            COUNT(DISTINCT CASE WHEN pr.treated != 0 THEN pr.reaction_time END) AS timepoint_count,
            MIN(CASE WHEN pr.treated != 0 THEN pr.reaction_time END) AS time_min,
            MAX(CASE WHEN pr.treated != 0 THEN pr.reaction_time END) AS time_max,
            GROUP_CONCAT(DISTINCT fv.valtype) AS valtypes,
            (
                SELECT GROUP_CONCAT(DISTINCT tfr.valtype)
                FROM probe_tc_fit_runs tfr
                WHERE tfr.rg_id = prg.rg_id
                  AND tfr.nt_id IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM probe_tc_fit_params tfp
                      WHERE tfp.fit_run_id = tfr.id
                        AND tfp.param_name IN ('kobs', 'log_kobs')
                        AND tfp.param_numeric IS NOT NULL
                  )
            ) AS fit_valtypes,
            (
                SELECT COUNT(DISTINCT tfr.nt_id)
                FROM probe_tc_fit_runs tfr
                WHERE tfr.rg_id = prg.rg_id
                  AND EXISTS (
                      SELECT 1 FROM probe_tc_fit_params tfp
                      WHERE tfp.fit_run_id = tfr.id
                        AND tfp.param_name IN ('kobs', 'log_kobs')
                        AND tfp.param_numeric IS NOT NULL
                  )
            ) AS kobs_site_count
        FROM probe_reaction_groups prg
        JOIN probe_reactions pr ON pr.rg_id = prg.rg_id
        JOIN meta_constructs mc ON mc.id = pr.construct_id
        JOIN meta_buffers mb ON mb.id = pr.buffer_id
        JOIN probe_fmod_values fv ON fv.rxn_id = pr.id
        GROUP BY prg.rg_id
        ORDER BY COALESCE(prg.rg_label, ''), prg.rg_id
        """
    ).fetchall())
    for group in reaction_groups:
        group["valtypes"] = _csv_values(group.get("valtypes"))
        group["fit_valtypes"] = _csv_values(group.get("fit_valtypes"))

    return {"fmod_runs": fmod_runs, "reaction_groups": reaction_groups}


def kinetic_rates(conn: Any, rg_ids: Sequence[int], valtype: str) -> Dict[str, Any]:
    """Return the preferred stored k_obs fit for each group and nucleotide."""
    selected = list(dict.fromkeys(int(rg_id) for rg_id in rg_ids))
    if not 1 <= len(selected) <= 3:
        raise ValueError("Choose between one and three reaction groups.")
    normalized_valtype = str(valtype or "").strip()
    if not normalized_valtype:
        raise ValueError("Choose a data type.")

    placeholders = ",".join("?" for _ in selected)
    rows = _dict_rows(conn.execute(
        f"""
        SELECT
            r.id AS fit_run_id,
            r.rg_id,
            rg.rg_label,
            r.nt_id,
            mn.site,
            UPPER(mn.base) AS base,
            r.fit_kind,
            r.model,
            r.created_at,
            p.param_name,
            p.param_numeric
        FROM probe_tc_fit_runs r
        JOIN probe_tc_fit_params p ON p.fit_run_id = r.id
        JOIN probe_reaction_groups rg ON rg.rg_id = r.rg_id
        JOIN meta_nucleotides mn ON mn.id = r.nt_id
        WHERE r.rg_id IN ({placeholders})
          AND (r.valtype = ? OR r.valtype IS NULL)
          AND p.param_name IN ('kobs', 'log_kobs', 'diag:r2')
          AND p.param_numeric IS NOT NULL
        ORDER BY mn.site, mn.id, r.rg_id, r.id DESC
        """,
        (*selected, normalized_valtype),
    ).fetchall())

    fits: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        fit = fits.setdefault(row["fit_run_id"], {
            "fit_run_id": row["fit_run_id"],
            "rg_id": row["rg_id"],
            "rg_label": row["rg_label"],
            "nt_id": row["nt_id"],
            "site": row["site"],
            "base": row["base"],
            "fit_kind": row["fit_kind"],
            "model": row["model"],
            "created_at": row["created_at"],
            "params": {},
        })
        fit["params"][row["param_name"]] = row["param_numeric"]

    priority = {"round3_constrained": 0, "round1_free": 1}
    best: Dict[tuple[int, int], Dict[str, Any]] = {}
    for fit in sorted(
        fits.values(),
        key=lambda item: (priority.get(item["fit_kind"], 2), -int(item["fit_run_id"])),
    ):
        params = fit.pop("params")
        try:
            log_kobs = float(params["log_kobs"]) if "log_kobs" in params else None
            kobs = float(params["kobs"]) if "kobs" in params else None
            if log_kobs is None and kobs is not None and kobs > 0:
                log_kobs = math.log(kobs)
            if kobs is None and log_kobs is not None:
                kobs = math.exp(log_kobs)
            if not (math.isfinite(float(log_kobs)) and math.isfinite(float(kobs)) and kobs > 0):
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        fit["log_kobs"] = log_kobs
        fit["kobs"] = kobs
        try:
            r2 = float(params["diag:r2"]) if "diag:r2" in params else None
            fit["r2"] = r2 if r2 is not None and math.isfinite(r2) else None
        except (TypeError, ValueError):
            fit["r2"] = None
        fit["site_base"] = f"{fit['site']}{fit['base']}"
        best.setdefault((int(fit["rg_id"]), int(fit["nt_id"])), fit)

    values = sorted(
        best.values(), key=lambda row: (int(row["site"]), int(row["nt_id"]), selected.index(int(row["rg_id"])))
    )
    return {"rg_ids": selected, "valtype": normalized_valtype, "values": values}


def modification_rates(conn: Any, run_ids: Sequence[int], valtype: str) -> Dict[str, Any]:
    selected = list(dict.fromkeys(int(run_id) for run_id in run_ids))
    if not 1 <= len(selected) <= 3:
        raise ValueError("Choose between one and three ShapeMapper runs.")
    normalized_valtype = str(valtype or "").strip()
    if not normalized_valtype:
        raise ValueError("Choose a data type.")

    placeholders = ",".join("?" for _ in selected)
    rows = _dict_rows(conn.execute(
        f"""
        SELECT
            fv.fmod_run_id,
            s.sample_name,
            fv.nt_id,
            mn.site,
            UPPER(mn.base) AS base,
            AVG(fv.fmod_val) AS fmod_val,
            MAX(fv.read_depth) AS read_depth,
            MAX(fv.outlier) AS outlier,
            COUNT(*) AS source_rows
        FROM probe_fmod_values fv
        JOIN probe_fmod_runs fr ON fr.id = fv.fmod_run_id
        JOIN sequencing_samples s ON s.id = fr.s_id
        JOIN meta_nucleotides mn ON mn.id = fv.nt_id
        WHERE fv.fmod_run_id IN ({placeholders})
          AND fv.valtype = ?
          AND fv.fmod_val IS NOT NULL
        GROUP BY fv.fmod_run_id, fv.nt_id
        ORDER BY mn.site, mn.id, fv.fmod_run_id
        """,
        (*selected, normalized_valtype),
    ).fetchall())
    for row in rows:
        row["site_base"] = f"{row['site']}{row['base']}"
    return {"run_ids": selected, "valtype": normalized_valtype, "values": rows}


def timecourse_options(conn: Any, rg_id: int) -> Dict[str, Any]:
    rows = _dict_rows(conn.execute(
        """
        SELECT DISTINCT
            fv.nt_id,
            mn.site,
            UPPER(mn.base) AS base,
            fv.valtype
        FROM probe_reactions pr
        JOIN probe_fmod_values fv ON fv.rxn_id = pr.id
        JOIN meta_nucleotides mn ON mn.id = fv.nt_id
        WHERE pr.rg_id = ? AND fv.fmod_val IS NOT NULL
        ORDER BY mn.site, mn.id, fv.valtype
        """,
        (int(rg_id),),
    ).fetchall())
    sites: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        entry = sites.setdefault(row["nt_id"], {
            "nt_id": row["nt_id"],
            "site": row["site"],
            "base": row["base"],
            "site_base": f"{row['site']}{row['base']}",
            "valtypes": [],
        })
        entry["valtypes"].append(row["valtype"])
    return {"rg_id": int(rg_id), "sites": list(sites.values())}


def _fit_curve(params: Dict[str, float], time_max: float, points: int = 200) -> List[Dict[str, float]]:
    log_kobs = float(params["log_kobs"])
    log_kdeg = float(params["log_kdeg"])
    log_fmod0 = float(params["log_fmod0"])
    kobs = math.exp(log_kobs)
    kdeg = math.exp(log_kdeg)
    fmod0 = math.exp(log_fmod0)
    upper = max(1.0, float(time_max))
    curve = []
    for index in range(points):
        time = upper * index / (points - 1)
        value = 1.0 - math.exp(-kobs * (1.0 - math.exp(-kdeg * time))) + fmod0
        curve.append({"time": time, "fmod_val": value})
    return curve


def _best_fits(conn: Any, rg_id: int, nt_ids: Sequence[int], valtype: str) -> Dict[int, Dict[str, Any]]:
    placeholders = ",".join("?" for _ in nt_ids)
    rows = _dict_rows(conn.execute(
        f"""
        SELECT
            r.id AS fit_run_id,
            r.nt_id,
            r.fit_kind,
            r.model,
            r.created_at,
            p.param_name,
            p.param_numeric,
            p.param_text
        FROM probe_tc_fit_runs r
        JOIN probe_tc_fit_params p ON p.fit_run_id = r.id
        WHERE r.rg_id = ?
          AND r.nt_id IN ({placeholders})
          AND (r.valtype = ? OR r.valtype IS NULL)
        ORDER BY r.id DESC
        """,
        (int(rg_id), *[int(value) for value in nt_ids], valtype),
    ).fetchall())
    runs: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        run = runs.setdefault(row["fit_run_id"], {
            "fit_run_id": row["fit_run_id"],
            "nt_id": row["nt_id"],
            "fit_kind": row["fit_kind"],
            "model": row["model"],
            "created_at": row["created_at"],
            "params": {},
        })
        value = row["param_numeric"] if row["param_numeric"] is not None else row["param_text"]
        run["params"][row["param_name"]] = value

    priority = {"round3_constrained": 0, "round1_free": 1}
    best: Dict[int, Dict[str, Any]] = {}
    candidates = sorted(
        runs.values(),
        key=lambda item: (priority.get(item["fit_kind"], 2), -int(item["fit_run_id"])),
    )
    for candidate in candidates:
        required = ("log_kobs", "log_kdeg", "log_fmod0")
        try:
            if not all(math.isfinite(float(candidate["params"][name])) for name in required):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        best.setdefault(int(candidate["nt_id"]), candidate)
    return best


def timecourse_data(
    conn: Any,
    rg_id: int,
    nt_ids: Sequence[int],
    valtype: str,
    *,
    include_fits: bool = True,
) -> Dict[str, Any]:
    selected = list(dict.fromkeys(int(nt_id) for nt_id in nt_ids))
    if not 1 <= len(selected) <= 3:
        raise ValueError("Choose between one and three nucleotide sites.")
    normalized_valtype = str(valtype or "").strip()
    if not normalized_valtype:
        raise ValueError("Choose a data type.")

    placeholders = ",".join("?" for _ in selected)
    rows = _dict_rows(conn.execute(
        f"""
        SELECT
            fv.nt_id,
            mn.site,
            UPPER(mn.base) AS base,
            pr.id AS reaction_id,
            pr.reaction_time,
            CASE WHEN pr.treated = 0 THEN 0.0 ELSE pr.reaction_time END AS plot_time,
            pr.treated,
            pr.temperature,
            pr.replicate,
            s.sample_name,
            s.to_drop,
            fv.fmod_run_id,
            fv.fmod_val,
            fv.read_depth,
            fv.outlier
        FROM probe_reactions pr
        JOIN sequencing_samples s ON s.id = pr.s_id
        JOIN probe_fmod_values fv ON fv.rxn_id = pr.id
        JOIN meta_nucleotides mn ON mn.id = fv.nt_id
        WHERE pr.rg_id = ?
          AND fv.nt_id IN ({placeholders})
          AND fv.valtype = ?
          AND fv.fmod_val IS NOT NULL
        ORDER BY mn.site, plot_time, pr.id, fv.fmod_run_id
        """,
        (int(rg_id), *selected, normalized_valtype),
    ).fetchall())
    for row in rows:
        row["site_base"] = f"{row['site']}{row['base']}"

    series: List[Dict[str, Any]] = []
    fit_by_nt = _best_fits(conn, rg_id, selected, normalized_valtype) if include_fits else {}
    for nt_id in selected:
        observations = [row for row in rows if int(row["nt_id"]) == nt_id]
        if not observations:
            continue
        item: Dict[str, Any] = {
            "nt_id": nt_id,
            "site": observations[0]["site"],
            "base": observations[0]["base"],
            "site_base": observations[0]["site_base"],
            "observations": observations,
            "fit": None,
        }
        fit = fit_by_nt.get(nt_id)
        if fit is not None:
            max_time = max(float(row["plot_time"]) for row in observations)
            item["fit"] = {
                "fit_run_id": fit["fit_run_id"],
                "fit_kind": fit["fit_kind"],
                "model": fit["model"],
                "created_at": fit["created_at"],
                "r2": fit["params"].get("diag:r2"),
                "curve": _fit_curve(fit["params"], max_time),
            }
        series.append(item)
    return {"rg_id": int(rg_id), "valtype": normalized_valtype, "series": series}
