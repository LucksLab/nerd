"""CSV exports for persisted NERD analysis results."""

from __future__ import annotations

import csv
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO


PROBE_TIMECOURSE_COLUMNS = [
    "site",
    "base",
    "construct",
    "temp",
    "rep",
    "buffer",
    "kobs",
    "kobs_err",
    "kdeg",
    "kdeg_err",
    "r2",
    "chisq",
]

MUT_COUNT_COLUMNS = [
    "sample",
    "site",
    "base",
    "valtype",
    "mutation_rate",
    "read_depth",
]

_ROUND_PRIORITY = {
    "round1_free": 1,
    "round2_global": 2,
    "round2_global_profiled": 2,
    "round3_constrained": 3,
}


def _numeric(params: Dict[str, Any], name: str) -> Optional[float]:
    value = params.get(name)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _linear_error(
    params: Dict[str, Any], direct_name: str, value_name: str, log_error_name: str
) -> Optional[float]:
    direct = _numeric(params, direct_name)
    if direct is not None:
        return direct
    value = _numeric(params, value_name)
    log_error = _numeric(params, log_error_name)
    if value is None or log_error is None:
        return None
    # First-order propagation for y = exp(log(y)).
    return abs(value * log_error)


def _parameter_maps(
    conn: sqlite3.Connection, fit_run_ids: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    if not fit_run_ids:
        return {}
    placeholders = ",".join("?" for _ in fit_run_ids)
    rows = conn.execute(
        "SELECT fit_run_id, param_name, param_numeric, param_text "
        f"FROM probe_tc_fit_params WHERE fit_run_id IN ({placeholders})",
        tuple(fit_run_ids),
    ).fetchall()
    result: Dict[int, Dict[str, Any]] = {int(run_id): {} for run_id in fit_run_ids}
    for row in rows:
        value = row["param_numeric"] if row["param_numeric"] is not None else row["param_text"]
        result[int(row["fit_run_id"])][str(row["param_name"])] = value
    return result


def probe_timecourse_rows(
    conn: sqlite3.Connection,
    *,
    task_id: Optional[int] = None,
    rg_ids: Optional[Sequence[int]] = None,
    all_results: bool = False,
) -> List[Dict[str, Any]]:
    """Return final per-nucleotide probe-timecourse fits ready for CSV output."""
    selected_rg_ids = sorted({int(value) for value in (rg_ids or [])})
    if sum((task_id is not None, bool(selected_rg_ids), bool(all_results))) != 1:
        raise ValueError("Choose exactly one selector: --task-id, --rg-id, or --all.")

    if task_id is not None:
        task = conn.execute(
            "SELECT task_name, scope_kind, scope_id FROM core_tasks WHERE id = ?",
            (int(task_id),),
        ).fetchone()
        if task is None:
            raise ValueError(f"Task {task_id} does not exist.")
        if task["task_name"] != "probe_timecourse":
            raise ValueError(f"Task {task_id} is {task['task_name']}, not probe_timecourse.")

    clauses = ["r.nt_id IS NOT NULL"]
    query_params: List[Any] = []
    if task_id is not None:
        linked = conn.execute(
            "SELECT 1 FROM probe_tc_fit_runs WHERE task_id = ? LIMIT 1",
            (int(task_id),),
        ).fetchone()
        if linked is not None:
            clauses.append("r.task_id = ?")
            query_params.append(int(task_id))
        else:
            # Fits written before task provenance was introduced have a NULL
            # task_id. Recover them only when task scope makes the ownership
            # unambiguous; otherwise direct the caller to an rg-id export.
            member_rows = conn.execute(
                "SELECT member_id FROM core_task_scope_members "
                "WHERE task_id = ? AND member_kind = 'rg' AND member_id IS NOT NULL",
                (int(task_id),),
            ).fetchall()
            legacy_rg_ids = {int(row["member_id"]) for row in member_rows}
            if not legacy_rg_ids and task["scope_kind"] == "rg" and task["scope_id"] is not None:
                legacy_rg_ids.add(int(task["scope_id"]))
            if not legacy_rg_ids:
                raise ValueError(
                    f"Task {task_id} has no linked fits or recorded reaction-group scope."
                )

            placeholders = ",".join("?" for _ in legacy_rg_ids)
            competing = conn.execute(
                """
                SELECT DISTINCT t.id
                FROM core_tasks t
                LEFT JOIN core_task_scope_members sm
                  ON sm.task_id = t.id AND sm.member_kind = 'rg'
                WHERE t.id <> ? AND t.task_name = 'probe_timecourse'
                  AND t.state = 'completed'
                  AND NOT EXISTS (
                      SELECT 1 FROM probe_tc_fit_runs linked WHERE linked.task_id = t.id
                  )
                  AND (
                      sm.member_id IN (%s)
                      OR (t.scope_kind = 'rg' AND t.scope_id IN (%s))
                  )
                """ % (placeholders, placeholders),
                (int(task_id), *sorted(legacy_rg_ids), *sorted(legacy_rg_ids)),
            ).fetchall()
            if competing:
                ids = ", ".join(str(row["id"]) for row in competing)
                raise ValueError(
                    f"Task {task_id} predates fit provenance and overlaps legacy "
                    f"probe_timecourse task(s) {ids}; export with --rg-id instead."
                )
            clauses.append("r.task_id IS NULL")
            clauses.append(f"r.rg_id IN ({placeholders})")
            query_params.extend(sorted(legacy_rg_ids))
    elif selected_rg_ids:
        placeholders = ",".join("?" for _ in selected_rg_ids)
        clauses.append(f"r.rg_id IN ({placeholders})")
        query_params.extend(selected_rg_ids)

    runs = conn.execute(
        """
        SELECT r.id, r.task_id, r.rg_id, r.nt_id, r.valtype, r.fit_kind,
               r.created_at, mn.site, mn.base, mc.disp_name AS construct,
               rx.temp, rx.rep, COALESCE(mb.disp_name, mb.name) AS buffer
        FROM probe_tc_fit_runs r
        JOIN meta_nucleotides mn ON mn.id = r.nt_id
        JOIN (
            SELECT rg_id, MIN(construct_id) AS construct_id,
                   MIN(buffer_id) AS buffer_id, MIN(temperature) AS temp,
                   MIN(replicate) AS rep
            FROM probe_reactions
            GROUP BY rg_id
        ) rx ON rx.rg_id = r.rg_id
        JOIN meta_constructs mc ON mc.id = rx.construct_id
        JOIN meta_buffers mb ON mb.id = rx.buffer_id
        WHERE %s
        ORDER BY r.rg_id, mn.site, r.id
        """ % " AND ".join(clauses),
        tuple(query_params),
    ).fetchall()
    if not runs:
        selector = f"task {task_id}" if task_id is not None else (
            "reaction group(s) " + ", ".join(str(value) for value in selected_rg_ids)
            if selected_rg_ids else "the project"
        )
        raise ValueError(f"No persisted probe_timecourse fits found for {selector}.")

    params_by_run = _parameter_maps(conn, [int(row["id"]) for row in runs])

    # A task can persist several fitting rounds. The requested CSV has no round
    # column, so keep the scientifically final successful round for each site.
    # Failed per-site attempts are persisted for diagnostics but contain no
    # rate parameters and do not belong in a fit-parameter export.
    chosen: Dict[tuple[int, int, str], Any] = {}
    for row in runs:
        if _numeric(params_by_run.get(int(row["id"]), {}), "kobs") is None:
            continue
        key = (int(row["rg_id"]), int(row["nt_id"]), str(row["valtype"] or ""))
        previous = chosen.get(key)
        score = (
            int(row["task_id"]) if row["task_id"] is not None else 0,
            _ROUND_PRIORITY.get(str(row["fit_kind"]), 0),
            int(row["id"]),
        )
        if previous is None:
            chosen[key] = row
            continue
        previous_score = (
            int(previous["task_id"]) if previous["task_id"] is not None else 0,
            _ROUND_PRIORITY.get(str(previous["fit_kind"]), 0),
            int(previous["id"]),
        )
        if score > previous_score:
            chosen[key] = row

    chosen_rows = sorted(
        chosen.values(), key=lambda row: (int(row["rg_id"]), int(row["site"]), str(row["valtype"] or ""))
    )
    if not chosen_rows:
        raise ValueError("No successful probe_timecourse site fits matched the selection.")

    # Global round parameters hold the shared kdeg uncertainty for global and
    # constrained fits. Map those onto the per-site rows as a fallback.
    global_runs = conn.execute(
        "SELECT id, task_id, rg_id, fit_kind FROM probe_tc_fit_runs WHERE nt_id IS NULL"
    ).fetchall()
    global_ids = [int(row["id"]) for row in global_runs]
    global_params = _parameter_maps(conn, global_ids)
    global_by_key: Dict[tuple[Optional[int], int, str], Dict[str, Any]] = {}
    shared_kdeg_by_scope: Dict[tuple[Optional[int], int], Dict[str, Any]] = {}
    for row in global_runs:
        run_params = global_params.get(int(row["id"]), {})
        key = (
            int(row["task_id"]) if row["task_id"] is not None else None,
            int(row["rg_id"]),
            str(row["fit_kind"]),
        )
        global_by_key[key] = run_params
        scope_key = key[:2]
        if run_params.get("kdeg_err") is not None or run_params.get("log_kdeg_err") is not None:
            shared_kdeg_by_scope[scope_key] = run_params

    exported: List[Dict[str, Any]] = []
    for row in chosen_rows:
        params = dict(params_by_run.get(int(row["id"]), {}))
        shared = global_by_key.get(
            (
                int(row["task_id"]) if row["task_id"] is not None else None,
                int(row["rg_id"]),
                str(row["fit_kind"]),
            ),
            {},
        )
        scope_shared = shared_kdeg_by_scope.get(
            (
                int(row["task_id"]) if row["task_id"] is not None else None,
                int(row["rg_id"]),
            ),
            {},
        )
        for name in ("kdeg", "kdeg_err", "log_kdeg_err"):
            if params.get(name) is None and shared.get(name) is not None:
                params[name] = shared[name]
            if params.get(name) is None and scope_shared.get(name) is not None:
                params[name] = scope_shared[name]
        exported.append(
            {
                "site": row["site"],
                "base": row["base"],
                "construct": row["construct"],
                "temp": row["temp"],
                "rep": row["rep"],
                "buffer": row["buffer"],
                "kobs": _numeric(params, "kobs"),
                "kobs_err": _linear_error(params, "kobs_err", "kobs", "log_kobs_err"),
                "kdeg": _numeric(params, "kdeg"),
                "kdeg_err": _linear_error(params, "kdeg_err", "kdeg", "log_kdeg_err"),
                "r2": _numeric(params, "diag:r2"),
                "chisq": _numeric(params, "diag:chisq"),
            }
        )
    return exported


def mut_count_rows(
    conn: sqlite3.Connection,
    *,
    samples: Optional[Sequence[str]] = None,
    all_samples: bool = False,
) -> List[Dict[str, Any]]:
    """Return mutation rates from the latest ingested run for each sample."""
    selected = [str(value) for value in (samples or []) if str(value).strip()]
    if bool(selected) == bool(all_samples):
        raise ValueError("Choose either one or more sample names or --all.")

    clauses: List[str] = []
    params: List[Any] = []
    if selected:
        placeholders = ",".join("?" for _ in selected)
        clauses.append(f"s.sample_name IN ({placeholders})")
        params.extend(selected)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        """
        SELECT s.sample_name AS sample, mn.site, mn.base, v.valtype,
               v.fmod_val AS mutation_rate, v.read_depth
        FROM sequencing_samples s
        JOIN probe_fmod_runs fr ON fr.s_id = s.id
        JOIN (
            SELECT s_id, MAX(id) AS run_id
            FROM probe_fmod_runs
            GROUP BY s_id
        ) latest ON latest.s_id = fr.s_id AND latest.run_id = fr.id
        JOIN probe_fmod_values v ON v.fmod_run_id = fr.id
        JOIN meta_nucleotides mn ON mn.id = v.nt_id
        %s
        ORDER BY s.sample_name, mn.site, v.valtype
        """ % where,
        tuple(params),
    ).fetchall()
    if selected:
        found = {str(row["sample"]) for row in rows}
        missing = sorted(set(selected) - found)
        if missing:
            raise ValueError("No mut_count results found for sample(s): %s." % ", ".join(missing))
    if not rows:
        raise ValueError("No persisted mut_count results found.")
    return [dict(row) for row in rows]


def write_csv(
    rows: Iterable[Dict[str, Any]],
    fieldnames: Sequence[str],
    output: Optional[Path],
    *,
    stdout: Optional[TextIO] = None,
) -> Optional[Path]:
    """Write rows to ``output`` or stdout when output is omitted."""
    handle: TextIO
    should_close = False
    if output is None:
        handle = stdout or sys.stdout
    else:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = output.open("w", encoding="utf-8", newline="")
        should_close = True
    try:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if should_close:
            handle.close()
    return output
