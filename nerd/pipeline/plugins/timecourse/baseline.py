"""
Baseline Python implementation of the three-round timecourse fitting engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from lmfit import Model, Parameters, minimize

from .base import (
    NucleotideSeries,
    PerNucleotideFit,
    RoundResult,
    TimecourseEngine,
    TimecourseRequest,
    TimecourseResult,
    register_timecourse_engine,
)

ROUND_FREE = "round1_free"
ROUND_GLOBAL = "round2_global"
ROUND_CONSTRAINED = "round3_constrained"


def _ensure_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(value, floor))


def _fmod_model(x: np.ndarray, log_kappa: float, log_kdeg: float, log_fmod_0: float) -> np.ndarray:
    kappa = math.exp(log_kappa)
    kdeg = math.exp(log_kdeg)
    fmod0 = math.exp(log_fmod_0)
    return 1.0 - np.exp(-kappa * (1.0 - np.exp(-kdeg * x))) + fmod0


def _fit_single_site(
    timepoints: Sequence[float],
    fmods: Sequence[float],
    *,
    log_kdeg_initial: Optional[float],
    fixed_log_kdeg: Optional[float],
) -> Tuple[Any, Dict[str, Any]]:
    time_arr = np.asarray(list(timepoints), dtype=float)
    fmod_arr = np.asarray(list(fmods), dtype=float)
    if time_arr.size < 3:
        raise ValueError("At least three timepoints are required for a fit.")

    if np.allclose(fmod_arr.max(), 0.0):
        raise ValueError("Non-zero fmod values are required for a fit.")

    model = Model(_fmod_model)

    kappa0 = max(1e-6, -math.log(max(1e-6, 1.0 - min(0.999, float(fmod_arr.max())))))
    fmod_initial = max(1e-6, float(fmod_arr.min()))

    params = model.make_params(
        log_kappa=math.log(kappa0),
        log_fmod_0=math.log(fmod_initial),
    )

    if fixed_log_kdeg is not None:
        params.add("log_kdeg", value=fixed_log_kdeg, vary=False)
    else:
        params.add("log_kdeg", value=log_kdeg_initial if log_kdeg_initial is not None else _safe_log(1e-3), vary=True)

    result = model.fit(fmod_arr, params, x=time_arr)
    diagnostics = {
        "r2": _ensure_float(getattr(result, "rsquared", float("nan"))),
        "chisq": _ensure_float(getattr(result, "chisqr", float("nan"))),
        "reduced_chisq": _ensure_float(getattr(result, "redchi", float("nan"))),

        "time_min": _ensure_float(time_arr.min()),
        "time_max": _ensure_float(time_arr.max()),
        "ndata": int(result.ndata),
        "nfree": int(result.nfree),
        "success": bool(result.success),
    }
    # Include covariance matrix if available from lmfit (var_names order)
    covar = getattr(result, "covar", None)
    if covar is not None:
        try:
            matrix = np.asarray(covar, dtype=float).tolist()
        except Exception:
            matrix = None
        diagnostics["covariance"] = {
            "params": list(getattr(result, "var_names", []) or []),
            "matrix": matrix,
        }
    return result, diagnostics


def _log_params_for_site(result: Any, diagnostics: Mapping[str, Any]) -> Dict[str, Any]:
    def _entry(param_name: str) -> Tuple[float, Optional[float]]:
        param = result.params.get(param_name)
        if param is None:
            return float("nan"), None
        return float(param.value), (float(param.stderr) if param.stderr is not None else None)

    log_kappa, log_kappa_err = _entry("log_kappa")
    log_kdeg, log_kdeg_err = _entry("log_kdeg")
    log_fmod0, log_fmod0_err = _entry("log_fmod_0")
    payload: Dict[str, Any] = {
        "log_kobs": log_kappa,
        "log_kdeg": log_kdeg,
        "log_fmod0": log_fmod0,
        "kobs": math.exp(log_kappa) if math.isfinite(log_kappa) else float("nan"),
        "kdeg": math.exp(log_kdeg) if math.isfinite(log_kdeg) else float("nan"),
        "fmod0": math.exp(log_fmod0) if math.isfinite(log_fmod0) else float("nan"),
    }
    if log_kappa_err is not None:
        payload["log_kobs_err"] = log_kappa_err
    if log_kdeg_err is not None:
        payload["log_kdeg_err"] = log_kdeg_err
    if log_fmod0_err is not None:
        payload["log_fmod0_err"] = log_fmod0_err
    return payload


def _create_global_params(kappa_logs: Sequence[float], kdeg_logs: Sequence[float], fmod_logs: Sequence[float]) -> Parameters:
    params = Parameters()
    if not kdeg_logs:
        log_kdeg = _safe_log(1e-3)
    else:
        log_kdeg = float(np.nanmean(np.asarray(kdeg_logs, dtype=float)))

    for idx, (log_kappa, log_fmod) in enumerate(zip(kappa_logs, fmod_logs), start=1):
        params.add(f"log_kappa_{idx}", value=float(log_kappa))
        params.add(f"log_kdeg_{idx}", value=float(log_kdeg))
        params.add(f"log_fmod0_{idx}", value=float(log_fmod))
        if idx > 1:
            params[f"log_kdeg_{idx}"].expr = "log_kdeg_1"
    return params


def _prepare_global_dataset(time_arrays: Sequence[np.ndarray], fmod_arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not time_arrays or not fmod_arrays:
        raise ValueError("No data provided for global fit.")
    reference = time_arrays[0]
    for candidate in time_arrays[1:]:
        if candidate.shape != reference.shape or not np.allclose(candidate, reference):
            raise ValueError("Global fit requires identical timepoints for all nucleotides.")
    x_data = reference
    y_data = np.vstack(fmod_arrays)
    return x_data, y_data


def _fmod_dataset(params: Parameters, idx: int, x: np.ndarray) -> np.ndarray:
    log_kappa = params[f"log_kappa_{idx}"]
    log_kdeg = params[f"log_kdeg_{idx}"]
    log_fmod = params[f"log_fmod0_{idx}"]
    return _fmod_model(x, log_kappa, log_kdeg, log_fmod)


def _global_objective(params: Parameters, x: np.ndarray, data: np.ndarray) -> np.ndarray:
    nrows, _ = data.shape
    residual = np.zeros_like(data)
    for i in range(nrows):
        residual[i, :] = data[i, :] - _fmod_dataset(params, i + 1, x)
    return residual.flatten()


@dataclass(slots=True)
class _SingleFitRecord:
    params: Dict[str, Any]
    diagnostics: Dict[str, Any]


@register_timecourse_engine
class BaselinePythonEngine(TimecourseEngine):
    """
    Three-round Python engine replicating the legacy workflow.
    """

    name = "python_baseline"
    version = "0.1.0"

    def run(self, request: TimecourseRequest) -> TimecourseResult:
        rounds_requested = [str(r).strip().lower() for r in request.rounds or ()]
        rounds_requested = [self._normalize_round(r) for r in rounds_requested]

        # Prepare per-nucleotide free fits (used by multiple rounds).
        single_fit_cache: Dict[Tuple[int, Optional[str]], _SingleFitRecord] = {}
        round_results: List[RoundResult] = []

        log_kdeg_initial = self._initial_log_kdeg(request)

        # Compute free fits once if needed by downstream rounds.
        if ROUND_FREE in rounds_requested or ROUND_GLOBAL in rounds_requested:
            for series in request.nucleotides:
                try:
                    fit_payload = self._compute_single_fit(
                        series,
                        log_kdeg_initial=log_kdeg_initial,
                        fixed_log_kdeg=None,
                    )
                    single_fit_cache[self._series_key(series)] = fit_payload
                except Exception as exc:  # noqa: BLE001
                    diagnostics = {
                        "status": "failed",
                        "reason": str(exc),
                    }
                    single_fit_cache[self._series_key(series)] = _SingleFitRecord(
                        params={},
                        diagnostics=diagnostics,
                    )

        # Round 1: free fits
        if ROUND_FREE in rounds_requested:
            round_results.append(
                self._round_from_single_fits(
                    ROUND_FREE,
                    single_fit_cache,
                    request.nucleotides,
                    notes=None,
                )
            )

        # Round 2: global fit
        global_log_kdeg: Optional[float] = None
        if ROUND_GLOBAL in rounds_requested:
            global_result = self._run_global_fit(request, single_fit_cache)
            round_results.append(global_result)
            if global_result.status == "completed":
                global_log_kdeg = _ensure_float(global_result.global_params.get("log_kdeg"))

        # Round 3: constrained fits (requires log_kdeg from round 2 or overrides)
        if ROUND_CONSTRAINED in rounds_requested:
            constrained_log_kdeg = self._resolve_constrained_log_kdeg(request, global_log_kdeg)

            if constrained_log_kdeg is None:
                round_results.append(
                    RoundResult(
                        round_id=ROUND_CONSTRAINED,
                        status="skipped",
                        per_nt=tuple(),
                        global_params={},
                        qc_metrics={},
                        notes="No constrained kdeg value supplied or produced by round 2.",
                    )
                )
            else:
                constrained_records: Dict[Tuple[int, Optional[str]], _SingleFitRecord] = {}
                for series in request.nucleotides:
                    try:
                        fit_payload = self._compute_single_fit(
                            series,
                            log_kdeg_initial=constrained_log_kdeg,
                            fixed_log_kdeg=constrained_log_kdeg,
                        )
                        constrained_records[self._series_key(series)] = fit_payload
                    except Exception as exc:  # noqa: BLE001
                        diagnostics = {
                            "status": "failed",
                            "reason": str(exc),
                        }
                        constrained_records[self._series_key(series)] = _SingleFitRecord(
                            params={},
                            diagnostics=diagnostics,
                        )

                round_results.append(
                    self._round_from_single_fits(
                        ROUND_CONSTRAINED,
                        constrained_records,
                        request.nucleotides,
                        notes=None,
                    )
                )

        metadata = dict(request.global_metadata)
        metadata.setdefault("rg_id", request.rg_id)
        metadata["rounds_requested"] = list(rounds_requested)

        return TimecourseResult(
            engine=self.name,
            engine_version=self.version,
            metadata=metadata,
            rounds=tuple(round_results),
            artifacts={},
        )

    @staticmethod
    def _series_key(series: NucleotideSeries) -> Tuple[int, Optional[str]]:
        valtype: Optional[str] = None
        meta = series.metadata or {}
        if isinstance(meta, Mapping):
            raw = meta.get("valtype")
            if raw not in (None, ""):
                valtype = str(raw)
        return (series.nt_id, valtype)

    @staticmethod
    def _normalize_round(value: str) -> str:
        tokens = value.replace("-", "_")
        if tokens in {"round1", "free", "free_fit"}:
            return ROUND_FREE
        if tokens in {"round2", "global"}:
            return ROUND_GLOBAL
        if tokens in {"round3", "constrained"}:
            return ROUND_CONSTRAINED
        return tokens

    def _initial_log_kdeg(self, request: TimecourseRequest) -> Optional[float]:
        meta = request.global_metadata or {}
        options = request.options or {}
        for key in ("log_kdeg_initial", "initial_log_kdeg"):
            if key in meta:
                return _ensure_float(meta[key])
            if key in options:
                return _ensure_float(options[key])
        for key in ("kdeg_initial", "initial_kdeg"):
            if key in meta:
                return _safe_log(_ensure_float(meta[key], default=1e-3))
            if key in options:
                return _safe_log(_ensure_float(options[key], default=1e-3))
        return None

    def _compute_single_fit(
        self,
        series: NucleotideSeries,
        *,
        log_kdeg_initial: Optional[float],
        fixed_log_kdeg: Optional[float],
    ) -> _SingleFitRecord:
        result, diagnostics = _fit_single_site(
            series.timepoints,
            series.fmod_values,
            log_kdeg_initial=log_kdeg_initial,
            fixed_log_kdeg=fixed_log_kdeg,
        )
        param_payload = _log_params_for_site(result, diagnostics)
        extras = dict(series.metadata or {})
        if extras:
            param_payload["metadata"] = extras
        return _SingleFitRecord(params=param_payload, diagnostics=diagnostics)

    def _round_from_single_fits(
        self,
        round_id: str,
        records: Mapping[Tuple[int, Optional[str]], _SingleFitRecord],
        series_list: Sequence[NucleotideSeries],
        *,
        notes: Optional[str],
    ) -> RoundResult:
        fits: List[PerNucleotideFit] = []
        successes = 0
        for series in series_list:
            # Extract valtype from series metadata
            valtype_val: str = ""
            meta = series.metadata or {}
            if isinstance(meta, Mapping):
                raw = meta.get("valtype")
                if raw not in (None, ""):
                    valtype_val = str(raw)
            
            record = records.get(self._series_key(series))
            if record is None:
                fits.append(PerNucleotideFit(nt_id=series.nt_id, valtype=valtype_val, params={}, diagnostics={"status": "missing"}))
                continue
            diagnostics = dict(record.diagnostics)
            status = diagnostics.get("status", "completed")
            if status == "failed":
                fits.append(PerNucleotideFit(nt_id=series.nt_id, valtype=valtype_val, params={}, diagnostics=diagnostics))
                continue
            successes += 1
            params = dict(record.params)
            diagnostics.update({"status": "completed"})
            fits.append(
                PerNucleotideFit(
                    nt_id=series.nt_id,
                    valtype=valtype_val,
                    params=params,
                    diagnostics=diagnostics,
                )
            )

        status = "completed" if successes else "failed"
        qc_metrics = {
            "n_total": len(series_list),
            "n_success": successes,
            "success_rate": successes / len(series_list) if series_list else 0.0,
        }
        return RoundResult(
            round_id=round_id,
            status=status,
            per_nt=tuple(fits),
            global_params={},
            qc_metrics=qc_metrics,
            notes=notes,
        )

    def _run_global_fit(
        self,
        request: TimecourseRequest,
        single_fit_cache: Mapping[Tuple[int, Optional[str]], _SingleFitRecord],
    ) -> RoundResult:
        selected_series = self._filter_series_for_global(request, single_fit_cache)
        if not selected_series:
            return RoundResult(
                round_id=ROUND_GLOBAL,
                status="skipped",
                per_nt=tuple(),
                global_params={},
                qc_metrics={},
                notes="No nucleotides satisfied the selection criteria for global fitting.",
            )

        time_arrays: List[np.ndarray] = []
        fmod_arrays: List[np.ndarray] = []
        log_kappas: List[float] = []
        log_kdegs: List[float] = []
        log_fmods: List[float] = []

        for series in selected_series:
            record = single_fit_cache.get(self._series_key(series))
            if record is None or not record.params:
                continue
            params = record.params
            time_arrays.append(np.asarray(series.timepoints, dtype=float))
            fmod_arrays.append(np.asarray(series.fmod_values, dtype=float))
            log_kappas.append(_ensure_float(params.get("log_kobs")))
            log_kdegs.append(_ensure_float(params.get("log_kdeg")))
            log_fmods.append(_ensure_float(params.get("log_fmod0")))

        if not time_arrays or not fmod_arrays:
            return RoundResult(
                round_id=ROUND_GLOBAL,
                status="failed",
                per_nt=tuple(),
                global_params={},
                qc_metrics={},
                notes="Insufficient per-nucleotide fits available for global fitting.",
            )

        kept_indices = self._select_consistent_timepoints(time_arrays)
        if not kept_indices:
            return RoundResult(
                round_id=ROUND_GLOBAL,
                status="failed",
                per_nt=tuple(),
                global_params={},
                qc_metrics={},
                notes="Global fit failed: No nucleotides shared a consistent set of timepoints.",
            )
        dropped_count = len(time_arrays) - len(kept_indices)

        if dropped_count:
            note_text = f"Excluded {dropped_count} nucleotides from global fit due to mismatched timepoints."
        else:
            note_text = None

        time_arrays = [time_arrays[idx] for idx in kept_indices]
        fmod_arrays = [fmod_arrays[idx] for idx in kept_indices]
        log_kappas = [log_kappas[idx] for idx in kept_indices]
        log_kdegs = [log_kdegs[idx] for idx in kept_indices]
        log_fmods = [log_fmods[idx] for idx in kept_indices]

        try:
            params = _create_global_params(log_kappas, log_kdegs, log_fmods)
            x_data, y_dataset = _prepare_global_dataset(time_arrays, fmod_arrays)
            out: Any = minimize(_global_objective, params, args=(x_data, y_dataset))
        except Exception as exc:  # noqa: BLE001
            return RoundResult(
                round_id=ROUND_GLOBAL,
                status="failed",
                per_nt=tuple(),
                global_params={},
                qc_metrics={},
                notes=f"Global fit failed: {exc}",
            )

        nrows, _ = y_dataset.shape
        predicted = np.zeros_like(y_dataset)
        for idx in range(nrows):
            predicted[idx, :] = _fmod_dataset(out.params, idx + 1, x_data)

        y_true = y_dataset.flatten()
        y_pred = predicted.flatten()
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

        log_kdeg = float(out.params["log_kdeg_1"].value)
        log_kdeg_err: Optional[float] = None
        stderr = out.params["log_kdeg_1"].stderr
        if stderr is not None:
            log_kdeg_err = float(stderr)

        global_params = {
            "log_kdeg": log_kdeg,
            "kdeg": math.exp(log_kdeg),
        }
        if log_kdeg_err is not None:
            global_params["log_kdeg_err"] = log_kdeg_err

        return RoundResult(
            round_id=ROUND_GLOBAL,
            status="completed",
            per_nt=tuple(),
            global_params=global_params,
            qc_metrics=            {
                "chisq": float(getattr(out, "chisqr", float("nan"))),
                "r2": r2,
                "ndata": int(getattr(out, "ndata", len(y_true))),
                "nfree": int(getattr(out, "nvarys", 0)),
                "n_sites": int(nrows),
            },
            notes=note_text,
        )

    def _filter_series_for_global(
        self,
        request: TimecourseRequest,
        single_fit_cache: Mapping[Tuple[int, Optional[str]], _SingleFitRecord],
    ) -> List[NucleotideSeries]:
        options = dict(request.options or {})
        filters = dict(options.get("global_filters") or {})

        bases_filter: Optional[Iterable[str]] = None
        mode = (options.get("global_selection") or "").strip().lower()
        if mode in {"ac_only", "a_c", "ac"}:
            bases_filter = {"A", "C"}
        elif mode in {"acg_only", "acg"}:
            bases_filter = {"A", "C", "G"}

        bases_override = filters.get("bases")
        if bases_override:
            bases_filter = {str(b).upper() for b in bases_override}

        min_r2 = _ensure_float(filters.get("r2_threshold"), default=0.0) if "r2_threshold" in filters else None

        selected: List[NucleotideSeries] = []
        for series in request.nucleotides:
            if bases_filter:
                base = str((series.metadata or {}).get("base", "")).upper()
                if base and base not in bases_filter:
                    continue

            record = single_fit_cache.get(self._series_key(series))
            if record is None or not record.params:
                continue
            diagnostics = record.diagnostics
            if min_r2 is not None:
                r2_val = _ensure_float(diagnostics.get("r2"))
                if not math.isfinite(r2_val) or r2_val < min_r2:
                    continue

            selected.append(series)

        return selected

    @staticmethod
    def _select_consistent_timepoints(time_arrays: Sequence[np.ndarray]) -> List[int]:
        best_indices: List[int] = []
        if not time_arrays:
            return best_indices

        for idx, candidate in enumerate(time_arrays):
            matches: List[int] = []
            for jdx, other in enumerate(time_arrays):
                if candidate.shape != other.shape:
                    continue
                if np.allclose(candidate, other, rtol=1e-5, atol=1e-8):
                    matches.append(jdx)
            if len(matches) > len(best_indices):
                best_indices = matches
        return best_indices

    def _resolve_constrained_log_kdeg(
        self,
        request: TimecourseRequest,
        global_log_kdeg: Optional[float],
    ) -> Optional[float]:
        if global_log_kdeg is not None and math.isfinite(global_log_kdeg):
            return global_log_kdeg

        options = request.options or {}
        meta = request.global_metadata or {}
        for src in (options, meta):
            if "constrained_log_kdeg" in src:
                return _ensure_float(src["constrained_log_kdeg"])
            if "constrained_kdeg" in src:
                return _safe_log(_ensure_float(src["constrained_kdeg"], default=1e-3))
        return None
