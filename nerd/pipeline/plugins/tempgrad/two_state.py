"""
Two-state melt fitting engine supporting free and global (shared baseline) modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from lmfit import Parameters, minimize

from .base import (
    SeriesFitResult,
    TempgradEngine,
    TempgradRequest,
    TempgradResult,
    register_tempgrad_engine,
)

#R_GAS_CONSTANT = 0.0019872041  # kcal mol^-1 K^-1
R_GAS_CONSTANT = 0.00831446261815324  # kJ mol^-1 K^-1


def _safe_array(values: Sequence[Any], dtype=float) -> np.ndarray:
    return np.asarray(list(values), dtype=dtype)


def melt_fit(inv_t: np.ndarray, a: float, b: float, c: float, d: float, f: float, g: float) -> np.ndarray:
    """
    Evaluate the two-state melt model at inverse temperature (1/K).
    """
    temp_k = 1.0 / inv_t
    Tg = g + 273.15
    K = np.exp((f / R_GAS_CONSTANT) * (1.0 / Tg - inv_t))
    frac_unfolded = 1.0 / (1.0 + K)
    frac_folded = 1.0 - frac_unfolded
    upper = a * inv_t + b
    lower = c * inv_t + d
    return frac_unfolded * upper + frac_folded * lower


def _initial_linear_guess(inv_t: np.ndarray, logk: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Simple linear fits to the top and bottom three points to seed the melt fit.
    Falls back to a global line if fewer than three points are available.
    """
    inv_t = np.asarray(inv_t, dtype=float)
    logk = np.asarray(logk, dtype=float)
    sort_idx = np.argsort(inv_t)
    inv_t_sorted = inv_t[sort_idx]
    logk_sorted = logk[sort_idx]

    if inv_t_sorted.size >= 3:
        x_top = inv_t_sorted[:3]
        y_top = logk_sorted[:3]
        slope_top, intercept_top = np.polyfit(x_top, y_top, 1)

        x_bot = inv_t_sorted[-3:]
        y_bot = logk_sorted[-3:]
        slope_bot, intercept_bot = np.polyfit(x_bot, y_bot, 1)
    else:
        slope_top, intercept_top = np.polyfit(inv_t_sorted, logk_sorted, 1)
        slope_bot, intercept_bot = slope_top, intercept_top

    return float(slope_top), float(intercept_top), float(slope_bot), float(intercept_bot)


def _fit_single_site(inv_t: np.ndarray, logk: np.ndarray, sigma: Optional[np.ndarray], weighted: bool) -> Any:
    slope_top, intercept_top, slope_bot, intercept_bot = _initial_linear_guess(inv_t, logk)

    params = Parameters()
    params.add("a", value=slope_top)
    params.add("b", value=intercept_top)
    params.add("c", value=slope_bot, vary=True)
    params.add("d", value=intercept_bot, vary=True)
    params.add("f", value=-400.0, min=-1e5, max=0.0)
    params.add("g", value=40.0, min=30.0, max=60.0)

    def _residual(p: Parameters) -> np.ndarray:
        y_pred = melt_fit(inv_t, p["a"].value, p["b"].value, p["c"].value, p["d"].value, p["f"].value, p["g"].value)
        resid = logk - y_pred
        if weighted and sigma is not None:
            valid = np.isfinite(sigma) & (sigma > 0)
            out = np.zeros_like(resid)
            out[valid] = resid[valid] / sigma[valid]
            out[~valid] = resid[~valid]
            return out
        return resid

    return minimize(_residual, params, method="least_squares")


def _build_global_params(
    site_data: Mapping[Any, Dict[str, np.ndarray]],
    shared_label: str,
    init_ab: Optional[Tuple[float, float]],
) -> Parameters:
    params = Parameters()
    if init_ab is None:
        params.add(f"a_{shared_label}", value=0.0)
        params.add(f"b_{shared_label}", value=0.0)
    else:
        params.add(f"a_{shared_label}", value=float(init_ab[0]))
        params.add(f"b_{shared_label}", value=float(init_ab[1]))

    top_slopes: List[float] = []
    top_inters: List[float] = []

    for site_id, payload in site_data.items():
        inv_t = payload["inv_t"]
        logk = payload["log_k"]
        slope_top, intercept_top, slope_bot, intercept_bot = _initial_linear_guess(inv_t, logk)
        top_slopes.append(slope_top)
        top_inters.append(intercept_top)

        params.add(f"c_{site_id}", value=slope_bot, vary=True)
        params.add(f"d_{site_id}", value=intercept_bot, vary=True)
        params.add(f"f_{site_id}", value=-400.0, min=-1e5, max=0.0)
        params.add(f"g_{site_id}", value=40.0, min=30.0, max=60.0)
    if init_ab is None:
        if top_slopes:
            params[f"a_{shared_label}"].set(value=float(np.median(top_slopes)))
            params[f"b_{shared_label}"].set(value=float(np.median(top_inters)))
        else:
            params[f"a_{shared_label}"].set(value=0.0)
            params[f"b_{shared_label}"].set(value=0.0)

    return params


def _global_residual(
    p: Parameters,
    site_data: Mapping[Any, Dict[str, np.ndarray]],
    shared_label: str,
    weighted: bool,
) -> np.ndarray:
    residuals: List[np.ndarray] = []
    a = p[f"a_{shared_label}"].value
    b = p[f"b_{shared_label}"].value
    for site_id, payload in site_data.items():
        c = p[f"c_{site_id}"].value
        d = p[f"d_{site_id}"].value
        f_val = p[f"f_{site_id}"].value
        g_val = p[f"g_{site_id}"].value

        inv_t = payload["inv_t"]
        logk = payload["log_k"]
        sigma = payload.get("sigma")

        y_pred = melt_fit(inv_t, a, b, c, d, f_val, g_val)
        resid = logk - y_pred
        if weighted and sigma is not None:
            valid = np.isfinite(sigma) & (sigma > 0)
            out = np.zeros_like(resid)
            out[valid] = resid[valid] / sigma[valid]
            out[~valid] = resid[~valid]
            residuals.append(out)
        else:
            residuals.append(resid)
    return np.concatenate(residuals)


def _fit_global(
    site_data: Mapping[Any, Dict[str, np.ndarray]],
    shared_label: str,
    init_ab: Optional[Tuple[float, float]],
    weighted: bool,
) -> Any:
    params = _build_global_params(site_data, shared_label, init_ab)
    return minimize(
        _global_residual,
        params,
        args=(site_data, shared_label, weighted),
        method="least_squares",
    )


@dataclass(slots=True)
class _FitBundle:
    params: Dict[str, float]
    diagnostics: Dict[str, Any]


def _series_payload(series) -> Dict[str, np.ndarray]:
    temps_c = _safe_array(series.x_values, dtype=float)
    logk = _safe_array(series.y_values, dtype=float)
    sigma = None
    meta_errs = series.metadata.get("log_kobs_err")
    if meta_errs is not None:
        sigma = _safe_array(meta_errs, dtype=float)
    else:
        src_data = series.metadata.get("src_kobs_data")
        if isinstance(src_data, Sequence):
            sigma_values: List[float] = []
            for record in src_data:
                err_val = None
                if isinstance(record, Mapping):
                    err_val = record.get("log_kobs2_err")
                if err_val in (None, ""):
                    sigma_values.append(float("nan"))
                else:
                    try:
                        sigma_values.append(float(err_val))
                    except (TypeError, ValueError):
                        sigma_values.append(float("nan"))
            if sigma_values:
                sigma = _safe_array(sigma_values, dtype=float)
    return {
        "inv_t": 1.0 / (temps_c + 273.15),
        "log_k": logk,
        "sigma": sigma,
        "temps_c": temps_c,
    }


def _param_stderr(result: Any, param_name: str) -> Optional[float]:
    param_obj = result.params.get(param_name)
    if param_obj is None:
        return None
    stderr = getattr(param_obj, "stderr", None)
    if stderr is not None and np.isfinite(stderr):
        return float(stderr)
    covar = getattr(result, "covar", None)
    var_names = getattr(result, "var_names", None)
    if covar is None or not var_names:
        return None
    try:
        idx = list(var_names).index(param_name)
    except ValueError:
        return None
    try:
        var = covar[idx][idx]
    except Exception:
        return None
    if var is None or not np.isfinite(var) or var < 0:
        return None
    return float(np.sqrt(var))


def _metadata_snapshot(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not metadata:
        return {}
    snapshot = dict(metadata)
    src_records = metadata.get("src_kobs_data")
    if isinstance(src_records, Sequence) and not isinstance(src_records, (str, bytes)):
        normalized_records: List[Any] = []
        for item in src_records:
            if isinstance(item, Mapping):
                normalized_records.append(dict(item))
            else:
                normalized_records.append(item)
        snapshot["src_kobs_data"] = normalized_records
    return snapshot

def _result_payload(
    temps_c: np.ndarray,
    inv_t: np.ndarray,
    logk: np.ndarray,
    sigma: Optional[np.ndarray],
    params: Mapping[str, float],
) -> Dict[str, Any]:
    preds = melt_fit(inv_t, params["a"], params["b"], params["c"], params["d"], params["f"], params["g"])
    residual = logk - preds
    if sigma is not None and np.any(np.isfinite(sigma) & (sigma > 0)):
        weights = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma, 1.0)
        chisq = float(np.sum((residual * weights) ** 2))
    else:
        chisq = float(np.sum(residual ** 2))
    diagnostics = {
        "chisq": chisq,
        "ndata": int(inv_t.size),
        "nfree": max(int(inv_t.size) - 6, 0),
    }
    return diagnostics


@register_tempgrad_engine
class TwoStateMeltEngine(TempgradEngine):
    """Two-state melt fitting with free and global Kobs modes."""

    name = "two_state_melt"
    version = "0.1.0"

    def run(self, request: TempgradRequest) -> TempgradResult:
        model = str(request.options.get("model") or "free_kadd").strip().lower()
        weighted = bool(request.options.get("weighted", False))
        shared_label = str(request.options.get("shared_label") or "global")

        fit_name = request.metadata.get("fit_name")

        if model not in {"free_kadd", "global_kadd"}:
            raise ValueError("two_state_melt.model must be 'free_kadd' or 'global_kadd'.")

        if model == "free_kadd":
            series_results = self._run_free_mode(request.series, weighted)
            global_params: Dict[str, Any] = {}
        else:
            series_results, global_params = self._run_global_mode(
                request.series,
                weighted=weighted,
                shared_label=shared_label,
                fit_name=fit_name,
            )

        metadata = dict(request.metadata)
        metadata.setdefault("model", model)

        return TempgradResult(
            engine=self.name,
            engine_version=self.version,
            metadata=metadata,
            series_results=tuple(series_results),
            global_params=global_params,
            artifacts={},
        )

    def _run_free_mode(self, series_list, weighted: bool) -> List[SeriesFitResult]:
        results: List[SeriesFitResult] = []
        for series in series_list:
            payload = _series_payload(series)
            result = _fit_single_site(payload["inv_t"], payload["log_k"], payload["sigma"], weighted)

            params: Dict[str, float] = {}
            for name in ("a", "b", "c", "d", "f", "g"):
                param_obj = result.params.get(name)
                if param_obj is None:
                    continue
                params[name] = float(param_obj.value)
                stderr = _param_stderr(result, name)
                params[f"{name}_err"] = None if stderr is None else float(stderr)
            diagnostics = _result_payload(
                payload["temps_c"],
                payload["inv_t"],
                payload["log_k"],
                payload["sigma"],
                params,
            )
            diagnostics.update(
                {
                    "success": bool(result.success),
                    "message": str(result.message),
                    "redchi": float(getattr(result, "redchi", np.nan)),
                }
            )
            diagnostics.update(_metadata_snapshot(series.metadata))
            results.append(
                SeriesFitResult(
                    series_id=series.series_id,
                    params=params,
                    diagnostics=diagnostics,
                )
            )
        return results

    def _run_global_mode(
        self,
        series_list,
        weighted: bool,
        shared_label: str,
        fit_name: Optional[str] = None,
    ) -> Tuple[List[SeriesFitResult], Dict[str, Any]]:
        # Group series by group identifier; default to metadata group_label or series_id.
        group_map: Dict[str, List[int]] = {}
        for idx, series in enumerate(series_list):
            group_label = series.metadata.get("group_label") or series.metadata.get("construct_name") or series.series_id
            group_map.setdefault(str(group_label), []).append(idx)

        series_results: Dict[int, SeriesFitResult] = {}
        global_params: Dict[str, Any] = {}

        for group_label, indices in group_map.items():
            site_data: Dict[str, Dict[str, np.ndarray]] = {}
            for idx in indices:
                series = series_list[idx]
                payload = _series_payload(series)
                site_id = str(series.metadata.get("nt_id") or series.series_id)
                site_data[site_id] = payload

            init_ab = None
            if fit_name:
                ab_meta = series_list[indices[0]].metadata.get("fit_seed_ab")
                if isinstance(ab_meta, (list, tuple)) and len(ab_meta) == 2:
                    init_ab = (float(ab_meta[0]), float(ab_meta[1]))

            result = _fit_global(site_data, shared_label, init_ab, weighted)

            a_val = float(result.params[f"a_{shared_label}"].value)
            b_val = float(result.params[f"b_{shared_label}"].value)
            global_params[group_label] = {"a": a_val, "b": b_val}

            for idx in indices:
                series = series_list[idx]
                site_key = str(series.metadata.get("nt_id") or series.series_id)
                payload = site_data[site_key]

                params: Dict[str, float] = {}
                param_specs = {
                    "a": f"a_{shared_label}",
                    "b": f"b_{shared_label}",
                    "c": f"c_{site_key}",
                    "d": f"d_{site_key}",
                    "f": f"f_{site_key}",
                    "g": f"g_{site_key}",
                }
                for out_name, param_name in param_specs.items():
                    param_obj = result.params.get(param_name)
                    if param_obj is None:
                        continue
                    params[out_name] = float(param_obj.value)
                    stderr = _param_stderr(result, param_name)
                    params[f"{out_name}_err"] = None if stderr is None else float(stderr)
                diagnostics = _result_payload(
                    payload["temps_c"],
                    payload["inv_t"],
                    payload["log_k"],
                    payload["sigma"],
                    params,
                )
                diagnostics.update(
                    {
                        "success": bool(result.success),
                        "message": str(result.message),
                        "redchi": float(getattr(result, "redchi", np.nan)),
                        "group": group_label,
                    }
                )
                diagnostics.update(_metadata_snapshot(series.metadata))
                series_results[idx] = SeriesFitResult(
                    series_id=series.series_id,
                    params=params,
                    diagnostics=diagnostics,
                )

        ordered_results = [series_results[idx] for idx in range(len(series_list))]
        return ordered_results, global_params
