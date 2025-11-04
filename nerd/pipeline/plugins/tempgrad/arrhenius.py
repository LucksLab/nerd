"""
Baseline Arrhenius fitting engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from lmfit import Model

from .base import (
    SeriesFitResult,
    TempgradEngine,
    TempgradRequest,
    TempgradResult,
    register_tempgrad_engine,
)

R_GAS_CONSTANT = 1.98720425864083  # cal mol^-1 K^-1


def _linear_model(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * x + intercept


@dataclass(slots=True)
class _FitPayload:
    params: Dict[str, Any]
    diagnostics: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)


@register_tempgrad_engine
class ArrheniusPythonEngine(TempgradEngine):
    """Log-linear Arrhenius regression."""

    name = "arrhenius_python"
    version = "0.1.0"

    def run(self, request: TempgradRequest) -> TempgradResult:
        results: List[SeriesFitResult] = []
        for series in request.series:
            fit_payload = self._fit_series(series, request.options)
            results.append(
                SeriesFitResult(
                    series_id=series.series_id,
                    params=fit_payload.params,
                    diagnostics=fit_payload.diagnostics,
                    artifacts=fit_payload.artifacts,
                )
            )
        metadata = dict(request.metadata)
        metadata.setdefault("mode", request.mode)
        return TempgradResult(
            engine=self.name,
            engine_version=self.version,
            metadata=metadata,
            series_results=tuple(results),
            global_params={},
            artifacts={},
        )

    def _fit_series(self, series, options: Dict[str, Any]) -> _FitPayload:
        temps = np.asarray(series.x_values, dtype=float)
        rates = np.asarray(series.y_values, dtype=float)
        if temps.size != rates.size:
            raise ValueError(f"Series {series.series_id} has mismatched temperature/rate lengths.")
        if temps.size < 2:
            raise ValueError(f"Series {series.series_id} requires at least two points for Arrhenius fit.")

        # Temperature handling
        temp_unit = str(options.get("temperature_unit") or series.metadata.get("temperature_unit") or "c").lower()
        if temp_unit in {"c", "celsius"}:
            temps_k = temps + 273.15
        elif temp_unit in {"k", "kelvin"}:
            temps_k = temps
        elif temp_unit in {"f", "fahrenheit"}:
            temps_k = (temps - 32.0) * 5.0 / 9.0 + 273.15
        else:
            raise ValueError(f"Unsupported temperature unit '{temp_unit}' for Arrhenius fit.")

        if np.any(temps_k <= 0):
            raise ValueError(f"Series {series.series_id} contains non-positive absolute temperatures.")

        if np.any(rates <= 0):
            raise ValueError(f"Series {series.series_id} contains non-positive rate constants.")

        inv_t_all = 1.0 / temps_k
        log_k_all = np.log(rates)

        sigma: Optional[np.ndarray] = None
        if series.weights is not None:
            sigma = np.asarray(series.weights, dtype=float)
            if sigma.shape != inv_t_all.shape:
                raise ValueError(f"Series {series.series_id} has mismatched weights.")

        mask = np.isfinite(inv_t_all) & np.isfinite(log_k_all)
        if sigma is not None:
            mask &= np.isfinite(sigma) & (sigma > 0)

        inv_t = inv_t_all[mask]
        log_k = log_k_all[mask]
        if sigma is not None:
            sigma = sigma[mask]

        if inv_t.size < 2 or log_k.size < 2:
            raise ValueError(f"Series {series.series_id} requires at least two valid points for Arrhenius fit.")

        weights = None
        if sigma is not None:
            with np.errstate(divide="ignore"):
                weights = 1.0 / sigma
            weights = np.asarray(weights, dtype=float)
            if not np.any(np.isfinite(weights) & (weights > 0)):
                weights = None

        # Initial guesses via simple regression
        slope_init, intercept_init = np.polyfit(inv_t, log_k, 1)
        model = Model(_linear_model)
        params = model.make_params(slope=slope_init, intercept=intercept_init)

        fit_kwargs: Dict[str, Any] = {"x": inv_t, "weights": weights, "nan_policy": "omit", "scale_covar": False}
        fit_result = model.fit(log_k, params, **fit_kwargs)

        slope = float(fit_result.params["slope"].value)
        slope_err = fit_result.params["slope"].stderr
        intercept = float(fit_result.params["intercept"].value)
        intercept_err = fit_result.params["intercept"].stderr

        log_preds = fit_result.best_fit
        ss_res = float(np.sum((log_k - log_preds) ** 2))
        ss_tot = float(np.sum((log_k - log_k.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
        rmse = math.sqrt(ss_res / float(log_k.size)) if log_k.size else float("nan")

        activation_energy = -slope * R_GAS_CONSTANT
        activation_energy_err = None
        if slope_err is not None:
            activation_energy_err = abs(slope_err) * R_GAS_CONSTANT

        params_payload: Dict[str, Any] = {
            "slope": slope,
            "intercept": intercept,
            "activation_energy_cal_per_mol": activation_energy,
        }
        if slope_err is not None:
            params_payload["slope_err"] = float(slope_err)
        if intercept_err is not None:
            params_payload["intercept_err"] = float(intercept_err)
        if activation_energy_err is not None:
            params_payload["activation_energy_err_cal_per_mol"] = activation_energy_err

        diagnostics = {
            "r2": r2,
            "chisq": float(getattr(fit_result, "chisqr", float("nan"))),
            "ndata": int(fit_result.ndata),
            "nfree": int(fit_result.nfree),
            "rmse_log_k": rmse,
        }
        diagnostics.update(series.metadata or {})

        artifacts: Dict[str, Any] = {}
        covar = getattr(fit_result, "covar", None)
        if covar is not None:
            try:
                artifacts["covariance_matrix"] = np.asarray(covar, dtype=float).tolist()
            except Exception:
                artifacts["covariance_matrix"] = None

        stderr_map: Dict[str, float] = {}
        for name, param in fit_result.params.items():
            stderr_val = getattr(param, "stderr", None)
            if stderr_val is None or not math.isfinite(stderr_val):
                continue
            stderr_map[name] = float(stderr_val)
        if stderr_map:
            artifacts["parameter_stderr"] = stderr_map

        return _FitPayload(params=params_payload, diagnostics=diagnostics, artifacts=artifacts)
