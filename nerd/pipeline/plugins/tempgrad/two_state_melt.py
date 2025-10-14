"""
Baseline two-state melt fitting engine.
"""

from __future__ import annotations

from dataclasses import dataclass
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

R_GAS_CONSTANT = 0.00198720425864083  # kcal mol^-1 K^-1


def _two_state_model(x: np.ndarray, a: float, b: float, c: float, d: float, f: float, g: float) -> np.ndarray:
    """
    Two-state melt response model operating on inverse temperature (1/K).
    """
    temp_k = 1.0 / x
    equilibrium = np.exp((f / R_GAS_CONSTANT) * (1.0 / (g + 273.15) - 1.0 / temp_k))
    q_total = 1.0 + equilibrium
    frac_unfolded = 1.0 / q_total
    frac_folded = equilibrium / q_total
    baseline_folded = a * x + b
    baseline_unfolded = c * x + d
    return frac_unfolded * baseline_unfolded + frac_folded * baseline_folded


@dataclass(slots=True)
class _FitPayload:
    params: Dict[str, Any]
    diagnostics: Dict[str, Any]


@register_tempgrad_engine
class TwoStateMeltEngine(TempgradEngine):
    """Two-state melt curve fitter."""

    name = "two_state_melt"
    version = "0.1.0"

    def run(self, request: TempgradRequest) -> TempgradResult:
        results: List[SeriesFitResult] = []
        for series in request.series:
            payload = self._fit_series(series, request.options)
            results.append(
                SeriesFitResult(
                    series_id=series.series_id,
                    params=payload.params,
                    diagnostics=payload.diagnostics,
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
        responses = np.asarray(series.y_values, dtype=float)
        if temps.size != responses.size:
            raise ValueError(f"Series {series.series_id} has mismatched temperature/response lengths.")
        if temps.size < 4:
            raise ValueError(f"Series {series.series_id} requires at least four points for melt fitting.")

        temp_unit = str(options.get("temperature_unit") or series.metadata.get("temperature_unit") or "c").lower()
        if temp_unit in {"c", "celsius"}:
            temps_k = temps + 273.15
        elif temp_unit in {"k", "kelvin"}:
            temps_k = temps
        elif temp_unit in {"f", "fahrenheit"}:
            temps_k = (temps - 32.0) * 5.0 / 9.0 + 273.15
        else:
            raise ValueError(f"Unsupported temperature unit '{temp_unit}' for melt fitting.")

        if np.any(temps_k <= 0):
            raise ValueError(f"Series {series.series_id} contains non-positive absolute temperatures.")

        inv_t = 1.0 / temps_k

        # Initial guesses: linear fits on low/high temperature segments.
        order = np.argsort(inv_t)
        inv_t_sorted = inv_t[order]
        resp_sorted = responses[order]

        window = max(3, min(5, len(inv_t_sorted) // 2))
        top_slice = slice(0, window)
        bottom_slice = slice(-window, None)

        def _linear_guess(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float]:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            return slope, intercept

        slope_top, intercept_top = _linear_guess(inv_t_sorted[top_slice], resp_sorted[top_slice])
        slope_bottom, intercept_bottom = _linear_guess(inv_t_sorted[bottom_slice], resp_sorted[bottom_slice])

        melt_midpoint = float(series.metadata.get("initial_tm_c", np.median(temps)))

        model = Model(_two_state_model)
        params = model.make_params(
            a=slope_bottom,
            b=intercept_bottom,
            c=slope_top,
            d=intercept_top,
            f=-5000.0,  # kcal/mol default
            g=melt_midpoint,
        )
        params["g"].vary = True

        weights = None
        if series.weights is not None:
            weights = np.asarray(series.weights, dtype=float)
            if weights.shape != inv_t.shape:
                raise ValueError(f"Series {series.series_id} has mismatched weights.")

        fit_result = model.fit(resp_sorted, params, x=inv_t_sorted, weights=weights, nan_policy="omit")

        diagnostics = {
            "chisq": float(getattr(fit_result, "chisqr", float("nan"))),
            "ndata": int(fit_result.ndata),
            "nfree": int(fit_result.nfree),
        }

        preds = fit_result.best_fit
        ss_res = float(np.sum((resp_sorted - preds) ** 2))
        ss_tot = float(np.sum((resp_sorted - resp_sorted.mean()) ** 2))
        diagnostics["r2"] = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

        payload_params: Dict[str, Any] = {}
        for name in ["a", "b", "c", "d", "f", "g"]:
            param = fit_result.params.get(name)
            if param is None:
                continue
            payload_params[name] = float(param.value)
            if param.stderr is not None:
                payload_params[f"{name}_err"] = float(param.stderr)

        diagnostics.update(series.metadata or {})

        return _FitPayload(params=payload_params, diagnostics=diagnostics)
