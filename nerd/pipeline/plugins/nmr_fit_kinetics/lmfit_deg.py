"""
Degradation (kdeg) fitting plugin using an lmfit exponential decay model.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from lmfit.models import ExponentialModel

from .base import FitRequest, FitResult, NmrFitPlugin, register_nmr_fit_plugin


def _load_trace(path: str | bytes) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns or "peak_integral" not in df.columns:
        raise ValueError(f"Trace '{path}' must contain 'time' and 'peak_integral' columns.")
    return df


@register_nmr_fit_plugin
class LmfitDegPlugin(NmrFitPlugin):
    name = "lmfit_deg"

    def fit(self, request: FitRequest) -> FitResult:
        csv_path = self._resolve_trace_path(request)
        df = _load_trace(csv_path)

        model = ExponentialModel()
        params = model.guess(df["peak_integral"], x=df["time"])
        result = model.fit(df["peak_integral"], params, x=df["time"])

        decay_param = result.params.get("decay")
        if decay_param is None or decay_param.value == 0:
            raise ValueError("Fit did not yield a valid decay parameter.")

        tau = float(decay_param.value)
        k_value = 1.0 / tau
        k_err = None
        if decay_param.stderr is not None:
            k_err = float(decay_param.stderr) / (tau ** 2)

        diagnostics: Dict[str, Optional[float]] = {
            "tau": tau,
            "amplitude": float(result.params.get("amplitude", 0.0).value) if "amplitude" in result.params else None,
            "decay_stderr": float(decay_param.stderr) if decay_param.stderr is not None else None,
            "aic": getattr(result, "aic", None),
            "bic": getattr(result, "bic", None),
        }

        return FitResult(
            k_value=k_value,
            k_error=k_err,
            r2=getattr(result, "rsquared", None),
            chisq=getattr(result, "chisqr", None),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _resolve_trace_path(request: FitRequest) -> str:
        for key in ("decay_trace", "trace", "dms_trace"):
            candidate = request.files.get(key)
            if candidate:
                return str(candidate)
        raise ValueError("lmfit_deg requires a 'decay_trace' file in the FitRequest.")

