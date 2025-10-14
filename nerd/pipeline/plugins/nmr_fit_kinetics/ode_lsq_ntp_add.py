"""
Adduction (kadd) kinetic fit using an lmfit-driven least-squares ODE solver.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from lmfit import Parameters, minimize
from scipy.integrate import solve_ivp

from .base import FitRequest, FitResult, NmrFitPlugin, register_nmr_fit_plugin


def _load_trace(path: str | bytes, *, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ("time", "peak"):
        if column not in df.columns:
            raise ValueError(f"{name} trace '{path}' must contain '{column}' column.")
    return df


@register_nmr_fit_plugin
class OdeLsqNtpAddPlugin(NmrFitPlugin):
    name = "ode_lsq_ntp_add"

    def fit(self, request: FitRequest) -> FitResult:
        peak_trace, dms_trace = self._resolve_trace_paths(request)
        ntp_conc = self._resolve_ntp_conc(request)

        peak_df = _load_trace(peak_trace, name="NTP reporter")
        dms_df = _load_trace(dms_trace, name="DMS")

        if len(dms_df) < len(peak_df):
            t_obs = list(dms_df["time"].values)
        else:
            t_obs = list(peak_df["time"].values)
        peak_df = peak_df[peak_df["time"].isin(t_obs)]
        dms_df = dms_df[dms_df["time"].isin(t_obs)]

        rnaconc = float(ntp_conc)
        dmsconc = float(request.metadata.get("dms_conc", 0.01564))

        u_data = (peak_df["peak"].values.astype(float) * rnaconc).tolist()
        s_data = (dms_df["peak"].values.astype(float) * dmsconc).tolist()
        m_data = ((1.0 - peak_df["peak"].values.astype(float)) * rnaconc).tolist()

        y_obs = np.array([u_data, s_data, m_data])
        y0 = [rnaconc, dmsconc, 0.0]

        def system(t, y, k_add, k_deg):
            U, S, M = y
            dUdt = -k_add * U * S
            dSdt = -k_add * U * S - k_deg * S
            dMdt = k_add * U * S
            return [dUdt, dSdt, dMdt]

        def solve_system(params, t, data):
            k_add = params["k_add"]
            k_deg = params["k_deg"]
            s_factor = params["S_factor"]

            adjusted = np.array(data, copy=True)
            adjusted[1] = adjusted[1] * s_factor

            fill_in = np.linspace(0, t[0], 51)[:-1]
            full_t = np.concatenate((fill_in, t))
            sol = solve_ivp(system, [0, t[-1]], y0, args=(k_add, k_deg), t_eval=full_t, vectorized=True)
            sol_observed = sol.y[:, 50:]
            return (sol_observed.ravel() - adjusted.ravel())

        t = np.array(t_obs, dtype=float)

        params = Parameters()
        params.add("k_add", value=float(self._options.get("k_add_init", 3e-3)), min=0.0)
        params.add("k_deg", value=float(self._options.get("k_deg_init", 1e-3)), min=0.0)
        params.add("S_factor", value=float(self._options.get("S_factor", 1.0)), vary=False)

        out = minimize(solve_system, params, args=(t, y_obs))

        k_add_param = out.params.get("k_add")
        if k_add_param is None:
            raise ValueError("Adduction fit missing k_add parameter.")

        k_value = float(k_add_param.value)
        k_err = None
        if k_add_param.stderr is not None and k_add_param.value:
            k_err = float(k_add_param.stderr) / (k_add_param.value ** 2)

        ss_res = float(np.sum(out.residual ** 2))
        ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot else None
        chi_sq = float(ss_res / (len(t) - len(out.params))) if len(t) > len(out.params) else None

        diagnostics: Dict[str, Optional[float]] = {
            "k_deg": float(out.params.get("k_deg").value) if "k_deg" in out.params else None,
            "S_factor": float(out.params.get("S_factor").value) if "S_factor" in out.params else None,
            "success": bool(out.success),
            "message": out.message,
        }

        return FitResult(
            k_value=k_value,
            k_error=k_err,
            r2=r_squared,
            chisq=chi_sq,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _resolve_trace_paths(request: FitRequest) -> tuple[str, str]:
        peak = request.files.get("peak_trace") or request.files.get("ntp_trace")
        dms = request.files.get("dms_trace")
        if not peak or not dms:
            raise ValueError("ode_lsq_ntp_add requires both 'peak_trace' and 'dms_trace' files.")
        return str(peak), str(dms)

    @staticmethod
    def _resolve_ntp_conc(request: FitRequest) -> float:
        for source in (request.metadata, request.params):
            if not source:
                continue
            if "ntp_conc" in source:
                return float(source["ntp_conc"])
            if "rna_conc" in source:
                return float(source["rna_conc"])
        raise ValueError("ode_lsq_ntp_add requires 'ntp_conc' (or 'rna_conc') in metadata/params.")

