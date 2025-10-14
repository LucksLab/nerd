"""
Compatibility helpers for invoking NMR fitting plugins.

The actual fitting implementations live under `nerd.pipeline.plugins.nmr_fit_kinetics`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from nerd.pipeline.plugins.nmr_fit_kinetics import FitRequest, FitResult, load_nmr_fit_plugin


def run_fit(
    plugin_name: str,
    *,
    reaction_id: Optional[int] = None,
    files: Mapping[str, Path | str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> FitResult:
    """
    Execute an NMR fit plugin by name.

    Args:
        plugin_name: Registered plugin identifier (e.g., 'lmfit_kdeg').
        reaction_id: Optional NMR reaction primary key.
        files: Mapping of role → staged path(s) required by the plugin.
        metadata: Arbitrary reaction metadata provided to the plugin.
        params: Plugin-specific overrides/options.
    """

    normalized_files = {k: Path(v) if not isinstance(v, Path) else v for k, v in (files or {}).items()}
    plugin = load_nmr_fit_plugin(plugin_name)
    request = FitRequest(
        reaction_id=reaction_id,
        files=normalized_files,
        metadata=metadata or {},
        params=params or {},
    )
    return plugin.fit(request)


def fit_exp_decay(time_series_path: Path | str) -> FitResult:
    """
    Backwards-compatible helper: fit degradation kinetics using the default plugin.
    """
    return run_fit("lmfit_deg", files={"decay_trace": time_series_path})


def fit_ode_adduction(
    *,
    peak_trace: Path | str,
    dms_trace: Path | str,
    ntp_conc: float,
    metadata: Optional[Mapping[str, Any]] = None,
) -> FitResult:
    """Backward-compatible helper for the adduction ODE fit."""
    meta = dict(metadata or {})
    meta.setdefault("ntp_conc", ntp_conc)
    return run_fit(
        "ode_lsq_ntp_add",
        files={"peak_trace": peak_trace, "dms_trace": dms_trace},
        metadata=meta,
    )
