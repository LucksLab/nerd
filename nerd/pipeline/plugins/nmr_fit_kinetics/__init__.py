"""
NMR kinetic fit plugin registry and bundled implementations.
"""

from __future__ import annotations

from .base import (  # noqa: F401
    FitRequest,
    FitResult,
    NmrFitPlugin,
    available_plugins,
    load_nmr_fit_plugin,
    register_nmr_fit_plugin,
)

# Register built-in plugins on import
from . import lmfit_deg  # noqa: F401
from . import ode_lsq_ntp_add  # noqa: F401

__all__ = [
    "FitRequest",
    "FitResult",
    "NmrFitPlugin",
    "available_plugins",
    "load_nmr_fit_plugin",
    "register_nmr_fit_plugin",
]

