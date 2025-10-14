"""
Stub engine placeholder for external R-based tempgrad fits.
"""

from __future__ import annotations

from typing import List

from nerd.utils.logging import get_logger

from .base import (
    SeriesFitResult,
    TempgradEngine,
    TempgradRequest,
    TempgradResult,
    register_tempgrad_engine,
)

log = get_logger(__name__)


@register_tempgrad_engine
class TempgradRIntegrationEngine(TempgradEngine):
    """Placeholder for future R integration."""

    name = "tempgrad_r_integration"
    version = "0.0.0"

    def run(self, request: TempgradRequest) -> TempgradResult:
        log.info(
            "TempgradRIntegrationEngine invoked for mode=%s but not yet implemented.",
            request.mode,
        )
        results: List[SeriesFitResult] = []
        for series in request.series:
            results.append(
                SeriesFitResult(
                    series_id=series.series_id,
                    params={},
                    diagnostics={
                        "status": "skipped",
                        "reason": "R integration engine not implemented.",
                    },
                )
            )

        metadata = dict(request.metadata)
        metadata.setdefault("mode", request.mode)
        metadata.setdefault("engine_placeholder", True)

        return TempgradResult(
            engine=self.name,
            engine_version=self.version,
            metadata=metadata,
            series_results=tuple(results),
            global_params={},
            artifacts={},
        )
