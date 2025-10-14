"""
Stub engine placeholder for future R-based timecourse fitting.
"""

from __future__ import annotations

from typing import Any, List

from nerd.utils.logging import get_logger

from .base import RoundResult, TimecourseEngine, TimecourseRequest, TimecourseResult, register_timecourse_engine

log = get_logger(__name__)


@register_timecourse_engine
class RIntegrationEngine(TimecourseEngine):
    """
    Placeholder engine that will eventually call out to an R implementation.
    """

    name = "r_integration"
    version = "0.0.0"

    def run(self, request: TimecourseRequest) -> TimecourseResult:
        log.info("RIntegrationEngine invoked for rg_id=%s but not yet implemented.", request.rg_id)

        rounds: List[RoundResult] = []
        for round_id in request.rounds or []:
            rounds.append(
                RoundResult(
                    round_id=str(round_id),
                    status="skipped",
                    per_nt=tuple(),
                    global_params={},
                    qc_metrics={},
                    notes="R integration engine not implemented yet.",
                )
            )

        metadata = dict(request.global_metadata or {})
        metadata.setdefault("rg_id", request.rg_id)
        metadata.setdefault("engine_placeholder", True)

        return TimecourseResult(
            engine=self.name,
            engine_version=self.version,
            metadata=metadata,
            rounds=tuple(rounds),
            artifacts={},
        )
