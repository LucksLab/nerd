"""
Base interfaces and registry helpers for probe timecourse fitting engines.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


@dataclass(slots=True)
class NucleotideSeries:
    """
    Observed timecourse data for a single nucleotide site.
    """

    nt_id: int
    timepoints: Sequence[float]
    fmod_values: Sequence[float]
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TimecourseRequest:
    """
    Inputs supplied to a timecourse engine.
    """

    rg_id: int
    rounds: Sequence[str]
    nucleotides: Sequence[NucleotideSeries]
    global_metadata: MutableMapping[str, Any] = field(default_factory=dict)
    options: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PerNucleotideFit:
    """
    Parameters and diagnostics produced for a single nucleotide.
    """

    nt_id: int
    params: MutableMapping[str, Any] = field(default_factory=dict)
    diagnostics: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoundResult:
    """
    Normalized payload for one fitting round.
    """

    round_id: str
    status: str
    per_nt: Sequence[PerNucleotideFit] = field(default_factory=tuple)
    global_params: MutableMapping[str, Any] = field(default_factory=dict)
    qc_metrics: MutableMapping[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None


@dataclass(slots=True)
class TimecourseResult:
    """
    Canonical output from a timecourse engine.
    """

    engine: str
    engine_version: str
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    rounds: Sequence[RoundResult] = field(default_factory=tuple)
    artifacts: MutableMapping[str, Any] = field(default_factory=dict)


class TimecourseEngine(abc.ABC):
    """Abstract base class for probe timecourse engines."""

    name: str = "abstract"

    def __init__(self, **kwargs: Any) -> None:
        self._options: Dict[str, Any] = dict(kwargs)

    @abc.abstractmethod
    def run(self, request: TimecourseRequest) -> TimecourseResult:
        """Execute the engine for a reaction group."""
        raise NotImplementedError

    def options(self) -> Dict[str, Any]:
        """Return engine configuration for diagnostics."""
        return dict(self._options)


_REGISTRY: Dict[str, type[TimecourseEngine]] = {}


def register_timecourse_engine(cls: type[TimecourseEngine]) -> type[TimecourseEngine]:
    """Decorator to register an engine class by name."""
    key = getattr(cls, "name", "") or cls.__name__
    norm = key.strip().lower()
    if not norm:
        raise ValueError(f"Cannot register timecourse engine with empty name: {cls}")
    if norm in _REGISTRY and _REGISTRY[norm] is not cls:
        raise ValueError(f"Timecourse engine '{norm}' already registered")
    _REGISTRY[norm] = cls
    return cls


def load_timecourse_engine(name: str, **kwargs: Any) -> TimecourseEngine:
    """Instantiate a registered timecourse engine."""
    norm = (name or "").strip().lower()
    if not norm:
        raise ValueError("Engine name is required.")
    try:
        cls = _REGISTRY[norm]
    except KeyError as exc:  # noqa: B904
        raise ValueError(f"Unknown timecourse engine: {name}") from exc
    return cls(**kwargs)


def available_timecourse_engines() -> Dict[str, type[TimecourseEngine]]:
    """Return the registered engines."""
    return dict(_REGISTRY)
