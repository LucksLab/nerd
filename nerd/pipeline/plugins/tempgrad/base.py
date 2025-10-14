"""
Base interfaces and registry helpers for temperature-gradient fitting engines.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


@dataclass(slots=True)
class TempgradSeries:
    """
    Input data for a single temperature-dependent fit.
    """

    series_id: str
    x_values: Sequence[float]
    y_values: Sequence[float]
    weights: Optional[Sequence[float]] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SeriesFitResult:
    """
    Output payload for a fitted series.
    """

    series_id: str
    params: MutableMapping[str, Any] = field(default_factory=dict)
    diagnostics: MutableMapping[str, Any] = field(default_factory=dict)
    artifacts: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TempgradRequest:
    """
    Request delivered to a tempgrad engine.
    """

    mode: str
    series: Sequence[TempgradSeries]
    options: MutableMapping[str, Any] = field(default_factory=dict)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TempgradResult:
    """
    Canonical engine output.
    """

    engine: str
    engine_version: str
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    series_results: Sequence[SeriesFitResult] = field(default_factory=tuple)
    global_params: MutableMapping[str, Any] = field(default_factory=dict)
    artifacts: MutableMapping[str, Any] = field(default_factory=dict)


class TempgradEngine(abc.ABC):
    """Abstract base class for temperature-gradient engines."""

    name: str = "abstract"
    version: str = "0.0.0"

    def __init__(self, **kwargs: Any) -> None:
        self._options = dict(kwargs)

    @abc.abstractmethod
    def run(self, request: TempgradRequest) -> TempgradResult:
        """Execute the engine and return the normalized result."""
        raise NotImplementedError

    def options(self) -> Dict[str, Any]:
        """Return engine configuration for diagnostics."""
        return dict(self._options)


_REGISTRY: Dict[str, type[TempgradEngine]] = {}


def register_tempgrad_engine(cls: type[TempgradEngine]) -> type[TempgradEngine]:
    """Decorator to register a tempgrad engine class by name."""
    key = getattr(cls, "name", "") or cls.__name__
    norm = key.strip().lower()
    if not norm:
        raise ValueError(f"Cannot register tempgrad engine with empty name: {cls}")
    if norm in _REGISTRY and _REGISTRY[norm] is not cls:
        raise ValueError(f"Tempgrad engine '{norm}' already registered")
    _REGISTRY[norm] = cls
    return cls


def load_tempgrad_engine(name: str, **kwargs: Any) -> TempgradEngine:
    """Instantiate a registered tempgrad engine."""
    norm = (name or "").strip().lower()
    if not norm:
        raise ValueError("Engine name is required.")
    try:
        cls = _REGISTRY[norm]
    except KeyError as exc:
        raise ValueError(f"Unknown tempgrad engine: {name}") from exc
    return cls(**kwargs)


def available_tempgrad_engines() -> Mapping[str, type[TempgradEngine]]:
    """Return the registered tempgrad engines."""
    return dict(_REGISTRY)
