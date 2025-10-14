"""
Base interfaces and registry helpers for NMR kinetic fit plugins.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


@dataclass(slots=True)
class FitRequest:
    """Encapsulates the inputs supplied to an NMR fit plugin."""

    reaction_id: Optional[int]
    files: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FitResult:
    """Normalized output from an NMR fit plugin."""

    k_value: float
    k_error: Optional[float]
    r2: Optional[float]
    chisq: Optional[float]
    diagnostics: MutableMapping[str, Any] = field(default_factory=dict)


class NmrFitPlugin(abc.ABC):
    """Abstract base class for NMR kinetic fit plugins."""

    name: str = "abstract"

    def __init__(self, **kwargs: Any) -> None:
        self._options: Dict[str, Any] = dict(kwargs)

    @abc.abstractmethod
    def fit(self, request: FitRequest) -> FitResult:
        """Execute the kinetic fit and return the normalized result."""
        raise NotImplementedError

    def options(self) -> Dict[str, Any]:
        """Return plugin configuration for diagnostics/logging."""
        return dict(self._options)


_REGISTRY: Dict[str, type[NmrFitPlugin]] = {}


def register_nmr_fit_plugin(cls: type[NmrFitPlugin]) -> type[NmrFitPlugin]:
    """Decorator to register a plugin class by name."""
    key = getattr(cls, "name", "") or cls.__name__
    norm = key.strip().lower()
    if not norm:
        raise ValueError(f"Cannot register plugin with empty name: {cls}")
    if norm in _REGISTRY and _REGISTRY[norm] is not cls:
        raise ValueError(f"NMR fit plugin '{norm}' already registered")
    _REGISTRY[norm] = cls
    return cls


def load_nmr_fit_plugin(name: str, **kwargs: Any) -> NmrFitPlugin:
    """Instantiate a plugin from the registry."""
    norm = (name or "").strip().lower()
    if not norm:
        raise ValueError("Plugin name is required.")
    try:
        cls = _REGISTRY[norm]
    except KeyError as exc:
        raise ValueError(f"Unknown NMR fit plugin: {name}") from exc
    return cls(**kwargs)


def available_plugins() -> Dict[str, type[NmrFitPlugin]]:
    """Return the registered plugins (name → class)."""
    return dict(_REGISTRY)

