from __future__ import annotations

from typing import Optional

from .base import MutCountPlugin
from .shapemapper import ShapeMapperPlugin


def load_mutcount_plugin(name: str, bin_path: Optional[str] = None, version: Optional[str] = None) -> MutCountPlugin:
    key = (name or "").strip().lower()
    if key in {"shapemapper", "shape_mapper", "shape-map", "shape"}:
        return ShapeMapperPlugin(bin_path=bin_path, version=version)
    raise ValueError(f"Unknown mut_count plugin: {name}")
