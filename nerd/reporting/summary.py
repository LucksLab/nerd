"""Stable, serializable task result contract and terminal renderers.

The schema is intentionally composed only of JSON-native values.  New optional
fields may be added in future schema versions; existing field meanings do not
change within a schema version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


SCHEMA_VERSION = "1.0"


def _json_value(value: Any) -> Any:
    """Convert supported values without leaking rows or exception objects."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError("Task summaries cannot serialize %s" % type(value).__name__)


@dataclass
class TaskIssue:
    code: str
    message: str
    item: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class ArtifactReference:
    kind: str
    path: str
    label: Optional[str] = None
    exists: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class TaskSummary:
    """Versioned public result for a workflow or task lifecycle operation."""

    status: str
    workflow: str
    schema_version: str = SCHEMA_VERSION
    task_id: Optional[int] = None
    source_task_id: Optional[int] = None
    label: Optional[str] = None
    plugin: Optional[str] = None
    engine: Optional[str] = None
    version: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    timings: Dict[str, Optional[float]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[TaskIssue] = field(default_factory=list)
    failures: List[TaskIssue] = field(default_factory=list)
    artifacts: List[ArtifactReference] = field(default_factory=list)
    log_path: Optional[str] = None
    database_path: Optional[str] = None
    next_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # Explicit order is part of the v1 human-readable JSON contract.
        return _json_value({
            "schema_version": self.schema_version,
            "status": self.status,
            "workflow": self.workflow,
            "task_id": self.task_id,
            "source_task_id": self.source_task_id,
            "label": self.label,
            "plugin": self.plugin,
            "engine": self.engine,
            "version": self.version,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "timings": dict(sorted(self.timings.items())),
            "counts": dict(sorted(self.counts.items())),
            "metrics": dict(sorted(self.metrics.items())),
            "warnings": [issue.to_dict() for issue in self.warnings],
            "failures": [issue.to_dict() for issue in self.failures],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "log_path": self.log_path,
            "database_path": self.database_path,
            "next_actions": list(self.next_actions),
        })


def render_json(summary: TaskSummary, *, indent: Optional[int] = 2) -> str:
    """Render deterministic JSON (stable top-level order and sorted map keys)."""
    return json.dumps(summary.to_dict(), indent=indent, ensure_ascii=False) + "\n"


def render_human(summary: TaskSummary) -> str:
    """Render a concise, color-independent terminal summary."""
    heading = "%s: %s" % (summary.workflow, summary.status)
    identity = []
    if summary.task_id is not None:
        identity.append("task %s" % summary.task_id)
    if summary.label:
        identity.append(summary.label)
    if identity:
        heading += " (%s)" % ", ".join(identity)
    lines = [heading]
    tool = summary.plugin or summary.engine
    if tool:
        lines.append("tool: %s%s" % (tool, " %s" % summary.version if summary.version else ""))
    if summary.counts:
        lines.append("counts: " + ", ".join(
            "%s=%s" % item for item in sorted(summary.counts.items())
        ))
    if summary.metrics:
        lines.append("metrics: " + ", ".join(
            "%s=%s" % item for item in sorted(summary.metrics.items())
        ))
    for issue in summary.warnings:
        lines.append("warning [%s]: %s" % (issue.code, issue.message))
    for issue in summary.failures:
        lines.append("failure [%s]: %s" % (issue.code, issue.message))
    if summary.duration_seconds is not None:
        lines.append("elapsed: %.3fs" % summary.duration_seconds)
    if summary.log_path:
        lines.append("log: %s" % summary.log_path)
    if summary.database_path:
        lines.append("database: %s" % summary.database_path)
    if summary.artifacts:
        lines.append("artifacts: " + ", ".join(item.path for item in summary.artifacts))
    if summary.next_actions:
        lines.append("next: " + "; ".join(summary.next_actions))
    return "\n".join(lines) + "\n"


def summary_exit_code(summary: TaskSummary) -> int:
    """Map contract states to CLI process semantics."""
    if summary.status in {"success", "completed", "cached", "skipped", "submitted", "queued", "running", "awaiting_collection"}:
        return 0
    if summary.status in {"partial_success", "warning"}:
        return 2
    return 1
