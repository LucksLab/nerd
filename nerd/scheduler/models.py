"""Scheduler-neutral state and job value objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AttemptState(str, Enum):
    PENDING = "pending"
    SUBMITTING = "submitting"
    QUEUED = "queued"
    RUNNING = "running"
    SCHEDULER_COMPLETED = "scheduler_completed"
    SCHEDULER_FAILED = "scheduler_failed"
    SUBMISSION_FAILED = "submission_failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    COLLECTING = "collecting"
    COMPLETED = "completed"
    VALIDATION_FAILED = "validation_failed"
    UNKNOWN = "unknown"


class TaskState(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    AWAITING_COLLECTION = "awaiting_collection"
    COLLECTING = "collecting"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobSpec:
    command: str
    workdir: Path
    env: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    remote_workdir: Optional[str] = None
    preamble: Optional[str] = None
    stage_in: List[Dict[str, str]] = field(default_factory=list)
    stage_out: List[str] = field(default_factory=list)
    controller_cwd: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["workdir"] = str(self.workdir)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobSpec":
        values = dict(data)
        values["workdir"] = Path(values["workdir"])
        return cls(**values)


@dataclass
class JobHandle:
    scheduler_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobStatus:
    state: AttemptState
    exit_code: Optional[int] = None
    signal: Optional[str] = None
    message: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
