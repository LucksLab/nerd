"""
Task registry for the pipeline.

This module exposes concrete task classes so the CLI can discover them without
each consumer having to know the individual module paths.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Task
from .create import CreateTask
from .drop import DropTask
from .mut_count import MutCountTask
from .nmr_create import NmrCreateTask
from .nmr_kinetic_fit import NmrKineticFitTask
from .timecourse import ProbeTimecourseTask
from .tempgrad_fit import TempgradFitTask

__all__ = [
    "CreateTask",
    "MutCountTask",
    "NmrCreateTask",
    "NmrKineticFitTask",
    "ProbeTimecourseTask",
    "TempgradFitTask",
    "DropTask",
    "TASK_REGISTRY",
]

TASK_REGISTRY: Dict[str, Type[Task]] = {
    "create": CreateTask,
    "drop": DropTask,
    "mut_count": MutCountTask,
    "nmr_create": NmrCreateTask,
    "nmr_kinetic_fit": NmrKineticFitTask,
    "probe_timecourse": ProbeTimecourseTask,
    "tempgrad_fit": TempgradFitTask,
}
