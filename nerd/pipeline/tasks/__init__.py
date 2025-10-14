"""
Task registry for the pipeline.

This module exposes concrete task classes so the CLI can discover them without
each consumer having to know the individual module paths.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Task
from .create import CreateTask
from .mut_count import MutCountTask
from .nmr_create import NmrCreateTask
from .nmr_deg_kinetics import NmrDegKineticsTask
from .nmr_add_kinetics import NmrAddKineticsTask
from .timecourse import ProbeTimecourseTask
from .tempgrad_fit import TempgradFitTask

__all__ = [
    "CreateTask",
    "MutCountTask",
    "NmrCreateTask",
    "NmrDegKineticsTask",
    "NmrAddKineticsTask",
    "ProbeTimecourseTask",
    "TempgradFitTask",
    "TASK_REGISTRY",
]

TASK_REGISTRY: Dict[str, Type[Task]] = {
    "create": CreateTask,
    "mut_count": MutCountTask,
    "nmr_create": NmrCreateTask,
    "nmr_deg_kinetics": NmrDegKineticsTask,
    "nmr_add_kinetics": NmrAddKineticsTask,
    "probe_timecourse": ProbeTimecourseTask,
    "probe_tc_kinetics": ProbeTimecourseTask,
    "tempgrad_fit": TempgradFitTask,
}
