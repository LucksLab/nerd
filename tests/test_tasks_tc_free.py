"""
Unit tests for round1_free behavior and confirmation that
ProbeTimecourseTask accepts round2_global_profiled without any task-layer
changes (round dispatch/persistence are generic over round_id strings).
"""

from __future__ import annotations

import math

import numpy as np

from nerd.pipeline.plugins.timecourse.baseline import BaselinePythonEngine, ROUND_FREE, _fmod_model
from nerd.pipeline.plugins.timecourse.base import NucleotideSeries, TimecourseRequest
from nerd.pipeline.tasks.timecourse import ProbeTimecourseTask


def test_round1_free_fits_each_nucleotide_independently():
    log_kdeg_true = [math.log(1e-3), math.log(5e-3)]
    x = [0.0, 30.0, 60.0, 120.0, 240.0]
    series = []
    for idx, lk in enumerate(log_kdeg_true, start=1):
        y = _fmod_model(np.asarray(x), math.log(1.0), lk, math.log(0.02))
        series.append(
            NucleotideSeries(nt_id=idx, timepoints=x, fmod_values=y.tolist(), metadata={"base": "A"})
        )

    engine = BaselinePythonEngine()
    request = TimecourseRequest(rg_id=1, rounds=[ROUND_FREE], nucleotides=series, global_metadata={}, options={})
    result = engine.run(request)
    free_round = next(r for r in result.rounds if r.round_id == ROUND_FREE)
    assert free_round.status == "completed"
    assert len(free_round.per_nt) == 2
    fitted = {fit.nt_id: fit.params["log_kdeg"] for fit in free_round.per_nt}
    for idx, lk in enumerate(log_kdeg_true, start=1):
        assert abs(fitted[idx] - lk) < 0.05


def test_task_prepare_accepts_round2_global_profiled_unmodified():
    task = ProbeTimecourseTask()
    cfg = {
        "probe_timecourse": {
            "engine": "python_baseline",
            "rounds": ["round1_free", "round2_global_profiled", "round3_constrained"],
            "rg_ids": [26],
            "valtype": "modrate",
        }
    }
    inputs, _ = task.prepare(cfg)
    assert inputs["rounds"] == ["round1_free", "round2_global_profiled", "round3_constrained"]
