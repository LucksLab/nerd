"""
Unit tests for the baseline Python timecourse fitting engine
(nerd.pipeline.plugins.timecourse.baseline.BaselinePythonEngine), covering
round-name normalization, the profiled (1D scalar) global round, and its
numerical-robustness regression versus the existing joint round2_global.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from nerd.pipeline.plugins.timecourse.baseline import (
    BaselinePythonEngine,
    ROUND_CONSTRAINED,
    ROUND_FREE,
    ROUND_GLOBAL,
    ROUND_GLOBAL_PROFILED,
    _fmod_model,
    _SingleFitRecord,
)
from nerd.pipeline.plugins.timecourse.base import NucleotideSeries, TimecourseRequest


TIMEPOINTS = [0.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0]


def _make_series(
    nt_id: int,
    log_kappa: float,
    log_kdeg: float,
    log_fmod0: float,
    *,
    seed: int = 0,
    noise: float = 0.0,
) -> NucleotideSeries:
    rng = np.random.default_rng(seed + nt_id)
    x = np.asarray(TIMEPOINTS, dtype=float)
    y = _fmod_model(x, log_kappa, log_kdeg, log_fmod0)
    if noise:
        y = y + rng.normal(scale=noise, size=y.shape)
    return NucleotideSeries(
        nt_id=nt_id,
        timepoints=x.tolist(),
        fmod_values=y.tolist(),
        metadata={"base": "A", "valtype": "modrate"},
    )


def _request(rounds: List[str], series: List[NucleotideSeries], **options) -> TimecourseRequest:
    return TimecourseRequest(
        rg_id=1,
        rounds=rounds,
        nucleotides=series,
        global_metadata={},
        options=dict(options),
    )


# ---------------------------------------------------------------------------
# (a) Round-name normalization aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "alias,expected",
    [
        ("round1", ROUND_FREE),
        ("free", ROUND_FREE),
        ("free_fit", ROUND_FREE),
        ("round2", ROUND_GLOBAL),
        ("global", ROUND_GLOBAL),
        ("round2b", ROUND_GLOBAL_PROFILED),
        ("global_profiled", ROUND_GLOBAL_PROFILED),
        ("profiled", ROUND_GLOBAL_PROFILED),
        ("round2_profiled", ROUND_GLOBAL_PROFILED),
        ("round2-profiled", ROUND_GLOBAL_PROFILED),  # hyphen normalization
        ("round3", ROUND_CONSTRAINED),
        ("constrained", ROUND_CONSTRAINED),
    ],
)
def test_normalize_round_aliases(alias, expected):
    assert BaselinePythonEngine._normalize_round(alias) == expected


def test_normalize_round_passes_through_canonical_ids():
    for canonical in (ROUND_FREE, ROUND_GLOBAL, ROUND_GLOBAL_PROFILED, ROUND_CONSTRAINED):
        assert BaselinePythonEngine._normalize_round(canonical) == canonical


# ---------------------------------------------------------------------------
# (b) Profiled round vs. joint round2_global on a small, well-behaved dataset
# ---------------------------------------------------------------------------

def test_profiled_round_matches_joint_global_on_small_dataset():
    true_log_kdeg = math.log(2.5e-3)
    series = [
        _make_series(1, math.log(0.8), true_log_kdeg, math.log(0.01), seed=1),
        _make_series(2, math.log(1.2), true_log_kdeg, math.log(0.02), seed=2),
        _make_series(3, math.log(0.5), true_log_kdeg, math.log(0.015), seed=3),
        _make_series(4, math.log(1.5), true_log_kdeg, math.log(0.03), seed=4),
    ]
    engine = BaselinePythonEngine()

    joint_request = _request([ROUND_FREE, ROUND_GLOBAL], series)
    joint_result = engine.run(joint_request)
    joint_round = next(r for r in joint_result.rounds if r.round_id == ROUND_GLOBAL)
    assert joint_round.status == "completed"

    profiled_request = _request([ROUND_FREE, ROUND_GLOBAL_PROFILED], series)
    profiled_result = engine.run(profiled_request)
    profiled_round = next(r for r in profiled_result.rounds if r.round_id == ROUND_GLOBAL_PROFILED)
    assert profiled_round.status == "completed"

    joint_log_kdeg = joint_round.global_params["log_kdeg"]
    profiled_log_kdeg = profiled_round.global_params["log_kdeg"]
    assert pytest.approx(joint_log_kdeg, abs=0.05) == profiled_log_kdeg

    # Profiled round should also emit per-nucleotide records for free.
    assert len(profiled_round.per_nt) == len(series)
    assert all(fit.diagnostics.get("status") == "completed" for fit in profiled_round.per_nt)


# ---------------------------------------------------------------------------
# (c) Large-N regression: round2_global overflows, round2_global_profiled doesn't
# ---------------------------------------------------------------------------

def _make_large_dataset(n: int, *, seed: int = 42) -> List[NucleotideSeries]:
    rng = np.random.default_rng(seed)
    series = []
    base_log_kdeg = math.log(3e-3)
    for i in range(n):
        log_kappa = math.log(float(rng.uniform(0.2, 3.0)))
        log_fmod0 = math.log(float(rng.uniform(0.005, 0.05)))
        # Small per-nucleotide jitter in local kdeg estimates (as in real
        # noisy round1 data) is what destabilizes the joint N-dim optimizer.
        jitter = float(rng.normal(scale=0.4))
        series.append(
            _make_series(
                i + 1,
                log_kappa,
                base_log_kdeg + jitter,
                log_fmod0,
                seed=seed + i,
                noise=0.01,
            )
        )
    return series


def test_profiled_round_succeeds_on_large_n():
    series = _make_large_dataset(200)
    engine = BaselinePythonEngine()
    request = _request([ROUND_FREE, ROUND_GLOBAL_PROFILED], series)
    result = engine.run(request)
    profiled_round = next(r for r in result.rounds if r.round_id == ROUND_GLOBAL_PROFILED)
    assert profiled_round.status == "completed"
    assert math.isfinite(profiled_round.global_params["log_kdeg"])
    # A handful of round1 free fits may fail to converge on noisy synthetic
    # data (excluded by _filter_series_for_global), so n_sites can be less
    # than len(series); it must still cover the large majority.
    assert 0 < profiled_round.qc_metrics["n_sites"] <= len(series)
    assert profiled_round.qc_metrics["n_sites"] == len(profiled_round.per_nt)
    assert profiled_round.qc_metrics["optimizer_success"] is True


def test_joint_global_overflows_with_extreme_seed_profiled_does_not():
    """
    Directly rigs the single_fit_cache -- as if round1's free fits had
    converged to a wildly extreme log_kdeg for every nucleotide, which can
    happen on degenerate/underdetermined timecourses -- to deterministically
    trigger the known overflow failure mode: _create_global_params seeds the
    joint fit's shared log_kdeg from np.nanmean(kdeg_logs), and math.exp() on
    an extreme seed (math.exp overflows above ~709.78) blows up immediately
    inside the very first residual evaluation. Reproducing this from
    "naturally jittered" synthetic data proved too optimizer-internals-
    dependent to be a reliable, deterministic regression test; rigging the
    cache isolates the exact failure condition instead.

    round2_global_profiled must survive the same rigged cache unscathed: its
    bounds are always hard-clamped to [-30, 10] regardless of how extreme the
    auto-derived seed is, so exp() can never overflow.
    """
    engine = BaselinePythonEngine()
    extreme_log_kdeg = 800.0
    series = [
        _make_series(i, math.log(1.0), math.log(1e-3), math.log(0.02), seed=i)
        for i in range(1, 6)
    ]
    rigged_cache = {
        engine._series_key(s): _SingleFitRecord(
            params={"log_kobs": math.log(1.0), "log_kdeg": extreme_log_kdeg, "log_fmod0": math.log(0.02)},
            diagnostics={"r2": 0.99},
        )
        for s in series
    }
    request = _request([ROUND_GLOBAL, ROUND_GLOBAL_PROFILED], series)

    global_result = engine._run_global_fit(request, rigged_cache)
    assert global_result.status == "failed"

    profiled_result = engine._run_global_fit_profiled(request, rigged_cache)
    assert profiled_result.status == "completed"
    assert math.isfinite(profiled_result.global_params["log_kdeg"])
    # Bounds must be hard-clamped even though the (rigged) seed estimate was extreme.
    assert profiled_result.qc_metrics["bounds_hi"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# (d) round3_constrained consumes log_kdeg from round2_global_profiled
# ---------------------------------------------------------------------------

def test_round3_consumes_profiled_log_kdeg():
    true_log_kdeg = math.log(4e-3)
    series = [
        _make_series(1, math.log(1.0), true_log_kdeg, math.log(0.02), seed=10),
        _make_series(2, math.log(0.6), true_log_kdeg, math.log(0.015), seed=11),
        _make_series(3, math.log(1.8), true_log_kdeg, math.log(0.03), seed=12),
    ]
    engine = BaselinePythonEngine()
    request = _request([ROUND_FREE, ROUND_GLOBAL_PROFILED, ROUND_CONSTRAINED], series)
    result = engine.run(request)

    profiled_round = next(r for r in result.rounds if r.round_id == ROUND_GLOBAL_PROFILED)
    constrained_round = next(r for r in result.rounds if r.round_id == ROUND_CONSTRAINED)

    assert constrained_round.status == "completed"
    expected_log_kdeg = profiled_round.global_params["log_kdeg"]
    for fit in constrained_round.per_nt:
        assert fit.diagnostics.get("status") == "completed"
        assert pytest.approx(fit.params["log_kdeg"], abs=1e-9) == expected_log_kdeg


def test_round3_prefers_profiled_over_joint_global_when_both_requested():
    true_log_kdeg = math.log(4e-3)
    series = [
        _make_series(1, math.log(1.0), true_log_kdeg, math.log(0.02), seed=20),
        _make_series(2, math.log(0.6), true_log_kdeg, math.log(0.015), seed=21),
        _make_series(3, math.log(1.8), true_log_kdeg, math.log(0.03), seed=22),
    ]
    engine = BaselinePythonEngine()
    request = _request(
        [ROUND_FREE, ROUND_GLOBAL, ROUND_GLOBAL_PROFILED, ROUND_CONSTRAINED],
        series,
    )
    result = engine.run(request)

    profiled_round = next(r for r in result.rounds if r.round_id == ROUND_GLOBAL_PROFILED)
    constrained_round = next(r for r in result.rounds if r.round_id == ROUND_CONSTRAINED)
    assert constrained_round.status == "completed"
    expected_log_kdeg = profiled_round.global_params["log_kdeg"]
    for fit in constrained_round.per_nt:
        assert pytest.approx(fit.params["log_kdeg"], abs=1e-9) == expected_log_kdeg


# ---------------------------------------------------------------------------
# (e) Bounds override + hard safety clamp
# ---------------------------------------------------------------------------

def test_profiled_bounds_override_is_respected():
    series = _make_large_dataset(20, seed=99)
    engine = BaselinePythonEngine()
    request = _request(
        [ROUND_FREE, ROUND_GLOBAL_PROFILED],
        series,
        global_profiled_bounds=[-12.0, -4.0],
    )
    result = engine.run(request)
    profiled_round = next(r for r in result.rounds if r.round_id == ROUND_GLOBAL_PROFILED)
    assert profiled_round.qc_metrics["bounds_lo"] == pytest.approx(-12.0)
    assert profiled_round.qc_metrics["bounds_hi"] == pytest.approx(-4.0)
    assert -12.0 <= profiled_round.global_params["log_kdeg"] <= -4.0


def test_profiled_bounds_hard_clamp_enforced_even_with_extreme_override():
    series = _make_large_dataset(10, seed=7)
    engine = BaselinePythonEngine()
    request = _request(
        [ROUND_FREE, ROUND_GLOBAL_PROFILED],
        series,
        global_profiled_bounds=[-1000.0, 1000.0],
    )
    result = engine.run(request)
    profiled_round = next(r for r in result.rounds if r.round_id == ROUND_GLOBAL_PROFILED)
    assert profiled_round.qc_metrics["bounds_lo"] == pytest.approx(-30.0)
    assert profiled_round.qc_metrics["bounds_hi"] == pytest.approx(10.0)
