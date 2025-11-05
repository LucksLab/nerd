import numpy as np
import pytest

from nerd.pipeline.plugins.tempgrad.arrhenius import (
    ArrheniusPythonEngine,
    R_GAS_CONSTANT,
)
from nerd.pipeline.plugins.tempgrad.base import TempgradRequest, TempgradSeries


def test_arrhenius_engine_returns_expected_activation_energy():
    series = TempgradSeries(
        series_id="series_1",
        x_values=[25.0, 35.0, 45.0, 55.0],
        y_values=[0.5, 0.9, 1.8, 3.4],
        metadata={"temperature_unit": "c"},
    )

    engine = ArrheniusPythonEngine()
    request = TempgradRequest(mode="arrhenius", series=[series], metadata={"source": "test"})
    result = engine.run(request)

    assert result.engine == "arrhenius_python"
    assert result.metadata["mode"] == "arrhenius"
    assert len(result.series_results) == 1

    fit = result.series_results[0]
    assert fit.series_id == "series_1"
    assert "activation_energy_cal_per_mol" in fit.params
    assert fit.params["activation_energy_cal_per_mol"] > 0
    assert fit.diagnostics["ndata"] == len(series.x_values)
    assert fit.diagnostics["r2"] <= 1.0

    temps_k = np.asarray(series.x_values) + 273.15
    inv_t = 1.0 / temps_k
    log_k = np.log(np.asarray(series.y_values))
    slope, _ = np.polyfit(inv_t, log_k, 1)
    expected_ea = -slope * R_GAS_CONSTANT

    assert pytest.approx(fit.params["activation_energy_cal_per_mol"], rel=1e-3) == expected_ea
