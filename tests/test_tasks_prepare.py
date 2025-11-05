from nerd.pipeline.tasks.tempgrad_fit import TempgradFitTask
from nerd.pipeline.tasks.timecourse import ProbeTimecourseTask


def test_probe_timecourse_prepare_defaults():
    task = ProbeTimecourseTask()
    cfg = {
        "probe_timecourse": {
            "rg_ids": [11, 12],
            "outliers": ["11:sample_a:15_A:modrate"],
        }
    }

    prepared, params = task.prepare(cfg)
    assert not params
    assert prepared["engine"] == "python_baseline"
    assert prepared["rounds"] == ["round1_free", "round2_global", "round3_constrained"]
    assert prepared["rg_ids"] == [11, 12]
    assert prepared["valtypes"] == ["modrate"]
    assert prepared["outliers"][0]["rg_id"] == 11
    assert prepared["outliers"][0]["valtype"] == "modrate"


def test_tempgrad_fit_prepare_selects_defaults():
    task = TempgradFitTask()
    cfg = {
        "tempgrad_fit": {
            "mode": "arrhenius",
            "filters": {"construct": "test"},
            "use_probe_tc": True,
            "group_by": ["construct", "buffer"],
        }
    }

    prepared, params = task.prepare(cfg)
    assert not params
    assert prepared["engine"] == "arrhenius_python"
    assert prepared["data_source"] == "probe_tc"
    assert prepared["group_by"] == ["construct", "buffer"]
    assert prepared["engine_options"] == {}
    assert prepared["overwrite"] is False
    assert prepared["mode"] == "arrhenius"
