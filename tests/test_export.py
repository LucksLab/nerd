from __future__ import annotations

import csv

import pytest

from nerd.cli import app
from nerd.db import api as db_api
from nerd.exporting import PROBE_TIMECOURSE_COLUMNS, mut_count_rows, probe_timecourse_rows


def _seed_metadata(conn):
    with conn:
        conn.execute(
            "INSERT INTO meta_buffers (id, name, pH, composition, disp_name) "
            "VALUES (1, 'tris', 8.0, '50 mM Tris', 'Tris pH 8')"
        )
        conn.execute(
            "INSERT INTO meta_constructs (id, family, name, version, sequence, disp_name) "
            "VALUES (1, 'switch', 'WT', 'v1', 'AU', 'WT construct')"
        )
        conn.execute(
            "INSERT INTO meta_nucleotides (id, construct_id, site, base, base_region) "
            "VALUES (1, 1, 10, 'A', '1'), (2, 1, 11, 'U', '1')"
        )
        conn.execute(
            "INSERT INTO sequencing_runs (id, run_name, date, sequencer, run_manager) "
            "VALUES (1, 'run1', '2026-01-01', 'NovaSeq', 'person')"
        )
        conn.execute(
            "INSERT INTO sequencing_samples "
            "(id, seqrun_id, sample_name, fq_dir, r1_file, r2_file) "
            "VALUES (1, 1, 'sample-a', '.', 'r1.fastq', 'r2.fastq')"
        )
        conn.execute("INSERT INTO probe_reaction_groups (rg_id, rg_label) VALUES (7, 'rg-seven')")
        conn.execute(
            "INSERT INTO probe_reactions "
            "(id, rg_id, s_id, construct_id, buffer_id, temperature, replicate, "
            "reaction_time, probe_concentration, probe, rt_protocol, done_by, treated) "
            "VALUES (1, 7, 1, 1, 1, 37.0, 2, 30.0, 10.0, 'DMS', 'rt', 'person', 1)"
        )


def _task(conn, name="probe_timecourse"):
    return db_api.begin_task(
        conn, name, "rg", 7, "local", "/tmp/out", "test", None
    )


def _fit(conn, task_id, fit_kind, nt_id, values):
    run_id = db_api.begin_probe_tc_fit_run(
        conn,
        fit_kind=fit_kind,
        task_id=task_id,
        rg_id=7,
        nt_id=nt_id,
        valtype="modrate",
        model="python_baseline",
    )
    db_api.record_probe_tc_fit_params(
        conn,
        fit_run_id=run_id,
        entries=[{"param_name": key, "param_numeric": value} for key, value in values.items()],
    )
    return run_id


def test_probe_timecourse_export_has_requested_columns_and_final_round(tmp_path):
    conn = db_api.connect(tmp_path / "nerd.sqlite")
    db_api.init_schema(conn)
    _seed_metadata(conn)
    task_id = _task(conn)
    _fit(conn, task_id, "round1_free", 1, {"kobs": 1.0, "kdeg": 0.1})
    _fit(
        conn,
        task_id,
        "round2_global_profiled",
        None,
        {"kdeg": 0.2, "log_kdeg_err": 0.25},
    )
    _fit(
        conn,
        task_id,
        "round3_constrained",
        1,
        {
            "kobs": 2.0,
            "log_kobs_err": 0.1,
            "kdeg": 0.2,
            "diag:r2": 0.98,
            "diag:chisq": 0.03,
        },
    )

    rows = probe_timecourse_rows(conn, task_id=task_id)

    assert list(rows[0]) == PROBE_TIMECOURSE_COLUMNS
    assert rows[0]["site"] == 10
    assert rows[0]["base"] == "A"
    assert rows[0]["construct"] == "WT construct"
    assert rows[0]["temp"] == 37.0
    assert rows[0]["rep"] == 2
    assert rows[0]["buffer"] == "Tris pH 8"
    assert rows[0]["kobs"] == 2.0
    assert rows[0]["kobs_err"] == pytest.approx(0.2)
    assert rows[0]["kdeg"] == 0.2
    assert rows[0]["kdeg_err"] == pytest.approx(0.05)
    assert rows[0]["r2"] == 0.98
    assert rows[0]["chisq"] == 0.03


def test_probe_timecourse_export_cli_by_task(tmp_path, cli_runner):
    db_path = tmp_path / "nerd.sqlite"
    conn = db_api.connect(db_path)
    db_api.init_schema(conn)
    _seed_metadata(conn)
    task_id = _task(conn)
    _fit(conn, task_id, "round1_free", 1, {"kobs": 1.0, "kdeg": 0.1})
    conn.close()
    output = tmp_path / "fits.csv"

    result = cli_runner.invoke(
        app,
        ["--db", str(db_path), "export", "probe_timecourse", "--task-id", str(task_id), "-o", str(output)],
    )

    assert result.exit_code == 0, result.output
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PROBE_TIMECOURSE_COLUMNS
        assert list(reader)[0]["site"] == "10"


def test_probe_timecourse_export_recovers_unambiguous_legacy_task(tmp_path):
    conn = db_api.connect(tmp_path / "nerd.sqlite")
    db_api.init_schema(conn)
    _seed_metadata(conn)
    task_id = _task(conn)
    db_api.record_task_scope_members(
        conn,
        task_id,
        [{"kind": "rg", "ref_id": 7, "label": "rg-seven"}],
    )
    _fit(conn, None, "round1_free", 1, {"kobs": 1.0, "kdeg": 0.1})
    _fit(conn, None, "round3_constrained", 1, {})

    rows = probe_timecourse_rows(conn, task_id=task_id)

    assert len(rows) == 1
    assert rows[0]["site"] == 10
    assert rows[0]["kobs"] == 1.0


def test_mut_count_export_selects_samples_and_latest_run(tmp_path):
    conn = db_api.connect(tmp_path / "nerd.sqlite")
    db_api.init_schema(conn)
    _seed_metadata(conn)
    with conn:
        conn.execute(
            "INSERT INTO probe_fmod_runs "
            "(id, software_name, software_version, run_args, run_datetime, output_dir, s_id) "
            "VALUES (1, 'shapemapper', '1', '', '2026-01-01', '/old', 1), "
            "(2, 'shapemapper', '1', '', '2026-01-02', '/new', 1)"
        )
        conn.execute(
            "INSERT INTO probe_fmod_values "
            "(nt_id, fmod_run_id, rxn_id, valtype, fmod_val, read_depth) "
            "VALUES (1, 1, 1, 'modrate', 0.1, 100), "
            "(1, 2, 1, 'modrate', 0.2, 200)"
        )

    rows = mut_count_rows(conn, samples=["sample-a"])

    assert rows == [
        {
            "sample": "sample-a",
            "site": 10,
            "base": "A",
            "valtype": "modrate",
            "mutation_rate": 0.2,
            "read_depth": 200,
        }
    ]


def test_export_requires_one_selector(tmp_path):
    conn = db_api.connect(tmp_path / "nerd.sqlite")
    db_api.init_schema(conn)
    with pytest.raises(ValueError, match="exactly one selector"):
        probe_timecourse_rows(conn)
    with pytest.raises(ValueError, match="either one or more"):
        mut_count_rows(conn, samples=[], all_samples=False)
