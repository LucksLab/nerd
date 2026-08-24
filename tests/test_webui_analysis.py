"""Focused coverage for the read-only Web UI analysis queries."""

from __future__ import annotations

import math
import sqlite3

import pytest

from nerd.db import api as db_api
from nerd.webui.analysis import (
    analysis_catalog, kinetic_rates, modification_rates, timecourse_data,
    timecourse_options,
)


@pytest.fixture
def analysis_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db_api.init_schema(conn)
    with conn:
        conn.execute(
            "INSERT INTO meta_buffers (id, name, pH, composition, disp_name) VALUES (1, 'HEPES', 7.5, '100 mM', 'HEPES 7.5')"
        )
        conn.execute(
            "INSERT INTO meta_constructs (id, family, name, version, sequence, disp_name) "
            "VALUES (1, 'switch', 'WT', 'v1', 'AC', 'switch WT')"
        )
        conn.executemany(
            "INSERT INTO meta_nucleotides (id, construct_id, site, base, base_region) VALUES (?, 1, ?, ?, '1')",
            [(1, 1, "A"), (2, 293, "C")],
        )
        conn.execute(
            "INSERT INTO sequencing_runs (id, run_name, date, sequencer, run_manager) "
            "VALUES (1, 'seq-1', '2026-08-01', 'NovaSeq', 'EKC')"
        )
        conn.executemany(
            "INSERT INTO sequencing_samples "
            "(id, seqrun_id, sample_name, fq_source, fq_dir, r1_file, r2_file, to_drop) "
            "VALUES (?, 1, ?, 'local', '/fq', 'r1', 'r2', ?)",
            [(1, "control", 0), (2, "time-30", 0), (3, "time-60", 1)],
        )
        conn.execute("INSERT INTO probe_reaction_groups (rg_id, rg_label) VALUES (7, 'WT_25C_rep1')")
        conn.executemany(
            "INSERT INTO probe_reactions "
            "(id, rg_id, s_id, construct_id, buffer_id, temperature, replicate, reaction_time, "
            "probe_concentration, probe, rt_protocol, done_by, treated) "
            "VALUES (?, 7, ?, 1, 1, 25, 1, ?, 0.5, 'DMS', 'Marathon', 'EKC', ?)",
            [(1, 1, 999.0, 0), (2, 2, 30.0, 1), (3, 3, 60.0, 1)],
        )
        conn.executemany(
            "INSERT INTO probe_fmod_runs "
            "(id, software_name, software_version, run_args, run_datetime, output_dir, s_id) "
            "VALUES (?, 'ShapeMapper', '2.3', '{}', ?, ?, ?)",
            [
                (11, "2026-08-01T10:00:00", "/out/11", 1),
                (12, "2026-08-01T10:01:00", "/out/12", 2),
                (13, "2026-08-01T10:02:00", "/out/13", 3),
            ],
        )
        values = []
        row_id = 1
        for run_id, rxn_id, base_value in [(11, 1, .01), (12, 2, .04), (13, 3, .08)]:
            for nt_id, offset in [(1, 0), (2, .01)]:
                for valtype, multiplier in [("modrate", 1), ("GAmodrate", 2)]:
                    values.append((row_id, nt_id, run_id, rxn_id, valtype, (base_value + offset) * multiplier, 1000, 1 if run_id == 12 and nt_id == 1 else 0))
                    row_id += 1
        conn.executemany(
            "INSERT INTO probe_fmod_values "
            "(id, nt_id, fmod_run_id, rxn_id, valtype, fmod_val, read_depth, outlier) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        cursor = conn.execute(
            "INSERT INTO probe_tc_fit_runs (fit_kind, rg_id, nt_id, valtype, model) "
            "VALUES ('round3_constrained', 7, 1, 'modrate', 'python_baseline')"
        )
        fit_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO probe_tc_fit_params (fit_run_id, param_name, param_numeric) VALUES (?, ?, ?)",
            [
                (fit_id, "log_kobs", math.log(1.2)),
                (fit_id, "log_kdeg", math.log(.01)),
                (fit_id, "log_fmod0", math.log(.01)),
                (fit_id, "diag:r2", .94),
            ],
        )
    yield conn
    conn.close()


def test_catalog_prioritizes_reaction_metadata_and_lists_available_types(analysis_conn):
    catalog = analysis_catalog(analysis_conn)

    run = next(item for item in catalog["fmod_runs"] if item["fmod_run_id"] == 12)
    assert run["sample_name"] == "time-30"
    assert run["construct_name"] == "switch WT"
    assert run["buffer_name"] == "HEPES 7.5"
    assert run["temperature"] == 25
    assert run["replicate"] == 1
    assert run["reaction_time"] == 30
    assert run["valtypes"] == ["GAmodrate", "modrate"]

    group = catalog["reaction_groups"][0]
    assert group["rg_label"] == "WT_25C_rep1"
    assert group["timepoint_count"] == 2
    assert group["time_min"] == 30
    assert group["time_max"] == 60
    assert group["fit_valtypes"] == ["modrate"]
    assert group["kobs_site_count"] == 1


def test_modification_rates_support_unlimited_runs_and_numeric_site_order(analysis_conn):
    result = modification_rates(analysis_conn, [11, 12, 13], "GAmodrate")

    assert result["valtype"] == "GAmodrate"
    assert [row["site_base"] for row in result["values"] if row["fmod_run_id"] == 11] == ["1A", "293C"]
    flagged = next(row for row in result["values"] if row["fmod_run_id"] == 12 and row["nt_id"] == 1)
    assert flagged["outlier"] == 1

    unlimited = modification_rates(analysis_conn, [11, 12, 13, 14], "modrate")
    assert unlimited["run_ids"] == [11, 12, 13, 14]


def test_timecourse_options_and_data_include_flags_and_stored_fit(analysis_conn):
    options = timecourse_options(analysis_conn, 7)
    assert [site["site_base"] for site in options["sites"]] == ["1A", "293C"]
    assert set(options["sites"][0]["valtypes"]) == {"modrate", "GAmodrate"}

    result = timecourse_data(analysis_conn, 7, [1, 2], "modrate")
    first = result["series"][0]
    assert [row["plot_time"] for row in first["observations"]] == [0.0, 30.0, 60.0]
    assert first["observations"][1]["outlier"] == 1
    assert first["observations"][2]["to_drop"] == 1
    assert first["fit"]["fit_kind"] == "round3_constrained"
    assert first["fit"]["r2"] == pytest.approx(.94)
    assert len(first["fit"]["curve"]) == 200
    assert first["fit"]["curve"][0]["fmod_val"] == pytest.approx(.01)
    assert result["series"][1]["fit"] is None


def test_kinetic_rates_return_logged_and_linear_values_for_unlimited_groups(analysis_conn):
    result = kinetic_rates(analysis_conn, [7], "modrate")

    assert result["rg_ids"] == [7]
    assert result["valtype"] == "modrate"
    assert len(result["values"]) == 1
    value = result["values"][0]
    assert value["site_base"] == "1A"
    assert value["rg_label"] == "WT_25C_rep1"
    assert value["log_kobs"] == pytest.approx(math.log(1.2))
    assert value["kobs"] == pytest.approx(1.2)
    assert value["log_kdeg"] == pytest.approx(math.log(.01))
    assert value["kdeg"] == pytest.approx(.01)
    assert value["r2"] == pytest.approx(.94)

    unlimited = kinetic_rates(analysis_conn, [1, 2, 3, 4], "modrate")
    assert unlimited["rg_ids"] == [1, 2, 3, 4]
