import json
from pathlib import Path
from typing import Dict, List

import pytest

from nerd.db import api as db_api
from nerd.pipeline.plugins.nmr_fit_kinetics import FitResult
from nerd.pipeline.tasks.base import TaskContext
from nerd.pipeline.tasks.nmr_add_kinetics import NmrAddKineticsTask
from nerd.pipeline.tasks.nmr_deg_kinetics import NmrDegKineticsTask


def make_ctx(tmp_path: Path, label: str = "NMR_Run") -> tuple[TaskContext, Path, Path]:
    db_path = tmp_path / "nmr.sqlite"
    conn = db_api.connect(db_path)
    db_api.init_schema(conn)

    output_dir = tmp_path / "out"
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    run_dir = tmp_path / "task_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = TaskContext(
        db=conn,
        backend="local",
        workdir=run_dir,
        threads=2,
        mem_gb=2,
        time="00:10:00",
        label=label,
        output_dir=str(output_dir),
    )
    return ctx, run_dir, label_dir


def seed_buffer(conn) -> int:
    buf_id = db_api.upsert_buffer(
        conn,
        {
            "name": "TestBuffer",
            "pH": 6.5,
            "composition": "Test buffer composition",
            "disp_name": "TestBufferDisp",
        },
    )
    assert buf_id is not None
    return buf_id


def insert_reaction(
    conn,
    *,
    buffer_id: int,
    reaction_type: str,
    substrate: str,
    substrate_conc: float,
    kinetic_dir: str,
) -> int:
    sql = """
        INSERT INTO nmr_reactions (
            reaction_type, temperature, replicate, num_scans, time_per_read,
            total_kinetic_reads, total_kinetic_time, probe, probe_conc,
            probe_solvent, substrate, substrate_conc, buffer_id, nmr_machine,
            kinetic_data_dir, mnova_analysis_dir, raw_fid_dir
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        reaction_type,
        25.0,
        1,
        8,
        2.0,
        10,
        600,
        "dms",
        0.01564,
        "water",
        substrate,
        substrate_conc,
        buffer_id,
        "TestMachine",
        kinetic_dir,
        None,
        None,
    )
    with conn:
        conn.execute(sql, params)
        row_id = conn.execute(
            "SELECT id FROM nmr_reactions WHERE kinetic_data_dir = ?",
            (kinetic_dir,),
        ).fetchone()[0]
    return int(row_id)


def write_trace_csv(path: Path, columns: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(columns.keys())
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("All columns must share the same length.")
    rows = zip(*(columns[h] for h in headers))
    with path.open("w", newline="") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(str(val) for val in row) + "\n")


def test_nmr_deg_task_records_rate(tmp_path, monkeypatch):
    ctx, run_dir, label_dir = make_ctx(tmp_path)
    buffer_id = seed_buffer(ctx.db)
    reaction_id = insert_reaction(
        ctx.db,
        buffer_id=buffer_id,
        reaction_type="deg",
        substrate="none",
        substrate_conc=0.0,
        kinetic_dir="deg_run",
    )

    trace_rel = Path("nmr") / "deg_trace.csv"
    write_trace_csv(
        label_dir / trace_rel,
        {
            "time": [0, 10, 20, 30],
            "peak_integral": [1.0, 0.7, 0.52, 0.4],
        },
    )
    db_api.register_nmr_trace_file(
        ctx.db,
        nmr_reaction_id=reaction_id,
        role="decay_trace",
        path=str(trace_rel),
    )

    captured: Dict[str, Dict] = {}

    def fake_loader(name: str, **kwargs):
        class StubPlugin:
            def fit(self_inner, request):
                captured["decay"] = {
                    "files": {k: str(v) for k, v in request.files.items()},
                    "metadata": dict(request.metadata),
                    "params": dict(request.params),
                }
                return FitResult(k_value=0.5, k_error=0.05, r2=0.9, chisq=0.01)

        return StubPlugin()

    monkeypatch.setattr("nerd.pipeline.plugins.nmr_fit_kinetics.load_nmr_fit_plugin", fake_loader)

    task = NmrDegKineticsTask()
    inputs, _ = task.prepare(
        {
            "nmr_deg_kinetics": {
                "reaction_ids": [reaction_id],
                "plugin": "stub",
                "species": "dms",
            }
        }
    )

    task.consume_outputs(ctx, inputs, {}, run_dir, task_id=None)

    row = ctx.db.execute("SELECT nmr_reaction_id, species, k_value FROM nmr_kinetic_rates").fetchone()
    assert row is not None
    assert row["nmr_reaction_id"] == reaction_id
    assert row["species"] == "dms"
    assert pytest.approx(row["k_value"], rel=1e-6) == 0.5

    assert "decay" in captured
    copied_path = Path(captured["decay"]["files"]["decay_trace"])
    assert copied_path.exists()
    result_file = run_dir / "results" / f"reaction_{reaction_id}.json"
    assert result_file.exists()
    payload = json.loads(result_file.read_text())
    assert payload["result"]["k_value"] == 0.5


def test_nmr_add_task_uses_substrate_metadata(tmp_path, monkeypatch):
    ctx, run_dir, label_dir = make_ctx(tmp_path)
    buffer_id = seed_buffer(ctx.db)
    reaction_id = insert_reaction(
        ctx.db,
        buffer_id=buffer_id,
        reaction_type="add",
        substrate="ATP",
        substrate_conc=0.25,
        kinetic_dir="add_run",
    )

    peak_rel = Path("nmr") / "peak.csv"
    dms_rel = Path("nmr") / "dms.csv"
    write_trace_csv(
        label_dir / peak_rel,
        {
            "time": [0, 5, 10],
            "peak": [0.2, 0.4, 0.6],
        },
    )
    write_trace_csv(
        label_dir / dms_rel,
        {
            "time": [0, 5, 10],
            "peak": [0.8, 0.7, 0.6],
        },
    )
    db_api.register_nmr_trace_file(
        ctx.db,
        nmr_reaction_id=reaction_id,
        role="peak_trace",
        path=str(peak_rel),
    )
    db_api.register_nmr_trace_file(
        ctx.db,
        nmr_reaction_id=reaction_id,
        role="dms_trace",
        path=str(dms_rel),
    )

    captured = {}

    def fake_loader(name: str, **kwargs):
        class StubPlugin:
            def fit(self_inner, request):
                captured["meta"] = dict(request.metadata)
                return FitResult(k_value=1.1, k_error=None, r2=None, chisq=None)

        return StubPlugin()

    monkeypatch.setattr("nerd.pipeline.plugins.nmr_fit_kinetics.load_nmr_fit_plugin", fake_loader)

    task = NmrAddKineticsTask()
    inputs, _ = task.prepare(
        {
            "nmr_add_kinetics": {
                "reaction_ids": [reaction_id],
                "plugin": "stub",
            }
        }
    )

    task.consume_outputs(ctx, inputs, {}, run_dir, task_id=None)

    row = ctx.db.execute(
        "SELECT nmr_reaction_id, species, k_value FROM nmr_kinetic_rates"
    ).fetchone()
    assert row is not None
    assert row["species"] == "ATP"
    assert pytest.approx(row["k_value"], rel=1e-6) == 1.1
    assert captured["meta"]["ntp_conc"] == 0.25
    assert captured["meta"]["substrate"] == "ATP"
