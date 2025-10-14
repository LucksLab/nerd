from pathlib import Path

import pytest

from nerd.db import api as db_api
from nerd.pipeline.tasks.base import TaskContext
from nerd.pipeline.tasks.nmr_create import NmrCreateTask


def make_ctx(tmp_path: Path, label: str = "NMR_Create") -> tuple[TaskContext, Path]:
    db_file = tmp_path / "nmr.sqlite"
    conn = db_api.connect(db_file)
    db_api.init_schema(conn)

    output_dir = tmp_path / "out"
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = TaskContext(
        db=conn,
        backend="local",
        workdir=run_dir,
        threads=1,
        mem_gb=1,
        time="00:10:00",
        label=label,
        output_dir=str(output_dir),
    )
    return ctx, label_dir


def seed_buffer(ctx: TaskContext) -> int:
    buffer_id = db_api.upsert_buffer(
        ctx.db,
        {
            "name": "Schwalbe_bistris",
            "pH": 6.5,
            "composition": "150 mM bis-tris, 15 mM phosphate, 25 mM KCl",
            "disp_name": "Schwalbe_bistris_pH6.5",
        },
    )
    assert buffer_id is not None
    return buffer_id


def test_nmr_create_ingests_reaction(tmp_path):
    ctx, label_dir = make_ctx(tmp_path)
    seed_buffer(ctx)

    trace_path = label_dir / "traces" / "deg_trace.csv"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("time,peak_integral\n0,1.0\n10,0.8\n")

    task = NmrCreateTask()
    cfg = {
        "run": {
            "label": ctx.label,
            "output_dir": ctx.output_dir,
        },
        "nmr_create": {
            "reactions": [
                {
                    "reaction_type": "deg",
                    "temperature": 25,
                    "replicate": 1,
                    "num_scans": 16,
                    "time_per_read": 2.0,
                    "total_kinetic_reads": 64,
                    "total_kinetic_time": 1280,
                    "probe": "dms",
                    "probe_conc": 0.015,
                    "probe_solvent": "etoh",
                    "substrate": "none",
                    "substrate_conc": 0.0,
                    "buffer": "Schwalbe_bistris_pH6.5",
                    "nmr_machine": "Ascend600",
                    "kinetic_data_dir": "runs/deg_25C_rep1",
                    "trace_files": {
                        "decay_trace": "traces/deg_trace.csv",
                    },
                }
            ]
        },
    }

    inputs, _ = task.prepare(cfg)
    task.consume_outputs(ctx, inputs, {}, ctx.workdir, task_id=None)

    row = ctx.db.execute(
        "SELECT reaction_type, buffer_id, kinetic_data_dir FROM nmr_reactions"
    ).fetchone()
    assert row is not None
    assert row["reaction_type"] == "deg"
    assert row["kinetic_data_dir"] == "runs/deg_25C_rep1"

    trace_row = ctx.db.execute(
        "SELECT role, path FROM nmr_trace_files"
    ).fetchone()
    assert trace_row is not None
    assert trace_row["role"] == "decay_trace"
    assert trace_row["path"] == "traces/deg_trace.csv"


def test_nmr_create_missing_buffer_raises(tmp_path):
    ctx, _ = make_ctx(tmp_path)

    task = NmrCreateTask()
    cfg = {
        "run": {"label": ctx.label, "output_dir": ctx.output_dir},
        "nmr_create": {
            "reactions": [
                {
                    "reaction_type": "deg",
                    "temperature": 25,
                    "replicate": 1,
                    "num_scans": 16,
                    "time_per_read": 2.0,
                    "total_kinetic_reads": 64,
                    "total_kinetic_time": 1280,
                    "probe": "dms",
                    "probe_conc": 0.015,
                    "probe_solvent": "etoh",
                    "substrate": "none",
                    "substrate_conc": 0.0,
                    "buffer": "UnknownBuffer",
                    "nmr_machine": "Ascend600",
                    "kinetic_data_dir": "runs/deg_missing_buffer",
                }
            ]
        },
    }

    inputs, _ = task.prepare(cfg)

    with pytest.raises(ValueError, match="Buffer 'UnknownBuffer' not found"):
        task.consume_outputs(ctx, inputs, {}, ctx.workdir, task_id=None)
