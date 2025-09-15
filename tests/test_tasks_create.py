import sqlite3
from pathlib import Path

from nerd.db import api as db_api
from nerd.pipeline.tasks.create import CreateTask
from nerd.pipeline.tasks.base import TaskContext


def make_ctx(tmp_path, db_path: Path, label: str = "TestLabel", output_dir: Path = None):
    if output_dir is None:
        output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = db_api.connect(db_path)
    db_api.init_schema(conn)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = TaskContext(
        db=conn,
        backend="local",
        workdir=run_dir,
        threads=2,
        mem_gb=1,
        time="00:01:00",
        label=label,
        output_dir=str(output_dir),
    )
    return ctx, run_dir


def minimal_create_inputs(rg_label: str = "65_1"):
    return {
        "construct": {
            "family": "Salm_4U_thermometer",
            "name": "WT",
            "version": 2,
            "sequence": "ggtgtaagggtgaagtgtaAGGTTGAACTTTTGAATAGTGATTCAGGAGGTTAATGGAAgtaaaggtaatgaaggtgaag",
            "disp_name": "4U_wt",
        },
        "buffer": {
            "name": "Schwalbe_bistris",
            "pH": 6.5,
            "composition": "150 mM bis-tris, 15 mM Kx(HPO4)y, 25 mM KCl",
            "disp_name": "pH6.5_A",
        },
        "sequencing_run": {
            "run_name": "230607_M05164_0144_000000000-L3FLD",
            "date": "20250607",
            "sequencer": "Illumina_MiSeq",
            "run_manager": "EKC",
        },
        "samples": [
            {
                "sample_name": "fourU_WT_65c_rep1_tp1",
                "fq_dir": ".",  # not validated in consume_outputs
                "r1_file": "R1.fastq.gz",
                "r2_file": "R2.fastq.gz",
                "reaction_group": rg_label,
                "temperature": 65,
                "replicate": 1,
                "reaction_time": 20,
                "probe": "dms",
                "probe_concentration": 0.01585,
                "RT": "MRT",
                "treated": 1,
                "done_by": "EKC",
            }
        ],
    }


def test_create_task_inserts_core_entities(tmp_path):
    db_file = tmp_path / "test.sqlite"
    ctx, run_dir = make_ctx(tmp_path, db_file, label="TestLabel", output_dir=tmp_path / "out")

    task = CreateTask()
    inputs = minimal_create_inputs(rg_label="65_1")

    # Execute consume_outputs directly to avoid file validations in prepare()
    task.consume_outputs(ctx, inputs, {}, run_dir)

    # Assertions: constructs, buffers, seq run, samples
    cur = ctx.db.cursor()
    assert cur.execute("SELECT COUNT(*) FROM constructs").fetchone()[0] == 1
    assert cur.execute("SELECT COUNT(*) FROM buffers").fetchone()[0] == 1
    assert cur.execute("SELECT COUNT(*) FROM sequencing_runs").fetchone()[0] == 1
    assert cur.execute("SELECT COUNT(*) FROM sequencing_samples").fetchone()[0] == 1

    # reaction_groups row exists with correct label
    rg = cur.execute("SELECT rg_id, rg_label FROM reaction_groups").fetchone()
    assert rg is not None
    assert rg[1] == "65_1"

    # probing_reactions linked to the group and sample
    row = cur.execute(
        """
        SELECT pr.rg_id, rg.rg_label, pr.s_id
        FROM probing_reactions pr
        JOIN reaction_groups rg ON rg.rg_id = pr.rg_id
        """
    ).fetchone()
    assert row is not None
    assert row[1] == "65_1"


def test_reaction_group_label_reuse(tmp_path):
    db_file = tmp_path / "test.sqlite"
    ctx, run_dir = make_ctx(tmp_path, db_file, label="TestLabel", output_dir=tmp_path / "out")

    task = CreateTask()
    inputs = minimal_create_inputs(rg_label="65_1")

    # First insertion
    task.consume_outputs(ctx, inputs, {}, run_dir)
    cur = ctx.db.cursor()
    rg1 = cur.execute("SELECT rg_id FROM reaction_groups WHERE rg_label=?", ("65_1",)).fetchone()[0]

    # Second insertion with same label should reuse rg_id
    task.consume_outputs(ctx, inputs, {}, run_dir)
    rg2 = cur.execute("SELECT rg_id FROM reaction_groups WHERE rg_label=?", ("65_1",)).fetchone()[0]
    assert rg1 == rg2

    # There may be multiple probing_reactions now; ensure at least 2 after two runs
    count_rxn = cur.execute("SELECT COUNT(*) FROM probing_reactions").fetchone()[0]
    assert count_rxn >= 2

    # Cleanup
    ctx.db.close()
