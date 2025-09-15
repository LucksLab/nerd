import sqlite3
from pathlib import Path

import json
import pytest
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


def test_derived_sample_inserts_with_parent(tmp_path):
    db_file = tmp_path / "test.sqlite"
    ctx, run_dir = make_ctx(tmp_path, db_file, label="TestLabel", output_dir=tmp_path / "out")

    # Seed a parent sequencing sample into the DB
    seqrun = {
        "run_name": "TEST_RUN",
        "date": "20250101",
        "sequencer": "Illumina_MiSeq",
        "run_manager": "EKC",
    }
    seqrun_id = db_api.upsert_sequencing_run(ctx.db, seqrun)
    samples = [
        {
            "sample_name": "parent_s1",
            "fq_dir": ".",
            "r1_file": "R1.fastq.gz",
            "r2_file": "R2.fastq.gz",
        }
    ]
    db_api.bulk_upsert_samples(ctx.db, seqrun_id, samples)

    task = CreateTask()
    inputs = {
        "derived_samples": [
            {
                "child_name": "parent_s1__subsample-n1000-s7",
                "parent_sample": "parent_s1",
                "kind": "subsample",
                "tool": "seqtk",
                "params": {"count": 1000, "seed": 7},
                "cmd_template": "seqtk sample -s {seed} {R1} {count} > {OUT_R1} && seqtk sample -s {seed} {R2} {count} > {OUT_R2}",
            }
        ]
    }

    # Execute consume_outputs for derived-only config
    task.consume_outputs(ctx, inputs, {}, run_dir)

    # Assert one derived_sample row exists and links to the parent
    cur = ctx.db.cursor()
    row = cur.execute(
        "SELECT parent_sample_id, child_name, kind, tool, cmd_template, params_json, cache_key FROM derived_samples"
    ).fetchone()
    assert row is not None
    parent_sample_id = cur.execute(
        "SELECT id FROM sequencing_samples WHERE sample_name = ?",
        ("parent_s1",),
    ).fetchone()[0]
    assert row[0] == parent_sample_id
    assert row[1] == "parent_s1__subsample-n1000-s7"
    assert row[2] == "subsample"
    assert row[3] == "seqtk"
    assert "{R1}" in row[4] and "{OUT_R2}" in row[4]
    params = json.loads(row[5])
    assert params["count"] == 1000 and params["seed"] == 7
    assert isinstance(row[6], str) and len(row[6]) >= 40  # cache_key looks like a long hash

    ctx.db.close()


def test_derived_sample_without_parent_raises(tmp_path):
    db_file = tmp_path / "test.sqlite"
    ctx, run_dir = make_ctx(tmp_path, db_file, label="TestLabel", output_dir=tmp_path / "out")

    task = CreateTask()
    inputs = {
        "derived_samples": [
            {
                "child_name": "missing_parent__subsample-n10-s1",
                "parent_sample": "missing_parent",
                "kind": "subsample",
                "tool": "seqtk",
                "params": {"count": 10, "seed": 1},
                "cmd_template": "seqtk sample -s {seed} {R1} {count} > {OUT_R1} && seqtk sample -s {seed} {R2} {count} > {OUT_R2}",
            }
        ]
    }

    with pytest.raises(ValueError) as exc:
        task.consume_outputs(ctx, inputs, {}, run_dir)
    assert "Could not resolve parent sample" in str(exc.value)
    ctx.db.close()
