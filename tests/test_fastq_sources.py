from pathlib import Path
import sqlite3

import pytest

from nerd.db import api as db_api
from nerd.fastq_sources import (
    FastqSourceError, list_remote_directory, normalize_source, profile_for_source,
    remote_profiles,
)
from nerd.pipeline.tasks.base import TaskContext
from nerd.pipeline.tasks.mut_count import MutCountTask
from nerd.pipeline.tasks.create import CreateTask


EXECUTORS = {
    "quest": {
        "type": "ssh_slurm", "host": "quest",
        "remote_base_dir": "/scratch/user/nerd-runs",
    },
    "local": {"type": "local"},
}


def test_source_normalization_and_remote_profile_choices():
    assert normalize_source(None) == "local"
    assert normalize_source("remote_HPC:quest") == "remote_hpc:quest"
    assert remote_profiles(EXECUTORS) == [{
        "value": "remote_hpc:quest", "label": "Remote HPC — quest",
        "alias": "quest", "host": "quest",
    }]
    assert profile_for_source("remote_hpc:quest", EXECUTORS).name == "quest"
    with pytest.raises(FastqSourceError, match="not configured"):
        profile_for_source("remote_hpc:other", EXECUTORS)


def test_remote_listing_uses_selected_profile(monkeypatch):
    observed = {}

    def fake_run(profile, command, timeout=30):
        observed["profile"] = profile.name
        observed["command"] = list(command)
        return type("Result", (), {
            "returncode": 0,
            "stdout": "/projects/run/b_R2.fastq.gz\n/projects/run/a_R1.fastq.gz\n",
            "stderr": "",
        })()

    monkeypatch.setattr("nerd.fastq_sources._remote_run", fake_run)
    names = list_remote_directory("/projects/run", profile_for_source("remote_hpc:quest", EXECUTORS))

    assert names == ["a_R1.fastq.gz", "b_R2.fastq.gz"]
    assert observed == {
        "profile": "quest",
        "command": ["find", "/projects/run", "-maxdepth", "1", "-type", "f", "-print"],
    }


def test_create_checks_remote_reads_through_declared_alias(tmp_path, monkeypatch):
    observed = {}

    def fake_check(directory, filenames, profile, known_files=None):
        observed.update(directory=directory, filenames=tuple(filenames), profile=profile.name)
        return set(filenames)

    monkeypatch.setattr("nerd.pipeline.tasks.create.check_remote_fastqs", fake_check)
    sample = {
        "sample_name": "sample", "fq_source": "remote_hpc:quest",
        "fq_dir": "/projects/run", "r1_file": "sample_R1.fastq.gz",
        "r2_file": "sample_R2.fastq.gz", "reaction_group": "group",
        "temperature": 25, "replicate": 1, "reaction_time": 60,
        "probe": "dms", "probe_concentration": 1, "rt_protocol": "rt",
        "treated": 1, "buffer": "buffer", "construct": "construct",
        "done_by": "user",
    }
    inputs, _ = CreateTask().prepare({
        "run": {"output_dir": str(tmp_path), "label": "label"},
        "executors": EXECUTORS,
        "create": {"samples": [sample]},
    })

    assert inputs["samples"][0]["fq_source"] == "remote_hpc:quest"
    assert observed == {
        "directory": "/projects/run",
        "filenames": ("sample_R1.fastq.gz", "sample_R2.fastq.gz"),
        "profile": "quest",
    }


def test_existing_database_is_migrated_with_local_default(tmp_path):
    database = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE sequencing_samples (id INTEGER PRIMARY KEY, seqrun_id INTEGER, sample_name TEXT, fq_dir TEXT, r1_file TEXT, r2_file TEXT, to_drop INTEGER DEFAULT 0, UNIQUE(seqrun_id, sample_name, fq_dir))")
    conn.execute("INSERT INTO sequencing_samples (seqrun_id, sample_name, fq_dir, r1_file, r2_file) VALUES (1, 'sample', '.', 'R1.fastq', 'R2.fastq')")
    conn.commit()

    db_api.init_schema(conn)

    assert conn.execute("SELECT fq_source FROM sequencing_samples").fetchone()[0] == "local"


def _remote_mut_count_context(tmp_path: Path):
    conn = db_api.connect(tmp_path / "nerd.sqlite")
    db_api.init_schema(conn)
    conn.execute("INSERT INTO sequencing_runs (run_name, date, sequencer, run_manager) VALUES ('run', '20260101', 'miseq', 'user')")
    conn.execute("INSERT INTO meta_constructs (family, name, version, sequence, disp_name) VALUES ('f', 'n', '1', 'AC', 'c')")
    conn.execute("INSERT INTO meta_nucleotides (construct_id, site, base, base_region) VALUES (1, 1, 'A', '1')")
    conn.execute("INSERT INTO sequencing_samples (seqrun_id, sample_name, fq_source, fq_dir, r1_file, r2_file) VALUES (1, 'sample', 'remote_hpc:quest', '/projects/run', 'sample_R1.fastq.gz', 'sample_R2.fastq.gz')")
    conn.execute("INSERT INTO probe_reaction_groups (rg_id, rg_label) VALUES (1, 'group')")
    conn.execute("INSERT INTO meta_buffers (name, pH, composition, disp_name) VALUES ('b', 7, 'x', 'b')")
    conn.execute("INSERT INTO probe_reactions (rg_id, s_id, construct_id, buffer_id, temperature, replicate, reaction_time, probe_concentration, probe, rt_protocol, done_by, treated) VALUES (1, 1, 1, 1, 25, 1, 1, 1, 'dms', 'rt', 'user', 1)")
    conn.commit()
    return TaskContext(
        db=conn, backend="ssh_slurm", workdir=tmp_path / "work", threads=1,
        mem_gb=1, time="00:10:00", label="label", output_dir=str(tmp_path),
        executor_profile="quest",
    )


def test_remote_fastqs_are_used_in_place_without_stage_in(tmp_path):
    ctx = _remote_mut_count_context(tmp_path)
    task = MutCountTask()
    command = task.command(ctx, {
        "samples": ["sample"], "plugin": "shapemapper", "dry_run": True,
        "tool": {},
    }, {})

    assert "/projects/run/sample_R1.fastq.gz" in command
    assert task.stage_in_pairs() == []


def test_reaction_group_uses_sample_id_when_name_exists_in_multiple_runs(tmp_path):
    ctx = _remote_mut_count_context(tmp_path)
    ctx.db.execute(
        "INSERT INTO sequencing_runs "
        "(run_name, date, sequencer, run_manager) "
        "VALUES ('other-run', '20260102', 'miseq', 'user')"
    )
    ctx.db.execute(
        "INSERT INTO sequencing_samples "
        "(seqrun_id, sample_name, fq_source, fq_dir, r1_file, r2_file) "
        "VALUES (2, 'sample', 'remote_hpc:quest', '/projects/wrong', "
        "'wrong_R1.fastq.gz', 'wrong_R2.fastq.gz')"
    )
    ctx.db.commit()

    task = MutCountTask()
    inputs = {
        "samples": [], "reaction_group": 1, "plugin": "shapemapper",
        "dry_run": True, "tool": {},
    }
    scope = task.resolve_scope(ctx, inputs)
    command = task.command(ctx, inputs, {})

    assert [(member.kind, member.ref_id) for member in scope.members] == [
        ("rg", 1), ("sample", 1),
    ]
    assert "/projects/run/sample_R1.fastq.gz" in command
    assert "/projects/wrong/wrong_R1.fastq.gz" not in command


def test_remote_fastqs_require_matching_executor_alias(tmp_path):
    ctx = _remote_mut_count_context(tmp_path)
    ctx.executor_profile = "other"
    with pytest.raises(RuntimeError, match="run mut_count with executor 'quest'"):
        MutCountTask().command(ctx, {
            "samples": ["sample"], "plugin": "shapemapper", "dry_run": True,
            "tool": {},
        }, {})
