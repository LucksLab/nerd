"""Regression coverage for replacing mutation-count values during collection."""
import sqlite3
from types import SimpleNamespace

import pytest

from nerd.db import schema
from nerd.pipeline.tasks.mut_count import MutCountTask


@pytest.mark.parametrize('overwrite,valid,ga_valid', [
    (True, True, True), (False, True, True), (True, False, True),
    (True, True, False), (True, False, False),
])
def test_overwrite_retires_only_successfully_replaced_sample(
    tmp_path, monkeypatch, overwrite, valid, ga_valid,
):
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    for sql in (schema.CREATE_PROBE_FMOD_RUNS, schema.CREATE_PROBE_FMOD_VALUES,
                schema.CREATE_PROBE_TC_FIT_RUNS):
        db.executescript(sql)
    db.executescript('''
        CREATE TABLE probe_reactions (id INTEGER, s_id INTEGER, construct_id INTEGER, treated INTEGER);
        CREATE TABLE meta_nucleotides (id INTEGER, construct_id INTEGER, site INTEGER, base TEXT, base_region TEXT);
        INSERT INTO probe_reactions VALUES (1, 1, 1, 1);
        INSERT INTO meta_nucleotides VALUES (1, 1, 1, 'A', 'test');
        INSERT INTO probe_fmod_runs VALUES (1, 'old', '', '', 'old', 'old1', 1);
        INSERT INTO probe_fmod_runs VALUES (2, 'old', '', '', 'old', 'old2', 1);
        INSERT INTO probe_fmod_runs VALUES (3, 'old', '', '', 'old', 'other', 2);
        INSERT INTO probe_tc_fit_runs (id, fit_kind, fmod_run_id) VALUES (1, 'free', 1);
        INSERT INTO probe_fmod_values (nt_id, fmod_run_id, rxn_id, valtype, fmod_val, read_depth)
        VALUES (1, 1, 1, 'modrate', .1, 100), (1, 2, 1, 'GAmodrate', .2, 100),
               (1, 3, 2, 'modrate', .3, 100);
    ''')
    sample_dir = tmp_path / 'artifacts' / 'sample'
    sample_dir.mkdir(parents=True)
    (sample_dir / 'sample_profile.txt').touch()
    (sample_dir / 'sample_profile.txtga').touch()
    def parse(path):
        ok = ga_valid if str(path).endswith('txtga') else valid
        return [{'Sequence': 'A' if ok else 'C', 'Modified_rate': '.9',
                 'Modified_read_depth': '200'}]
    monkeypatch.setattr('nerd.pipeline.plugins.mutcount.load_mutcount_plugin',
                        lambda *a, **kw: SimpleNamespace(parse_profile=parse))
    task = MutCountTask()
    monkeypatch.setattr(task, '_resolve_sample_names', lambda *a: (['sample'], {}))
    monkeypatch.setattr(task, '_sample_row_for_name', lambda *a: {'id': 1})
    inputs = {'plugin': 'shapemapper', 'overwrite': overwrite}
    if not valid and not ga_valid:
        with pytest.raises(ValueError):
            task.consume_outputs(SimpleNamespace(db=db), inputs, {}, tmp_path)
    else:
        task.consume_outputs(SimpleNamespace(db=db), inputs, {}, tmp_path)
    replaced = overwrite and valid and ga_valid
    old_values = db.execute('SELECT count(*) FROM probe_fmod_values WHERE fmod_run_id IN (1, 2)').fetchone()[0]
    assert old_values == (0 if replaced else 2)
    assert db.execute('SELECT count(*) FROM probe_fmod_values WHERE fmod_run_id = 3').fetchone()[0] == 1
    assert db.execute('SELECT fmod_run_id FROM probe_tc_fit_runs').fetchone()[0] == 1
    assert db.execute('SELECT count(*) FROM probe_fmod_runs WHERE id = 2').fetchone()[0] == (0 if replaced else 1)
    if replaced:
        task.consume_outputs(SimpleNamespace(db=db), inputs, {}, tmp_path)
        assert db.execute(
            "SELECT count(*) FROM probe_fmod_values v JOIN probe_fmod_runs r "
            "ON r.id = v.fmod_run_id WHERE r.s_id = 1"
        ).fetchone()[0] == 2
