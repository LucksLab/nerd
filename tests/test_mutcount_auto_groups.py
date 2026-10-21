import sqlite3
from types import SimpleNamespace

import pytest

from nerd.pipeline.tasks.mut_count import MutCountTask


def plan(rows, monkeypatch):
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE sequencing_samples (id INTEGER, sample_name TEXT, to_drop INTEGER);
        CREATE TABLE probe_reactions (s_id INTEGER, rg_id INTEGER);
        CREATE TABLE probe_fmod_runs (s_id INTEGER);
        INSERT INTO sequencing_samples VALUES (1, 'done', 0), (2, 'a', 0), (3, 'b', 0), (4, 'dropped', 1), (5, 'standalone', 0);
        INSERT INTO probe_fmod_runs VALUES (1);
    ''')
    conn.executemany('INSERT INTO probe_reactions VALUES (?, ?)', rows)
    task = MutCountTask()
    def group(ctx, value):
        samples = conn.execute('SELECT id, sample_name FROM sequencing_samples JOIN probe_reactions ON s_id=id WHERE rg_id=?', (value,)).fetchall()
        return value, 'group-%s' % value, [(r[0], r[1]) for r in samples]
    monkeypatch.setattr(task, '_fetch_reaction_group_info', group)
    cfg = {'mut_count': {'plugin': 'shapemapper', 'missing_only': True}}
    inputs, params = task.prepare(cfg)
    units = task.plan_work_units(SimpleNamespace(db=conn), cfg, inputs, params)
    conn.close()
    return units


def test_missing_only_fans_out_pending_groups(monkeypatch):
    units = plan([(1, 10), (2, 20), (3, 30), (4, 40)], monkeypatch)
    assert [u.key for u in units] == ['rg-20', 'rg-30', 'ungrouped']
    assert [u.scope_id for u in units] == [20, 30, None]
    assert units[-1].config['mut_count']['samples'] == ['standalone']
    assert all(u.config['mut_count']['missing_only'] is True for u in units)
    assert 'samples' not in units[0].config['mut_count']


def test_single_pending_group_still_uses_group_scope(monkeypatch):
    units = plan([(1, 10), (2, 20), (3, 20), (4, 40), (5, 20)], monkeypatch)
    assert [u.key for u in units] == ['rg-20']
    assert units[0].config['mut_count']['reaction_group'] == 20


def test_no_pending_samples_does_not_plan_empty_job():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE sequencing_samples (id INTEGER, sample_name TEXT, to_drop INTEGER);
        CREATE TABLE probe_reactions (s_id INTEGER, rg_id INTEGER);
        CREATE TABLE probe_fmod_runs (s_id INTEGER);
    ''')
    task = MutCountTask()
    cfg = {'mut_count': {'plugin': 'shapemapper', 'missing_only': True}}
    inputs, params = task.prepare(cfg)
    with pytest.raises(ValueError, match='No eligible samples'):
        task.plan_work_units(SimpleNamespace(db=conn), cfg, inputs, params)
    conn.close()
