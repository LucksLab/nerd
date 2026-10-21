import sqlite3
from types import SimpleNamespace

import pytest

from nerd.pipeline.tasks.timecourse import ProbeTimecourseTask


@pytest.fixture
def ctx():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE probe_reaction_groups (rg_id INTEGER, rg_label TEXT);
        CREATE TABLE probe_tc_fit_runs (rg_id INTEGER);
        CREATE TABLE probe_reactions (id INTEGER, rg_id INTEGER, s_id INTEGER);
        CREATE TABLE sequencing_samples (id INTEGER, to_drop INTEGER);
        CREATE TABLE probe_fmod_values (rxn_id INTEGER);
        INSERT INTO probe_reaction_groups VALUES (1, 'done'), (2, 'pending'), (3, 'empty'), (4, 'dropped'), (5, 'pending-two');
        INSERT INTO probe_tc_fit_runs VALUES (1);
        INSERT INTO sequencing_samples VALUES (1, 0), (2, 0), (4, 1), (5, 0);
        INSERT INTO probe_reactions VALUES (1, 1, 1), (2, 2, 2), (4, 4, 4), (5, 5, 5);
        INSERT INTO probe_fmod_values VALUES (1), (2), (4), (5);
    ''')
    yield SimpleNamespace(db=conn)
    conn.close()


def test_automatic_missing_groups_fan_out(ctx):
    task = ProbeTimecourseTask()
    cfg = {'probe_timecourse': {'missing_only': True}}
    inputs, params = task.prepare(cfg)
    units = task.plan_work_units(ctx, cfg, inputs, params)
    assert [u.key for u in units] == ['rg-2', 'rg-5']
    assert units[0].config['probe_timecourse']['rg_ids'] == [2]
    assert units[0].scope_id == 2


def test_explicit_group_labels_filter_missing(ctx):
    task = ProbeTimecourseTask()
    inputs, _ = task.prepare({'probe_timecourse': {
        'reaction_groups': ['done', 'pending'], 'missing_only': True,
    }})
    assert task._resolve_groups(ctx, inputs) == [2]
    assert task.resolve_scope(ctx, inputs).scope_id == 2


def test_explicit_ids_and_labels_without_missing(ctx):
    task = ProbeTimecourseTask()
    inputs, _ = task.prepare({'probe_timecourse': {'reaction_groups': [1, 'pending', 1]}})
    assert task._resolve_groups(ctx, inputs) == [1, 2]


def test_include_dropped_and_no_pending(ctx):
    task = ProbeTimecourseTask()
    inputs, _ = task.prepare({'probe_timecourse': {'missing_only': True, 'include_dropped_samples': True}})
    assert task._resolve_groups(ctx, inputs) == [2, 4, 5]
    inputs, _ = task.prepare({'probe_timecourse': {'missing_only': True, 'reaction_groups': ['done']}})
    with pytest.raises(ValueError, match='No eligible'):
        task._resolve_groups(ctx, inputs)


def test_unknown_group_fails(ctx):
    task = ProbeTimecourseTask()
    inputs, _ = task.prepare({'probe_timecourse': {'reaction_groups': ['typo']}})
    with pytest.raises(ValueError, match='not found'):
        task._resolve_groups(ctx, inputs)
