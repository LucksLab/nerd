import sqlite3
from types import SimpleNamespace

import pytest

from nerd.cli import app
from nerd.db import api as db_api
from nerd.pipeline.tasks.mut_count import MutCountTask
from nerd.sheetbuilder.pattern.compile import compile_pattern, parse_name
from nerd.sheetbuilder.pattern.tokens import TokenRegistry


@pytest.mark.parametrize('raw, expected', [('rep1', '1'), ('rep2', '2'), ('REP12', '12'), ('2', '2'), ('a', 'a')])
def test_replicate_prefix(raw, expected):
    registry = TokenRegistry.load()
    pattern = compile_pattern('[replicate]_[construct]', registry)
    result = parse_name(raw + '_RNA', pattern, registry)
    assert result['values']['replicate'] == expected
    assert result['raw']['replicate'] == raw


def test_missing_only_selection():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE sequencing_samples (id INTEGER, sample_name TEXT, to_drop INTEGER);
        CREATE TABLE probe_fmod_runs (s_id INTEGER);
        INSERT INTO sequencing_samples VALUES (1, 'done', 0), (2, 'new', 0), (3, 'dropped', 1), (4, 'other', 0);
        INSERT INTO probe_fmod_runs VALUES (1);
    ''')
    task = MutCountTask()
    inputs, _ = task.prepare({'mut_count': {'plugin': 'shapemapper', 'missing_only': True}})
    assert task._resolve_sample_names(SimpleNamespace(db=conn), inputs)[0] == ['new', 'other']
    inputs = {'missing_only': True, 'samples': ['done', 'new', 'dropped']}
    assert task._resolve_sample_names(SimpleNamespace(db=conn), inputs)[0] == ['new']
    conn.close()


def test_reaction_group_listing(cli_runner, tmp_path):
    path = tmp_path / 'nerd.sqlite'
    conn = db_api.connect(path)
    db_api.init_schema(conn)
    conn.execute("INSERT INTO probe_reaction_groups (rg_id, rg_label) VALUES (7, 'test-group')")
    conn.commit()
    conn.close()
    result = cli_runner.invoke(app, ['db', 'reaction-groups', '--db', str(path)])
    assert result.exit_code == 0, result.output
    assert 'test-group' in result.output
    assert 'without mut_count' in result.output
