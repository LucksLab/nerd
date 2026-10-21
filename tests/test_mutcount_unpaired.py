from pathlib import Path
import shlex

import pytest

from nerd.configuration import ConfigValidationError, _validate_plugin
from nerd.pipeline.plugins.mutcount.shapemapper import ShapeMapperPlugin
from nerd.pipeline.tasks.mut_count import MutCountTask


@pytest.mark.parametrize('mode,flags', [(None, ['--R1', 'read 1.fq', '--R2', 'read 2.fq']), ('R1', ['--U', 'read 1.fq']), ('R2', ['--U', 'read 2.fq'])])
def test_read_selection(mode, flags):
    args = shlex.split(ShapeMapperPlugin().command(
        sample_name='s', r1_path=Path('read 1.fq'), r2_path=Path('read 2.fq'),
        fasta_path=Path('target.fa'), out_dir=Path('out'), options={'unpaired_mode': mode},
    ))
    assert args[args.index('--modified') + 1:] == flags


@pytest.mark.parametrize('mode', ['r1', 'R3', True, ['R1']])
def test_invalid_mode(mode):
    with pytest.raises(ConfigValidationError, match='unpaired_mode'):
        _validate_plugin('mut_count', {'plugin': 'shapemapper', 'unpaired_mode': mode})


