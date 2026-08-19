# nerd/utils/config.py
"""
Configuration loading utility.
Handles loading YAML files, including support for `include:` directives
and command-line overrides.
"""

from copy import deepcopy
import yaml
from pathlib import Path
from typing import Dict, Any, List, Union


class LoadedConfig(dict):
    """A config mapping that retains its source and pre-normalization values."""

    def __init__(self, data: Dict[str, Any], source_path: Path):
        super().__init__(data)
        self.source_path = source_path
        self.base_dir = source_path.parent
        self.hash_data = deepcopy(data)


_PATH_KEYS = {
    "output_dir", "fq_dir", "nt_info", "path", "script_path",
    "shapemapper_path", "sif_path", "cache_dir", "samples_yaml",
    "samples_config", "config_path", "yaml", "source", "constructs",
    "buffers", "sequencing_runs",
}
_PATH_LIST_KEYS = {"search_roots"}


def _absolute(value: str, base_dir: Path) -> str:
    if not value.strip():
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _normalize_paths(value: Any, base_dir: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _normalize_paths(v, base_dir, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        if key in _PATH_LIST_KEYS:
            return [_absolute(str(item), base_dir) for item in value]
        return [_normalize_paths(item, base_dir) for item in value]
    if isinstance(value, str) and (
        key in _PATH_KEYS or key in _PATH_LIST_KEYS or key.endswith("_path")
    ):
        return _absolute(value, base_dir)
    return value

def load_config(path: Union[str, Path]) -> LoadedConfig:
    """
    Loads a YAML configuration file.
    
    Args:
        path: The path to the YAML file.

    Returns:
        A dictionary containing the configuration.
    """
    source_path = Path(path).expanduser().resolve()
    with source_path.open('r') as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be a mapping.")
    loaded = LoadedConfig(raw, source_path)
    loaded.update(_normalize_paths(raw, loaded.base_dir))
    return loaded
