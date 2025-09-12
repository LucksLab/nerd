# nerd/utils/config.py
"""
Configuration loading utility.
Handles loading YAML files, including support for `include:` directives
and command-line overrides.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Union

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    
    Args:
        path: The path to the YAML file.

    Returns:
        A dictionary containing the configuration.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)