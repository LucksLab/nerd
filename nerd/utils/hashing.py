# nerd/utils/hashing.py
"""
Utilities for generating consistent hashes from configurations.
"""

import json
import hashlib
from typing import Dict, Any

def config_hash(cfg: Dict[str, Any], length: int = 7) -> str:
    """
    Creates a short, deterministic hash from a configuration dictionary.

    Args:
        cfg: The configuration dictionary.
        length: The desired length of the hash.

    Returns:
        A hexadecimal hash string.
    """
    # Convert the dict to a sorted JSON string to ensure consistency
    config_string = json.dumps(cfg, sort_keys=True, ensure_ascii=True)
    
    # Use SHA256 for hashing
    sha256 = hashlib.sha256(config_string.encode('utf-8')).hexdigest()
    
    # Truncate to the desired length
    return sha256[:length]