# nerd/utils/config.py

import tomllib  # Python 3.11+
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "config" / "defaults.toml"

def load_config():
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)
    

### USE IN MODULES ###
# from nerd.utils.config import load_config

# config = load_config()

# # Use default DB path
# db_path = config["paths"]["db_file"]

# # Use max iterations for fitting
# max_iters = config["fitting"]["max_nfev"]

# # Use unit for Arrhenius plots
# unit = config["arrhenius"]["temperature_unit"]