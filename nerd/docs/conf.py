# -- Project info -----------------------------------------------------
project = "nerd"
author = "Edric Choi"
version = "0.1.0"
release = "0.1.0"

# -- Paths ------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath("../nerd"))  # Point to your source code

# -- Extensions -------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

autodoc_typehints = "description"

# -- Napoleon settings for NumPy/Google style docstrings --------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- HTML theme -------------------------------------------------------
html_theme = "sphinx_rtd_theme"