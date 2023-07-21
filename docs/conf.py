"""Sphinx configuration."""
project = "NKHDF5"
author = "Ankit Khambhati"
copyright = "2023, Ankit Khambhati"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
