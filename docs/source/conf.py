"""Sphinx configuration for the epydemix documentation."""

import shutil
import sys
from pathlib import Path

import tomllib

DOCS_SOURCE = Path(__file__).parent
REPO_ROOT = DOCS_SOURCE.parents[1]

# Document the code in this checkout, whatever version happens to be installed
sys.path.insert(0, str(REPO_ROOT))

with open(REPO_ROOT / "pyproject.toml", "rb") as f:
    release = tomllib.load(f)["project"]["version"]
version = ".".join(release.split(".")[:2])

project = "epydemix"
author = "The epydemix developers"
copyright = "2026, The epydemix developers"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build"]

# The tutorial notebooks live in the top-level tutorials/ folder (so that the GitHub
# and Colab links keep working). Copy them next to tutorials/index.md at build time.
for notebook in sorted((REPO_ROOT / "tutorials").glob("*.ipynb")):
    shutil.copy2(notebook, DOCS_SOURCE / "tutorials" / notebook.name)

# Notebooks are shown with the outputs saved in them: nothing is executed during the build
nb_execution_mode = "off"
nb_merge_streams = True

# MyST: allow ::: fences for admonitions and generate anchors for headings
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
# The tutorial notebooks skip heading levels in places; don't fail the build on that
suppress_warnings = ["myst.header"]

autodoc_member_order = "bysource"
# Show the constructor arguments once, on the class itself
autoclass_content = "both"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

html_theme = "furo"
html_title = f"epydemix {release}"
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "epydemix-logo.png",
    "dark_logo": "epydemix-logo-dark.png",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/epistorm/epydemix",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#0b7f76",
        "color-brand-content": "#0b7f76",
    },
    "dark_css_variables": {
        "color-brand-primary": "#40d0c0",
        "color-brand-content": "#40d0c0",
    },
}
