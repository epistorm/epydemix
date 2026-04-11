# epydemix/io/__init__.py

from .bundle import add_figure_to_manifest, load_bundle, save_bundle
from .inspect import inspect_bundle

__all__ = [
    "save_bundle",
    "load_bundle",
    "add_figure_to_manifest",
    "inspect_bundle",
]
