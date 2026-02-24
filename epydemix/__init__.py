# epydemix/__init__.py

from importlib.metadata import PackageNotFoundError, version

from .model.epimodel import EpiModel, simulate
from .model.predefined_models import load_predefined_model

try:
    __version__ = version("epydemix")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["EpiModel", "simulate", "load_predefined_model"]
