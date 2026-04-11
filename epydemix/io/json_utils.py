"""JSON encoding utilities for numpy/pandas types."""

import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


class NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and pandas types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()[:10]
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)


def _round_recursive(obj: Any, precision: int) -> Any:
    """Recursively round floats in nested structures."""
    if isinstance(obj, dict):
        return {k: _round_recursive(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_recursive(v, precision) for v in obj]
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, np.floating):
        return round(float(obj), precision)
    if isinstance(obj, np.ndarray):
        return [round(float(x), precision) for x in obj.flat]
    return obj


def to_json(obj: Any, precision: int = 6, indent: int = 2) -> str:
    """Serialize an object to JSON with numpy/pandas support and rounding.

    Args:
        obj: The object to serialize.
        precision: Decimal places for floating point numbers.
        indent: JSON indentation level.
    """
    rounded = _round_recursive(obj, precision)
    return json.dumps(rounded, cls=NumpySafeEncoder, indent=indent)


def round_values(d: dict, precision: int) -> dict:
    """Recursively round float values in a nested dict."""
    return _round_recursive(d, precision)
