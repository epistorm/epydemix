"""Inspection engine for querying .epx bundles.

Each function reads from the on-disk Parquet files, computes an answer,
and returns a compact dictionary suitable for JSON serialization to stdout.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .bundle import load_bundle, load_bundle_dataframe
from .json_utils import round_values


def inspect_bundle(
    path: str,
    command: str,
    **kwargs,
) -> Dict[str, Any]:
    """Dispatch an inspect command against a bundle.

    Args:
        path: Path to the .epx bundle directory.
        command: One of ``"manifest"``, ``"quantiles"``, ``"summary"``,
            ``"peak"``, ``"posterior"``, ``"fit"``, ``"export"``.
        **kwargs: Command-specific arguments.

    Returns:
        A dict suitable for JSON serialization.
    """
    commands = {
        "manifest": _cmd_manifest,
        "quantiles": _cmd_quantiles,
        "summary": _cmd_summary,
        "peak": _cmd_peak,
        "posterior": _cmd_posterior,
        "fit": _cmd_fit,
    }

    if command not in commands:
        raise ValueError(
            f"Unknown inspect command '{command}'. "
            f"Available: {list(commands.keys())}"
        )

    return commands[command](path, **kwargs)


# ---------------------------------------------------------------------------
# Time-slice helper
# ---------------------------------------------------------------------------

def _apply_time_slice(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Filter a DataFrame to a time window.

    Args:
        df: DataFrame with a date column.
        start: Start date (inclusive), ISO format string.
        end: End date (inclusive), ISO format string.
        date_col: Name of the date column.
    """
    if start is not None:
        start_dt = pd.to_datetime(start)
        df = df[df[date_col] >= start_dt]
    if end is not None:
        end_dt = pd.to_datetime(end)
        df = df[df[date_col] <= end_dt]
    return df


def _resample_grouped(
    df: pd.DataFrame,
    freq: str,
    value_cols: List[str],
    agg: str = "last",
) -> pd.DataFrame:
    """Resample time-series data grouped by sim_id.

    Args:
        df: DataFrame with sim_id, date, and value columns.
        freq: Pandas frequency string (e.g. 'W', 'M').
        value_cols: Columns to aggregate.
        agg: Aggregation method.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    resampled = (
        df.groupby("sim_id")
        .apply(
            lambda g: g.set_index("date")[value_cols].resample(freq).agg(agg),
            include_groups=False,
        )
        .reset_index()
    )
    return resampled


def _resolve_variables(
    manifest: Dict,
    variables: Optional[List[str]],
    file_key: str = "compartments",
) -> List[str]:
    """Resolve variable names from manifest, defaulting to _total vars."""
    if variables:
        return variables

    files = manifest.get("files", {})
    if file_key in files:
        cols = files[file_key].get("columns", {})
        # Default to _total variables (most useful for agents)
        total_vars = [c for c in cols if c.endswith("_total")]
        if total_vars:
            return total_vars
        # Fall back to all non-index columns
        return [c for c in cols if c not in ("sim_id", "date")]
    return []


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_manifest(path: str, **kwargs) -> Dict[str, Any]:
    """Return the manifest."""
    return load_bundle(path)


def _cmd_quantiles(
    path: str,
    variables: Optional[List[str]] = None,
    quantiles: Optional[List[float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    resample: Optional[str] = None,
    precision: int = 2,
    **kwargs,
) -> Dict[str, Any]:
    """Compute quantiles across simulations for selected variables.

    Args:
        variables: Variable names to compute quantiles for. Defaults to _total vars.
        quantiles: Quantile levels. Defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
        start: Start date for time slice.
        end: End date for time slice.
        resample: Temporal resampling frequency (e.g. 'W', 'M').
        precision: Decimal places for rounding.
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    manifest = load_bundle(path)
    variables = _resolve_variables(manifest, variables)
    cols_to_load = ["sim_id", "date"] + variables

    df = load_bundle_dataframe(path, "compartments", columns=cols_to_load)
    df = _apply_time_slice(df, start, end)

    if resample:
        df = _resample_grouped(df, resample, variables)

    # Compute quantiles
    dates = sorted(df["date"].unique())
    result: Dict[str, Any] = {
        "dates": [str(d)[:10] for d in dates],
    }

    for var in variables:
        pivoted = df.pivot(index="date", columns="sim_id", values=var)
        var_result = {}
        for q in quantiles:
            vals = pivoted.quantile(q, axis=1).values
            var_result[str(q)] = [round(float(v), precision) for v in vals]
        result[var] = var_result

    return result


def _cmd_summary(
    path: str,
    variables: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    precision: int = 2,
    **kwargs,
) -> Dict[str, Any]:
    """Compute summary statistics for selected variables.

    Returns peak date/value, cumulative totals, and final state
    as quantiles across simulations.
    """
    manifest = load_bundle(path)
    variables = _resolve_variables(manifest, variables)
    cols_to_load = ["sim_id", "date"] + variables

    df = load_bundle_dataframe(path, "compartments", columns=cols_to_load)
    df = _apply_time_slice(df, start, end)

    result = {}
    for var in variables:
        pivoted = df.pivot(index="date", columns="sim_id", values=var)
        dates = pivoted.index

        # Peak: per-simulation max, then quantiles of peak values and dates
        peak_values = pivoted.max(axis=0)
        peak_indices = pivoted.idxmax(axis=0)

        # Final value across simulations
        final_values = pivoted.iloc[-1]

        var_summary = {
            "peak_date_median": str(peak_indices.median())[:10] if len(peak_indices) > 0 else None,
            "peak_value": {
                "0.05": round(float(peak_values.quantile(0.05)), precision),
                "0.50": round(float(peak_values.quantile(0.50)), precision),
                "0.95": round(float(peak_values.quantile(0.95)), precision),
            },
            "final_value": {
                "0.05": round(float(final_values.quantile(0.05)), precision),
                "0.50": round(float(final_values.quantile(0.50)), precision),
                "0.95": round(float(final_values.quantile(0.95)), precision),
            },
            "mean_across_time": {
                "0.50": round(float(pivoted.quantile(0.5, axis=1).mean()), precision),
            },
        }
        result[var] = var_summary

    return result


def _cmd_peak(
    path: str,
    variables: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    precision: int = 2,
    **kwargs,
) -> Dict[str, Any]:
    """Find peak timing and magnitude for selected variables."""
    manifest = load_bundle(path)
    variables = _resolve_variables(manifest, variables)
    cols_to_load = ["sim_id", "date"] + variables

    df = load_bundle_dataframe(path, "compartments", columns=cols_to_load)
    df = _apply_time_slice(df, start, end)

    result = {}
    for var in variables:
        pivoted = df.pivot(index="date", columns="sim_id", values=var)

        peak_values = pivoted.max(axis=0)
        peak_dates = pivoted.idxmax(axis=0)

        # Convert peak dates to sortable format for quantile estimation
        peak_dates_ts = pd.to_datetime(peak_dates)
        sorted_dates = peak_dates_ts.sort_values()
        n = len(sorted_dates)

        result[var] = {
            "peak_date": {
                "0.05": str(sorted_dates.iloc[max(0, int(0.05 * n))])[:10] if n > 0 else None,
                "0.50": str(sorted_dates.iloc[int(0.5 * n)])[:10] if n > 0 else None,
                "0.95": str(sorted_dates.iloc[min(n - 1, int(0.95 * n))])[:10] if n > 0 else None,
            },
            "peak_value": {
                "0.05": round(float(peak_values.quantile(0.05)), precision),
                "0.50": round(float(peak_values.quantile(0.50)), precision),
                "0.95": round(float(peak_values.quantile(0.95)), precision),
            },
        }

    return result


def _cmd_posterior(
    path: str,
    generation: Optional[int] = None,
    precision: int = 4,
    **kwargs,
) -> Dict[str, Any]:
    """Summarize posterior distributions from calibration results."""
    df = load_bundle_dataframe(path, "posterior")

    if generation is not None:
        df = df[df["generation"] == generation]
    else:
        # Use last generation
        max_gen = df["generation"].max()
        df = df[df["generation"] == max_gen]

    param_cols = [c for c in df.columns if c != "generation"]
    result = {}
    for col in param_cols:
        vals = df[col].dropna()
        result[col] = {
            "mean": round(float(vals.mean()), precision),
            "std": round(float(vals.std()), precision),
            "ci95": [
                round(float(vals.quantile(0.025)), precision),
                round(float(vals.quantile(0.975)), precision),
            ],
            "median": round(float(vals.median()), precision),
        }

    return result


def _cmd_fit(
    path: str,
    variables: Optional[List[str]] = None,
    quantiles: Optional[List[float]] = None,
    precision: int = 2,
    **kwargs,
) -> Dict[str, Any]:
    """Get calibration fit trajectories (observed vs. simulated quantiles).

    This reads the trajectories from the calibration bundle and computes
    quantiles of the accepted simulations.
    """
    if quantiles is None:
        quantiles = [0.05, 0.5, 0.95]

    bundle_path = Path(path)
    manifest = load_bundle(path)

    # Try to load observed data
    observed_path = bundle_path / "observed_data.parquet"
    observed = None
    if observed_path.exists():
        observed = pd.read_parquet(observed_path)

    # Load trajectories
    traj_path = bundle_path / "trajectories.parquet"
    if not traj_path.exists():
        return {"error": "No trajectories file found in calibration bundle."}

    traj_df = pd.read_parquet(traj_path)

    if variables is None:
        variables = list(traj_df["variable"].unique())

    result: Dict[str, Any] = {}
    for var in variables:
        var_df = traj_df[traj_df["variable"] == var]
        if var_df.empty:
            continue

        pivoted = var_df.pivot(index="timestep", columns="sim_id", values="value")
        var_result = {}
        for q in quantiles:
            vals = pivoted.quantile(q, axis=1).values
            var_result[str(q)] = [round(float(v), precision) for v in vals]
        result[var] = var_result

    if observed is not None:
        result["observed"] = observed.to_dict(orient="list")

    return result
