"""Inspection engine for querying .epx bundles.

Each function reads from the on-disk Parquet files, computes an answer,
and returns a compact dictionary suitable for JSON serialization to stdout.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .bundle import load_bundle, load_bundle_dataframe


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
    **kwargs,
) -> Dict[str, Any]:
    """Compute quantiles across simulations for selected variables.

    Args:
        variables: Variable names to compute quantiles for. Defaults to _total vars.
        quantiles: Quantile levels. Defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
        start: Start date for time slice.
        end: End date for time slice.
        resample: Temporal resampling frequency (e.g. 'W', 'M').
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
            var_result[str(q)] = [float(v) for v in vals]
        result[var] = var_result

    return result


def _cmd_summary(
    path: str,
    variables: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
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
                "0.05": float(peak_values.quantile(0.05)),
                "0.50": float(peak_values.quantile(0.50)),
                "0.95": float(peak_values.quantile(0.95)),
            },
            "final_value": {
                "0.05": float(final_values.quantile(0.05)),
                "0.50": float(final_values.quantile(0.50)),
                "0.95": float(final_values.quantile(0.95)),
            },
            "mean_across_time": {
                "0.50": float(pivoted.quantile(0.5, axis=1).mean()),
            },
        }
        result[var] = var_summary

    return result


def _cmd_peak(
    path: str,
    variables: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
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
                "0.05": float(peak_values.quantile(0.05)),
                "0.50": float(peak_values.quantile(0.50)),
                "0.95": float(peak_values.quantile(0.95)),
            },
        }

    return result


def _cmd_posterior(
    path: str,
    generation: Optional[int] = None,
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
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "ci95": [
                float(vals.quantile(0.025)),
                float(vals.quantile(0.975)),
            ],
            "median": float(vals.median()),
        }

    return result


def _cmd_fit(
    path: str,
    variables: Optional[List[str]] = None,
    quantiles: Optional[List[float]] = None,
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

    # The trajectory "variable" column may be "data" (the internal key used by
    # ABCSampler) rather than the user-facing variable name (e.g. "Infected_total").
    # Build a mapping from user-facing name → trajectory key so that both work.
    traj_vars = set(traj_df["variable"].unique())
    target_variable = manifest.get("calibration", {}).get("target_variable")

    if variables is None:
        # Default: expose the target variable name if known, else raw keys
        if target_variable and "data" in traj_vars:
            variables = [target_variable]
        else:
            variables = list(traj_vars)

    # Map user-requested variable names to actual keys in the parquet.
    # "data" is the internal fallback key written by ABCSampler when only one
    # target variable is calibrated.
    def _resolve_traj_key(var: str) -> str:
        if var in traj_vars:
            return var
        if "data" in traj_vars and var == target_variable:
            return "data"
        return var  # will produce empty df → skipped below

    result: Dict[str, Any] = {}
    for var in variables:
        traj_key = _resolve_traj_key(var)
        var_df = traj_df[traj_df["variable"] == traj_key]
        if var_df.empty:
            continue

        pivoted = var_df.pivot(index="timestep", columns="sim_id", values="value")
        var_result = {}
        for q in quantiles:
            vals = pivoted.quantile(q, axis=1).values
            var_result[str(q)] = [float(v) for v in vals]
        result[var] = var_result

    if observed is not None:
        # observed_data.parquet is in long format: [timestep, variable, value].
        # Reshape to {"var_name": [values...]} so agents can do fit["observed"][VAR].
        obs_dict: Dict[str, Any] = {}
        for var_name, grp in observed.groupby("variable"):
            grp_sorted = grp.sort_values("timestep")
            obs_dict[str(var_name)] = [
                float(v) for v in grp_sorted["value"].values
            ]
        result["observed"] = obs_dict

    return result


# ---------------------------------------------------------------------------
# Cross-bundle comparison
# ---------------------------------------------------------------------------

# Built-in metric registry — each metric is a function that takes
# (comp_df, manifest, **kwargs) and returns a dict with "value" and
# optionally "ci90" (a two-element list).

def _metric_attack_rate(comp, manifest, **kw):
    """Fraction of population ever infected (recovered + dead)."""
    # Find terminal compartments (R-like + D-like)
    total_cols = [c for c in comp.columns if c.endswith("_total")
                  and c not in ("sim_id", "date")]
    # Heuristic: susceptible is the only compartment that decreases monotonically
    # Attack rate = 1 - S_final / S_initial
    s_col = None
    for c in total_cols:
        if c.startswith("S") or c.startswith("Susceptible"):
            s_col = c
            break
    if s_col is None:
        return {"value": None, "note": "no susceptible compartment found"}

    first_day = comp.groupby("sim_id")[s_col].first()
    last_day = comp.groupby("sim_id")[s_col].last()
    attack = 1.0 - last_day / first_day
    return {
        "median": float(attack.median()) * 100,
        "ci90": [float(attack.quantile(0.05)) * 100,
                 float(attack.quantile(0.95)) * 100],
        "units": "percent",
    }


def _metric_peak(comp, manifest, variable=None, **kw):
    """Peak value of a variable across simulations."""
    var = variable or _default_var(comp, prefix="I")
    if var not in comp.columns:
        return {"value": None, "note": f"{var} not found"}
    peaks = comp.groupby("sim_id")[var].max()
    return {
        "median": float(peaks.median()),
        "ci90": [float(peaks.quantile(0.05)),
                 float(peaks.quantile(0.95))],
        "variable": var,
    }


def _metric_peak_date(comp, manifest, variable=None, **kw):
    """Date of peak value."""
    var = variable or _default_var(comp, prefix="I")
    if var not in comp.columns:
        return {"value": None, "note": f"{var} not found"}
    comp_sorted = comp.sort_values("date")
    peak_dates = comp_sorted.groupby("sim_id").apply(
        lambda g: g.loc[g[var].idxmax(), "date"], include_groups=False
    )
    peak_dates = pd.to_datetime(peak_dates).sort_values()
    n = len(peak_dates)
    return {
        "median": str(peak_dates.iloc[int(0.5 * n)])[:10] if n > 0 else None,
        "ci90": [
            str(peak_dates.iloc[max(0, int(0.05 * n))])[:10],
            str(peak_dates.iloc[min(n - 1, int(0.95 * n))])[:10],
        ] if n > 0 else None,
        "variable": var,
    }


def _metric_total_deaths(comp, manifest, **kw):
    """Total deaths at end of simulation."""
    d_col = None
    for c in comp.columns:
        if c.endswith("_total") and (c.startswith("D") or c.startswith("Dead")):
            d_col = c
            break
    if d_col is None:
        return {"value": None, "note": "no death compartment found"}
    final = comp.groupby("sim_id")[d_col].last()
    return {
        "median": float(final.median()),
        "ci90": [float(final.quantile(0.05)),
                 float(final.quantile(0.95))],
        "variable": d_col,
    }


def _metric_days_over(comp, manifest, variable=None, threshold=None, **kw):
    """Number of days the median of a variable exceeds a threshold."""
    if threshold is None:
        return {"value": None, "note": "threshold required (e.g. days_over:500)"}
    threshold = float(threshold)
    var = variable or _default_var(comp, prefix="H")
    if var not in comp.columns:
        return {"value": None, "note": f"{var} not found"}
    median_ts = comp.groupby("date")[var].median()
    days = int((median_ts > threshold).sum())
    return {
        "value": days,
        "threshold": threshold,
        "variable": var,
    }


def _metric_final_value(comp, manifest, variable=None, **kw):
    """Final value of a variable across simulations."""
    var = variable or _default_var(comp, prefix="I")
    if var not in comp.columns:
        return {"value": None, "note": f"{var} not found"}
    final = comp.groupby("sim_id")[var].last()
    return {
        "median": float(final.median()),
        "ci90": [float(final.quantile(0.05)),
                 float(final.quantile(0.95))],
        "variable": var,
    }


def _default_var(comp, prefix="I"):
    """Find the first _total column matching a prefix."""
    for c in comp.columns:
        if c.endswith("_total") and c.split("_")[0].startswith(prefix):
            return c
    # Fall back to first _total column that isn't S
    for c in comp.columns:
        if c.endswith("_total") and not c.startswith("S"):
            return c
    return None


COMPARE_METRICS = {
    "attack_rate": _metric_attack_rate,
    "peak": _metric_peak,
    "peak_date": _metric_peak_date,
    "total_deaths": _metric_total_deaths,
    "days_over": _metric_days_over,
    "final_value": _metric_final_value,
}


def compare_bundles(
    bundles: Dict[str, str],
    metrics: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare multiple bundles on a set of metrics.

    Args:
        bundles: Mapping of scenario name → bundle path.
        metrics: List of metric names to compute. Each can be a plain name
            (e.g. ``"attack_rate"``) or include a parameter after a colon
            (e.g. ``"days_over:500"``).  Defaults to all standard metrics
            (excluding ``days_over`` which requires a threshold).
        variables: Optional list of variable names to pass to variable-specific
            metrics (peak, peak_date, final_value).  If not provided, metrics
            will auto-detect the most relevant variable.

    Returns:
        A dict with structure ``{scenario_name: {metric_name: result}}``.
    """
    if metrics is None:
        metrics = ["attack_rate", "peak", "peak_date", "total_deaths"]

    # Parse metric specs (name or name:param)
    parsed_metrics = []
    for m in metrics:
        if ":" in m:
            name, param = m.split(":", 1)
            parsed_metrics.append((name, param))
        else:
            parsed_metrics.append((m, None))

    # Validate metric names
    for name, _ in parsed_metrics:
        if name not in COMPARE_METRICS:
            raise ValueError(
                f"Unknown metric '{name}'. "
                f"Available: {list(COMPARE_METRICS.keys())}"
            )

    result = {}
    for scenario_name, bundle_path in bundles.items():
        manifest = load_bundle(bundle_path)
        comp = load_bundle_dataframe(bundle_path, "compartments")

        scenario_result = {}
        for metric_name, metric_param in parsed_metrics:
            fn = COMPARE_METRICS[metric_name]
            kwargs = {}
            if variables:
                kwargs["variable"] = variables[0]
            if metric_param is not None:
                kwargs["threshold"] = metric_param
            scenario_result[metric_name] = fn(comp, manifest, **kwargs)

        result[scenario_name] = scenario_result

    return result
