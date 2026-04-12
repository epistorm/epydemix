"""Write and read .epx result bundles (Parquet + manifest.json).

Requires the ``io`` extra: ``pip install epydemix[io]``
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError:
    raise ImportError(
        "The epydemix.io module requires pyarrow. "
        "Install it with: pip install epydemix[io]"
    )

from .json_utils import NumpySafeEncoder
from .manifest import build_calibration_manifest, build_simulation_manifest


def save_bundle(
    results,
    path: str,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Save simulation or calibration results as an .epx bundle.

    Creates a directory containing Parquet data files and a manifest.json.

    Args:
        results: A SimulationResults or CalibrationResults object.
        path: Path for the bundle directory (e.g. ``"results.epx"``).
        config: Optional config dict to include for reproducibility.

    Returns:
        The manifest dictionary.
    """
    from ..calibration.calibration_results import CalibrationResults
    from ..model.simulation_results import SimulationResults

    import sys

    path = Path(path)
    if path.exists() and (path / "manifest.json").exists():
        print(
            f"Warning: overwriting existing bundle at {path}",
            file=sys.stderr,
        )
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(results, SimulationResults):
        return _save_simulation_bundle(results, path, config)
    elif isinstance(results, CalibrationResults):
        return _save_calibration_bundle(results, path, config)
    else:
        raise TypeError(
            f"Expected SimulationResults or CalibrationResults, got {type(results)}"
        )


def _save_simulation_bundle(
    results,
    path: Path,
    config: Optional[Dict],
) -> Dict[str, Any]:
    """Write simulation results to an .epx bundle."""
    file_sizes = {}

    # Build compartments DataFrame
    comp_rows = []
    for sim_id, traj in enumerate(results.trajectories):
        df = pd.DataFrame(traj.compartments)
        df["sim_id"] = sim_id
        df["date"] = traj.dates
        comp_rows.append(df)
    comp_df = pd.concat(comp_rows, ignore_index=True)

    # Reorder columns: sim_id, date, then data columns
    data_cols = [c for c in comp_df.columns if c not in ("sim_id", "date")]
    comp_df = comp_df[["sim_id", "date"] + data_cols]

    comp_path = path / "compartments.parquet"
    comp_df.to_parquet(comp_path, index=False, engine="pyarrow")
    file_sizes["compartments"] = os.path.getsize(comp_path) / (1024 * 1024)

    # Build transitions DataFrame
    trans_rows = []
    for sim_id, traj in enumerate(results.trajectories):
        df = pd.DataFrame(traj.transitions)
        df["sim_id"] = sim_id
        df["date"] = traj.dates
        trans_rows.append(df)
    trans_df = pd.concat(trans_rows, ignore_index=True)

    data_cols = [c for c in trans_df.columns if c not in ("sim_id", "date")]
    trans_df = trans_df[["sim_id", "date"] + data_cols]

    trans_path = path / "transitions.parquet"
    trans_df.to_parquet(trans_path, index=False, engine="pyarrow")
    file_sizes["transitions"] = os.path.getsize(trans_path) / (1024 * 1024)

    # Build parameters DataFrame (per-simulation parameter values)
    param_rows = []
    for sim_id, traj in enumerate(results.trajectories):
        row = {"sim_id": sim_id}
        for k, v in traj.parameters.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                row[k] = float(v)
        param_rows.append(row)
    if param_rows:
        param_df = pd.DataFrame(param_rows)
        param_path = path / "parameters.parquet"
        param_df.to_parquet(param_path, index=False, engine="pyarrow")
        file_sizes["parameters"] = os.path.getsize(param_path) / (1024 * 1024)

    # Save config if provided
    if config:
        try:
            import yaml
            config_path = path / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            config_path = path / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, cls=NumpySafeEncoder)

    # Create figures/ subdirectory for agent-produced visualizations
    (path / "figures").mkdir(exist_ok=True)

    # Build and write manifest
    manifest = build_simulation_manifest(
        results, str(path), config=config, file_sizes=file_sizes
    )
    manifest["figures"] = {}
    manifest_path = path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpySafeEncoder)

    return manifest


def _save_calibration_bundle(
    results,
    path: Path,
    config: Optional[Dict],
) -> Dict[str, Any]:
    """Write calibration results to an .epx bundle."""
    file_sizes = {}

    # Save posterior distributions (all generations)
    posterior_rows = []
    for gen, df in results.posterior_distributions.items():
        gen_df = df.copy()
        gen_df["generation"] = gen
        posterior_rows.append(gen_df)
    if posterior_rows:
        posterior_df = pd.concat(posterior_rows, ignore_index=True)
        # Move generation column to front
        cols = ["generation"] + [c for c in posterior_df.columns if c != "generation"]
        posterior_df = posterior_df[cols]
        post_path = path / "posterior.parquet"
        posterior_df.to_parquet(post_path, index=False, engine="pyarrow")
        file_sizes["posterior"] = os.path.getsize(post_path) / (1024 * 1024)

    # Save distances
    dist_rows = []
    for gen, dists in results.distances.items():
        for i, d in enumerate(dists):
            dist_rows.append({"generation": gen, "particle_id": i, "distance": d})
    if dist_rows:
        dist_df = pd.DataFrame(dist_rows)
        dist_path = path / "distances.parquet"
        dist_df.to_parquet(dist_path, index=False, engine="pyarrow")
        file_sizes["distances"] = os.path.getsize(dist_path) / (1024 * 1024)

    # Resolve the target variable name from the config (used to label trajectories
    # and observed data with the user-facing variable name rather than the internal
    # "data" key used by ABCSampler).
    target_variable = None
    if config:
        target_variable = config.get("calibration", {}).get("target_variable")

    def _resolve_var_name(internal_key: str) -> str:
        """Map internal 'data' key to the actual target variable name if known."""
        if internal_key == "data" and target_variable:
            return target_variable
        return internal_key

    # Save selected trajectories (last generation only, as compartment arrays)
    selected = results.get_selected_trajectories()
    if selected is not None and len(selected) > 0:
        traj_rows = []
        for sim_id, sim_data in enumerate(selected):
            for key, values in sim_data.items():
                if isinstance(values, np.ndarray) and values.ndim == 1:
                    var_name = _resolve_var_name(key)
                    for t, val in enumerate(values):
                        traj_rows.append({
                            "sim_id": sim_id,
                            "timestep": t,
                            "variable": var_name,
                            "value": float(val),
                        })
        if traj_rows:
            traj_df = pd.DataFrame(traj_rows)
            traj_path = path / "trajectories.parquet"
            traj_df.to_parquet(traj_path, index=False, engine="pyarrow")
            file_sizes["trajectories"] = os.path.getsize(traj_path) / (1024 * 1024)

    # Save particle weights (needed for posterior-weighted projection sampling)
    weight_rows = []
    for gen, w in results.weights.items():
        w_arr = np.asarray(w, dtype=float)
        for i, wt in enumerate(w_arr):
            weight_rows.append({"generation": gen, "particle_id": i, "weight": wt})
    if weight_rows:
        weight_df = pd.DataFrame(weight_rows)
        weight_path = path / "weights.parquet"
        weight_df.to_parquet(weight_path, index=False, engine="pyarrow")
        file_sizes["weights"] = os.path.getsize(weight_path) / (1024 * 1024)

    # Save observed data used for calibration so the fit command can compare it
    # against the accepted trajectories.
    if results.observed_data is not None:
        obs_rows = []
        for internal_key, values in results.observed_data.items():
            var_name = _resolve_var_name(internal_key)
            for t, val in enumerate(np.asarray(values)):
                obs_rows.append({
                    "timestep": t,
                    "variable": var_name,
                    "value": float(val),
                })
        if obs_rows:
            obs_df = pd.DataFrame(obs_rows)
            obs_path = path / "observed_data.parquet"
            obs_df.to_parquet(obs_path, index=False, engine="pyarrow")
            file_sizes["observed_data"] = os.path.getsize(obs_path) / (1024 * 1024)

    # Save config
    if config:
        try:
            import yaml
            with open(path / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            with open(path / "config.json", "w") as f:
                json.dump(config, f, indent=2, cls=NumpySafeEncoder)

    # Create figures/ subdirectory for agent-produced visualizations
    (path / "figures").mkdir(exist_ok=True)

    # Build and write manifest
    manifest = build_calibration_manifest(
        results, str(path), config=config, file_sizes=file_sizes
    )
    manifest["figures"] = {}
    with open(path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpySafeEncoder)

    return manifest


def load_bundle(path: str) -> Dict[str, Any]:
    """Load a bundle's manifest and return it.

    This does NOT load the full data — it reads only the manifest.
    Use :func:`load_bundle_dataframe` to load specific Parquet files.

    Args:
        path: Path to the .epx bundle directory.

    Returns:
        The manifest dictionary.

    Raises:
        FileNotFoundError: If the bundle or manifest doesn't exist.
    """
    path = Path(path)
    manifest_path = path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json found in '{path}'. Is this a valid .epx bundle?"
        )

    with open(manifest_path, "r") as f:
        return json.load(f)


def load_bundle_dataframe(
    path: str,
    file_key: str,
    columns: Optional[List[str]] = None,
    filters: Optional[List] = None,
) -> pd.DataFrame:
    """Load a specific Parquet file from a bundle.

    Args:
        path: Path to the .epx bundle directory.
        file_key: Key in the manifest's ``files`` section, e.g.
            ``"compartments"``, ``"transitions"``, ``"parameters"``,
            ``"posterior"``.
        columns: Optional list of columns to load (for efficiency).
        filters: Optional pyarrow filters for row-group pruning, e.g.
            ``[("sim_id", "<", 10)]``.

    Returns:
        A pandas DataFrame with the requested data.
    """
    bundle_path = Path(path)
    manifest = load_bundle(path)

    if file_key not in manifest.get("files", {}):
        available = list(manifest.get("files", {}).keys())
        raise KeyError(
            f"File key '{file_key}' not found in bundle. Available: {available}"
        )

    file_info = manifest["files"][file_key]
    file_path = bundle_path / file_info["path"]

    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    return pd.read_parquet(file_path, columns=columns, filters=filters)


def add_figure_to_manifest(
    bundle_path: str,
    filename: str,
    description: str,
    variables: Optional[List[str]] = None,
) -> None:
    """Register a figure in the bundle's manifest.

    Call this after saving a plot to ``<bundle>/figures/<filename>``.
    It updates ``manifest.json`` so that subsequent ``inspect manifest``
    calls include the figure metadata.

    Args:
        bundle_path: Path to the ``.epx`` bundle directory.
        filename: Name of the figure file inside ``figures/``
            (e.g. ``"epidemic_curve.png"``).
        description: One-line description of what the figure shows.
        variables: Optional list of variable names depicted in the figure.

    Raises:
        FileNotFoundError: If the bundle, manifest, or figure file is missing.
    """
    bp = Path(bundle_path)
    manifest_path = bp / "manifest.json"
    figure_path = bp / "figures" / filename

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json found in '{bp}'. Is this a valid .epx bundle?"
        )
    if not figure_path.exists():
        raise FileNotFoundError(
            f"Figure file not found: {figure_path}. "
            f"Save it to <bundle>/figures/ before registering."
        )

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if "figures" not in manifest:
        manifest["figures"] = {}

    entry: Dict[str, Any] = {"description": description}
    if variables:
        entry["variables"] = variables

    manifest["figures"][filename] = entry

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpySafeEncoder)
