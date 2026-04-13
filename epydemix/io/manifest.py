"""Build manifest.json metadata from simulation and calibration results."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from importlib.metadata import version, PackageNotFoundError


def _get_version() -> str:
    try:
        return version("epydemix")
    except PackageNotFoundError:
        return "unknown"


def _scalar_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only JSON-safe scalar parameters."""
    result = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool)):
            result[k] = v
        elif isinstance(v, np.integer):
            result[k] = int(v)
        elif isinstance(v, np.floating):
            result[k] = float(v)
        # Skip arrays, matrices, etc.
    return result


def _describe_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Build column schema from a DataFrame."""
    cols = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "int" in dtype:
            dtype_str = "int32"
        elif "float" in dtype:
            dtype_str = "float64"
        elif "datetime" in dtype or "date" in dtype:
            dtype_str = "date"
        elif "object" in dtype:
            dtype_str = "string"
        else:
            dtype_str = dtype
        cols[col] = {"dtype": dtype_str}
    return cols


def build_simulation_manifest(
    results,  # SimulationResults
    bundle_path: str,
    config: Optional[Dict] = None,
    file_sizes: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build a manifest dict for simulation results.

    Args:
        results: A SimulationResults object.
        bundle_path: Path to the .epx bundle directory.
        config: Optional config dict that produced this run.
        file_sizes: Optional dict of filename → size_mb.
    """
    if not results.trajectories:
        raise ValueError("Cannot build manifest from empty SimulationResults")

    traj0 = results.trajectories[0]
    dates = traj0.dates
    comp_names = list(traj0.compartments.keys())
    trans_names = list(traj0.transitions.keys())
    n_timesteps = len(dates)
    n_sims = results.Nsim

    # Extract base compartment names (without demographic suffixes)
    base_compartments = sorted(set(
        name.rsplit("_", 1)[0] for name in comp_names
        if name.endswith("_total")
    ))

    # Demographic groups from non-total compartment names
    demo_groups = []
    if base_compartments:
        prefix = base_compartments[0] + "_"
        demo_groups = [
            name[len(prefix):] for name in comp_names
            if name.startswith(prefix) and not name.endswith("_total")
        ]

    manifest = {
        "type": "SimulationResults",
        "epydemix_version": _get_version(),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": {
            "compartments": base_compartments,
        },
        "population": {
            "demographic_groups": demo_groups,
        },
        "simulation": {
            "n_simulations": n_sims,
            "start_date": str(dates[0])[:10] if len(dates) > 0 else None,
            "end_date": str(dates[-1])[:10] if len(dates) > 0 else None,
            "n_timesteps": n_timesteps,
        },
        "parameters_used": _scalar_params(results.parameters),
        "files": {
            "compartments": {
                "path": "compartments.parquet",
                "description": "Compartment counts per simulation, date, and demographic group.",
                "shape": [n_sims * n_timesteps, len(comp_names) + 2],
                "index": ["sim_id", "date"],
                "columns": {
                    "sim_id": {"dtype": "int32", "description": "Simulation index (0 to n_simulations-1)"},
                    "date": {"dtype": "date", "description": "Simulation date"},
                    **{name: {"dtype": "float64"} for name in comp_names},
                },
            },
            "transitions": {
                "path": "transitions.parquet",
                "description": "Daily transition counts per simulation and date.",
                "shape": [n_sims * n_timesteps, len(trans_names) + 2],
                "index": ["sim_id", "date"],
                "columns": {
                    "sim_id": {"dtype": "int32"},
                    "date": {"dtype": "date"},
                    **{name: {"dtype": "float64"} for name in trans_names},
                },
            },
            "parameters": {
                "path": "parameters.parquet",
                "description": "Parameter values per simulation.",
            },
        },
        "usage_hints": {
            "inspect_cli": "Use `epydemix inspect <bundle> <command>` for common queries (quantiles, summary, peak, export).",
            "custom_python": (
                "Load files with `pd.read_parquet('<bundle>/compartments.parquet')`. "
                "Filter by sim_id or select specific columns to keep memory manageable."
            ),
            "parquet_row_groups": (
                "Files are partitioned by sim_id for efficient partial reads via "
                "`pd.read_parquet(..., filters=[('sim_id', '<', 10)])`."
            ),
        },
    }

    if config:
        manifest["files"]["config"] = {
            "path": "config.yaml",
            "description": "The configuration that produced this run.",
        }

    # Add file sizes if available
    if file_sizes:
        for key, size in file_sizes.items():
            if key in manifest["files"]:
                manifest["files"][key]["size_mb"] = round(size, 2)

    return manifest


def build_calibration_manifest(
    results,  # CalibrationResults
    bundle_path: str,
    config: Optional[Dict] = None,
    file_sizes: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build a manifest dict for calibration results."""
    generations = list(results.posterior_distributions.keys())
    n_generations = len(generations)

    # Get posterior info from last generation
    posterior_df = results.get_posterior_distribution()
    param_names = list(posterior_df.columns) if posterior_df is not None else []
    n_particles = len(posterior_df) if posterior_df is not None else 0

    target_variable = None
    if config:
        target_variable = config.get("calibration", {}).get("target_variable")

    calibration_section = {
        "strategy": results.calibration_strategy,
        "n_generations": n_generations,
        "n_particles": n_particles,
        "calibrated_parameters": param_names,
    }
    if target_variable:
        calibration_section["target_variable"] = target_variable

    # Extract population metadata from config if available
    population_section = None
    if config:
        pop_cfg = config.get("population", {})
        if pop_cfg:
            population_section = {}
            if "name" in pop_cfg:
                population_section["name"] = pop_cfg["name"]
            if "size" in pop_cfg:
                population_section["size"] = pop_cfg["size"]
            if "contact_layers" in pop_cfg:
                population_section["contact_layers"] = pop_cfg["contact_layers"]

    manifest = {
        "type": "CalibrationResults",
        "epydemix_version": _get_version(),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "calibration": calibration_section,
        "population": population_section,
        "parameters_used": _scalar_params(results.calibration_params),
        "files": {
            "posterior": {
                "path": "posterior.parquet",
                "description": "Parameter posterior samples per generation.",
                "columns": {
                    "generation": {"dtype": "int32"},
                    **{name: {"dtype": "float64"} for name in param_names},
                },
            },
            "distances": {
                "path": "distances.parquet",
                "description": "Distance values per generation.",
            },
            "weights": {
                "path": "weights.parquet",
                "description": "Particle weights per generation "
                               "(used for posterior-weighted sampling in projections).",
                "columns": {
                    "generation": {"dtype": "int32"},
                    "particle_id": {"dtype": "int32"},
                    "weight": {"dtype": "float64"},
                },
            },
            "trajectories": {
                "path": "trajectories.parquet",
                "description": "Accepted simulated trajectories (last generation). "
                               "Long format: [sim_id, timestep, variable, value]. "
                               "Use `epydemix inspect <bundle> fit` to query.",
                "columns": {
                    "sim_id": {"dtype": "int64"},
                    "timestep": {"dtype": "int64"},
                    "variable": {"dtype": "str"},
                    "value": {"dtype": "float64"},
                },
            },
            "observed_data": {
                "path": "observed_data.parquet",
                "description": "Observed data used for calibration. "
                               "Long format: [timestep, variable, value].",
                "columns": {
                    "timestep": {"dtype": "int64"},
                    "variable": {"dtype": "str"},
                    "value": {"dtype": "float64"},
                },
            },
        },
        "usage_hints": {
            "inspect_cli": "Use `epydemix inspect <bundle> posterior` or `epydemix inspect <bundle> fit` for common queries.",
            "custom_python": "Load with `pd.read_parquet('<bundle>/posterior.parquet')`.",
        },
    }

    if config:
        manifest["files"]["config"] = {
            "path": "config.yaml",
            "description": "The configuration that produced this run.",
        }

    if file_sizes:
        for key, size in file_sizes.items():
            if key in manifest["files"]:
                manifest["files"][key]["size_mb"] = round(size, 2)

    return manifest
