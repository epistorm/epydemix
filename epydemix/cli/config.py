"""Load and validate YAML/JSON configs, build EpiModel instances from them."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..model.epimodel import EpiModel
from ..model.predefined_models import SUPPORTED_MODELS, load_predefined_model


def load_config(path: str) -> Dict[str, Any]:
    """Load a config from a YAML or JSON file.

    Args:
        path: Path to the config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ImportError: If YAML is needed but pyyaml is not installed.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = filepath.suffix.lower()
    with open(filepath, "r") as f:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Loading YAML configs requires pyyaml. "
                    "Install it with: pip install epydemix[cli]"
                )
            return yaml.safe_load(f)
        elif suffix == ".json":
            return json.load(f)
        else:
            # Try YAML first, fall back to JSON
            content = f.read()
            try:
                import yaml
                return yaml.safe_load(content)
            except (ImportError, Exception):
                return json.loads(content)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a config dict and return structured errors/warnings.

    Returns:
        A dict with ``valid`` (bool), ``errors`` (list), ``warnings`` (list).
    """
    errors = []
    warnings = []

    # Must have model section
    if "model" not in config:
        errors.append("Missing required section: 'model'")

    # Must have simulation section
    if "simulation" not in config:
        errors.append("Missing required section: 'simulation'")
    else:
        sim = config["simulation"]
        if "start_date" not in sim:
            errors.append("simulation.start_date is required")
        if "end_date" not in sim:
            errors.append("simulation.end_date is required")

    # Model section validation
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "custom")
    if model_type != "custom" and model_type not in SUPPORTED_MODELS:
        errors.append(
            f"Unknown model type '{model_type}'. "
            f"Supported: {SUPPORTED_MODELS + ['custom']}"
        )

    if model_type == "custom":
        if "compartments" not in model_cfg:
            errors.append("Custom model requires 'model.compartments'")
        if "transitions" not in model_cfg:
            errors.append("Custom model requires 'model.transitions'")

    # Parameters section
    if "parameters" not in config:
        warnings.append("No 'parameters' section — will use model defaults")

    # Initial conditions
    if "initial_conditions" not in config:
        warnings.append(
            "No 'initial_conditions' section — will use default "
            "(small fraction in first infectious compartment)"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def build_model_from_config(config: Dict[str, Any]) -> EpiModel:
    """Build an EpiModel from a validated config dict.

    Args:
        config: A config dictionary (as loaded from YAML/JSON).

    Returns:
        A configured EpiModel ready to run simulations.
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "custom")
    params = config.get("parameters", {})

    # Build model
    if model_type in SUPPORTED_MODELS:
        model = load_predefined_model(model_type, **params)
    else:
        # Custom model
        compartments = model_cfg.get("compartments", [])
        model = EpiModel(
            compartments=compartments,
            parameters=params,
            use_default_population=True,
        )
        # Add transitions
        for tr in model_cfg.get("transitions", []):
            tr_params = tr["params"]
            # Normalize params: YAML list → tuple for mediated transitions
            if isinstance(tr_params, list):
                tr_params = tuple(tr_params)
            model.add_transition(
                source=tr["source"],
                target=tr["target"],
                kind=tr["kind"],
                params=tr_params,
            )

    # Population
    pop_cfg = config.get("population", {})
    if "name" in pop_cfg and pop_cfg["name"] != "default":
        model.import_epydemix_population(
            population_name=pop_cfg["name"],
            layers=pop_cfg.get("contact_layers"),
        )

    # Override parameters (if different from what predefined model set)
    if model_type in SUPPORTED_MODELS and params:
        for name, value in params.items():
            model.parameters[name] = value

    # Interventions
    for intv in config.get("interventions", []):
        model.add_intervention(
            layer_name=intv["layer"],
            start_date=intv["start_date"],
            end_date=intv["end_date"],
            reduction_factor=intv.get("reduction"),
        )

    # Parameter overrides (time-varying)
    for ovr in config.get("overrides", []):
        model.override_parameter(
            start_date=ovr["start_date"],
            end_date=ovr["end_date"],
            parameter_name=ovr["parameter"],
            value=ovr["value"],
        )

    return model


def build_initial_conditions(
    config: Dict[str, Any],
    model: EpiModel,
) -> Optional[Dict[str, np.ndarray]]:
    """Build initial conditions dict from config.

    Args:
        config: The full config dict.
        model: The configured model (to know compartments and population size).

    Returns:
        Initial conditions dict, or None to use defaults.
    """
    ic_cfg = config.get("initial_conditions")
    if not ic_cfg:
        return None

    n_groups = len(model.population.Nk)
    pop_sizes = model.population.Nk

    ic_dict = {}
    for comp_name, fraction in ic_cfg.items():
        if comp_name in model.compartments:
            ic_dict[comp_name] = np.array(pop_sizes) * fraction
        else:
            # Try to find a match (handle case differences)
            for real_name in model.compartments:
                if real_name.lower() == comp_name.lower():
                    ic_dict[real_name] = np.array(pop_sizes) * fraction
                    break

    return ic_dict if ic_dict else None


def run_from_config(config: Dict[str, Any]) -> Tuple[Any, Dict]:
    """Build and run a simulation from a config dict.

    Returns:
        Tuple of (SimulationResults, manifest_dict).
    """
    model = build_model_from_config(config)
    sim_cfg = config.get("simulation", {})
    output_cfg = config.get("output", {})

    ic = build_initial_conditions(config, model)

    results = model.run_simulations(
        start_date=sim_cfg["start_date"],
        end_date=sim_cfg["end_date"],
        Nsim=sim_cfg.get("n_simulations", 100),
        dt=sim_cfg.get("dt", 1.0),
        initial_conditions_dict=ic,
    )

    return results, config
