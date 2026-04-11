"""Load and validate YAML/JSON configs, build EpiModel instances from them."""

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..model.epimodel import EpiModel
from ..model.predefined_models import SUPPORTED_MODELS, load_predefined_model


def _load_raw(path: str) -> Dict[str, Any]:
    """Load a single config file without resolving inheritance."""
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
            return yaml.safe_load(f) or {}
        elif suffix == ".json":
            return json.load(f)
        else:
            content = f.read()
            try:
                import yaml
                return yaml.safe_load(content) or {}
            except (ImportError, Exception):
                return json.loads(content)


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Deep-merge *overlay* onto *base*, returning a new dict.

    - Dicts are merged recursively.
    - All other types (lists, scalars) in *overlay* replace the base value.

    This means lists like ``overrides`` and ``interventions`` are replaced
    wholesale, giving the overlay full control.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


_MAX_INHERITANCE_DEPTH = 10


def load_config(path: str) -> Dict[str, Any]:
    """Load a config from a YAML or JSON file, resolving inheritance.

    If the config contains a ``base`` key, the referenced config is loaded
    first and the current config is deep-merged on top of it.  Inheritance
    chains are followed up to 10 levels deep.  The ``base`` key is resolved
    relative to the directory of the file that contains it.

    Args:
        path: Path to the config file.

    Returns:
        Fully resolved config dictionary (``base`` key removed).

    Raises:
        FileNotFoundError: If any file in the chain doesn't exist.
        ValueError: If the inheritance chain exceeds the depth limit.
    """
    return _resolve_config(path, depth=0, seen=set())


def _resolve_config(path: str, depth: int, seen: set) -> Dict[str, Any]:
    """Recursive config loader with cycle and depth detection."""
    real = str(Path(path).resolve())
    if real in seen:
        raise ValueError(f"Circular config inheritance detected: {real}")
    if depth > _MAX_INHERITANCE_DEPTH:
        raise ValueError(
            f"Config inheritance too deep (>{_MAX_INHERITANCE_DEPTH} levels)"
        )

    seen = seen | {real}
    config = _load_raw(path)

    base_ref = config.pop("base", None)
    if base_ref is not None:
        # Resolve relative to the directory of the current config file
        base_path = str((Path(path).parent / base_ref).resolve())
        base_config = _resolve_config(base_path, depth + 1, seen)
        config = _deep_merge(base_config, config)

    return config


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

    # Population config (read early so size is available at model construction)
    pop_cfg = config.get("population", {})

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
            default_population_size=pop_cfg.get("size", 100_000),
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
