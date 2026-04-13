"""Load and validate YAML/JSON configs, build EpiModel instances from them."""

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        else:
            _BUILTIN_KINDS = {"spontaneous", "mediated", "scheduled"}
            for i, tr in enumerate(model_cfg.get("transitions", [])):
                kind = tr.get("kind")
                if kind == "scheduled" and "schedule" not in tr:
                    errors.append(
                        f"transitions[{i}]: kind 'scheduled' requires a 'schedule' field "
                        "(path to a CSV file or an inline list)"
                    )
                if kind in _BUILTIN_KINDS and kind != "scheduled" and "params" not in tr:
                    errors.append(
                        f"transitions[{i}]: kind '{kind}' requires a 'params' field"
                    )
                if kind not in _BUILTIN_KINDS:
                    warnings.append(
                        f"transitions[{i}]: kind '{kind}' is not a built-in kind "
                        f"({sorted(_BUILTIN_KINDS)}); ensure it is registered via "
                        "register_transition_kind() before running"
                    )

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


def _load_schedule(
    schedule_spec,
    config_dir: Optional[Path],
    start_date: str,
    end_date: str,
    n_groups: int,
    group_names: Optional[list] = None,
) -> np.ndarray:
    """Load and align a vaccination (or any dose) schedule to simulation dates.

    The ``schedule_spec`` can be:

    * A file path (str) to a CSV whose first column is a date index and the
      remaining columns are daily doses per demographic group.  The file is
      resolved relative to *config_dir*.  If the CSV columns match
      ``group_names`` (in any order), they are reordered to match the model's
      group ordering; otherwise columns are taken positionally.
    * An inline list of numbers (broadcast to all groups) or a list of lists
      (one inner list per timestep, one value per group).

    Missing dates are filled with zero.  A single-column CSV is broadcast to
    all groups.  Returns an array of shape ``(T, n_groups)``.
    """
    import pandas as pd
    from ..utils.utils import compute_simulation_dates

    dates = compute_simulation_dates(start_date, end_date)
    T = len(dates)

    if isinstance(schedule_spec, list):
        arr = np.array(schedule_spec, dtype=float)
        if arr.ndim == 1:
            # flat list → broadcast across groups
            if len(arr) != T:
                raise ValueError(
                    f"Inline schedule has {len(arr)} entries but simulation has {T} timesteps"
                )
            arr = np.tile(arr[:, np.newaxis], (1, n_groups))
        elif arr.ndim == 2:
            if arr.shape[0] != T:
                raise ValueError(
                    f"Inline schedule has {arr.shape[0]} rows but simulation has {T} timesteps"
                )
            if arr.shape[1] == 1 and n_groups > 1:
                arr = np.tile(arr, (1, n_groups))
            elif arr.shape[1] != n_groups:
                raise ValueError(
                    f"Inline schedule has {arr.shape[1]} columns but model has {n_groups} groups"
                )
        return arr

    # File path
    obs_path = Path(schedule_spec)
    if not obs_path.is_absolute() and config_dir is not None:
        obs_path = config_dir / obs_path
    if not obs_path.exists():
        raise FileNotFoundError(f"Schedule file not found: {obs_path}")

    df = pd.read_csv(obs_path, index_col=0, parse_dates=True)
    date_index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    df = df.reindex(date_index, fill_value=0.0)

    # If the CSV has named columns that match the population group names, reorder
    # them to match the model's group ordering rather than relying on column position.
    if group_names is not None and len(df.columns) > 1:
        csv_cols = [str(c) for c in df.columns]
        group_names_str = [str(g) for g in group_names]
        if set(csv_cols) == set(group_names_str):
            df = df[group_names_str]

    arr = df.values.astype(float)
    if arr.shape[1] == 1 and n_groups > 1:
        arr = np.tile(arr, (1, n_groups))
    elif arr.shape[1] != n_groups:
        raise ValueError(
            f"Schedule file has {arr.shape[1]} columns but model has {n_groups} demographic groups"
        )
    return arr


def build_model_from_config(
    config: Dict[str, Any],
    config_dir: Optional[Path] = None,
) -> EpiModel:
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
        # Apply population size from config (predefined models default to 100,000)
        if "size" in pop_cfg and "name" not in pop_cfg:
            model.population.Nk = np.array([pop_cfg["size"]], dtype=float)
    else:
        # Custom model
        compartments = model_cfg.get("compartments", [])
        model = EpiModel(
            compartments=compartments,
            parameters=params,
            use_default_population=True,
            default_population_size=pop_cfg.get("size", 100_000),
        )

    # Population — load before transitions so n_groups is correct when
    # schedule files are resolved (e.g. kind: scheduled needs the real n_groups).
    if "name" in pop_cfg and pop_cfg["name"] != "default":
        model.import_epydemix_population(
            population_name=pop_cfg["name"],
            contact_layers=pop_cfg.get("contact_layers"),
        )

    if model_type not in SUPPORTED_MODELS:
        # Add transitions now that the population (and its n_groups) is known
        sim_cfg = config.get("simulation", {})
        for tr in model_cfg.get("transitions", []):
            tr_kind = tr["kind"]
            if tr_kind == "scheduled":
                # Load dose schedule and build params tuple
                dose_array = _load_schedule(
                    tr["schedule"],
                    config_dir=config_dir,
                    start_date=sim_cfg["start_date"],
                    end_date=sim_cfg["end_date"],
                    n_groups=len(model.population.Nk),
                    group_names=list(model.population.Nk_names),
                )
                eligible = tr.get("eligible")
                tr_params = (dose_array, eligible) if eligible else (dose_array,)
            else:
                tr_params = tr["params"]
                # Normalize params: YAML list → tuple for mediated transitions
                if isinstance(tr_params, list):
                    tr_params = tuple(tr_params)
            model.add_transition(
                source=tr["source"],
                target=tr["target"],
                kind=tr_kind,
                params=tr_params,
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


def run_from_config(
    config: Dict[str, Any],
    config_dir: Optional[Path] = None,
) -> Tuple[Any, Dict]:
    """Build and run a simulation from a config dict.

    Args:
        config: The full config dict.
        config_dir: Directory of the config file, used to resolve relative
            paths inside the config (e.g. ``schedule`` files for ``scheduled``
            transitions).

    Returns:
        Tuple of (SimulationResults, manifest_dict).
    """
    model = build_model_from_config(config, config_dir=config_dir)
    sim_cfg = config.get("simulation", {})

    ic = build_initial_conditions(config, model)

    results = model.run_simulations(
        start_date=sim_cfg["start_date"],
        end_date=sim_cfg["end_date"],
        Nsim=sim_cfg.get("n_simulations", 100),
        dt=sim_cfg.get("dt", 1.0),
        initial_conditions_dict=ic,
    )

    return results, config


# ---------------------------------------------------------------------------
# Calibration support
# ---------------------------------------------------------------------------

# Supported scipy.stats distribution constructors, keyed by YAML name.
_DISTRIBUTION_MAP = {
    "uniform": "uniform",
    "normal": "norm",
    "lognormal": "lognorm",
    "truncnorm": "truncnorm",
    "beta": "beta",
    "gamma": "gamma_dist",  # avoid clash with gamma parameter name
    "expon": "expon",
}


def build_prior(spec: Dict[str, Any]) -> Any:
    """Build a scipy.stats frozen distribution from a YAML prior spec.

    Supported distributions and their parameters:

    - ``uniform``:   ``low``, ``high``
    - ``normal``:    ``mean``, ``std``
    - ``lognormal``: ``shape`` (sigma), ``scale`` (exp(mu))
    - ``truncnorm``: ``mean``, ``std``, ``low``, ``high``
    - ``beta``:      ``a``, ``b``
    - ``gamma``:     ``a`` (shape), ``scale``
    - ``expon``:     ``scale``

    Returns:
        A frozen ``scipy.stats`` distribution.

    Raises:
        ValueError: If the distribution name is unknown.
        ImportError: If scipy is not installed.
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError(
            "Calibration requires scipy. Install it with: pip install scipy"
        )

    dist_name = spec.get("distribution", "uniform")
    if dist_name not in _DISTRIBUTION_MAP:
        raise ValueError(
            f"Unknown distribution '{dist_name}'. "
            f"Supported: {list(_DISTRIBUTION_MAP.keys())}"
        )

    if dist_name == "uniform":
        low = float(spec["low"])
        high = float(spec["high"])
        return stats.uniform(loc=low, scale=high - low)

    elif dist_name == "normal":
        return stats.norm(loc=float(spec["mean"]), scale=float(spec["std"]))

    elif dist_name == "lognormal":
        return stats.lognorm(s=float(spec["shape"]), scale=float(spec.get("scale", 1.0)))

    elif dist_name == "truncnorm":
        mean = float(spec["mean"])
        std = float(spec["std"])
        low = float(spec["low"])
        high = float(spec["high"])
        a = (low - mean) / std
        b = (high - mean) / std
        return stats.truncnorm(a=a, b=b, loc=mean, scale=std)

    elif dist_name == "beta":
        return stats.beta(a=float(spec["a"]), b=float(spec["b"]))

    elif dist_name == "gamma":
        return stats.gamma(a=float(spec["a"]), scale=float(spec.get("scale", 1.0)))

    elif dist_name == "expon":
        return stats.expon(scale=float(spec.get("scale", 1.0)))

    raise ValueError(f"Unhandled distribution: {dist_name}")


def build_priors(cal_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a dict of frozen scipy distributions from the calibration config.

    Args:
        cal_cfg: The ``calibration`` section of the config.

    Returns:
        Dict mapping parameter name → frozen scipy distribution.
    """
    priors_cfg = cal_cfg.get("priors", {})
    if not priors_cfg:
        raise ValueError("calibration.priors is required and must not be empty")
    return {name: build_prior(spec) for name, spec in priors_cfg.items()}


def load_observed_data(
    cal_cfg: Dict[str, Any],
    config_dir: Optional[Path] = None,
) -> np.ndarray:
    """Load observed data from the calibration config.

    The ``observed_data`` field can be:
    - A string path to a CSV file (resolved relative to *config_dir*).
    - A list of numbers (inline data).

    When a CSV file is used, ``observed_column`` selects which column to
    extract.  If omitted and the CSV has exactly two columns, the second
    column is used (assuming the first is a date/index).

    Args:
        cal_cfg: The ``calibration`` section of the config.
        config_dir: Directory of the config file, for resolving relative paths.

    Returns:
        1-D numpy array of observed values.
    """
    import pandas as pd

    obs = cal_cfg.get("observed_data")
    if obs is None:
        raise ValueError("calibration.observed_data is required")

    if isinstance(obs, list):
        return np.array(obs, dtype=float)

    # It's a file path
    obs_path = Path(obs)
    if not obs_path.is_absolute() and config_dir is not None:
        obs_path = config_dir / obs_path

    if not obs_path.exists():
        raise FileNotFoundError(f"Observed data file not found: {obs_path}")

    df = pd.read_csv(obs_path)
    col = cal_cfg.get("observed_column")
    if col is not None:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {obs_path}. "
                f"Available: {list(df.columns)}"
            )
        return df[col].values.astype(float)

    # Auto-select: if two columns, take the second; else take the last
    if len(df.columns) == 2:
        return df.iloc[:, 1].values.astype(float)
    return df.iloc[:, -1].values.astype(float)


def _make_simulation_function(
    config: Dict[str, Any],
    config_dir: Optional[Path] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a simulation function suitable for ABCSampler.

    The returned callable takes a parameter dict and returns
    ``{"data": 1D_array}`` — the format expected by epydemix distance
    functions.

    The target variable (which compartment/column to extract) is read
    from ``calibration.target_variable`` and defaults to the first
    ``_total`` column that starts with ``I`` (infected).
    """
    from ..model.epimodel import simulate

    model = build_model_from_config(config, config_dir=config_dir)
    sim_cfg = config.get("simulation", {})
    ic = build_initial_conditions(config, model)
    target = config.get("calibration", {}).get("target_variable")

    def sim_fn(params: Dict[str, Any]) -> Dict[str, Any]:
        # Update model parameters with sampled values
        for name, value in params.items():
            model.parameters[name] = value

        results = simulate(
            epimodel=model,
            start_date=sim_cfg["start_date"],
            end_date=sim_cfg["end_date"],
            dt=sim_cfg.get("dt", 1.0),
            initial_conditions_dict=ic,
        )

        # Extract target variable
        var = target
        if var is None:
            # Auto-detect: first I-like _total column
            for key in results.compartments:
                if key.endswith("_total") and key.split("_")[0].startswith("I"):
                    var = key
                    break
            if var is None:
                # Fall back to first _total
                for key in results.compartments:
                    if key.endswith("_total"):
                        var = key
                        break

        if var is None or var not in results.compartments:
            raise ValueError(
                f"Cannot find target variable '{var}' in simulation output. "
                f"Available: {list(results.compartments.keys())}"
            )

        return {"data": results.compartments[var]}

    return sim_fn


def _resolve_distance_function(name: str) -> Callable:
    """Look up a distance function by name from epydemix.calibration.metrics."""
    from ..calibration import metrics as m

    available = {
        "rmse": m.rmse,
        "mae": m.mae,
        "wmape": m.wmape,
        "mape": m.mape,
        "ae": m.ae,
    }
    if name not in available:
        raise ValueError(
            f"Unknown distance function '{name}'. "
            f"Available: {list(available.keys())}"
        )
    return available[name]


def validate_calibration_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a calibration config.  Returns ``{valid, errors, warnings}``."""
    errors = []
    warnings = []

    # Must have calibration section
    cal = config.get("calibration")
    if not cal:
        errors.append("Missing required section: 'calibration'")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check priors
    priors = cal.get("priors")
    if not priors:
        errors.append("calibration.priors is required and must not be empty")
    else:
        for name, spec in priors.items():
            if "distribution" not in spec:
                warnings.append(
                    f"Prior '{name}' has no 'distribution'; defaults to uniform"
                )
            dist = spec.get("distribution", "uniform")
            if dist not in _DISTRIBUTION_MAP:
                errors.append(
                    f"Prior '{name}': unknown distribution '{dist}'. "
                    f"Supported: {list(_DISTRIBUTION_MAP.keys())}"
                )

    # Check observed data
    if "observed_data" not in cal:
        errors.append("calibration.observed_data is required")

    # Check strategy
    strategy = cal.get("strategy", "smc")
    if strategy not in ("smc", "rejection", "top_fraction"):
        errors.append(
            f"calibration.strategy '{strategy}' is not valid. "
            f"Must be one of: smc, rejection, top_fraction"
        )

    # Check distance
    dist = cal.get("distance", "rmse")
    valid_distances = ("rmse", "mae", "wmape", "mape", "ae")
    if dist not in valid_distances:
        errors.append(
            f"calibration.distance '{dist}' is not valid. "
            f"Must be one of: {list(valid_distances)}"
        )

    # Also run base simulation config validation
    base_result = validate_config(config)
    errors.extend(base_result["errors"])
    warnings.extend(base_result["warnings"])

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def calibrate_from_config(
    config: Dict[str, Any],
    config_dir: Optional[Path] = None,
) -> Tuple[Any, Dict]:
    """Build and run a calibration from a config dict.

    Args:
        config: Fully resolved config dictionary.
        config_dir: Directory of the config file (for resolving relative
            paths to observed data files).

    Returns:
        Tuple of (CalibrationResults, config_dict).
    """
    from ..calibration.abc import ABCSampler

    cal_cfg = config.get("calibration", {})

    # Build components
    priors = build_priors(cal_cfg)
    observed = load_observed_data(cal_cfg, config_dir=config_dir)
    sim_fn = _make_simulation_function(config, config_dir=config_dir)
    distance_fn = _resolve_distance_function(cal_cfg.get("distance", "rmse"))

    # Fixed parameters = parameters section minus any that appear in priors
    fixed_params = {
        k: v for k, v in config.get("parameters", {}).items()
        if k not in priors
    }

    sampler = ABCSampler(
        simulation_function=sim_fn,
        priors=priors,
        parameters=fixed_params,
        observed_data=observed,
        distance_function=distance_fn,
    )

    # Extract strategy-specific kwargs
    strategy = cal_cfg.get("strategy", "smc")
    strategy_kwargs = {"verbose": False}  # suppress stdout from ABC

    if strategy == "smc":
        if "num_particles" in cal_cfg:
            strategy_kwargs["num_particles"] = int(cal_cfg["num_particles"])
        if "num_generations" in cal_cfg:
            strategy_kwargs["num_generations"] = int(cal_cfg["num_generations"])
        if "epsilon_quantile_level" in cal_cfg:
            strategy_kwargs["epsilon_quantile_level"] = float(
                cal_cfg["epsilon_quantile_level"]
            )
        if "minimum_epsilon" in cal_cfg:
            strategy_kwargs["minimum_epsilon"] = float(cal_cfg["minimum_epsilon"])
        if "total_simulations_budget" in cal_cfg:
            strategy_kwargs["total_simulations_budget"] = int(
                cal_cfg["total_simulations_budget"]
            )

    elif strategy == "rejection":
        if "epsilon" in cal_cfg:
            strategy_kwargs["epsilon"] = float(cal_cfg["epsilon"])
        if "num_particles" in cal_cfg:
            strategy_kwargs["num_particles"] = int(cal_cfg["num_particles"])
        if "total_simulations_budget" in cal_cfg:
            strategy_kwargs["total_simulations_budget"] = int(
                cal_cfg["total_simulations_budget"]
            )

    elif strategy == "top_fraction":
        if "top_fraction" in cal_cfg:
            strategy_kwargs["top_fraction"] = float(cal_cfg["top_fraction"])
        if "Nsim" in cal_cfg:
            strategy_kwargs["Nsim"] = int(cal_cfg["Nsim"])
        elif "n_simulations" in cal_cfg:
            strategy_kwargs["Nsim"] = int(cal_cfg["n_simulations"])

    results = sampler.calibrate(strategy=strategy, **strategy_kwargs)
    return results, config


# ---------------------------------------------------------------------------
# Projection support
# ---------------------------------------------------------------------------


def validate_projection_config(
    config: Dict[str, Any],
    calibration_bundle: str,
) -> Dict[str, Any]:
    """Validate a projection config.  Returns ``{valid, errors, warnings}``.

    Args:
        config: Fully resolved projection config dict.
        calibration_bundle: Path to the calibration .epx bundle.
    """
    errors: List[str] = []
    warnings: List[str] = []

    bundle_path = Path(calibration_bundle)
    if not bundle_path.exists():
        errors.append(f"Calibration bundle not found: {calibration_bundle}")
    elif not (bundle_path / "manifest.json").exists():
        errors.append(f"No manifest.json in bundle: {calibration_bundle}")
    elif not (bundle_path / "posterior.parquet").exists():
        errors.append(
            f"No posterior.parquet in bundle — is this a calibration bundle?"
        )

    # Must have simulation section (inherited or explicit)
    if "simulation" not in config:
        errors.append("Missing required section: 'simulation'")
    else:
        sim = config["simulation"]
        if "start_date" not in sim:
            errors.append("simulation.start_date is required")
        if "end_date" not in sim:
            errors.append("simulation.end_date is required")

    # Must have model section
    if "model" not in config:
        errors.append("Missing required section: 'model'")

    # Projection-specific settings
    proj = config.get("projection", {})
    n_sim = proj.get("n_simulations", 200)
    if not isinstance(n_sim, int) or n_sim < 1:
        errors.append("projection.n_simulations must be a positive integer")

    gen = proj.get("generation", -1)
    if not isinstance(gen, int):
        errors.append("projection.generation must be an integer")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def project_from_config(
    config: Dict[str, Any],
    calibration_bundle: str,
    config_dir: Optional[Path] = None,
) -> Tuple[Any, Dict]:
    """Run forward projections by sampling parameters from a calibration posterior.

    Reads the posterior (and optional weights) from a saved calibration bundle,
    samples parameter rows, and runs forward simulations with the (possibly
    overridden) config.

    Args:
        config: Fully resolved config dict — typically the calibration bundle's
            stored config deep-merged with a projection overlay that changes
            dates, adds interventions/overrides, etc.
        calibration_bundle: Path to the calibration .epx bundle.

    Returns:
        Tuple of (SimulationResults, config_dict).
    """
    import pandas as pd

    from ..io.bundle import load_bundle_dataframe
    from ..model.epimodel import simulate

    bundle_path = Path(calibration_bundle)

    # --- load posterior ---------------------------------------------------
    posterior_df = pd.read_parquet(bundle_path / "posterior.parquet")

    # Determine which generation to use
    proj_cfg = config.get("projection", {})
    gen_req = proj_cfg.get("generation", -1)

    if "generation" in posterior_df.columns:
        available_gens = sorted(posterior_df["generation"].unique())
        if gen_req == -1:
            gen = max(available_gens)
        else:
            gen = gen_req
        posterior_df = posterior_df[posterior_df["generation"] == gen].drop(
            columns=["generation"]
        )
    # else: no generation column (single-generation result); use as-is

    param_names = list(posterior_df.columns)

    # --- load weights (optional) ------------------------------------------
    weights_path = bundle_path / "weights.parquet"
    if weights_path.exists():
        weights_df = pd.read_parquet(weights_path)
        if "generation" in weights_df.columns:
            weights_df = weights_df[weights_df["generation"] == gen]
        w = weights_df["weight"].values.astype(float)
        w = w / w.sum()
    else:
        # Uniform weights (old bundles without weights.parquet)
        w = np.ones(len(posterior_df)) / len(posterior_df)

    # --- build model and simulation params --------------------------------
    model = build_model_from_config(config, config_dir=config_dir)
    sim_cfg = config.get("simulation", {})
    ic = build_initial_conditions(config, model)
    n_simulations = proj_cfg.get("n_simulations", 200)

    # --- sample and simulate ----------------------------------------------
    from ..model.simulation_results import SimulationResults

    all_trajectories = []
    posterior_arr = posterior_df.values  # (n_particles, n_params)

    for _ in range(n_simulations):
        idx = np.random.choice(len(posterior_arr), p=w)
        sampled_params = dict(zip(param_names, posterior_arr[idx]))

        # Update model parameters
        for name, value in sampled_params.items():
            model.parameters[name] = value

        traj = simulate(
            epimodel=model,
            start_date=sim_cfg["start_date"],
            end_date=sim_cfg["end_date"],
            dt=sim_cfg.get("dt", 1.0),
            initial_conditions_dict=ic,
        )
        all_trajectories.append(traj)

    # Assemble SimulationResults
    results = SimulationResults(
        trajectories=all_trajectories,
        parameters=config.get("parameters", {}),
    )

    return results, config
