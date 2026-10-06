"""Load and validate YAML/JSON configs, build EpiModel instances from them."""

import ast
import copy
import difflib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..model.epimodel import EpiModel
from ..model.predefined_models import SUPPORTED_MODELS, load_predefined_model

# Default simulation counts when not specified in config
DEFAULT_N_SIMULATIONS = 100
DEFAULT_N_PROJECTIONS = 200

# Optional structural module flags accepted under the `model:` block of a
# predefined-backbone config. Numeric module rates (e.g. waning_rate,
# mortality_rate) stay in the `parameters:` block.
_MODEL_MODULE_FIELDS = ("waning_immunity", "vaccination", "outcome")
_VALID_OUTCOMES = ("deaths", "hospitalization")


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
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
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


def _check_ic_normalization(ic_cfg: Dict[str, Any], tol: float = 1e-4) -> List[str]:
    """Return error strings for any demographic-group path whose IC fractions don't sum to 1.

    Works without population data by reconstructing every "path" that appears in
    the config: the ``"default"`` path (covers plain scalars and dict defaults) plus
    every named group key found in any compartment dict (e.g. ``"65+"``).
    For each path the contribution from each compartment is:
      - the scalar value, if the compartment entry is a scalar;
      - the group-specific value if the group key appears in the dict;
      - the ``"default"`` value in the dict if the group key is absent;
      - 0.0 if neither is present.

    If any compartment uses count mode (int scalar or dict with ``unit: count``),
    normalization cannot be checked without population data and the check is skipped.
    """
    # Skip normalization check if any compartment uses count mode.
    for val in ic_cfg.values():
        if isinstance(val, int):
            return []
        if isinstance(val, dict) and val.get("unit") == "count":
            return []

    errors = []

    # Collect every named group key (excluding reserved keys) across all compartment dicts.
    named_keys: set = set()
    for val in ic_cfg.values():
        if isinstance(val, dict):
            named_keys.update(k for k in val if k not in ("default", "unit"))

    for path in {"default"} | named_keys:
        total = 0.0
        all_numeric = True
        for val in ic_cfg.values():
            try:
                if isinstance(val, (int, float)):
                    total += float(val)
                elif isinstance(val, dict):
                    raw = val[path] if path in val else val.get("default", 0.0)
                    total += float(raw)
            except (TypeError, ValueError):
                all_numeric = False
                break

        if not all_numeric:
            continue  # non-numeric values are already caught by the type check

        if abs(total - 1.0) > tol:
            label = (
                "default demographic groups" if path == "default" else f"group '{path}'"
            )
            errors.append(
                f"initial_conditions: fractions for {label} sum to {total:.6g}, expected 1.0"
            )

    return errors


def _validate_seed(sim_cfg: Dict[str, Any], errors: list) -> None:
    """Check an optional ``simulation.seed``. Booleans are ints in Python."""
    seed = sim_cfg.get("seed")
    if seed is None:
        return
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append("simulation.seed must be an integer")
    elif seed < 0:
        errors.append("simulation.seed must be non-negative")


# The runtime infers the output frequency from the simulation dates, which
# needs at least three of them ("Need at least 3 dates to infer frequency").
_MIN_SIMULATION_STEPS = 3

# Placeholder for a parameter reference that carries no value (a prior).
_NO_VALUE = object()


def _is_number(value: Any) -> bool:
    """True for int and float values. Booleans are ints in Python; excluded."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_simulation_window(sim_cfg: Dict[str, Any], errors: list) -> None:
    """Check the simulation dates, time step and replicate count.

    Each of these otherwise surfaces only once a run is under way, and some
    surface as unhelpful internals: an inverted date range fails as an
    unsigned-integer overflow.
    """
    import datetime

    import pandas as pd

    from ..utils.utils import compute_simulation_dates

    dates = {}
    for key in ("start_date", "end_date"):
        raw = sim_cfg.get(key)
        if raw is None:
            continue  # a missing date is reported by the caller
        if not isinstance(raw, (str, datetime.date)):
            errors.append(
                f"simulation.{key} must be a date written as YYYY-MM-DD, got {raw!r}"
            )
            continue
        try:
            stamp = pd.Timestamp(raw)
        except (ValueError, TypeError):
            stamp = pd.NaT
        if pd.isna(stamp):
            errors.append(
                f"simulation.{key} {raw!r} is not a valid date; "
                "write it as YYYY-MM-DD"
            )
        else:
            dates[key] = stamp

    dt = sim_cfg.get("dt", 1.0)
    dt_ok = _is_number(dt) and dt > 0
    if not dt_ok:
        errors.append(f"simulation.dt must be a positive number of days, got {dt!r}")

    n_sims = sim_cfg.get("n_simulations")
    if n_sims is not None and not (
        isinstance(n_sims, int) and not isinstance(n_sims, bool) and n_sims >= 1
    ):
        errors.append(
            f"simulation.n_simulations must be a positive integer, got {n_sims!r}"
        )

    if len(dates) == 2:
        start, end = dates["start_date"], dates["end_date"]
        if end <= start:
            errors.append(
                f"simulation.end_date ({end.date()}) must be after "
                f"simulation.start_date ({start.date()})"
            )
        elif dt_ok:
            n_steps = len(compute_simulation_dates(start, end, dt=dt))
            if n_steps < _MIN_SIMULATION_STEPS:
                errors.append(
                    f"the simulation window gives {n_steps} time steps at "
                    f"dt={dt}; at least {_MIN_SIMULATION_STEPS} are needed. "
                    "Move end_date later or reduce dt."
                )


def _numeric_leaves(value: Any) -> Optional[List[float]]:
    """Return the numbers in a parameter value, or None if it is malformed.

    The accepted shapes mirror what the model accepts at run time: a scalar, a
    list of numbers (time-varying), or a list of lists (time x group).
    """
    if _is_number(value):
        return [value]
    if isinstance(value, list) and value:
        if all(_is_number(x) for x in value):
            return list(value)
        if all(
            isinstance(row, list) and row and all(_is_number(x) for x in row)
            for row in value
        ):
            return [x for row in value for x in row]
    return None


def _parameter_value_problem(value: Any) -> Optional[str]:
    """Explain why a parameter value cannot be used, or return None if it can."""
    if _numeric_leaves(value) is not None:
        return None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return f"is the text {value!r}, not a number"
        mantissa = value.lower().split("e")[0]
        if "e" in value.lower() and "." not in mantissa:
            # PyYAML follows YAML 1.1: 3e-1 is a string, 3.0e-1 is a float.
            return (
                f"is the text {value!r}, not a number: YAML reads exponent "
                "notation as a number only when the mantissa has a decimal "
                f"point. Write {number!r} or "
                f"{value.lower().replace('e', '.0e', 1)}"
            )
        return f"is the quoted text {value!r}; remove the quotes"
    if value is None:
        return "has no value"
    return (
        "must be a number, a non-empty list of numbers (time-varying), or a "
        f"list of lists of numbers (time x group); got {value!r}"
    )


def _parameter_uses(config: Dict[str, Any], params: Dict[str, Any]) -> list:
    """Every place a config names a model parameter.

    Returns ``(section, name, value, value_label)`` tuples. ``value`` is
    ``_NO_VALUE`` for references that carry none, such as calibration priors.
    """
    uses = [
        ("parameters", name, value, f"parameters.{name}")
        for name, value in params.items()
    ]
    priors = (config.get("calibration") or {}).get("priors")
    if isinstance(priors, dict):
        uses += [("calibration.priors", name, _NO_VALUE, None) for name in priors]
    for i, ovr in enumerate(config.get("overrides") or []):
        if isinstance(ovr, dict) and "parameter" in ovr:
            uses.append((
                f"overrides[{i}].parameter",
                ovr["parameter"],
                ovr.get("value", _NO_VALUE),
                f"overrides[{i}].value",
            ))
    return uses


def _inactive_parameter_hint(name: str) -> Optional[str]:
    """Name the model setting that would activate a known parameter.

    ``load_predefined_model`` accepts every backbone's and module's parameters
    and silently ignores those the chosen model does not use, so e.g.
    ``vaccine_efficacy`` without ``model.vaccination: true`` runs without
    vaccinating anyone.
    """
    from ..parameters import predefined_specs as ps

    owners = [
        (ps.waning_immunity_specs(), "set model.waning_immunity: true to use it"),
        (ps.vaccination_specs(), "set model.vaccination: true to use it"),
        (ps.outcome_specs("deaths"), "set model.outcome: deaths to use it"),
        (ps.outcome_specs("hospitalization"),
         "set model.outcome: hospitalization to use it"),
        (ps.seir_specs(), "it needs model.type SEIR or SEIAR"),
        (ps.seiar_specs(), "it needs model.type SEIAR"),
    ]
    for specs, hint in owners:
        if name in {spec.name for spec in specs}:
            return hint
    return None


def _build_predefined(model_type: str, model_cfg: Dict[str, Any]) -> Optional[EpiModel]:
    """Build the predefined model a config describes, without its parameters.

    It is built with the config's own backbone and module flags, so its
    registry and compartments are exactly those the run will use. Returns
    None when the module combination is invalid, which is reported separately.
    """
    module_kwargs = {k: model_cfg[k] for k in _MODEL_MODULE_FIELDS if k in model_cfg}
    try:
        return load_predefined_model(model_type, **module_kwargs)
    except (ValueError, TypeError):
        return None


def _validate_predefined_parameters(registry, uses: list, errors: list) -> None:
    """Check parameter names and values against the model's registry."""
    known = registry.names
    for section, name, value, value_label in uses:
        if name not in registry:
            hint = _inactive_parameter_hint(name)
            if hint is None:
                close = difflib.get_close_matches(str(name), known, n=1)
                hint = f"did you mean '{close[0]}'?" if close else None
            message = f"{section}: '{name}' is not a parameter of this model"
            if hint:
                message += f" ({hint})"
            errors.append(f"{message}. Parameters: {', '.join(known)}")
            continue

        numbers = _numeric_leaves(value) if value is not _NO_VALUE else None
        if not numbers:
            continue  # absent, or malformed and reported elsewhere
        spec = registry.get(name)
        units = f" ({spec.units})" if spec.units else ""
        low, high = min(numbers), max(numbers)
        if spec.min is not None and low < spec.min:
            errors.append(
                f"{value_label} has value {low}, below the minimum of "
                f"{spec.min}{units} for '{name}'"
            )
        if spec.max is not None and high > spec.max:
            errors.append(
                f"{value_label} has value {high}, above the maximum of "
                f"{spec.max}{units} for '{name}'"
            )


def _validate_ic_compartments(
    ic_cfg: Dict[str, Any], compartments: list, predefined: bool, errors: list
) -> None:
    """Every initial-condition key must name a compartment of the model.

    ``build_initial_conditions`` skips names it cannot resolve. One unresolved
    name drops that compartment's value, so a typo in the infected compartment
    starts the run with nobody infected; if no name resolves, the whole block
    is replaced by defaults. Both used to run without complaint.
    """
    listing = ", ".join(compartments)
    for name in ic_cfg:
        if _resolve_compartment(str(name), compartments) is not None:
            continue
        prefix = [c for c in compartments if c.lower().startswith(str(name).lower())]
        close = prefix or difflib.get_close_matches(str(name), compartments, n=1)
        message = (
            f"initial_conditions: '{name}' is not a compartment of this model, "
            "so its value would be ignored"
        )
        if predefined and len(str(name)) <= 2:
            message += " (predefined models use full compartment names"
            message += f", e.g. '{close[0]}')" if close else ")"
        elif close:
            message += f" (did you mean '{close[0]}'?)"
        errors.append(f"{message}. Compartments: {listing}")


def _rate_problem(
    rate: Any, parameter_names: set, compartments: set
) -> Optional[str]:
    """Explain why a transition rate cannot be evaluated, or return None.

    A rate is a number, a parameter name, or an arithmetic expression over
    parameters. It is evaluated with only the parameters in scope, so a
    compartment name inside a rate is an error too.
    """
    if _is_number(rate):
        return None
    if not isinstance(rate, str):
        return (
            "must be a number, a parameter name, or an arithmetic expression "
            f"over parameters; got {rate!r}"
        )
    if rate in parameter_names:
        return None
    try:
        tree = ast.parse(rate, mode="eval")
    except SyntaxError:
        return f"{rate!r} is not a valid expression"
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    missing = sorted(names - parameter_names)
    if not missing:
        return None
    if missing == [rate]:
        message = f"{rate!r} is not a defined parameter"
    else:
        listed = ", ".join(repr(m) for m in missing)
        verb = "is not a defined parameter" if len(missing) == 1 else (
            "are not defined parameters"
        )
        message = f"{rate!r} uses {listed}, which {verb}"
    if set(missing) & compartments:
        message += " (a rate can use parameters only, not compartments)"
    defined = ", ".join(sorted(parameter_names)) or "none"
    return f"{message}. Defined parameters: {defined}"


def _validate_custom_references(
    model_cfg: Dict[str, Any], parameter_names: set, errors: list
) -> None:
    """Check that transitions refer only to declared compartments and parameters."""
    compartments = model_cfg.get("compartments")
    transitions = model_cfg.get("transitions")
    if not isinstance(compartments, list) or not isinstance(transitions, list):
        return
    declared = set(compartments)
    listing = ", ".join(str(c) for c in compartments)

    for i, tr in enumerate(transitions):
        if not isinstance(tr, dict):
            errors.append(f"transitions[{i}] must be a mapping")
            continue
        for end in ("source", "target"):
            name = tr.get(end)
            if name is None:
                errors.append(f"transitions[{i}]: '{end}' is required")
            elif name not in declared:
                errors.append(
                    f"transitions[{i}].{end} '{name}' is not a declared "
                    f"compartment. Compartments: {listing}"
                )

        kind, params = tr.get("kind"), tr.get("params")
        if kind == "spontaneous" and params is not None:
            problem = _rate_problem(params, parameter_names, declared)
            if problem:
                errors.append(f"transitions[{i}].params {problem}")
        elif kind == "mediated" and params is not None:
            if not (isinstance(params, (list, tuple)) and len(params) == 2):
                errors.append(
                    f"transitions[{i}].params must be [rate, infecting "
                    f"compartment] for a mediated transition; got {params!r}"
                )
                continue
            problem = _rate_problem(params[0], parameter_names, declared)
            if problem:
                errors.append(f"transitions[{i}].params[0] {problem}")
            if params[1] not in declared:
                errors.append(
                    f"transitions[{i}].params[1] '{params[1]}' is not a "
                    f"declared compartment. Compartments: {listing}"
                )
        elif kind == "scheduled":
            for name in tr.get("eligible") or []:
                if name not in declared:
                    errors.append(
                        f"transitions[{i}].eligible '{name}' is not a declared "
                        f"compartment. Compartments: {listing}"
                    )


def resolve_seed(sim_cfg: Dict[str, Any]) -> int:
    """Return the seed for a run, drawing one if the config does not set it.

    A seed is always resolved and always recorded, rather than being an opt-in
    the caller has to remember: reproducibility that depends on someone electing
    to ask for it is not reproducibility.
    """
    seed = sim_cfg.get("seed")
    if seed is None:
        # 32 bits, not 64: the seed is written to manifest.json, and integers
        # above 2**53 do not survive a round trip through consumers that parse
        # JSON numbers as IEEE-754 doubles. A silently altered seed would defeat
        # the purpose of recording it.
        seed = int.from_bytes(os.urandom(4), "big")
    return int(seed)


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
        _validate_seed(sim, errors)
        _validate_simulation_window(sim, errors)

    # Model section validation
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "custom")
    if model_type != "custom" and model_type not in SUPPORTED_MODELS:
        errors.append(
            f"Unknown model type '{model_type}'. "
            f"Supported: {SUPPORTED_MODELS + ['custom']}"
        )

    # Optional modular extensions for predefined backbones
    if model_type in SUPPORTED_MODELS:
        wi = model_cfg.get("waning_immunity")
        if wi is not None and not isinstance(wi, bool):
            errors.append("model.waning_immunity must be a boolean")
        vac = model_cfg.get("vaccination")
        if vac is not None and not isinstance(vac, bool):
            errors.append("model.vaccination must be a boolean")
        oc = model_cfg.get("outcome")
        if oc is not None and oc not in _VALID_OUTCOMES:
            errors.append(
                f"model.outcome must be one of {list(_VALID_OUTCOMES)} or omitted"
            )
        # Compatibility guardrails — mirror runtime ValueErrors raised by
        # load_predefined_model so they surface at `epydemix validate` time.
        if model_type == "SIS" and wi is True:
            errors.append(
                "model.waning_immunity is not compatible with model.type 'SIS' "
                "(SIS has no 'Recovered' compartment)"
            )
        if model_type == "SIS" and oc == "hospitalization":
            errors.append(
                "model.outcome 'hospitalization' is not compatible with "
                "model.type 'SIS' (no 'Recovered' compartment for the "
                "Hospitalized → Recovered transition)"
            )

    if model_type == "custom":
        if "compartments" not in model_cfg:
            errors.append("Custom model requires 'model.compartments'")
        if "transitions" not in model_cfg:
            errors.append("Custom model requires 'model.transitions'")
        else:
            _BUILTIN_KINDS = {"spontaneous", "mediated", "scheduled"}
            for i, tr in enumerate(model_cfg.get("transitions") or []):
                if not isinstance(tr, dict):
                    continue  # reported by _validate_custom_references
                kind = tr.get("kind")
                if kind == "scheduled" and "schedule" not in tr:
                    errors.append(
                        f"transitions[{i}]: kind 'scheduled' requires a 'schedule' field "
                        "(path to a CSV file or an inline list)"
                    )
                if (
                    kind in _BUILTIN_KINDS
                    and kind != "scheduled"
                    and "params" not in tr
                ):
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
    params = config.get("parameters", {})
    if "parameters" not in config:
        warnings.append("No 'parameters' section — will use model defaults")
    elif not isinstance(params, dict):
        errors.append("'parameters' must be a mapping of parameter name to value")
        params = {}

    uses = _parameter_uses(config, params)
    for _, name, value, value_label in uses:
        if value is _NO_VALUE:
            continue
        problem = _parameter_value_problem(value)
        if problem:
            errors.append(f"{value_label} {problem}")

    compartments = None
    if model_type in SUPPORTED_MODELS:
        model = _build_predefined(model_type, model_cfg)
        if model is not None:
            compartments = list(model.compartments)
            _validate_predefined_parameters(model.parameter_registry, uses, errors)
    elif isinstance(model_cfg.get("compartments"), list):
        compartments = model_cfg["compartments"]

    ic_cfg = config.get("initial_conditions")
    if isinstance(ic_cfg, dict) and compartments is not None:
        _validate_ic_compartments(
            ic_cfg, compartments, model_type in SUPPORTED_MODELS, errors
        )

    if model_type == "custom":
        # Parameters are defined by the parameters section and, in a
        # calibration config, by the priors; overrides only refer to them.
        defined = {
            name for section, name, _, _ in uses
            if section in ("parameters", "calibration.priors")
        }
        _validate_custom_references(model_cfg, defined, errors)
        for section, name, _, _ in uses:
            if section.startswith("overrides") and name not in defined:
                errors.append(
                    f"{section}: '{name}' is not a defined parameter, so the "
                    "override would have no effect. Defined parameters: "
                    f"{', '.join(sorted(defined)) or 'none'}"
                )

    # Initial conditions
    if "initial_conditions" not in config:
        warnings.append(
            "No 'initial_conditions' section — will use default "
            "(small fraction in first infectious compartment)"
        )
    else:
        ic_cfg = config["initial_conditions"]
        for comp, val in ic_cfg.items():
            if isinstance(val, dict):
                unit_val = val.get("unit")
                if unit_val is not None and unit_val not in ("fraction", "count"):
                    errors.append(
                        f"initial_conditions[{comp!r}]['unit'] must be 'fraction' or 'count'"
                    )
                for k, v in val.items():
                    if k in ("default", "unit"):
                        continue
                    if not isinstance(v, (int, float)):
                        errors.append(
                            f"initial_conditions[{comp!r}][{k!r}] must be a number"
                        )
        errors.extend(_check_ic_normalization(ic_cfg))

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
        module_kwargs = {
            k: model_cfg[k] for k in _MODEL_MODULE_FIELDS if k in model_cfg
        }
        model = load_predefined_model(model_type, **params, **module_kwargs)
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


def _resolve_compartment(name: str, compartments: list) -> Optional[str]:
    """Return the canonical compartment name matching *name* (case-insensitive).

    Returns ``None`` if no match is found.
    """
    if name in compartments:
        return name
    name_lower = name.lower()
    for real in compartments:
        if real.lower() == name_lower:
            return real
    return None


def build_initial_conditions(
    config: Dict[str, Any],
    model: EpiModel,
) -> Optional[Dict[str, np.ndarray]]:
    """Build initial conditions dict from config.

    Each compartment entry in ``initial_conditions`` can be:

    * A **float scalar** — fraction applied proportionally to every group:
      ``ic[comp] = Nk * fraction``.
    * An **int scalar** — absolute count distributed proportionally across
      groups: ``ic[comp] = Nk * (count / N_total)``.
    * A **dict** for per-group control.  Optional ``unit`` key (``"fraction"``
      or ``"count"``, default ``"fraction"``) sets how values are interpreted.
      Use ``"default"`` for groups not explicitly listed (0.0 / 0 when absent).
      Every other key must be a group name from ``population.Nk_names``::

          # fraction mode (default)
          Recovered:
            default: 0.0
            "65+": 0.75   # 75 % of the 65+ group starts immune

          # count mode
          Infected:
            unit: count
            default: 0
            "65+": 5      # exactly 5 individuals in the 65+ group

    Args:
        config: The full config dict.
        model: The configured model (to know compartments and population size).

    Returns:
        Initial conditions dict, or None to use defaults.
    """
    ic_cfg = config.get("initial_conditions")
    if not ic_cfg:
        return None

    pop_sizes = np.array(model.population.Nk, dtype=float)
    group_names = list(model.population.Nk_names)
    n_groups = len(pop_sizes)
    total_pop = pop_sizes.sum()

    ic_dict = {}
    for comp_name, value in ic_cfg.items():
        resolved = _resolve_compartment(comp_name, model.compartments)
        if resolved is None:
            continue

        if isinstance(value, dict):
            unit = value.get("unit", "fraction")
            if unit == "count":
                default_count = float(value.get("default", 0.0))
                counts = np.full(n_groups, default_count)
                for group_key, count in value.items():
                    if group_key in ("unit", "default"):
                        continue
                    if group_key not in group_names:
                        valid = ", ".join(f'"{g}"' for g in group_names)
                        raise ValueError(
                            f"initial_conditions[{comp_name!r}]: unknown group "
                            f"{group_key!r}. Valid groups: {valid}"
                        )
                    counts[group_names.index(group_key)] = float(count)
                ic_dict[resolved] = counts
            else:  # fraction mode
                default_frac = float(value.get("default", 0.0))
                fracs = np.full(n_groups, default_frac)
                for group_key, frac in value.items():
                    if group_key in ("unit", "default"):
                        continue
                    if group_key not in group_names:
                        valid = ", ".join(f'"{g}"' for g in group_names)
                        raise ValueError(
                            f"initial_conditions[{comp_name!r}]: unknown group "
                            f"{group_key!r}. Valid groups: {valid}"
                        )
                    fracs[group_names.index(group_key)] = float(frac)
                ic_dict[resolved] = pop_sizes * fracs
        elif isinstance(value, int):
            # count mode: distribute proportionally across groups
            ic_dict[resolved] = pop_sizes * (float(value) / total_pop)
        else:
            ic_dict[resolved] = pop_sizes * float(value)

    if ic_dict and set(ic_dict.keys()) == set(model.compartments):
        total_fracs = sum(counts / pop_sizes for counts in ic_dict.values())
        for group, s in zip(group_names, total_fracs):
            if abs(s - 1.0) > 1e-4:
                raise ValueError(
                    f"initial_conditions: fractions for group '{group}' sum to "
                    f"{s:.6g}, expected 1.0"
                )

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

    # Resolve a seed even when the config omits one, and write it back into the
    # config that gets stored in the bundle, so the stored config re-runs to the
    # same realizations rather than merely the same setup.
    seed = resolve_seed(sim_cfg)
    config = copy.deepcopy(config)
    config.setdefault("simulation", {})["seed"] = seed

    results = model.run_simulations(
        start_date=sim_cfg["start_date"],
        end_date=sim_cfg["end_date"],
        Nsim=sim_cfg.get("n_simulations", DEFAULT_N_SIMULATIONS),
        dt=sim_cfg.get("dt", 1.0),
        initial_conditions_dict=ic,
        rng=np.random.default_rng(seed),
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
        return stats.lognorm(
            s=float(spec["shape"]), scale=float(spec.get("scale", 1.0))
        )

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
                f"Column '{col}' not found in {obs_path}. Available: {list(df.columns)}"
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

        # Extract target variable — search compartments first, then
        # transitions (e.g. "Susceptible_to_Infected_total" for incidence).
        var = target
        if var is None:
            # Auto-detect: first I-like _total compartment column
            for key in results.compartments:
                if key.endswith("_total") and key.split("_")[0].startswith("I"):
                    var = key
                    break
            if var is None:
                # Fall back to first _total compartment
                for key in results.compartments:
                    if key.endswith("_total"):
                        var = key
                        break

        if var is not None and var in results.compartments:
            return {"data": results.compartments[var]}
        if var is not None and var in results.transitions:
            return {"data": results.transitions[var]}

        all_keys = list(results.compartments.keys()) + list(results.transitions.keys())
        raise ValueError(
            f"Cannot find target variable '{var}' in simulation output. "
            f"Available: {all_keys}"
        )

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
            f"Unknown distance function '{name}'. Available: {list(available.keys())}"
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
        k: v for k, v in config.get("parameters", {}).items() if k not in priors
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
        errors.append("No posterior.parquet in bundle — is this a calibration bundle?")

    # Must have simulation section (inherited or explicit)
    if "simulation" not in config:
        errors.append("Missing required section: 'simulation'")
    else:
        sim = config["simulation"]
        if "start_date" not in sim:
            errors.append("simulation.start_date is required")
        if "end_date" not in sim:
            errors.append("simulation.end_date is required")
        _validate_seed(sim, errors)
        _validate_simulation_window(sim, errors)

    # Must have model section
    if "model" not in config:
        errors.append("Missing required section: 'model'")

    # Projection-specific settings
    proj = config.get("projection", {})
    n_sim = proj.get("n_simulations", DEFAULT_N_PROJECTIONS)
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
    n_simulations = proj_cfg.get("n_simulations", DEFAULT_N_PROJECTIONS)

    # --- sample and simulate ----------------------------------------------
    from ..model.simulation_results import SimulationResults

    # One generator drives both the posterior draw and the trajectories, so a
    # projection is reproducible end to end from the recorded seed.
    seed = resolve_seed(sim_cfg)
    rng = np.random.default_rng(seed)
    config = copy.deepcopy(config)
    config.setdefault("simulation", {})["seed"] = seed

    all_trajectories = []
    posterior_arr = posterior_df.values  # (n_particles, n_params)

    for _ in range(n_simulations):
        idx = rng.choice(len(posterior_arr), p=w)
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
            rng=rng,
        )
        all_trajectories.append(traj)

    # Assemble SimulationResults
    results = SimulationResults(
        trajectories=all_trajectories,
        parameters=config.get("parameters", {}),
    )

    return results, config
