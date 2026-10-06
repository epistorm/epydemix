"""Load disease-specific default parameter sets from YAML files."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

from .spec import ParameterSpec

# Directory containing the YAML default files
_DEFAULTS_DIR = Path(__file__).parent / "defaults"


@dataclass
class DefaultParameterSet:
    """A loaded set of default parameters for a specific disease/scenario.

    Attributes:
        name: Human-readable name, e.g. ``"COVID-19 (Omicron-like)"``.
        model_type: Recommended predefined model type, e.g. ``"SEIR"``.
        source: Provenance / citation for the parameter values.
        description: Longer description of the scenario.
        parameters: Dictionary of parameter name → metadata dict with
            keys ``value``, ``range``, ``units``, ``description``.
    """

    name: str
    model_type: str
    source: str
    description: str
    parameters: Dict[str, Dict[str, Any]]

    def as_params(self) -> Dict[str, Any]:
        """Return a flat dict of parameter name → default value.

        Suitable for passing to ``load_predefined_model(**defaults.as_params())``.
        """
        return {name: info["value"] for name, info in self.parameters.items()}

    def as_specs(self, tags: Optional[List[str]] = None) -> List[ParameterSpec]:
        """Convert to a list of :class:`ParameterSpec` objects.

        Args:
            tags: Additional tags to add to every spec (e.g. the disease name).

        Note:
            All parameters are assigned ``kind="rate"`` because the shipped
            disease defaults (COVID-19, influenza, measles) only contain
            rate-type parameters.  If a future YAML default includes non-rate
            parameters, add an optional ``kind`` field to the YAML and read
            it here with ``info.get("kind", "rate")``.
        """
        specs = []
        for name, info in self.parameters.items():
            rng = info.get("range", [None, None])
            spec = ParameterSpec(
                name=name,
                description=info.get("description", ""),
                kind=info.get("kind", "rate"),
                default=info["value"],
                min=rng[0] if rng else None,
                max=rng[1] if len(rng) > 1 else None,
                units=info.get("units"),
                tags=list(tags) if tags else [],
            )
            specs.append(spec)
        return specs

    def as_priors(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter ranges as (min, max) tuples.

        Useful for setting up uniform priors in calibration::

            from scipy import stats
            defaults = load_defaults("covid19")
            priors = {
                name: stats.uniform(lo, hi - lo)
                for name, (lo, hi) in defaults.as_priors().items()
            }
        """
        result = {}
        skipped = []
        for name, info in self.parameters.items():
            rng = info.get("range")
            if rng and len(rng) == 2:
                result[name] = (rng[0], rng[1])
            else:
                skipped.append(name)
        if skipped:
            warnings.warn(
                f"as_priors() skipped parameters without a [min, max] range: "
                f"{', '.join(skipped)}",
                stacklevel=2,
            )
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "source": self.source,
            "description": self.description,
            "parameters": self.parameters,
        }


def load_defaults(
    disease: str, defaults_dir: Optional[str] = None
) -> DefaultParameterSet:
    """Load a default parameter set by disease name.

    Args:
        disease: Disease identifier, e.g. ``"covid19"``, ``"influenza"``,
            ``"measles"``. This matches the YAML filename (without extension).
        defaults_dir: Optional path to the directory containing YAML files.
            Defaults to the built-in ``epydemix/parameters/defaults/`` directory.

    Returns:
        A :class:`DefaultParameterSet` with the loaded values.

    Raises:
        FileNotFoundError: If no YAML file exists for the given disease.
    """
    if yaml is None:
        raise ImportError(
            "Loading parameter defaults requires pyyaml. "
            "Install it with: pip install epydemix[cli]"
        )

    search_dir = Path(defaults_dir) if defaults_dir else _DEFAULTS_DIR
    filepath = search_dir / f"{disease}.yaml"

    if not filepath.exists():
        available = get_available_defaults(defaults_dir)
        raise FileNotFoundError(
            f"No defaults file found for '{disease}'. "
            f"Available: {available}. "
            f"Searched in: {search_dir}"
        )

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    return DefaultParameterSet(
        name=data.get("name", disease),
        model_type=data.get("model_type", "unknown"),
        source=data.get("source", ""),
        description=data.get("description", ""),
        parameters=data.get("parameters", {}),
    )


def get_available_defaults(defaults_dir: Optional[str] = None) -> List[str]:
    """List available disease default identifiers.

    Args:
        defaults_dir: Optional path to search. Defaults to the built-in directory.

    Returns:
        List of disease identifiers (YAML filenames without extension).
    """
    search_dir = Path(defaults_dir) if defaults_dir else _DEFAULTS_DIR
    if not search_dir.exists():
        return []
    return sorted(
        f.stem for f in search_dir.glob("*.yaml") if not f.name.startswith("_")
    )
