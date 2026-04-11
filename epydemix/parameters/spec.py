"""Parameter specification for introspectable parameter metadata."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Valid values for the 'kind' field
PARAMETER_KINDS = ("rate", "probability", "count", "duration", "proportion", "dimensionless")

# Valid values for the 'shape' field
PARAMETER_SHAPES = ("scalar", "time_varying", "age_structured", "time_x_age")


@dataclass
class ParameterSpec:
    """Metadata describing a model parameter.

    This is the fundamental unit of the parameter registry. It stores
    everything an LLM agent (or a JSON Schema validator) needs to know
    about a parameter: its name, meaning, valid range, units, and
    relationship to other parameters.

    Attributes:
        name: Parameter identifier, e.g. ``"transmission_rate"``.
        description: Human/LLM-readable explanation of what the parameter
            controls and how it affects the model.
        kind: Semantic category. One of ``"rate"``, ``"probability"``,
            ``"count"``, ``"duration"``, ``"proportion"``, or ``"dimensionless"``.
        dtype: Expected Python/numpy type. One of ``"float"``, ``"int"``, ``"array"``.
        default: Default value used if the user does not specify one.
        min: Minimum valid value (inclusive). ``None`` means unbounded below.
        max: Maximum valid value (inclusive). ``None`` means unbounded above.
        units: Physical units string, e.g. ``"1/days"``, ``"days"``, ``"dimensionless"``.
        required: Whether this parameter must be provided for the model to run.
        shape: Expected dimensionality of the value. One of ``"scalar"``,
            ``"time_varying"``, ``"age_structured"``, ``"time_x_age"``.
        depends_on: Names of other parameters this one is related to
            (e.g. ``recovery_rate`` depends on ``infectious_period``).
        tags: Freeform labels for filtering, e.g. ``["transmission", "SIR"]``.
    """

    name: str
    description: str
    kind: str
    dtype: str = "float"
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    units: Optional[str] = None
    required: bool = True
    shape: str = "scalar"
    depends_on: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in PARAMETER_KINDS:
            raise ValueError(
                f"Invalid parameter kind '{self.kind}'. "
                f"Must be one of: {PARAMETER_KINDS}"
            )
        if self.shape not in PARAMETER_SHAPES:
            raise ValueError(
                f"Invalid parameter shape '{self.shape}'. "
                f"Must be one of: {PARAMETER_SHAPES}"
            )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"Parameter '{self.name}': min ({self.min}) > max ({self.max})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (JSON-safe)."""
        d: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "dtype": self.dtype,
            "required": self.required,
            "shape": self.shape,
        }
        if self.default is not None:
            d["default"] = self.default
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.units is not None:
            d["units"] = self.units
        if self.depends_on:
            d["depends_on"] = self.depends_on
        if self.tags:
            d["tags"] = self.tags
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParameterSpec":
        """Deserialize from a plain dictionary."""
        return cls(
            name=d["name"],
            description=d["description"],
            kind=d["kind"],
            dtype=d.get("dtype", "float"),
            default=d.get("default"),
            min=d.get("min"),
            max=d.get("max"),
            units=d.get("units"),
            required=d.get("required", True),
            shape=d.get("shape", "scalar"),
            depends_on=d.get("depends_on", []),
            tags=d.get("tags", []),
        )

    def to_json_schema_property(self) -> Dict[str, Any]:
        """Return a JSON Schema property definition for this parameter."""
        dtype_map = {"float": "number", "int": "integer", "array": "array"}
        prop: Dict[str, Any] = {
            "type": dtype_map.get(self.dtype, "number"),
            "description": self.description,
        }
        if self.units:
            prop["description"] += f" Units: {self.units}."
        if self.default is not None:
            prop["default"] = self.default
        if self.min is not None:
            prop["minimum"] = self.min
        if self.max is not None:
            prop["maximum"] = self.max
        return prop

    def describe(self) -> str:
        """Return a natural-language description suitable for an LLM prompt."""
        parts = [f"{self.name}: {self.description}"]
        if self.units:
            parts.append(f"Units: {self.units}.")
        if self.min is not None or self.max is not None:
            lo = str(self.min) if self.min is not None else "-inf"
            hi = str(self.max) if self.max is not None else "inf"
            parts.append(f"Valid range: [{lo}, {hi}].")
        if self.default is not None:
            parts.append(f"Default: {self.default}.")
        if self.shape != "scalar":
            parts.append(f"Shape: {self.shape}.")
        return " ".join(parts)
