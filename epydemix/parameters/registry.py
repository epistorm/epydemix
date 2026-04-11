"""Parameter registry for introspectable, queryable parameter metadata."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .spec import ParameterSpec


@dataclass
class ValidationError:
    """A single validation issue."""

    parameter: str
    message: str
    severity: str = "error"  # "error" or "warning"

    def to_dict(self) -> Dict[str, str]:
        return {
            "parameter": self.parameter,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """Result of validating a parameter dictionary against the registry."""

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


class ParameterRegistry:
    """Central registry of parameter metadata.

    Stores :class:`ParameterSpec` objects and exposes methods for querying,
    validating, and exporting parameter metadata. Designed to be queryable
    by LLM agents and exportable as JSON Schema.

    Example::

        registry = ParameterRegistry()
        registry.register(ParameterSpec(
            name="transmission_rate",
            description="Rate of disease transmission per contact",
            kind="rate",
            default=0.3,
            min=0,
            max=10,
            units="1/days",
        ))
        schema = registry.to_json_schema(title="SIR Parameters")
    """

    def __init__(self) -> None:
        self._specs: Dict[str, ParameterSpec] = {}

    def register(self, spec: ParameterSpec) -> None:
        """Register a parameter specification.

        If a spec with the same name already exists, it is replaced.

        Args:
            spec: The parameter specification to register.
        """
        self._specs[spec.name] = spec

    def get(self, name: str) -> ParameterSpec:
        """Retrieve a parameter spec by name.

        Args:
            name: The parameter name.

        Raises:
            KeyError: If the parameter is not registered.
        """
        if name not in self._specs:
            raise KeyError(
                f"Parameter '{name}' is not registered. "
                f"Available: {list(self._specs.keys())}"
            )
        return self._specs[name]

    def has(self, name: str) -> bool:
        """Check whether a parameter is registered."""
        return name in self._specs

    def list(self, tags: Optional[List[str]] = None) -> List[ParameterSpec]:
        """List registered parameter specs, optionally filtered by tags.

        Args:
            tags: If provided, only return specs that have at least one
                of the given tags.

        Returns:
            List of matching :class:`ParameterSpec` objects.
        """
        specs = list(self._specs.values())
        if tags:
            tag_set = set(tags)
            specs = [s for s in specs if tag_set & set(s.tags)]
        return specs

    @property
    def names(self) -> List[str]:
        """List of all registered parameter names."""
        return list(self._specs.keys())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __iter__(self):
        return iter(self._specs.values())

    def validate(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate a parameter dictionary against registered specs.

        Checks:
        - All required parameters are present
        - Values are within declared [min, max] bounds
        - No unknown parameters (warning, not error)

        Args:
            params: Dictionary of parameter name → value to validate.

        Returns:
            A :class:`ValidationResult` with errors and warnings.
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        # Check required params are present
        for spec in self._specs.values():
            if spec.required and spec.name not in params:
                if spec.default is not None:
                    # Has a default, just warn
                    warnings.append(ValidationError(
                        parameter=spec.name,
                        message=(
                            f"Required parameter '{spec.name}' not provided, "
                            f"will use default: {spec.default}"
                        ),
                        severity="warning",
                    ))
                else:
                    errors.append(ValidationError(
                        parameter=spec.name,
                        message=f"Required parameter '{spec.name}' is missing.",
                        severity="error",
                    ))

        # Check values
        for name, value in params.items():
            if name not in self._specs:
                warnings.append(ValidationError(
                    parameter=name,
                    message=(
                        f"Parameter '{name}' is not in the registry. "
                        f"It will still be passed to the model."
                    ),
                    severity="warning",
                ))
                continue

            spec = self._specs[name]
            # Only validate bounds for scalar numeric values
            if isinstance(value, (int, float)):
                if spec.min is not None and value < spec.min:
                    errors.append(ValidationError(
                        parameter=name,
                        message=(
                            f"Value {value} is below minimum {spec.min} "
                            f"for parameter '{name}'."
                        ),
                    ))
                if spec.max is not None and value > spec.max:
                    errors.append(ValidationError(
                        parameter=name,
                        message=(
                            f"Value {value} is above maximum {spec.max} "
                            f"for parameter '{name}'."
                        ),
                    ))

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def to_json_schema(self, title: Optional[str] = None) -> Dict[str, Any]:
        """Export the registry as a JSON Schema document.

        The schema can be used by agents to validate parameter configs
        before submitting them.

        Args:
            title: Optional title for the schema.

        Returns:
            A JSON-serializable dictionary conforming to JSON Schema
            draft 2020-12.
        """
        properties = {}
        required = []

        for spec in self._specs.values():
            properties[spec.name] = spec.to_json_schema_property()
            if spec.required:
                required.append(spec.name)

        schema: Dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": properties,
        }
        if title:
            schema["title"] = title
        if required:
            schema["required"] = required

        return schema

    def describe(self, name: Optional[str] = None) -> str:
        """Return LLM-friendly natural-language description(s).

        Args:
            name: If provided, describe a single parameter. Otherwise
                describe all registered parameters.

        Returns:
            A string suitable for including in an LLM prompt.
        """
        if name is not None:
            return self.get(name).describe()

        parts = [f"This model has {len(self._specs)} parameters:\n"]
        for spec in self._specs.values():
            parts.append(f"- {spec.describe()}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all specs to a dictionary."""
        return {name: spec.to_dict() for name, spec in self._specs.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParameterRegistry":
        """Deserialize a registry from a dictionary."""
        registry = cls()
        for name, spec_dict in d.items():
            # Ensure name consistency
            spec_dict["name"] = name
            registry.register(ParameterSpec.from_dict(spec_dict))
        return registry

    def defaults_dict(self) -> Dict[str, Any]:
        """Return a dict of parameter name → default value for all specs
        that have defaults."""
        return {
            spec.name: spec.default
            for spec in self._specs.values()
            if spec.default is not None
        }
