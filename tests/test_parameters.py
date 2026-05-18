"""Tests for the parameter registry, specs, and defaults catalog."""

import json

import pytest

from epydemix.parameters import (
    ParameterRegistry,
    ParameterSpec,
    ValidationResult,
    get_available_defaults,
    load_defaults,
)
from epydemix.parameters.predefined_specs import (
    MODEL_SPEC_FACTORIES,
    seir_specs,
    sir_specs,
    sis_specs,
)


# ---------------------------------------------------------------------------
# ParameterSpec tests
# ---------------------------------------------------------------------------


class TestParameterSpec:
    def test_basic_creation(self):
        spec = ParameterSpec(
            name="beta",
            description="Transmission rate",
            kind="rate",
            default=0.3,
            min=0,
            max=10,
            units="1/days",
        )
        assert spec.name == "beta"
        assert spec.kind == "rate"
        assert spec.default == 0.3

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="Invalid parameter kind"):
            ParameterSpec(name="x", description="x", kind="bogus")

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Invalid parameter shape"):
            ParameterSpec(name="x", description="x", kind="rate", shape="4d_tensor")

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="min.*>.*max"):
            ParameterSpec(name="x", description="x", kind="rate", min=5, max=1)

    def test_to_dict_round_trip(self):
        spec = ParameterSpec(
            name="gamma",
            description="Recovery rate",
            kind="rate",
            default=0.1,
            min=0,
            max=5,
            units="1/days",
            tags=["recovery", "SIR"],
        )
        d = spec.to_dict()
        restored = ParameterSpec.from_dict(d)
        assert restored.name == spec.name
        assert restored.description == spec.description
        assert restored.default == spec.default
        assert restored.min == spec.min
        assert restored.max == spec.max
        assert restored.tags == spec.tags

    def test_to_json_schema_property(self):
        spec = ParameterSpec(
            name="beta",
            description="Transmission rate",
            kind="rate",
            default=0.3,
            min=0,
            max=10,
            units="1/days",
        )
        prop = spec.to_json_schema_property()
        assert prop["type"] == "number"
        assert prop["minimum"] == 0
        assert prop["maximum"] == 10
        assert prop["default"] == 0.3
        assert "1/days" in prop["description"]

    def test_describe(self):
        spec = ParameterSpec(
            name="beta",
            description="Transmission rate",
            kind="rate",
            default=0.3,
            min=0,
            max=10,
            units="1/days",
        )
        desc = spec.describe()
        assert "beta" in desc
        assert "Transmission rate" in desc
        assert "1/days" in desc
        assert "0.3" in desc

    def test_optional_fields_omitted_in_dict(self):
        spec = ParameterSpec(name="x", description="x", kind="rate")
        d = spec.to_dict()
        assert "default" not in d
        assert "min" not in d
        assert "max" not in d
        assert "units" not in d
        assert "depends_on" not in d
        assert "tags" not in d


# ---------------------------------------------------------------------------
# ParameterRegistry tests
# ---------------------------------------------------------------------------


class TestParameterRegistry:
    @pytest.fixture
    def registry(self):
        reg = ParameterRegistry()
        reg.register(ParameterSpec(
            name="beta",
            description="Transmission rate",
            kind="rate",
            default=0.3,
            min=0,
            max=10,
            units="1/days",
            required=True,
            tags=["transmission"],
        ))
        reg.register(ParameterSpec(
            name="gamma",
            description="Recovery rate",
            kind="rate",
            default=0.1,
            min=0,
            max=5,
            units="1/days",
            required=True,
            tags=["recovery"],
        ))
        return reg

    def test_register_and_get(self, registry):
        spec = registry.get("beta")
        assert spec.name == "beta"
        assert spec.default == 0.3

    def test_get_missing_raises(self, registry):
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_has(self, registry):
        assert registry.has("beta")
        assert not registry.has("nonexistent")

    def test_len_and_contains(self, registry):
        assert len(registry) == 2
        assert "beta" in registry
        assert "nonexistent" not in registry

    def test_names(self, registry):
        assert set(registry.names) == {"beta", "gamma"}

    def test_list_all(self, registry):
        specs = registry.list()
        assert len(specs) == 2

    def test_list_filtered_by_tag(self, registry):
        specs = registry.list(tags=["transmission"])
        assert len(specs) == 1
        assert specs[0].name == "beta"

    def test_list_no_match(self, registry):
        specs = registry.list(tags=["nonexistent_tag"])
        assert len(specs) == 0

    def test_validate_valid_params(self, registry):
        result = registry.validate({"beta": 0.3, "gamma": 0.1})
        assert result.valid
        assert len(result.errors) == 0

    def test_validate_missing_required(self, registry):
        # beta is missing, but has a default → warning
        # gamma is missing, but has a default → warning
        result = registry.validate({})
        # Both have defaults, so they should be warnings not errors
        assert result.valid  # still valid because defaults exist
        assert len(result.warnings) == 2

    def test_validate_missing_required_no_default(self):
        reg = ParameterRegistry()
        reg.register(ParameterSpec(
            name="beta",
            description="Transmission rate",
            kind="rate",
            required=True,
            # no default
        ))
        result = reg.validate({})
        assert not result.valid
        assert len(result.errors) == 1
        assert result.errors[0].parameter == "beta"

    def test_validate_out_of_range(self, registry):
        result = registry.validate({"beta": -1, "gamma": 0.1})
        assert not result.valid
        assert any(e.parameter == "beta" for e in result.errors)

    def test_validate_above_max(self, registry):
        result = registry.validate({"beta": 100, "gamma": 0.1})
        assert not result.valid

    def test_validate_unknown_param_warns(self, registry):
        result = registry.validate({"beta": 0.3, "gamma": 0.1, "extra": 42})
        assert result.valid  # unknown params are just warnings
        assert any(w.parameter == "extra" for w in result.warnings)

    def test_to_json_schema(self, registry):
        schema = registry.to_json_schema(title="Test Model")
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == "Test Model"
        assert "beta" in schema["properties"]
        assert "gamma" in schema["properties"]
        assert set(schema["required"]) == {"beta", "gamma"}
        # Verify it's JSON-serializable
        json.dumps(schema)

    def test_describe_single(self, registry):
        desc = registry.describe("beta")
        assert "beta" in desc
        assert "Transmission" in desc

    def test_describe_all(self, registry):
        desc = registry.describe()
        assert "2 parameters" in desc
        assert "beta" in desc
        assert "gamma" in desc

    def test_to_dict_from_dict_round_trip(self, registry):
        d = registry.to_dict()
        restored = ParameterRegistry.from_dict(d)
        assert set(restored.names) == set(registry.names)
        assert restored.get("beta").default == registry.get("beta").default

    def test_defaults_dict(self, registry):
        defaults = registry.defaults_dict()
        assert defaults == {"beta": 0.3, "gamma": 0.1}

    def test_replace_on_re_register(self, registry):
        registry.register(ParameterSpec(
            name="beta",
            description="Updated",
            kind="rate",
            default=0.5,
        ))
        assert registry.get("beta").default == 0.5
        assert registry.get("beta").description == "Updated"

    def test_validation_result_to_dict(self):
        result = ValidationResult(valid=True, errors=[], warnings=[])
        d = result.to_dict()
        assert d["valid"] is True
        assert d["errors"] == []


# ---------------------------------------------------------------------------
# Predefined specs tests
# ---------------------------------------------------------------------------


class TestPredefinedSpecs:
    def test_sir_specs(self):
        specs = sir_specs()
        names = {s.name for s in specs}
        assert names == {"transmission_rate", "recovery_rate"}
        for s in specs:
            assert s.min is not None
            assert s.max is not None
            assert s.units == "1/days"

    def test_seir_specs(self):
        specs = seir_specs()
        names = {s.name for s in specs}
        assert names == {"transmission_rate", "incubation_rate", "recovery_rate"}

    def test_sis_specs(self):
        specs = sis_specs()
        names = {s.name for s in specs}
        assert names == {"transmission_rate", "recovery_rate"}

    def test_custom_defaults_propagated(self):
        specs = sir_specs(transmission_rate=0.5, recovery_rate=0.2)
        spec_map = {s.name: s for s in specs}
        assert spec_map["transmission_rate"].default == 0.5
        assert spec_map["recovery_rate"].default == 0.2

    def test_all_model_types_have_factories(self):
        assert set(MODEL_SPEC_FACTORIES.keys()) == {"SIR", "SEIR", "SIS", "SEIAR"}


# ---------------------------------------------------------------------------
# Defaults catalog tests
# ---------------------------------------------------------------------------


class TestDefaultsCatalog:
    def test_available_defaults(self):
        available = get_available_defaults()
        assert "covid19" in available
        assert "influenza" in available
        assert "measles" in available

    def test_load_covid19(self):
        defaults = load_defaults("covid19")
        assert defaults.name == "COVID-19 (Omicron-like)"
        assert defaults.model_type == "SEIR"
        assert "transmission_rate" in defaults.parameters
        assert "incubation_rate" in defaults.parameters
        assert "recovery_rate" in defaults.parameters

    def test_load_influenza(self):
        defaults = load_defaults("influenza")
        assert defaults.model_type == "SEIR"
        params = defaults.as_params()
        assert all(isinstance(v, float) for v in params.values())

    def test_load_measles(self):
        defaults = load_defaults("measles")
        assert defaults.model_type == "SEIR"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="No defaults file found"):
            load_defaults("bubonic_plague")

    def test_as_params(self):
        defaults = load_defaults("covid19")
        params = defaults.as_params()
        assert isinstance(params, dict)
        assert params["transmission_rate"] == 0.4
        assert params["incubation_rate"] == 0.33
        assert params["recovery_rate"] == 0.14

    def test_as_specs(self):
        defaults = load_defaults("covid19")
        specs = defaults.as_specs(tags=["covid19"])
        assert len(specs) == 3
        for s in specs:
            assert isinstance(s, ParameterSpec)
            assert "covid19" in s.tags

    def test_as_priors(self):
        defaults = load_defaults("covid19")
        priors = defaults.as_priors()
        assert "transmission_rate" in priors
        lo, hi = priors["transmission_rate"]
        assert lo == 0.2
        assert hi == 0.8

    def test_to_dict(self):
        defaults = load_defaults("covid19")
        d = defaults.to_dict()
        assert d["name"] == "COVID-19 (Omicron-like)"
        assert "parameters" in d
        # Should be JSON-serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# Integration: predefined models get registries
# ---------------------------------------------------------------------------


class TestPredefinedModelIntegration:
    def test_sir_model_has_registry(self):
        from epydemix.model.predefined_models import create_sir

        model = create_sir(0.3, 0.1)
        assert len(model.parameter_registry) == 2
        assert model.parameter_registry.has("transmission_rate")
        assert model.parameter_registry.has("recovery_rate")

    def test_seir_model_has_registry(self):
        from epydemix.model.predefined_models import create_seir

        model = create_seir(0.3, 0.2, 0.1)
        assert len(model.parameter_registry) == 3
        assert model.parameter_registry.has("incubation_rate")

    def test_sis_model_has_registry(self):
        from epydemix.model.predefined_models import create_sis

        model = create_sis(0.3, 0.1)
        assert len(model.parameter_registry) == 2

    def test_load_predefined_model_has_registry(self):
        from epydemix import load_predefined_model

        model = load_predefined_model("SEIR")
        assert len(model.parameter_registry) == 3
        schema = model.parameter_registry.to_json_schema(title="SEIR")
        assert "transmission_rate" in schema["properties"]

    def test_epimodel_starts_with_empty_registry(self):
        from epydemix import EpiModel

        model = EpiModel(compartments=["S", "I", "R"])
        assert len(model.parameter_registry) == 0

    def test_register_parameter_spec_method(self):
        from epydemix import EpiModel

        model = EpiModel()
        model.register_parameter_spec(ParameterSpec(
            name="custom_rate",
            description="A custom rate",
            kind="rate",
            default=0.5,
        ))
        assert model.parameter_registry.has("custom_rate")
