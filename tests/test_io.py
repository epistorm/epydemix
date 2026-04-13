"""Tests for the bundle I/O and inspection engine."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from epydemix import EpiModel, load_predefined_model
from epydemix.io.bundle import load_bundle, load_bundle_dataframe, save_bundle
from epydemix.io.inspect import inspect_bundle
from epydemix.io.json_utils import NumpySafeEncoder, to_json
from epydemix.model.simulation_output import Trajectory
from epydemix.model.simulation_results import SimulationResults
from epydemix.population import Population


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_population():
    """A minimal single-group population."""
    pop = Population()
    pop.add_population([10000])
    pop.add_contact_matrix(np.array([[1.0]]))
    return pop


@pytest.fixture
def sir_results(simple_population):
    """Run a small SIR simulation and return results."""
    model = load_predefined_model("SIR", transmission_rate=0.3, recovery_rate=0.1)
    model.set_population(simple_population)
    initial_conditions = {
        "Susceptible": np.array([9900]),
        "Infected": np.array([100]),
        "Recovered": np.array([0]),
    }
    results = model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-20",
        initial_conditions_dict=initial_conditions,
        Nsim=10,
    )
    return results


@pytest.fixture
def bundle_path(sir_results, tmp_path):
    """Save SIR results to a temp bundle and return the path."""
    path = str(tmp_path / "test_results.epx")
    save_bundle(sir_results, path)
    return path


# ---------------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------------

class TestJsonUtils:
    def test_numpy_encoder(self):
        data = {
            "int": np.int64(42),
            "float": np.float64(3.14159),
            "array": np.array([1.0, 2.0, 3.0]),
            "bool": np.bool_(True),
            "timestamp": pd.Timestamp("2023-01-01"),
        }
        result = json.dumps(data, cls=NumpySafeEncoder)
        parsed = json.loads(result)
        assert parsed["int"] == 42
        assert isinstance(parsed["float"], float)
        assert parsed["bool"] is True
        assert parsed["timestamp"] == "2023-01-01"

    def test_to_json(self):
        data = {"value": np.float64(3.14159265)}
        result = to_json(data, precision=2)
        parsed = json.loads(result)
        assert parsed["value"] == 3.14


# ---------------------------------------------------------------------------
# Bundle save/load
# ---------------------------------------------------------------------------

class TestBundleSaveLoad:
    def test_save_creates_files(self, sir_results, tmp_path):
        path = str(tmp_path / "results.epx")
        manifest = save_bundle(sir_results, path)

        assert os.path.isdir(path)
        assert os.path.exists(os.path.join(path, "manifest.json"))
        assert os.path.exists(os.path.join(path, "compartments.parquet"))
        assert os.path.exists(os.path.join(path, "transitions.parquet"))
        assert os.path.exists(os.path.join(path, "parameters.parquet"))

    def test_manifest_structure(self, sir_results, tmp_path):
        path = str(tmp_path / "results.epx")
        manifest = save_bundle(sir_results, path)

        assert manifest["type"] == "SimulationResults"
        assert "epydemix_version" in manifest
        assert "simulation" in manifest
        assert manifest["simulation"]["n_simulations"] == 10
        assert "files" in manifest
        assert "compartments" in manifest["files"]
        assert "usage_hints" in manifest

    def test_manifest_has_column_schema(self, sir_results, tmp_path):
        path = str(tmp_path / "results.epx")
        manifest = save_bundle(sir_results, path)

        comp_info = manifest["files"]["compartments"]
        assert "columns" in comp_info
        assert "sim_id" in comp_info["columns"]
        assert "date" in comp_info["columns"]
        # Should have _total columns
        total_cols = [c for c in comp_info["columns"] if c.endswith("_total")]
        assert len(total_cols) > 0

    def test_load_manifest(self, bundle_path):
        manifest = load_bundle(bundle_path)
        assert manifest["type"] == "SimulationResults"
        assert manifest["simulation"]["n_simulations"] == 10

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            load_bundle(str(tmp_path / "nonexistent.epx"))

    def test_load_dataframe(self, bundle_path):
        df = load_bundle_dataframe(bundle_path, "compartments")
        assert "sim_id" in df.columns
        assert "date" in df.columns
        # Should have data for 10 simulations
        assert df["sim_id"].nunique() == 10

    def test_load_dataframe_with_columns(self, bundle_path):
        # Load only specific columns
        df = load_bundle_dataframe(
            bundle_path, "compartments",
            columns=["sim_id", "date", "Infected_total"],
        )
        assert list(df.columns) == ["sim_id", "date", "Infected_total"]

    def test_load_dataframe_invalid_key(self, bundle_path):
        with pytest.raises(KeyError, match="not found in bundle"):
            load_bundle_dataframe(bundle_path, "nonexistent")

    def test_save_with_config(self, sir_results, tmp_path):
        path = str(tmp_path / "results.epx")
        config = {"model": {"type": "SIR"}, "parameters": {"beta": 0.3}}
        manifest = save_bundle(sir_results, path, config=config)
        assert "config" in manifest["files"]

    def test_results_save_method(self, sir_results, tmp_path):
        path = str(tmp_path / "results.epx")
        manifest = sir_results.save(path)
        assert manifest["type"] == "SimulationResults"
        assert os.path.exists(os.path.join(path, "manifest.json"))

    def test_results_load_method(self, bundle_path):
        manifest = SimulationResults.load(bundle_path)
        assert manifest["type"] == "SimulationResults"

    def test_manifest_is_valid_json(self, bundle_path):
        """Verify the written manifest.json is valid JSON."""
        with open(os.path.join(bundle_path, "manifest.json")) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_parquet_data_integrity(self, sir_results, bundle_path):
        """Verify parquet data matches original simulation."""
        df = load_bundle_dataframe(bundle_path, "compartments")

        # Check sim 0 matches the original trajectory
        sim0 = df[df["sim_id"] == 0].sort_values("date")
        traj0 = sir_results.trajectories[0]

        for comp_name in traj0.compartments:
            if comp_name in sim0.columns:
                np.testing.assert_array_almost_equal(
                    sim0[comp_name].values,
                    traj0.compartments[comp_name],
                    decimal=5,
                )

    def test_invalid_type_raises(self, tmp_path):
        with pytest.raises(TypeError, match="Expected SimulationResults"):
            save_bundle("not a result", str(tmp_path / "bad.epx"))


# ---------------------------------------------------------------------------
# Inspection engine
# ---------------------------------------------------------------------------

class TestInspectManifest:
    def test_manifest_command(self, bundle_path):
        result = inspect_bundle(bundle_path, "manifest")
        assert result["type"] == "SimulationResults"
        assert "files" in result

    def test_invalid_command_raises(self, bundle_path):
        with pytest.raises(ValueError, match="Unknown inspect command"):
            inspect_bundle(bundle_path, "bogus_command")


class TestInspectQuantiles:
    def test_basic_quantiles(self, bundle_path):
        result = inspect_bundle(
            bundle_path, "quantiles",
            variables=["Infected_total"],
            quantiles=[0.05, 0.5, 0.95],
        )
        assert "dates" in result
        assert "Infected_total" in result
        assert "0.5" in result["Infected_total"]
        assert len(result["dates"]) == len(result["Infected_total"]["0.5"])

    def test_quantiles_default_variables(self, bundle_path):
        """Should default to _total variables."""
        result = inspect_bundle(bundle_path, "quantiles")
        # Should have at least one _total variable
        data_keys = [k for k in result if k != "dates"]
        assert len(data_keys) > 0
        assert all("_total" in k for k in data_keys)

    def test_quantiles_time_slice(self, bundle_path):
        full = inspect_bundle(
            bundle_path, "quantiles",
            variables=["Infected_total"],
        )
        sliced = inspect_bundle(
            bundle_path, "quantiles",
            variables=["Infected_total"],
            start="2023-01-05",
            end="2023-01-15",
        )
        assert len(sliced["dates"]) < len(full["dates"])
        assert sliced["dates"][0] >= "2023-01-05"
        assert sliced["dates"][-1] <= "2023-01-15"

    def test_quantiles_full_precision(self, bundle_path):
        """Inspection API returns full-precision floats; rounding is CLI-only."""
        result = inspect_bundle(
            bundle_path, "quantiles",
            variables=["Infected_total"],
            quantiles=[0.5],
        )
        # All values should be plain floats (no internal rounding)
        for val in result["Infected_total"]["0.5"]:
            assert isinstance(val, float)


class TestInspectSummary:
    def test_basic_summary(self, bundle_path):
        result = inspect_bundle(
            bundle_path, "summary",
            variables=["Infected_total"],
        )
        assert "Infected_total" in result
        summary = result["Infected_total"]
        assert "peak_value" in summary
        assert "0.50" in summary["peak_value"]
        assert "final_value" in summary
        assert "peak_date_median" in summary

    def test_summary_time_slice(self, bundle_path):
        result = inspect_bundle(
            bundle_path, "summary",
            variables=["Infected_total"],
            start="2023-01-05",
            end="2023-01-10",
        )
        assert "Infected_total" in result


class TestInspectPeak:
    def test_basic_peak(self, bundle_path):
        result = inspect_bundle(
            bundle_path, "peak",
            variables=["Infected_total"],
        )
        assert "Infected_total" in result
        peak = result["Infected_total"]
        assert "peak_date" in peak
        assert "peak_value" in peak
        assert "0.50" in peak["peak_value"]
        assert "0.50" in peak["peak_date"]

    def test_peak_time_slice(self, bundle_path):
        result = inspect_bundle(
            bundle_path, "peak",
            variables=["Infected_total"],
            start="2023-01-01",
            end="2023-01-10",
        )
        assert "Infected_total" in result


# ---------------------------------------------------------------------------
# JSON serialization of inspect results
# ---------------------------------------------------------------------------

class TestInspectJsonSafety:
    """Verify all inspect commands return JSON-serializable results."""

    def test_manifest_is_json_safe(self, bundle_path):
        result = inspect_bundle(bundle_path, "manifest")
        json.dumps(result)  # should not raise

    def test_quantiles_is_json_safe(self, bundle_path):
        result = inspect_bundle(
            bundle_path, "quantiles",
            variables=["Infected_total"],
        )
        json.dumps(result)

    def test_summary_is_json_safe(self, bundle_path):
        result = inspect_bundle(
            bundle_path, "summary",
            variables=["Infected_total"],
        )
        json.dumps(result)

    def test_peak_is_json_safe(self, bundle_path):
        result = inspect_bundle(
            bundle_path, "peak",
            variables=["Infected_total"],
        )
        json.dumps(result)
