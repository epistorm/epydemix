"""Tests for the CLI (Click commands) and config loader."""

import json
import os
import tempfile

import numpy as np
import pytest
from click.testing import CliRunner

from epydemix import load_predefined_model
from epydemix.cli.main import cli
from epydemix.cli.config import load_config, validate_config, build_model_from_config
from epydemix.io.bundle import save_bundle
from epydemix.population import Population


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_population():
    pop = Population()
    pop.add_population([10000])
    pop.add_contact_matrix(np.array([[1.0]]))
    return pop


def _make_bundle(tmp_path, name="results.epx"):
    """Run a short SIR simulation and save as a bundle."""
    model = load_predefined_model("SIR", transmission_rate=0.3, recovery_rate=0.1)
    model.set_population(_simple_population())
    results = model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-20",
        initial_conditions_dict={
            "Susceptible": np.array([9900]),
            "Infected": np.array([100]),
            "Recovered": np.array([0]),
        },
        Nsim=5,
    )
    path = str(tmp_path / name)
    save_bundle(results, path)
    return path


MINIMAL_CONFIG = {
    "model": {"type": "SIR"},
    "simulation": {
        "start_date": "2023-01-01",
        "end_date": "2023-01-30",
        "n_simulations": 5,
    },
    "parameters": {
        "transmission_rate": 0.3,
        "recovery_rate": 0.1,
    },
}


def _write_yaml(tmp_path, cfg, name="config.yaml"):
    import yaml
    path = tmp_path / name
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return str(path)


def _write_json(tmp_path, cfg, name="config.json"):
    path = tmp_path / name
    with open(path, "w") as f:
        json.dump(cfg, f)
    return str(path)


# ---------------------------------------------------------------------------
# Config loader tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(path)
        assert cfg["model"]["type"] == "SIR"

    def test_load_json(self, tmp_path):
        path = _write_json(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(path)
        assert cfg["simulation"]["start_date"] == "2023-01-01"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/file.yaml")


class TestValidateConfig:
    def test_valid_config(self):
        result = validate_config(MINIMAL_CONFIG)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_model(self):
        cfg = {"simulation": {"start_date": "2023-01-01", "end_date": "2023-01-30"}}
        result = validate_config(cfg)
        assert result["valid"] is False
        assert any("model" in e for e in result["errors"])

    def test_missing_simulation(self):
        cfg = {"model": {"type": "SIR"}}
        result = validate_config(cfg)
        assert result["valid"] is False
        assert any("simulation" in e for e in result["errors"])

    def test_missing_dates(self):
        cfg = {"model": {"type": "SIR"}, "simulation": {}}
        result = validate_config(cfg)
        assert result["valid"] is False
        assert any("start_date" in e for e in result["errors"])
        assert any("end_date" in e for e in result["errors"])

    def test_unknown_model_type(self):
        cfg = {
            "model": {"type": "UNKNOWN"},
            "simulation": {"start_date": "2023-01-01", "end_date": "2023-01-30"},
        }
        result = validate_config(cfg)
        assert result["valid"] is False
        assert any("Unknown" in e or "UNKNOWN" in e for e in result["errors"])

    def test_custom_model_needs_compartments(self):
        cfg = {
            "model": {"type": "custom"},
            "simulation": {"start_date": "2023-01-01", "end_date": "2023-01-30"},
        }
        result = validate_config(cfg)
        assert result["valid"] is False
        assert any("compartments" in e for e in result["errors"])

    def test_warnings_for_missing_params(self):
        cfg = {
            "model": {"type": "SIR"},
            "simulation": {"start_date": "2023-01-01", "end_date": "2023-01-30"},
        }
        result = validate_config(cfg)
        assert result["valid"] is True
        assert len(result["warnings"]) > 0


class TestBuildModel:
    def test_build_sir(self):
        model = build_model_from_config(MINIMAL_CONFIG)
        assert "Susceptible" in model.compartments
        assert "Infected" in model.compartments
        assert "Recovered" in model.compartments


# ---------------------------------------------------------------------------
# CLI command tests (using Click's CliRunner)
# ---------------------------------------------------------------------------

class TestCLIModels:
    def test_models_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "models" in data
        assert "SIR" in data["models"]
        assert "SEIR" in data["models"]


class TestCLISchema:
    def test_schema_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["schema", "SIR"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "properties" in data

    def test_schema_describe(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["schema", "SIR", "--format", "describe"])
        assert result.exit_code == 0
        assert "transmission_rate" in result.output

    def test_schema_unknown_model(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["schema", "UNKNOWN"])
        assert result.exit_code != 0


class TestCLIDefaults:
    def test_defaults_list(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["defaults"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "defaults" in data
        assert "covid19" in data["defaults"]

    def test_defaults_detail(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["defaults", "covid19"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["model_type"] == "SEIR"
        assert "transmission_rate" in data["parameters"]

    def test_defaults_unknown(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["defaults", "nonexistent"])
        assert result.exit_code != 0


class TestCLIValidate:
    def test_validate_valid(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_CONFIG)
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", path])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["valid"] is True

    def test_validate_invalid(self, tmp_path):
        path = _write_yaml(tmp_path, {"model": {"type": "SIR"}})
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", path])
        assert result.exit_code != 0


class TestCLIRun:
    def test_run_sir(self, tmp_path):
        config_path = _write_yaml(tmp_path, MINIMAL_CONFIG)
        output_path = str(tmp_path / "out.epx")
        runner = CliRunner()
        result = runner.invoke(cli, ["run", config_path, "--output", output_path])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # CliRunner mixes stderr into output; extract the JSON object
        # by finding the first '{' in the output
        raw = result.output
        json_start = raw.index("{")
        data = json.loads(raw[json_start:])
        assert "model" in data
        assert os.path.isdir(output_path)

    def test_run_invalid_config(self, tmp_path):
        path = _write_yaml(tmp_path, {"model": {"type": "SIR"}})
        runner = CliRunner()
        result = runner.invoke(cli, ["run", path])
        assert result.exit_code != 0


class TestCLIInspect:
    def test_inspect_manifest(self, tmp_path):
        bundle = _make_bundle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", bundle, "manifest"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "model" in data
        assert "files" in data

    def test_inspect_quantiles(self, tmp_path):
        bundle = _make_bundle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "inspect", bundle, "quantiles",
            "-v", "Infected_total",
            "-q", "0.05,0.5,0.95",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "dates" in data
        assert "Infected_total" in data

    def test_inspect_summary(self, tmp_path):
        bundle = _make_bundle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", bundle, "summary", "-v", "Infected_total"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "Infected_total" in data

    def test_inspect_peak(self, tmp_path):
        bundle = _make_bundle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", bundle, "peak", "-v", "Infected_total"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "Infected_total" in data
        assert "peak_date" in data["Infected_total"]

    def test_inspect_csv_format(self, tmp_path):
        bundle = _make_bundle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "inspect", bundle, "quantiles",
            "-v", "Infected_total",
            "-q", "0.5",
            "--format", "csv",
        ])
        assert result.exit_code == 0
        # CSV output should have headers with "date"
        lines = result.output.strip().split("\n")
        assert "date" in lines[0]

    def test_inspect_time_slice(self, tmp_path):
        bundle = _make_bundle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "inspect", bundle, "quantiles",
            "-v", "Infected_total",
            "--start", "2023-01-05",
            "--end", "2023-01-15",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        dates = data["dates"]
        assert all(d >= "2023-01-05" for d in dates)
        assert all(d <= "2023-01-15" for d in dates)

    def test_inspect_unknown_command(self, tmp_path):
        bundle = _make_bundle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", bundle, "nonexistent"])
        assert result.exit_code != 0


class TestCLIPopulations:
    def test_populations_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["populations"])
        # May fail if population data files are not installed;
        # just verify it returns valid JSON (either success or structured error)
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert "populations" in data
        else:
            # Structured error on stderr — verify it's a handled failure
            assert result.exit_code == 1
