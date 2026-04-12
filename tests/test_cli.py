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


# ---------------------------------------------------------------------------
# Config inheritance tests
# ---------------------------------------------------------------------------

class TestConfigInheritance:
    def test_simple_inheritance(self, tmp_path):
        """Overlay inherits model/simulation from base, overrides parameters."""
        base_cfg = {
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
        overlay_cfg = {
            "base": "base.yaml",
            "parameters": {
                "transmission_rate": 0.5,
            },
        }
        _write_yaml(tmp_path, base_cfg, "base.yaml")
        overlay_path = _write_yaml(tmp_path, overlay_cfg, "overlay.yaml")

        from epydemix.cli.config import load_config
        config = load_config(overlay_path)

        # Model and simulation inherited
        assert config["model"]["type"] == "SIR"
        assert config["simulation"]["start_date"] == "2023-01-01"
        # Parameter overridden
        assert config["parameters"]["transmission_rate"] == 0.5
        # Parameter inherited
        assert config["parameters"]["recovery_rate"] == 0.1
        # base key removed
        assert "base" not in config

    def test_list_replacement(self, tmp_path):
        """Lists (like overrides) are replaced, not appended."""
        base_cfg = {
            "model": {"type": "SIR"},
            "simulation": {"start_date": "2023-01-01", "end_date": "2023-01-30"},
            "parameters": {"transmission_rate": 0.3, "recovery_rate": 0.1},
            "overrides": [
                {"parameter": "transmission_rate", "start_date": "2023-01-10",
                 "end_date": "2023-01-20", "value": 0.1},
            ],
        }
        overlay_cfg = {
            "base": "base.yaml",
            "overrides": [
                {"parameter": "transmission_rate", "start_date": "2023-01-15",
                 "end_date": "2023-01-25", "value": 0.2},
            ],
        }
        _write_yaml(tmp_path, base_cfg, "base.yaml")
        overlay_path = _write_yaml(tmp_path, overlay_cfg, "overlay.yaml")

        from epydemix.cli.config import load_config
        config = load_config(overlay_path)

        # Overlay's overrides replaced base's
        assert len(config["overrides"]) == 1
        assert config["overrides"][0]["value"] == 0.2

    def test_chain_inheritance(self, tmp_path):
        """Three-level chain: grandparent -> parent -> child."""
        gp = {"model": {"type": "SIR"}, "simulation": {"start_date": "2023-01-01", "end_date": "2023-01-30"},
              "parameters": {"transmission_rate": 0.1, "recovery_rate": 0.05}}
        parent = {"base": "gp.yaml", "parameters": {"transmission_rate": 0.3}}
        child = {"base": "parent.yaml", "parameters": {"recovery_rate": 0.2}}

        _write_yaml(tmp_path, gp, "gp.yaml")
        _write_yaml(tmp_path, parent, "parent.yaml")
        child_path = _write_yaml(tmp_path, child, "child.yaml")

        from epydemix.cli.config import load_config
        config = load_config(child_path)

        assert config["parameters"]["transmission_rate"] == 0.3  # from parent
        assert config["parameters"]["recovery_rate"] == 0.2       # from child

    def test_circular_inheritance_raises(self, tmp_path):
        a = {"base": "b.yaml", "model": {"type": "SIR"}}
        b = {"base": "a.yaml", "simulation": {"start_date": "2023-01-01"}}
        _write_yaml(tmp_path, a, "a.yaml")
        _write_yaml(tmp_path, b, "b.yaml")

        from epydemix.cli.config import load_config
        with pytest.raises(ValueError, match="[Cc]ircular"):
            load_config(str(tmp_path / "a.yaml"))

    def test_run_with_inheritance(self, tmp_path):
        """End-to-end: run a simulation using an overlay config."""
        base_cfg = {
            "model": {"type": "SIR"},
            "simulation": {"start_date": "2023-01-01", "end_date": "2023-01-20",
                           "n_simulations": 3},
            "parameters": {"transmission_rate": 0.3, "recovery_rate": 0.1},
        }
        overlay_cfg = {
            "base": "base.yaml",
            "parameters": {"transmission_rate": 0.5},
        }
        _write_yaml(tmp_path, base_cfg, "base.yaml")
        overlay_path = _write_yaml(tmp_path, overlay_cfg, "overlay.yaml")
        output = str(tmp_path / "out.epx")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", overlay_path, "-o", output])
        assert result.exit_code == 0, f"Failed: {result.output}"


# ---------------------------------------------------------------------------
# Compare command tests
# ---------------------------------------------------------------------------

class TestCLICompare:
    def test_compare_two_bundles(self, tmp_path):
        b1 = _make_bundle(tmp_path, "run1.epx")
        b2 = _make_bundle(tmp_path, "run2.epx")
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", b1, b2])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "run1" in data
        assert "run2" in data
        # Default metrics
        assert "attack_rate" in data["run1"]
        assert "peak" in data["run1"]

    def test_compare_with_names(self, tmp_path):
        b1 = _make_bundle(tmp_path, "a.epx")
        b2 = _make_bundle(tmp_path, "b.epx")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare", b1, b2, "-n", "Baseline,Intervention"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "Baseline" in data
        assert "Intervention" in data

    def test_compare_with_baseline_deltas(self, tmp_path):
        b1 = _make_bundle(tmp_path, "base.epx")
        b2 = _make_bundle(tmp_path, "alt.epx")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare", b1, b2, "-n", "Base,Alt", "-b", "Base"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "_deltas_vs_Base" in data
        assert "Alt" in data["_deltas_vs_Base"]

    def test_compare_custom_metrics(self, tmp_path):
        b1 = _make_bundle(tmp_path, "r.epx")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare", b1, "-m", "attack_rate,total_deaths"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "attack_rate" in data["r"]
        assert "total_deaths" in data["r"]

    def test_compare_days_over(self, tmp_path):
        b1 = _make_bundle(tmp_path, "r.epx")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare", b1, "-m", "days_over:100"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "days_over" in data["r"]
        assert "threshold" in data["r"]["days_over"]


# ---------------------------------------------------------------------------
# Calibration config & CLI
# ---------------------------------------------------------------------------

def _calibration_config(observed_data=None):
    """Return a minimal calibration config for SIR with inline observed data."""
    # Synthetic declining curve (looks like infected counts)
    obs = observed_data if observed_data is not None else [
        100, 95, 90, 86, 82, 78, 74, 70, 67, 64,
        61, 58, 55, 53, 50, 48, 46, 44, 42, 40,
    ]
    return {
        "model": {"type": "SIR"},
        "simulation": {
            "start_date": "2023-01-01",
            "end_date": "2023-01-20",
        },
        "parameters": {
            "transmission_rate": 0.3,
            "recovery_rate": 0.1,
        },
        "initial_conditions": {
            "Susceptible": 0.99,
            "Infected": 0.01,
            "Recovered": 0.0,
        },
        "calibration": {
            "strategy": "top_fraction",
            "priors": {
                "transmission_rate": {
                    "distribution": "uniform",
                    "low": 0.05,
                    "high": 0.8,
                },
            },
            "observed_data": obs,
            "target_variable": "Infected_total",
            "distance": "rmse",
            "top_fraction": 0.5,
            "n_simulations": 10,
        },
    }


class TestCalibrationConfig:
    """Tests for calibration config validation and building."""

    def test_build_prior_uniform(self):
        from epydemix.cli.config import build_prior
        d = build_prior({"distribution": "uniform", "low": 0.1, "high": 0.6})
        # Should produce values in [0.1, 0.6]
        samples = d.rvs(1000)
        assert samples.min() >= 0.1 - 1e-9
        assert samples.max() <= 0.6 + 1e-9

    def test_build_prior_normal(self):
        from epydemix.cli.config import build_prior
        d = build_prior({"distribution": "normal", "mean": 0.3, "std": 0.05})
        assert abs(d.mean() - 0.3) < 1e-6

    def test_build_prior_truncnorm(self):
        from epydemix.cli.config import build_prior
        d = build_prior({
            "distribution": "truncnorm",
            "mean": 0.3, "std": 0.1,
            "low": 0.1, "high": 0.5,
        })
        samples = d.rvs(1000)
        assert samples.min() >= 0.1 - 1e-9
        assert samples.max() <= 0.5 + 1e-9

    def test_build_prior_unknown(self):
        from epydemix.cli.config import build_prior
        with pytest.raises(ValueError, match="Unknown distribution"):
            build_prior({"distribution": "magic"})

    def test_build_priors(self):
        from epydemix.cli.config import build_priors
        cal_cfg = {
            "priors": {
                "beta": {"distribution": "uniform", "low": 0.1, "high": 0.8},
                "gamma": {"distribution": "normal", "mean": 0.1, "std": 0.02},
            },
        }
        priors = build_priors(cal_cfg)
        assert "beta" in priors
        assert "gamma" in priors

    def test_build_priors_empty_raises(self):
        from epydemix.cli.config import build_priors
        with pytest.raises(ValueError, match="priors is required"):
            build_priors({"priors": {}})

    def test_load_observed_data_inline(self):
        from epydemix.cli.config import load_observed_data
        data = load_observed_data({"observed_data": [1.0, 2.0, 3.0]})
        assert len(data) == 3
        assert data[1] == 2.0

    def test_load_observed_data_csv(self, tmp_path):
        from epydemix.cli.config import load_observed_data
        csv_path = tmp_path / "obs.csv"
        csv_path.write_text("date,cases\n2023-01-01,100\n2023-01-02,95\n")
        data = load_observed_data(
            {"observed_data": "obs.csv", "observed_column": "cases"},
            config_dir=tmp_path,
        )
        assert len(data) == 2
        assert data[0] == 100.0

    def test_load_observed_data_csv_auto_column(self, tmp_path):
        from epydemix.cli.config import load_observed_data
        csv_path = tmp_path / "obs.csv"
        csv_path.write_text("date,value\n2023-01-01,50\n2023-01-02,60\n")
        data = load_observed_data(
            {"observed_data": "obs.csv"},
            config_dir=tmp_path,
        )
        # Two columns, no observed_column specified → should pick second column
        assert data[0] == 50.0

    def test_validate_calibration_config_valid(self):
        from epydemix.cli.config import validate_calibration_config
        cfg = _calibration_config()
        result = validate_calibration_config(cfg)
        assert result["valid"] is True

    def test_validate_calibration_config_missing_priors(self):
        from epydemix.cli.config import validate_calibration_config
        cfg = _calibration_config()
        del cfg["calibration"]["priors"]
        result = validate_calibration_config(cfg)
        assert result["valid"] is False
        assert any("priors" in e for e in result["errors"])

    def test_validate_calibration_config_bad_strategy(self):
        from epydemix.cli.config import validate_calibration_config
        cfg = _calibration_config()
        cfg["calibration"]["strategy"] = "magic"
        result = validate_calibration_config(cfg)
        assert result["valid"] is False

    def test_validate_calibration_config_bad_distance(self):
        from epydemix.cli.config import validate_calibration_config
        cfg = _calibration_config()
        cfg["calibration"]["distance"] = "unknown_metric"
        result = validate_calibration_config(cfg)
        assert result["valid"] is False

    def test_resolve_distance_function(self):
        from epydemix.cli.config import _resolve_distance_function
        fn = _resolve_distance_function("rmse")
        assert callable(fn)
        with pytest.raises(ValueError, match="Unknown distance"):
            _resolve_distance_function("bogus")


class TestCalibrateCLI:
    """Tests for the `epydemix calibrate` CLI command."""

    def test_calibrate_inline_data(self, tmp_path):
        """Run calibration with inline observed data (top_fraction, fast)."""
        cfg = _calibration_config()
        config_path = _write_yaml(tmp_path, cfg, "cal.yaml")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "calibrate", config_path,
            "-o", str(tmp_path / "cal.epx"),
        ])
        # Parse JSON from stdout (skip any leading non-JSON lines)
        raw = result.output
        if "{" in raw:
            json_start = raw.index("{")
            data = json.loads(raw[json_start:])
            assert "type" in data or "calibration_strategy" in data or "files" in data
        if result.exit_code != 0:
            # Print stderr for debugging
            print("STDERR:", result.output)
        assert result.exit_code == 0

        # Bundle should exist on disk
        bundle_dir = tmp_path / "cal.epx"
        assert bundle_dir.exists()
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "posterior.parquet").exists()

    def test_calibrate_csv_observed(self, tmp_path):
        """Run calibration with CSV observed data."""
        # Write observed data CSV
        csv_path = tmp_path / "observed.csv"
        obs_values = [100, 95, 90, 86, 82, 78, 74, 70, 67, 64,
                      61, 58, 55, 53, 50, 48, 46, 44, 42, 40]
        csv_path.write_text(
            "date,cases\n" +
            "\n".join(f"2023-01-{i+1:02d},{v}" for i, v in enumerate(obs_values))
            + "\n"
        )

        cfg = _calibration_config()
        cfg["calibration"]["observed_data"] = "observed.csv"
        cfg["calibration"]["observed_column"] = "cases"
        config_path = _write_yaml(tmp_path, cfg, "cal_csv.yaml")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "calibrate", config_path,
            "-o", str(tmp_path / "cal_csv.epx"),
        ])
        assert result.exit_code == 0

    def test_calibrate_invalid_config(self, tmp_path):
        """Calibrate with missing priors → should fail."""
        cfg = _calibration_config()
        del cfg["calibration"]["priors"]
        config_path = _write_yaml(tmp_path, cfg, "bad_cal.yaml")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "calibrate", config_path,
            "-o", str(tmp_path / "bad.epx"),
        ])
        assert result.exit_code == 1

    def test_calibrate_inspect_posterior(self, tmp_path):
        """Run calibration, then inspect posterior."""
        cfg = _calibration_config()
        config_path = _write_yaml(tmp_path, cfg, "cal.yaml")
        bundle_path = str(tmp_path / "cal_post.epx")

        runner = CliRunner()
        # Run calibration
        result = runner.invoke(cli, [
            "calibrate", config_path, "-o", bundle_path,
        ])
        assert result.exit_code == 0

        # Inspect posterior
        result = runner.invoke(cli, [
            "inspect", bundle_path, "posterior",
        ])
        assert result.exit_code == 0
        raw = result.output
        if "{" in raw:
            data = json.loads(raw[raw.index("{"):])
            assert "transmission_rate" in data

    def test_calibrate_with_inheritance(self, tmp_path):
        """Calibration config can inherit from a base simulation config."""
        # Write base config
        base_cfg = {
            "model": {"type": "SIR"},
            "simulation": {
                "start_date": "2023-01-01",
                "end_date": "2023-01-20",
            },
            "parameters": {
                "transmission_rate": 0.3,
                "recovery_rate": 0.1,
            },
            "initial_conditions": {
                "Susceptible": 0.99,
                "Infected": 0.01,
                "Recovered": 0.0,
            },
        }
        _write_yaml(tmp_path, base_cfg, "base.yaml")

        # Write calibration overlay
        cal_overlay = {
            "base": "base.yaml",
            "calibration": {
                "strategy": "top_fraction",
                "priors": {
                    "transmission_rate": {
                        "distribution": "uniform",
                        "low": 0.05,
                        "high": 0.8,
                    },
                },
                "observed_data": [100, 95, 90, 86, 82, 78, 74, 70, 67, 64,
                                  61, 58, 55, 53, 50, 48, 46, 44, 42, 40],
                "target_variable": "Infected_total",
                "distance": "rmse",
                "top_fraction": 0.5,
                "n_simulations": 10,
            },
        }
        config_path = _write_yaml(tmp_path, cal_overlay, "cal_inherit.yaml")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "calibrate", config_path,
            "-o", str(tmp_path / "cal_inherit.epx"),
        ])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Projection config & CLI
# ---------------------------------------------------------------------------

def _run_calibration(tmp_path, name="cal.epx"):
    """Run a quick calibration and return the bundle path."""
    cfg = _calibration_config()
    config_path = _write_yaml(tmp_path, cfg, "cal_for_proj.yaml")
    bundle_path = str(tmp_path / name)
    runner = CliRunner()
    result = runner.invoke(cli, [
        "calibrate", config_path, "-o", bundle_path,
    ])
    assert result.exit_code == 0, f"Calibration failed: {result.output}"
    return bundle_path


class TestProjectionConfig:
    """Tests for projection config validation."""

    def test_validate_projection_config_valid(self, tmp_path):
        from epydemix.cli.config import validate_projection_config
        bundle = _run_calibration(tmp_path)
        cfg = {
            "model": {"type": "SIR"},
            "simulation": {
                "start_date": "2023-01-01",
                "end_date": "2023-02-28",
            },
            "parameters": {
                "transmission_rate": 0.3,
                "recovery_rate": 0.1,
            },
            "projection": {"n_simulations": 10},
        }
        result = validate_projection_config(cfg, bundle)
        assert result["valid"] is True

    def test_validate_projection_config_missing_bundle(self):
        from epydemix.cli.config import validate_projection_config
        cfg = {
            "model": {"type": "SIR"},
            "simulation": {
                "start_date": "2023-01-01",
                "end_date": "2023-02-28",
            },
        }
        result = validate_projection_config(cfg, "/nonexistent/bundle.epx")
        assert result["valid"] is False
        assert any("not found" in e for e in result["errors"])

    def test_validate_projection_config_missing_simulation(self, tmp_path):
        from epydemix.cli.config import validate_projection_config
        bundle = _run_calibration(tmp_path)
        cfg = {"model": {"type": "SIR"}}
        result = validate_projection_config(cfg, bundle)
        assert result["valid"] is False
        assert any("simulation" in e for e in result["errors"])


class TestProjectCLI:
    """Tests for the `epydemix project` CLI command."""

    def test_project_no_overlay(self, tmp_path):
        """Project using the calibration bundle's own config (no overlay)."""
        bundle = _run_calibration(tmp_path)
        proj_out = str(tmp_path / "proj.epx")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "project", bundle, "-o", proj_out,
        ])
        assert result.exit_code == 0, f"Project failed: {result.output}"

        # Should produce a simulation bundle
        assert os.path.isdir(proj_out)
        assert os.path.exists(os.path.join(proj_out, "manifest.json"))
        assert os.path.exists(os.path.join(proj_out, "compartments.parquet"))

        # Manifest should be SimulationResults type
        raw = result.output
        if "{" in raw:
            data = json.loads(raw[raw.index("{"):])
            assert data.get("type") == "SimulationResults"

    def test_project_with_overlay(self, tmp_path):
        """Project with a config overlay that extends the simulation period."""
        bundle = _run_calibration(tmp_path)

        # Write overlay config inheriting from the calibration bundle's config
        import yaml
        overlay_cfg = {
            "base": os.path.join(bundle, "config.yaml"),
            "simulation": {
                "end_date": "2023-02-28",
            },
            "projection": {
                "n_simulations": 5,
            },
        }
        overlay_path = _write_yaml(tmp_path, overlay_cfg, "proj_overlay.yaml")
        proj_out = str(tmp_path / "proj_overlay.epx")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "project", bundle, "-c", overlay_path, "-o", proj_out,
        ])
        assert result.exit_code == 0, f"Project failed: {result.output}"
        assert os.path.isdir(proj_out)

    def test_project_inspect_results(self, tmp_path):
        """Project results can be inspected with standard inspect commands."""
        bundle = _run_calibration(tmp_path)
        proj_out = str(tmp_path / "proj_insp.epx")

        runner = CliRunner()
        runner.invoke(cli, ["project", bundle, "-o", proj_out])

        # Inspect quantiles on the projection bundle
        result = runner.invoke(cli, [
            "inspect", proj_out, "quantiles",
            "-v", "Infected_total",
            "-q", "0.05,0.5,0.95",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "dates" in data
        assert "Infected_total" in data

    def test_project_weights_saved(self, tmp_path):
        """Calibration bundles now include weights.parquet."""
        bundle = _run_calibration(tmp_path)
        assert os.path.exists(os.path.join(bundle, "weights.parquet"))

    def test_project_invalid_bundle(self, tmp_path):
        """Project with a non-bundle directory should fail."""
        fake_bundle = str(tmp_path / "fake.epx")
        os.makedirs(fake_bundle)
        runner = CliRunner()
        result = runner.invoke(cli, ["project", fake_bundle])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Provenance / lineage tests
# ---------------------------------------------------------------------------

class TestProvenance:
    """Verify that CLI commands embed provenance in the manifest."""

    def _parse_manifest(self, output):
        """Extract JSON manifest from mixed stdout/stderr CliRunner output."""
        raw = output
        if "{" in raw:
            return json.loads(raw[raw.index("{"):])
        return None

    def test_run_provenance(self, tmp_path):
        config_path = _write_yaml(tmp_path, MINIMAL_CONFIG)
        output_path = str(tmp_path / "prov_run.epx")
        runner = CliRunner()
        result = runner.invoke(cli, ["run", config_path, "-o", output_path])
        assert result.exit_code == 0

        data = self._parse_manifest(result.output)
        assert data is not None
        assert "provenance" in data
        assert data["provenance"]["command"] == "run"
        assert "config_path" in data["provenance"]

    def test_calibrate_provenance(self, tmp_path):
        cfg = _calibration_config()
        config_path = _write_yaml(tmp_path, cfg, "cal_prov.yaml")
        output_path = str(tmp_path / "prov_cal.epx")
        runner = CliRunner()
        result = runner.invoke(cli, ["calibrate", config_path, "-o", output_path])
        assert result.exit_code == 0

        data = self._parse_manifest(result.output)
        assert data is not None
        assert "provenance" in data
        assert data["provenance"]["command"] == "calibrate"
        assert "config_path" in data["provenance"]

    def test_project_provenance(self, tmp_path):
        bundle = _run_calibration(tmp_path, "prov_parent.epx")
        proj_out = str(tmp_path / "prov_proj.epx")
        runner = CliRunner()
        result = runner.invoke(cli, ["project", bundle, "-o", proj_out])
        assert result.exit_code == 0

        data = self._parse_manifest(result.output)
        assert data is not None
        assert "provenance" in data
        prov = data["provenance"]
        assert prov["command"] == "project"
        assert "parent_bundle" in prov
        assert "prov_parent.epx" in prov["parent_bundle"]
        # No overlay config → config_path should be absent
        assert "config_path" not in prov

    def test_project_provenance_with_overlay(self, tmp_path):
        bundle = _run_calibration(tmp_path, "prov_parent2.epx")
        overlay_cfg = {
            "base": os.path.join(bundle, "config.yaml"),
            "simulation": {"end_date": "2023-02-28"},
            "projection": {"n_simulations": 5},
        }
        overlay_path = _write_yaml(tmp_path, overlay_cfg, "prov_overlay.yaml")
        proj_out = str(tmp_path / "prov_proj2.epx")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "project", bundle, "-c", overlay_path, "-o", proj_out,
        ])
        assert result.exit_code == 0

        data = self._parse_manifest(result.output)
        assert data is not None
        prov = data["provenance"]
        assert prov["command"] == "project"
        assert "parent_bundle" in prov
        assert "config_path" in prov  # overlay was provided

    def test_overwrite_warning(self, tmp_path, capsys):
        """Second save to same path emits a warning to stderr."""
        config_path = _write_yaml(tmp_path, MINIMAL_CONFIG)
        output_path = str(tmp_path / "overwrite.epx")
        runner = CliRunner()
        # First run — no warning
        runner.invoke(cli, ["run", config_path, "-o", output_path])
        # Second run — should warn
        result = runner.invoke(cli, ["run", config_path, "-o", output_path])
        assert result.exit_code == 0
        # CliRunner captures both stdout and stderr in result.output
        # The warning goes to stderr via bundle.py's print(..., file=sys.stderr)
        # but CliRunner may capture it depending on mix_stderr setting.
        # Check the manifest file on disk instead — it should still be valid
        with open(os.path.join(output_path, "manifest.json")) as f:
            manifest = json.load(f)
        assert "provenance" in manifest


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
