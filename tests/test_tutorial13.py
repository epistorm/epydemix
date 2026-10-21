"""The workflow of tutorial 13 (the command-line interface), run through the CLI.

The notebook uses a named population and data downloaded from GitHub. Like the
other tutorial tests, this one swaps those for local stand-ins (a flat
population, synthetic incidence) so it runs offline.
"""

import json
import os

import pytest
import yaml
from click.testing import CliRunner

from epydemix.cli.main import cli
from epydemix.io import add_figure_to_manifest, load_bundle_dataframe

BASELINE = {
    "model": {"type": "SEIR"},
    "parameters": {
        "transmission_rate": 0.35,
        "incubation_rate": 0.33,
        "recovery_rate": 0.14,
    },
    "population": {"size": 100000},
    "simulation": {
        "start_date": "2024-10-01",
        "end_date": "2025-01-31",
        "n_simulations": 10,
        "seed": 42,
    },
    "initial_conditions": {
        "Susceptible": 0.9998,
        "Exposed": 0.0001,
        "Infected": 0.0001,
        "Recovered": 0.0,
    },
}


def _write(path, config):
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return str(path)


def _cli(*args, expect=0):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == expect, result.output
    return result


def _json(*args, expect=0):
    """Run a command that prints only JSON and return it parsed."""
    return json.loads(_cli(*args, expect=expect).output)


@pytest.fixture
def workdir(tmp_path):
    _write(tmp_path / "baseline.yaml", BASELINE)
    return tmp_path


def test_discovery():
    assert "SEIR" in _json("models")["models"]
    assert "transmission_rate" in _cli("schema", "SEIR", "--format", "describe").output
    covid = _json("defaults", "covid19")
    assert covid["model_type"] == "SEIR"
    assert {"transmission_rate", "incubation_rate", "recovery_rate"} <= set(
        covid["parameters"]
    )


def test_validation_catches_common_mistakes(workdir):
    assert _json("validate", workdir / "baseline.yaml")["valid"]

    broken = yaml.safe_load(yaml.safe_dump(BASELINE))
    broken["parameters"]["transmision_rate"] = broken["parameters"].pop(
        "transmission_rate"
    )
    broken["parameters"]["recovery_rate"] = 14
    broken["initial_conditions"]["S"] = broken["initial_conditions"].pop("Susceptible")
    report = _json("validate", _write(workdir / "broken.yaml", broken), expect=1)

    assert not report["valid"]
    errors = " ".join(report["errors"])
    assert "did you mean 'transmission_rate'" in errors
    assert "above the maximum" in errors
    assert "'S' is not a compartment" in errors


def test_run_inspect_and_compare_scenarios(workdir):
    _cli("run", workdir / "baseline.yaml", "-o", workdir / "baseline.epx")
    assert sorted(os.listdir(workdir / "baseline.epx")) == [
        "compartments.parquet",
        "config.yaml",
        "figures",
        "manifest.json",
        "parameters.parquet",
        "transitions.parquet",
    ]

    peak = _json("inspect", workdir / "baseline.epx", "peak", "-v", "Infected_total")
    assert set(peak["Infected_total"]) == {"peak_date", "peak_value"}

    weekly = _cli(
        "inspect", workdir / "baseline.epx", "quantiles",
        "-v", "Infected_total", "-q", "0.05,0.5,0.95",
        "--start", "2024-11-01", "--end", "2024-11-30",
        "--resample", "W", "--format", "csv",
    ).output.strip().splitlines()  # fmt: skip
    assert weekly[0].startswith("date,Infected_total_0.05")
    assert len(weekly) > 1

    # Scenarios inherit from the baseline and only add an override
    names = ["baseline"]
    for name, start, end in [
        ("early", "2024-10-15", "2024-11-26"),
        ("late", "2024-11-15", "2024-12-27"),
    ]:
        overlay = {
            "base": "baseline.yaml",
            "overrides": [
                {
                    "parameter": "transmission_rate",
                    "start_date": start,
                    "end_date": end,
                    "value": 0.2,
                }
            ],
        }
        path = _write(workdir / f"{name}.yaml", overlay)
        assert _json("validate", path)["valid"]
        _cli("run", path, "-o", workdir / f"{name}.epx")
        names.append(name)

    comparison = _json(
        "compare",
        *[workdir / f"{n}.epx" for n in names],
        "-m", "attack_rate,peak,peak_date",
        "-b", "baseline",
    )  # fmt: skip
    assert set(comparison["_deltas_vs_baseline"]) == {"early", "late"}
    for name in names:
        assert set(comparison[name]) == {"attack_rate", "peak", "peak_date"}

    # Back to Python: read a bundle with pandas and store a figure in it
    df = load_bundle_dataframe(
        str(workdir / "early.epx"),
        "compartments",
        columns=["sim_id", "date", "Infected_total"],
    )
    runs = df.pivot(index="date", columns="sim_id", values="Infected_total")
    assert runs.shape[1] == BASELINE["simulation"]["n_simulations"]

    (workdir / "baseline.epx" / "figures" / "scenarios.png").write_bytes(b"")
    add_figure_to_manifest(
        str(workdir / "baseline.epx"), "scenarios.png", "scenarios", ["Infected_total"]
    )
    manifest = _json("inspect", workdir / "baseline.epx", "manifest")
    assert "scenarios.png" in manifest["figures"]


def test_calibrate_and_project(workdir):
    # Synthetic observed incidence: one seeded run of an SIR model
    truth = {
        "model": {"type": "SIR"},
        "parameters": {"transmission_rate": 0.3, "recovery_rate": 0.15},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "n_simulations": 1,
            "seed": 1,
        },
        "initial_conditions": {
            "Susceptible": 0.999,
            "Infected": 0.001,
            "Recovered": 0.0,
        },
    }
    _cli("run", _write(workdir / "truth.yaml", truth), "-o", workdir / "truth.epx")
    incidence = load_bundle_dataframe(
        str(workdir / "truth.epx"),
        "transitions",
        columns=["date", "Susceptible_to_Infected_total"],
    )
    incidence.rename(columns={"Susceptible_to_Infected_total": "data"}).to_csv(
        workdir / "incidence_data.csv", index=False
    )

    calibration = {key: value for key, value in truth.items() if key != "parameters"}
    calibration["simulation"] = {**truth["simulation"], "seed": 7}
    del calibration["simulation"]["n_simulations"]
    calibration["calibration"] = {
        "strategy": "smc",
        "priors": {
            "transmission_rate": {"distribution": "uniform", "low": 0.1, "high": 0.5},
            "recovery_rate": {"distribution": "uniform", "low": 0.1, "high": 0.2},
        },
        "observed_data": "incidence_data.csv",
        "observed_column": "data",
        "target_variable": "Susceptible_to_Infected_total",
        "distance": "rmse",
        "num_particles": 20,
        "num_generations": 2,
    }
    path = _write(workdir / "calibration.yaml", calibration)
    assert _json("validate", path)["valid"]
    _cli("calibrate", path, "-o", workdir / "calibration.epx")

    posterior = _json("inspect", workdir / "calibration.epx", "posterior")
    for name in ("transmission_rate", "recovery_rate"):
        low, high = posterior[name]["ci95"]
        assert low <= posterior[name]["median"] <= high

    fit = _json(
        "inspect", workdir / "calibration.epx", "fit",
        "-v", "Susceptible_to_Infected_total", "-q", "0.05,0.5,0.95",
    )  # fmt: skip
    assert len(fit["Susceptible_to_Infected_total"]["0.5"]) == len(incidence)
    assert "Susceptible_to_Infected_total" in fit["observed"]

    # A replay of the calibration period, and a counterfactual overlay
    overlay = {
        "overrides": [
            {
                "parameter": "transmission_rate",
                "start_date": "2024-01-15",
                "end_date": "2024-02-29",
                "value": 0.1,
            }
        ]
    }
    _cli("project", workdir / "calibration.epx", "-o", workdir / "as_observed.epx")
    _cli(
        "project", workdir / "calibration.epx",
        "-c", _write(workdir / "earlier_measures.yaml", overlay),
        "-o", workdir / "earlier_measures.epx",
    )  # fmt: skip
    comparison = _json(
        "compare", workdir / "as_observed.epx", workdir / "earlier_measures.epx",
        "-m", "attack_rate,peak", "-b", "as_observed",
    )  # fmt: skip
    delta = comparison["_deltas_vs_as_observed"]["earlier_measures"]
    assert delta["attack_rate"] < 0
