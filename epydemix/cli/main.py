"""Click CLI for epydemix — the primary interface for LLM agents.

Usage::

    epydemix run config.yaml --output results.epx
    epydemix inspect results.epx quantiles --variables I_total --quantiles 0.05,0.5,0.95
    epydemix schema SEIR
    epydemix validate config.yaml
    epydemix models
    epydemix defaults
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from ..io.json_utils import NumpySafeEncoder


def _print_json(data, precision=None):
    """Print a dict as JSON to stdout."""
    if precision is not None:
        from ..io.json_utils import _round_recursive
        data = _round_recursive(data, precision)
    click.echo(json.dumps(data, indent=2, cls=NumpySafeEncoder))


def _error_json(code, message, details=None):
    """Print a structured error to stderr and exit."""
    err = {"error": True, "code": code, "message": message}
    if details:
        err["details"] = details
    click.echo(json.dumps(err, indent=2), err=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="epydemix")
def cli():
    """Epydemix: epidemic modeling, simulation, and calibration.

    Agent-friendly CLI for driving epydemix from configs and inspecting results.
    Every command prints structured JSON to stdout and diagnostics/warnings to
    stderr, so stdout is always safe to parse or pipe.

    Run `epydemix COMMAND --help` for a given command's options and examples.
    """
    pass


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="results.epx",
              help="Path for the output .epx bundle.")
def run(config_path, output):
    """Run a simulation from a YAML/JSON config file.

    Writes an .epx bundle and prints the manifest to stdout.
    """
    from .config import load_config, run_from_config, validate_config
    from ..io.bundle import save_bundle

    try:
        config = load_config(config_path)
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))

    validation = validate_config(config)
    if not validation["valid"]:
        _error_json("INVALID_CONFIG", "Config validation failed.",
                    details=validation["errors"])

    # Print warnings to stderr
    for w in validation.get("warnings", []):
        click.echo(f"Warning: {w}", err=True)

    try:
        config_dir = Path(config_path).parent
        results, used_config = run_from_config(config, config_dir=config_dir)
        prov = {
            "command": "run",
            "config_path": str(Path(config_path).resolve()),
        }
        manifest = save_bundle(results, output, config=used_config, provenance=prov)
        click.echo(f"Bundle saved to: {output}", err=True)
        _print_json(manifest)
    except Exception as e:
        _error_json("RUNTIME_ERROR", str(e))


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="calibration.epx",
              help="Path for the output .epx bundle.")
def calibrate(config_path, output):
    """Run ABC calibration from a YAML/JSON config file.

    The config must include a 'calibration' section with priors,
    observed data, and strategy settings.  See AGENT.md for full
    config reference.

    Writes an .epx calibration bundle and prints the manifest to stdout.
    """
    from .config import (
        load_config,
        validate_calibration_config,
        calibrate_from_config,
    )
    from ..io.bundle import save_bundle

    try:
        config = load_config(config_path)
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))

    validation = validate_calibration_config(config)
    if not validation["valid"]:
        _error_json("INVALID_CONFIG", "Calibration config validation failed.",
                    details=validation["errors"])

    for w in validation.get("warnings", []):
        click.echo(f"Warning: {w}", err=True)

    try:
        config_dir = Path(config_path).parent
        results, used_config = calibrate_from_config(config, config_dir=config_dir)
        prov = {
            "command": "calibrate",
            "config_path": str(Path(config_path).resolve()),
        }
        manifest = save_bundle(results, output, config=used_config, provenance=prov)
        click.echo(f"Calibration bundle saved to: {output}", err=True)
        _print_json(manifest)
    except Exception as e:
        _error_json("RUNTIME_ERROR", str(e))


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("calibration_bundle", type=click.Path(exists=True))
@click.option("--config", "-c", "config_path", default=None,
              type=click.Path(exists=True),
              help="Projection overlay config (inherits from calibration bundle's config).")
@click.option("--output", "-o", default="projection.epx",
              help="Path for the output .epx bundle.")
def project(calibration_bundle, config_path, output):
    """Run forward projections from a calibration posterior.

    Samples parameter sets from the calibration posterior (weighted),
    runs forward simulations with the (possibly overridden) config,
    and saves results as a standard simulation bundle.

    \b
    The calibration bundle's stored config is always the implicit base.
    A projection overlay only needs to specify what changes — no ``base:``
    key is required:

    \b
      simulation:
        end_date: "2025-06-30"    # extend beyond calibration period
      overrides:
        - parameter: transmission_rate
          start_date: "2025-04-01"
          end_date: "2025-06-30"
          value: 0.15
      projection:
        n_simulations: 200

    If --config is omitted, the calibration bundle's own config is used
    (i.e. replay with posterior samples over the same period).

    \b
    Examples:
      epydemix project calibration.epx -o projection.epx
      epydemix project calibration.epx -c projection.yaml -o projection.epx
    """
    from .config import (
        load_config,
        validate_projection_config,
        project_from_config,
    )
    from ..io.bundle import save_bundle

    # Build the effective config
    try:
        import yaml

        from .config import _deep_merge

        bundle_p = Path(calibration_bundle)
        cfg_yaml = bundle_p / "config.yaml"
        cfg_json = bundle_p / "config.json"
        if cfg_yaml.exists():
            with open(cfg_yaml) as f:
                bundle_config = yaml.safe_load(f) or {}
        elif cfg_json.exists():
            with open(cfg_json) as f:
                bundle_config = json.load(f)
        else:
            _error_json(
                "CONFIG_LOAD_ERROR",
                "No config.yaml or config.json found in calibration bundle.",
            )

        # Strip the calibration section — not applicable to projections.
        bundle_config.pop("calibration", None)

        if config_path is not None:
            # The overlay only needs to specify what changes (end_date,
            # overrides, projection settings, …).  The calibration bundle's
            # config is used as the implicit base, so ``base:`` in the overlay
            # is not required.  If the overlay does carry a ``base:`` key it is
            # still resolved normally — the bundle config is simply merged on
            # top of whatever that chain produces.
            overlay = load_config(config_path)
            config = _deep_merge(bundle_config, overlay)
        else:
            # No overlay — replay calibration period with posterior samples.
            config = bundle_config
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))

    validation = validate_projection_config(config, calibration_bundle)
    if not validation["valid"]:
        _error_json("INVALID_CONFIG", "Projection config validation failed.",
                    details=validation["errors"])

    for w in validation.get("warnings", []):
        click.echo(f"Warning: {w}", err=True)

    try:
        proj_config_dir = Path(config_path).parent if config_path is not None else None
        results, used_config = project_from_config(
            config, calibration_bundle, config_dir=proj_config_dir
        )
        prov = {
            "command": "project",
            "parent_bundle": str(Path(calibration_bundle).resolve()),
        }
        if config_path is not None:
            prov["config_path"] = str(Path(config_path).resolve())
        manifest = save_bundle(results, output, config=used_config, provenance=prov)
        click.echo(f"Projection bundle saved to: {output}", err=True)
        _print_json(manifest)
    except Exception as e:
        _error_json("RUNTIME_ERROR", str(e))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path):
    """Validate a config file without running it.

    Automatically detects calibration configs (those with a 'calibration'
    section) and validates the calibration-specific fields as well.

    Prints validation result as JSON to stdout.
    """
    from .config import load_config, validate_calibration_config, validate_config

    try:
        config = load_config(config_path)
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))

    if "calibration" in config:
        result = validate_calibration_config(config)
    else:
        result = validate_config(config)
    _print_json(result)

    if not result["valid"]:
        sys.exit(1)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@cli.command(name="inspect")
@click.argument("bundle_path", type=click.Path(exists=True))
@click.argument("command")
@click.option("--variables", "-v", default=None,
              help="Comma-separated list of variables.")
@click.option("--quantiles", "-q", default=None,
              help="Comma-separated quantile levels (e.g. 0.05,0.5,0.95).")
@click.option("--start", default=None,
              help="Start date for time slice (ISO format).")
@click.option("--end", default=None,
              help="End date for time slice (ISO format).")
@click.option("--resample", default=None,
              help="Temporal resampling frequency (e.g. W, M).")
@click.option("--round", "precision", type=int, default=6,
              help="Decimal precision for numeric output (default: 6).")
@click.option("--generation", type=int, default=None,
              help="Calibration generation (for posterior/fit commands).")
@click.option("--format", "output_format", default="json",
              type=click.Choice(["json", "csv", "tsv"]),
              help="Output format.")
def inspect_cmd(bundle_path, command, variables, quantiles, start, end,
                resample, precision, generation, output_format):
    """Inspect an .epx result bundle.

    \b
    Commands:
      manifest    Return the manifest JSON
      quantiles   Compute quantiles across simulations
      summary     Summary statistics (peak, final state)
      peak        Peak timing and magnitude
      posterior   Calibration posterior summary
      fit         Calibration fit vs. observed data

    \b
    Examples:
      epydemix inspect results.epx manifest
      epydemix inspect results.epx quantiles -v I_total -q 0.05,0.5,0.95
      epydemix inspect results.epx summary -v I_total --start 2020-03-01
      epydemix inspect results.epx peak -v I_total
      epydemix inspect calibration.epx posterior
    """
    from ..io.inspect import inspect_bundle

    # Parse comma-separated options
    vars_list = variables.split(",") if variables else None
    quant_list = [float(q) for q in quantiles.split(",")] if quantiles else None

    kwargs = {
        "variables": vars_list,
        "start": start,
        "end": end,
        "resample": resample,
    }
    if quant_list is not None:
        kwargs["quantiles"] = quant_list
    if generation is not None:
        kwargs["generation"] = generation

    try:
        result = inspect_bundle(bundle_path, command, **kwargs)
    except Exception as e:
        _error_json("INSPECT_ERROR", str(e))

    if output_format == "json":
        # manifest returns raw JSON — skip rounding to preserve exact parameter values
        effective_precision = None if command == "manifest" else precision
        _print_json(result, precision=effective_precision)
    elif output_format in ("csv", "tsv"):
        _print_tabular(result, output_format)


def _print_tabular(data, fmt):
    """Best-effort conversion of inspect result to CSV/TSV."""
    import pandas as pd

    sep = "," if fmt == "csv" else "\t"

    if "dates" in data:
        # Time-series result — build a DataFrame
        rows = {"date": data["dates"]}
        for key, val in data.items():
            if key == "dates":
                continue
            if isinstance(val, dict):
                for q, values in val.items():
                    rows[f"{key}_{q}"] = values
            elif isinstance(val, list):
                rows[key] = val
        df = pd.DataFrame(rows)
        click.echo(df.to_csv(index=False, sep=sep))
    else:
        # Key-value result — print as flat table
        click.echo(json.dumps(data, indent=2, cls=NumpySafeEncoder))


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("model_name")
@click.option("--format", "fmt", default="json-schema",
              type=click.Choice(["json-schema", "describe"]),
              help="Output format.")
@click.option("--waning-immunity", "waning_immunity", is_flag=True, default=False,
              help="Include the waning-immunity module (adds waning_rate).")
@click.option("--vaccination", is_flag=True, default=False,
              help="Include the vaccination module "
                   "(adds vaccination_rate, vaccine_efficacy).")
@click.option("--outcome", default=None,
              type=click.Choice(["deaths", "hospitalization"]),
              help="Include an outcome module "
                   "(deaths → mortality_rate; "
                   "hospitalization → hospitalization_rate, hospitalization_recovery_rate).")
def schema(model_name, fmt, waning_immunity, vaccination, outcome):
    """Show the parameter schema for a predefined model.

    \b
    Examples:
      epydemix schema SEIR
      epydemix schema SEIAR
      epydemix schema SEIR --waning-immunity --outcome deaths
      epydemix schema SIR --format describe
    """
    from ..model.predefined_models import SUPPORTED_MODELS, load_predefined_model

    model_name = model_name.upper()
    if model_name not in SUPPORTED_MODELS:
        _error_json("UNKNOWN_MODEL",
                     f"Unknown model '{model_name}'. Available: {SUPPORTED_MODELS}")

    try:
        model = load_predefined_model(
            model_name,
            waning_immunity=waning_immunity,
            vaccination=vaccination,
            outcome=outcome,
        )
    except ValueError as e:
        _error_json("INVALID_MODULE", str(e))

    registry = model.parameter_registry

    if fmt == "json-schema":
        _print_json(registry.to_json_schema(title=f"{model_name} Model Parameters"))
    else:
        click.echo(registry.describe())


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

@cli.command()
def models():
    """List available predefined models."""
    from ..model.predefined_models import SUPPORTED_MODELS
    _print_json({"models": SUPPORTED_MODELS})


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("disease", required=False)
def defaults(disease):
    """List available disease defaults, or show one in detail.

    \b
    Examples:
      epydemix defaults              # list all
      epydemix defaults covid19      # show covid19 details
    """
    from ..parameters.defaults_loader import get_available_defaults, load_defaults

    if disease is None:
        available = get_available_defaults()
        _print_json({"defaults": available})
    else:
        try:
            d = load_defaults(disease)
            _print_json(d.to_dict())
        except FileNotFoundError as e:
            _error_json("UNKNOWN_DEFAULTS", str(e))


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("bundles", nargs=-1, required=True,
                type=click.Path(exists=True))
@click.option("--metrics", "-m", default=None,
              help="Comma-separated metrics (e.g. attack_rate,peak,total_deaths,days_over:500).")
@click.option("--variables", "-v", default=None,
              help="Comma-separated variable names for variable-specific metrics.")
@click.option("--names", "-n", default=None,
              help="Comma-separated scenario names (same order as bundles). "
                   "Defaults to bundle directory names.")
@click.option("--round", "precision", type=int, default=6,
              help="Decimal precision for numeric output (default: 6).")
@click.option("--baseline", "-b", default=None,
              help="Scenario name to use as baseline for delta computation.")
def compare(bundles, metrics, variables, names, precision, baseline):
    """Compare multiple .epx bundles on standard metrics.

    \b
    Built-in metrics:
      attack_rate    % of population ever infected
      peak           Peak value of a variable (default: first I-like _total)
      peak_date      Date of peak value
      total_deaths   Final value of death compartment
      days_over:N    Days the median of a variable exceeds threshold N
      final_value    Final value of a variable

    \b
    Examples:
      epydemix compare baseline.epx early.epx late.epx
      epydemix compare *.epx -m attack_rate,peak,total_deaths,days_over:500
      epydemix compare baseline.epx early.epx -n Baseline,Early -b Baseline
    """
    from ..io.inspect import compare_bundles

    # Parse names
    if names:
        name_list = names.split(",")
        if len(name_list) != len(bundles):
            _error_json("COMPARE_ERROR",
                        f"Got {len(name_list)} names but {len(bundles)} bundles.")
    else:
        # Default: use directory stem
        name_list = [Path(b).stem for b in bundles]

    bundle_map = dict(zip(name_list, bundles))

    # Parse options
    metric_list = metrics.split(",") if metrics else None
    var_list = variables.split(",") if variables else None

    try:
        result = compare_bundles(
            bundle_map,
            metrics=metric_list,
            variables=var_list,
        )
    except Exception as e:
        _error_json("COMPARE_ERROR", str(e))

    # Compute deltas if baseline specified
    if baseline:
        if baseline not in result:
            _error_json("COMPARE_ERROR",
                        f"Baseline '{baseline}' not found. "
                        f"Available: {list(result.keys())}")
        base_data = result[baseline]
        deltas = {}
        for scenario, metrics_data in result.items():
            if scenario == baseline:
                continue
            scenario_delta = {}
            for metric_name, metric_val in metrics_data.items():
                base_val = base_data.get(metric_name, {})
                # Compute delta on median or value fields (numeric only).
                # Rounding is handled by _print_json at the output layer.
                if "median" in metric_val and "median" in base_val:
                    bm = base_val["median"]
                    sm = metric_val["median"]
                    if (bm is not None and sm is not None
                            and isinstance(bm, (int, float))
                            and isinstance(sm, (int, float))):
                        scenario_delta[metric_name] = sm - bm
                elif "value" in metric_val and "value" in base_val:
                    bv = base_val["value"]
                    sv = metric_val["value"]
                    if (bv is not None and sv is not None
                            and isinstance(bv, (int, float))
                            and isinstance(sv, (int, float))):
                        scenario_delta[metric_name] = sv - bv
            deltas[scenario] = scenario_delta
        result["_deltas_vs_" + baseline] = deltas

    _print_json(result, precision=precision)


# ---------------------------------------------------------------------------
# populations
# ---------------------------------------------------------------------------

@cli.command()
def populations():
    """List available population datasets."""
    try:
        from ..population.population import get_available_locations
        locations = get_available_locations()["location"].tolist()
        _print_json({"populations": locations})
    except Exception as e:
        _error_json("POPULATION_ERROR", str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
