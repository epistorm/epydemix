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
        results, used_config = run_from_config(config)
        manifest = save_bundle(results, output, config=used_config)
        click.echo(f"Bundle saved to: {output}", err=True)
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

    Prints validation result as JSON to stdout.
    """
    from .config import load_config, validate_config

    try:
        config = load_config(config_path)
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))

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
@click.option("--round", "precision", type=int, default=2,
              help="Decimal precision for numeric output.")
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
        "precision": precision,
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
        _print_json(result, precision=precision)
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
def schema(model_name, fmt):
    """Show the parameter schema for a predefined model.

    \b
    Examples:
      epydemix schema SEIR
      epydemix schema SIR --format describe
    """
    from ..model.predefined_models import SUPPORTED_MODELS, load_predefined_model

    model_name = model_name.upper()
    if model_name not in SUPPORTED_MODELS:
        _error_json("UNKNOWN_MODEL",
                     f"Unknown model '{model_name}'. Available: {SUPPORTED_MODELS}")

    model = load_predefined_model(model_name)
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
