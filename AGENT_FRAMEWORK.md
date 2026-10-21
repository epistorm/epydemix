# Epydemix Agent Framework

The agent framework is a layer on top of the core epydemix library: a CLI, parameter registry, and structured output format that let an LLM agent (or any automation pipeline) discover, configure, run, calibrate, and inspect epidemic models without reading source code or holding Python objects in memory. For the core `EpiModel` Python API, see the [main README](README.md) and the [documentation](https://epydemix.readthedocs.io/en/latest/).

Ask an LLM agent to "run an SEIR model of COVID-19 in Italy and compare three intervention timings," and a typical modeling library forces it to read source code to find available models, guess at function signatures, and dig results out of in-memory DataFrames. The agent framework closes that gap with five properties: **discoverability** (query models/parameters via the CLI instead of reading code), **declarative input** (a single validated YAML config per run, not a chain of API calls), **statelessness** (every command is a self-contained function of files on disk — nothing needs to survive between tool calls), **structured output** (results land as self-describing Parquet + JSON bundles), and **inspectability** (targeted questions like "when did infections peak?" return compact JSON, not millions of rows).

## Install

```bash
pip install "epydemix[agent]"
```

This installs the `epydemix` console script and its CLI dependencies (`click`, `pyyaml`, `pyarrow`).

## Quick start

```bash
# Discover available models and their parameters
epydemix models
epydemix schema SEIR

# Browse literature-sourced disease presets
epydemix defaults covid19

# Validate a config, then run it
epydemix validate config.yaml
epydemix run config.yaml --output results.epx

# Ask questions about the results
epydemix inspect results.epx summary -v I_total
epydemix inspect results.epx quantiles -v I_total -q 0.05,0.5,0.95
epydemix inspect results.epx peak -v I_total

# Calibrate against observed data, then project forward
epydemix calibrate cal_config.yaml -o calibration.epx
epydemix project calibration.epx -c projection.yaml -o projection.epx
```

Every command prints structured JSON to stdout and diagnostics to stderr, so results are easy to parse programmatically.

## Learn more

- **[AGENT.md](AGENT.md)** — the full reference: every CLI command, the complete YAML config format, and the output bundle structure. This is the contract an LLM agent reads before driving epydemix.
- **[AGENT_EXAMPLES.md](AGENT_EXAMPLES.md)** — visualization recipes and end-to-end workflow examples.
- **[Tutorial 13](tutorials/13_Command_Line_Interface.ipynb)** — a notebook walkthrough of the CLI: discovery, validation, scenarios, calibration, and projections.

## Citation

The agent framework is described in a [preprint on arXiv](https://arxiv.org/abs/2609.28692). If you use it in your work, please cite:

```
@misc{gozzi2026agent,
  title={Driving Epidemic Models with AI Agents: the Epydemix Agent Framework},
  author={Gozzi, Nicol{\`o} and Cattuto, Ciro and Vespignani, Alessandro},
  year={2026},
  eprint={2609.28692},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2609.28692}
}
```

For the core epydemix library, please also cite the [Epydemix paper](https://doi.org/10.1371/journal.pcbi.1013735) (see the [main README](README.md#citation)).
