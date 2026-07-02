# Epydemix, the ABC of Epidemics
[![GitHub stars](https://img.shields.io/github/stars/epistorm/epydemix.svg?style=social)](https://github.com/epistorm/epydemix/stargazers)
[![PyPI Downloads](https://static.pepy.tech/badge/epydemix)](https://pepy.tech/projects/epydemix)
[![Read the Docs](https://readthedocs.org/projects/epydemix/badge/?version=latest)](https://epydemix.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Codecov](https://codecov.io/gh/epistorm/epydemix/branch/main/graph/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/epydemix.svg)](https://pypi.org/project/epydemix/)
[![PLOS Computational Biology](https://img.shields.io/badge/Published%20in-PLOS%20Computational%20Biology-blue?logo=plos)](https://doi.org/10.1371/journal.pcbi.1013735)

![Alt text](https://raw.githubusercontent.com/epistorm/epydemix/main/tutorials/img/epydemix-logo.png)

**[Documentation](https://epydemix.readthedocs.io/en/latest/)** | **[Website](https://www.epydemix.org/)** | **[Tutorials](https://github.com/epistorm/epydemix/tree/main/tutorials)**

**Epydemix** is a Python package for epidemic modeling. This branch adds an **agent framework** on top of it: a CLI, parameter registry, and structured output format that let an LLM agent (or any automation pipeline) discover, configure, run, calibrate, and inspect epidemic models without reading source code or holding Python objects in memory. The core epydemix library and its documentation continue to live on the [`main` branch](https://github.com/epistorm/epydemix/tree/main); this README covers the agent framework specifically.

---

## Epydemix Agent Framework

Ask an LLM agent to "run an SEIR model of COVID-19 in Italy and compare three intervention timings," and a typical modeling library forces it to read source code to find available models, guess at function signatures, and dig results out of in-memory DataFrames. The agent framework closes that gap with five properties: **discoverability** (query models/parameters via the CLI instead of reading code), **declarative input** (a single validated YAML config per run, not a chain of API calls), **statelessness** (every command is a self-contained function of files on disk — nothing needs to survive between tool calls), **structured output** (results land as self-describing Parquet + JSON bundles), and **inspectability** (targeted questions like "when did infections peak?" return compact JSON, not millions of rows). See [AGENT_FRIENDLY_DESIGN.md](AGENT_FRIENDLY_DESIGN.md) for the full design rationale.

### Install

This layer isn't published to PyPI yet — install from source:

```bash
git clone https://github.com/epistorm/epydemix.git
cd epydemix
git checkout agent-framework
pip install -e ".[agent]"
```

This installs the `epydemix` console script and its CLI dependencies (`click`, `pyyaml`, `pyarrow`).

### Quick start

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

### Learn more

- **[AGENT.md](AGENT.md)** — the full reference: every CLI command, the complete YAML config format, and the output bundle structure. This is the contract an LLM agent reads before driving epydemix.
- **[AGENT_EXAMPLES.md](AGENT_EXAMPLES.md)** — visualization recipes and end-to-end workflow examples.
- **[AGENT_FRIENDLY_DESIGN.md](AGENT_FRIENDLY_DESIGN.md)** — the design essay behind this layer: why CLI over MCP, why bundles instead of in-memory objects, and the other key decisions.

---

## Epydemix Core Library

Epydemix is also usable directly as a Python library — in notebooks, scripts, or any Python codebase — via its `EpiModel` API. It provides tools to create, calibrate, and analyze epidemic models using different compartmental models, contact layers, and calibration techniques, and pairs with the [epydemix-data](https://github.com/epistorm/epydemix-data/) package for population and contact matrix data. For the complete documentation, tutorials, and installation guide for the core library, see the [`main` branch](https://github.com/epistorm/epydemix/tree/main) or [Read the Docs](https://epydemix.readthedocs.io/en/latest/).

```bash
pip install epydemix
```

12 tutorial notebooks cover everything from a first SIR model to age-structured populations, interventions, ABC calibration, multi-strain models, and the predefined model backbones — see the [tutorials folder](./tutorials) (runnable directly in Google Colab) and the [full documentation](https://epydemix.readthedocs.io/en/latest/) on Read the Docs.

---
## Citation
The paper describing the development of Epydemix is available [here](https://doi.org/10.1371/journal.pcbi.1013735).
To reference our work, please use the following citation:
```
@article{gozzi2025epydemix,
  title={Epydemix: An open-source Python package for epidemic modeling with integrated approximate Bayesian calibration},
  author={Gozzi, Nicol{\'o} and Chinazzi, Matteo and Davis, Jessica T and Gioannini, Corrado and Rossi, Luca and Ajelli, Marco and Perra, Nicola and Vespignani, Alessandro},
  journal={PLOS Computational Biology},
  volume={21},
  number={11},
  pages={e1013735},
  year={2025},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

---
## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for more details.

---
## Contributors

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) and open issues or pull requests on GitHub. For questions and general discussion, visit our [GitHub Discussions](https://github.com/epistorm/epydemix/discussions).

<a href="https://github.com/epistorm/epydemix/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=epistorm/epydemix" />
</a>

---
## Changelog

See the [CHANGELOG](./CHANGELOG.md) file for details on past releases and updates.

---
## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer at `epydemix@isi.it`.
