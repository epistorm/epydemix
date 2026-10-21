# Command line and AI agents

epydemix also comes with an `epydemix` command-line tool. It covers the whole workflow
of the Python API, from choosing a model to calibration and projections, but every step
is a command that reads and writes files:

- a simulation is described by one **YAML config**, checked by a validator before
  anything runs;
- results are saved as a **bundle**, a folder of Parquet files with a JSON manifest that
  describes them;
- every command prints **JSON**, and targeted queries ("when is the peak?") return a few
  lines rather than every trajectory.

This is the **agent framework** of epydemix: it lets an LLM agent, or any automated
pipeline, run epidemic models without reading source code or keeping Python objects
in memory. It is just as usable by hand, from a terminal.

## Installation

The CLI needs a few extra dependencies (`click`, `pyyaml`, `pyarrow`):

```bash
pip install "epydemix[agent]"
epydemix --help
```

## Discovering models and parameters

```bash
epydemix models                          # predefined models: SIR, SEIR, SIS, SEIAR
epydemix schema SEIR --format describe   # parameters, meaning, units, valid range
epydemix schema SEIR --vaccination       # the same with a module added
epydemix defaults covid19                # literature-based parameter values
epydemix populations                     # the 400+ available populations
```

`epydemix schema` without `--format describe` prints a JSON Schema, convenient for
programs.

:::{tip}
Disease presets assume homogeneous mixing. On a real population, the reproduction number
is $R_0 = \beta \rho(C) / \gamma$, where $\rho(C)$ is the spectral radius of the overall
contact matrix, so a preset transmission rate usually needs rescaling:

```python
import numpy as np
from epydemix.population import load_epydemix_population

italy = load_epydemix_population("Italy")
C = sum(italy.contact_matrices.values())
rho = np.max(np.abs(np.linalg.eigvals(C)))   # about 13.9 for Italy
transmission_rate = 2.5 * 0.14 / rho         # R0 = 2.5 with recovery_rate = 0.14
```
:::

## Configs

A config is a YAML file. This one runs an SEIR model on Italy's population:

```yaml
model:
  type: SEIR

parameters:
  transmission_rate: 0.025
  incubation_rate: 0.33
  recovery_rate: 0.14

population:
  name: Italy
  contact_layers: [home, work, school, community]

simulation:
  start_date: "2024-10-01"
  end_date: "2025-03-31"
  n_simulations: 50
  seed: 42

initial_conditions:
  Susceptible: 0.99999
  Exposed: 0.000005
  Infected: 0.000005
  Recovered: 0.0

overrides:                     # optional: change a parameter for a period
  - parameter: transmission_rate
    start_date: "2024-12-15"
    end_date: "2025-01-26"
    value: 0.015
```

Other sections add contact `interventions`, custom models (`type: custom` with your own
compartments and transitions), and the modules of the {doc}`predefined models <models>`.
[`AGENT.md`](https://github.com/epistorm/epydemix/blob/main/AGENT.md) documents every
field.

`epydemix validate` checks a config without running it: parameter names and ranges,
compartment names, dates, and the structure of custom models. Each error says how to fix
it:

```bash
epydemix validate config.yaml
```

```json
{
  "valid": false,
  "errors": [
    "parameters: 'transmision_rate' is not a parameter of this model (did you mean 'transmission_rate'?). Parameters: transmission_rate, incubation_rate, recovery_rate",
    "parameters.recovery_rate has value 14, above the maximum of 10 (1/days) for 'recovery_rate'"
  ],
  "warnings": []
}
```

`epydemix run` performs the same checks and refuses to run an invalid config.

### Scenarios

A config can inherit from another with `base:` and only state what changes. Dictionaries
are merged key by key, anything else (such as a list) replaces the inherited value. This
is the convenient way to set up scenarios:

```yaml
# early.yaml: the baseline with an earlier reduction in transmission
base: baseline.yaml
overrides:
  - parameter: transmission_rate
    start_date: "2024-11-15"
    end_date: "2024-12-27"
    value: 0.015
```

## Running and inspecting

```bash
epydemix run baseline.yaml -o baseline.epx
```

The bundle `baseline.epx` contains:

- `compartments.parquet` and `transitions.parquet`: one row per simulation and date,
  with a column per compartment (or transition) and age group, plus `_total` columns;
- `manifest.json`: what the bundle contains, the parameters used, the column schema of
  every file, and its provenance (command and config that produced it);
- `config.yaml`: the config that produced it, seed included;
- `figures/`: a folder for plots of these results.

`epydemix inspect` answers questions about a bundle:

```bash
epydemix inspect baseline.epx peak -v Infected_total
epydemix inspect baseline.epx summary -v Infected_total
epydemix inspect baseline.epx quantiles -v Infected_total -q 0.05,0.5,0.95 --resample W
epydemix inspect baseline.epx quantiles -v Susceptible_to_Infected_total --format csv
```

Variables can be compartments (e.g. `Infected_total`) or transitions (e.g.
`Susceptible_to_Infected_total`, daily new infections). `--start` and `--end` select a
time window, and when resampling, a compartment keeps its last value in each period while
a transition is summed. `epydemix compare` computes the same metrics on several bundles,
with the differences from a baseline:

```bash
epydemix compare baseline.epx early.epx late.epx -m attack_rate,peak,peak_date -b baseline
```

## Calibration and projections

A calibration config is a simulation config with a `calibration` section. Parameters with
a prior are estimated, the others are fixed:

```yaml
calibration:
  strategy: smc                      # smc, rejection or top_fraction
  priors:
    transmission_rate: {distribution: uniform, low: 0.010, high: 0.030}
    recovery_rate: {distribution: uniform, low: 0.15, high: 0.25}
  observed_data: incidence_data.csv
  observed_column: data
  target_variable: Susceptible_to_Infected_total
  distance: rmse
  num_particles: 100
  num_generations: 4
```

```bash
epydemix calibrate calibration.yaml -o calibration.epx
epydemix inspect calibration.epx posterior
epydemix inspect calibration.epx fit -v Susceptible_to_Infected_total
```

A projection draws parameters from the posterior and runs new simulations. Without a
config it replays the calibration period; with `-c` it applies an overlay to the
calibration's config, for example a longer period or an intervention:

```bash
epydemix project calibration.epx -o replay.epx
epydemix project calibration.epx -c what_if.yaml -o what_if.epx
```

Projections are ordinary simulation bundles, so `inspect` and `compare` work on them.
See {doc}`calibration` for the methods behind these commands.

## Reproducibility

Every `run`, `calibrate` and `project` is seeded. If the config sets
`simulation.seed`, it is used; otherwise a seed is drawn. Either way it is recorded in the
manifest and in the bundle's `config.yaml`, so re-running that config reproduces the
bundle exactly:

```bash
epydemix run baseline.epx/config.yaml -o replay.epx    # identical trajectories
```

## Using bundles from Python

Bundles are plain Parquet files, readable by any tool. {doc}`epydemix.io <../api/io>` has
helpers to read them and to store figures in them:

```python
import matplotlib.pyplot as plt
from epydemix.io import add_figure_to_manifest, load_bundle_dataframe

df = load_bundle_dataframe("baseline.epx", "compartments",
                           columns=["sim_id", "date", "Infected_total"])
runs = df.pivot(index="date", columns="sim_id", values="Infected_total")
ax = runs.median(axis=1).plot()

plt.savefig("baseline.epx/figures/infected.png")
add_figure_to_manifest("baseline.epx", "infected.png", "Median infected", ["Infected_total"])
```

## Using an agent

Any LLM agent that can run shell commands (Claude Code, Codex, Gemini CLI, ...) can drive
epydemix. Install `epydemix[agent]` where the agent runs, and point it to
[`AGENT.md`](https://github.com/epistorm/epydemix/blob/main/AGENT.md): the reference
written for agents, with every command, the full config format, the bundle layout, and
pitfalls to avoid.
[`AGENT_EXAMPLES.md`](https://github.com/epistorm/epydemix/blob/main/AGENT_EXAMPLES.md)
adds plotting recipes and worked examples. For instance:

> *Read AGENT.md, then use epydemix to simulate a COVID-19-like epidemic in Italy with
> R0 = 2.5, compare a six-week reduction in transmission starting in mid-November,
> mid-December or mid-January, and save a figure of the three scenarios.*

The configs, seeds, and provenance the agent leaves in its bundles make each result
possible to check and reproduce afterwards.

The {doc}`command-line tutorial <../tutorials/13_Command_Line_Interface>` goes through a
complete example.
