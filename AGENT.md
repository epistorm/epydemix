# Driving Epydemix Programmatically

Epydemix is a Python library for compartmental epidemic modeling (SIR, SEIR, SIS, and custom models) with age-structured populations, contact matrices, and ABC calibration. This document is the contract for LLM agents (and any automation pipeline) that drive epydemix through its CLI.

All CLI commands print structured JSON to **stdout** and diagnostics to **stderr**. Parse stdout only.

## How to work with epydemix

Read this file before doing anything. Then follow these principles:

- **Use the CLI for discovery and inspection.** `epydemix models`, `epydemix schema`, `epydemix defaults`, `epydemix inspect` — these are your primary tools. Do not read source code to figure out what's available.
- **Use Python for analysis beyond what the CLI offers.** The CLI covers common queries (quantiles, summary, peak). For anything else (attack rates, scenario comparisons, custom metrics), write Python against the Parquet files. The manifest gives you full column schemas.
- **Store figures inside the bundle.** Save plots to `<bundle>/figures/` and register them with `add_figure_to_manifest`. See the Visualization Recipes section.
- **Work step by step.** Discover → build config → validate → run → inspect → analyze → visualize.
- **Never use `cd`.** Always use absolute paths. `cd` persists across shell invocations and silently corrupts all subsequent relative paths.

## Quick Start

```bash
# 1. Discover available models
epydemix models
# → {"models": ["SIR", "SEIR", "SIS"]}

# 1b. List named population datasets (461 countries/regions)
epydemix populations
# → {"populations": ["Afghanistan", "Albania", ..., "Zimbabwe"]}
# For a flat single-group population of arbitrary size, use `population.size`
# in the config instead (see Config Reference). Default size is 100,000.

# 2. Inspect a model's parameter space
epydemix schema SEIR
# → JSON Schema with all parameters, bounds, defaults, descriptions

# 3. Browse disease presets
epydemix defaults
# → {"defaults": ["covid19", "influenza", "measles"]}
epydemix defaults covid19
# → full parameter set with literature values and ranges

# 4. Write a YAML config (see Config Reference below)

# 5. Validate before running
epydemix validate config.yaml
# → {"valid": true, "errors": [], "warnings": [...]}

# 6. Run the simulation
epydemix run config.yaml --output results.epx
# → manifest JSON to stdout; results.epx/ directory on disk

# 7. Read what you got
epydemix inspect results.epx manifest

# 8. Ask questions about the results
epydemix inspect results.epx summary -v I_total
epydemix inspect results.epx quantiles -v I_total -q 0.05,0.5,0.95
epydemix inspect results.epx peak -v I_total

# 9. Zoom into a time window
epydemix inspect results.epx quantiles -v I_total --start 2020-03-01 --end 2020-05-01

# 10. For anything the canned queries can't do, write Python against the Parquet files
#     (the manifest gives you full column schemas — see Output Reference)

# 11. Project forward from a calibration posterior
epydemix calibrate cal_config.yaml -o calibration.epx
# Overlay config is passed with -c/--config, NOT as a positional argument
epydemix project calibration.epx -c projection.yaml -o projection.epx
# Omit -c to replay the calibration period with posterior samples
epydemix project calibration.epx -o projection.epx
# NOTE: do NOT run `epydemix validate` on a projection overlay — it is a partial
# diff, not a standalone config. Validation happens inside `epydemix project`.
```

**Working directory:** All CLI commands resolve paths relative to the current working directory. Use absolute paths for all bundle inputs and outputs.

## Available Models

**Predefined:** SIR, SEIR, SIS — use `epydemix schema <MODEL>` to see their parameters.

**Custom:** define any compartmental model by specifying compartments and transitions in the config. See "Building Custom Models" below.

## Parameter Discovery

```bash
# JSON Schema with bounds, defaults, descriptions, units
epydemix schema SEIR

# Human-readable parameter descriptions
epydemix schema SIR --format describe

# Disease presets with literature-sourced parameter values
epydemix defaults              # list available
epydemix defaults covid19      # full detail for one disease
```

The JSON Schema output can be used to validate parameter values before running. Each parameter includes `minimum`, `maximum`, `default`, and `description` fields.

## Config Reference

The config is a YAML or JSON file — the single artifact you construct to run a simulation.

```yaml
# ── Model ────────────────────────────────────────────────────────
model:
  type: "SEIR"                      # "SIR", "SEIR", "SIS", or "custom"
  # The fields below are only needed for type: custom
  compartments: ["S", "E", "I", "R"]
  transitions:
    - source: S
      target: E
      kind: mediated                # see Building Custom Models
      params: ["transmission_rate", "I"]
    - source: E
      target: I
      kind: spontaneous
      params: "incubation_rate"
    - source: I
      target: R
      kind: spontaneous
      params: "recovery_rate"

# ── Parameters ───────────────────────────────────────────────────
parameters:
  transmission_rate: 0.3
  incubation_rate: 0.2
  recovery_rate: 0.1

# ── Population ───────────────────────────────────────────────────
population:
  name: "Italy"                     # an epydemix population dataset
  contact_layers: ["home", "work", "school", "community"]
  # If omitted entirely, a single-group population of 100,000 is used.
  # Use `size` to change the flat-population size (no named dataset needed):
  size: 500000                       # optional; default 100,000

# ── Simulation ───────────────────────────────────────────────────
simulation:
  start_date: "2020-01-01"         # ISO date, required
  end_date: "2020-06-30"           # ISO date, required
  n_simulations: 100               # default: 100
  dt: 1.0                          # timestep in days, default: 1.0

# ── Initial conditions ───────────────────────────────────────────
# Fractions of the population in each compartment. Must sum to 1.0.
initial_conditions:
  S: 0.999
  E: 0.0005
  I: 0.0005
  R: 0.0

# ── Interventions (optional) ─────────────────────────────────────
# Reduce contact on a specific layer for a time window.
interventions:
  - layer: "school"
    start_date: "2020-03-10"
    end_date: "2020-06-01"
    reduction: 0.8                  # 80% reduction in school contacts

# ── Parameter overrides (optional) ───────────────────────────────
# Change a parameter value for a time window (e.g., lockdown effect).
overrides:
  - parameter: transmission_rate
    start_date: "2020-03-15"
    end_date: "2020-05-15"
    value: 0.15
```

**Required sections:** `model`, `simulation` (with `start_date` and `end_date`).

**Optional sections:** `parameters` (defaults used if omitted), `population`, `initial_conditions`, `interventions`, `overrides`, `base`.

Custom models can also use `kind: scheduled` transitions (see Building Custom Models) to drive compartment flows from a daily dose CSV — useful for vaccination campaigns.

### Config inheritance

A config can inherit from another using the `base` field. The overlay is deep-merged on top of the base: dicts merge recursively, everything else (lists, scalars) replaces. This means a scenario overlay only needs to specify what changes.

```yaml
# seirhd_early.yaml — inherits model, parameters, ICs from baseline
base: seirhd_baseline.yaml
overrides:
  - parameter: beta
    start_date: "2024-10-01"
    end_date: "2025-02-01"
    value: 0.27
```

Chains are supported (child → parent → grandparent, up to 10 levels). The `base` path is resolved relative to the file that contains it. Circular references are detected and rejected.

This is the recommended way to set up scenario sweeps: write one complete base config, then one small overlay per scenario. Validate and run each overlay normally — the inheritance is resolved at load time and is transparent to the rest of the pipeline.

## Output Reference

### Bundle structure

Running `epydemix run` produces a `.epx` directory:

```
results.epx/
  manifest.json         # metadata + full Parquet schemas (see below)
  compartments.parquet  # (sim_id, date, S_total, I_total, ..., S_0-4, S_5-19, ...)
  transitions.parquet   # (sim_id, date, S_to_I_total, ..., S_to_I_0-4, ...)
  parameters.parquet    # (sim_id) — only populated for calibration runs; empty for plain run
```

Column naming convention: `{Compartment}_{group}` for age-group-specific columns, `{Compartment}_total` for the sum across all groups. Transition columns follow the same pattern: `{Source}_to_{Target}_{group}` and `{Source}_to_{Target}_total`.

**Note:** `parameters.parquet` contains only `sim_id` for plain `run` and `project` commands (parameters are fixed, not varied). It is populated with sampled parameter values only for `calibrate` runs.

### Manifest

The `manifest.json` is the bridge between the opaque Parquet files and the agent. It contains:

- `model`: compartments (alphabetically sorted list)
- `population`: `demographic_groups` — ordered list of group names matching Parquet column order
- `simulation`: n_simulations, start/end dates, n_timesteps
- `parameters_used`: scalar parameter values (exact, unrounded — `epydemix inspect manifest` skips rounding entirely so these are preserved at full precision)
- `files`: for each Parquet file, the full column schema with dtypes and descriptions
- `usage_hints`: instructions for CLI inspection and custom Python access
- `provenance`: lineage information recording how the bundle was produced

**The `files` section is your schema reference.** Use it to write correct `pd.read_parquet()` calls when the canned inspect commands are insufficient.

Two things to get right when reading `files`:

- **Keys are file stems, not full filenames.** Use `m['files']['compartments']`, not `m['files']['compartments.parquet']`.
- **`columns` is a dict keyed by column name, not a list.** Iterate with `.items()` or index directly by name.

```python
import json, subprocess
manifest = json.loads(subprocess.check_output(
    ["epydemix", "inspect", "results.epx", "manifest"]))

# List all column names in compartments file
cols = list(manifest["files"]["compartments"]["columns"].keys())

# Get dtype for a specific column
dtype = manifest["files"]["compartments"]["columns"]["Infected_total"]["dtype"]
```

**Getting population size:** the manifest does not include a `size` or `name` field. Derive total population size from the Parquet data: `comp_df.groupby("date")[["S_total","I_total",...]].sum(axis=1).iloc[0]`, or sum across all `_total` compartment columns on any single row.

### Provenance

Every bundle produced by the CLI includes a `provenance` key in the manifest that records the operation that created it:

```json
{
  "provenance": {
    "command": "project",
    "parent_bundle": "/abs/path/to/calibration.epx",
    "config_path": "/abs/path/to/projection.yaml"
  }
}
```

Fields vary by command:

| Field | Present in | Description |
|-------|-----------|-------------|
| `command` | all | The CLI command that created this bundle (`run`, `calibrate`, `project`) |
| `config_path` | all | Absolute path to the config file used |
| `parent_bundle` | `project` only | Absolute path to the calibration bundle the projection sampled from |

Use provenance to trace the dependency chain: a projection bundle points back to its calibration bundle, and both point to their config files. This makes it possible to reconstruct the full lineage of any result.

### Reading Parquet with custom Python

```python
import pandas as pd

# Read only the columns you need
df = pd.read_parquet("results.epx/compartments.parquet",
                     columns=["sim_id", "date", "I_total"])

# Partial read: first 10 simulations only
df = pd.read_parquet("results.epx/compartments.parquet",
                     filters=[("sim_id", "<", 10)])
```

### Population Python API

When writing analysis scripts that load population data directly (outside the CLI), use:

```python
from epydemix.population import load_epydemix_population

pop = load_epydemix_population("United_States_Utah")

pop.Nk_names   # np.ndarray of group name strings, e.g. ['0-4', '5-19', '20-49', '50-64', '65+']
               # This is the authoritative group order — matches Parquet column order
pop.Nk         # np.ndarray of group population sizes (same order as Nk_names)
pop.num_groups # int — number of demographic groups
pop.total_population  # int — sum of Nk
```

**Do not use `population.age_groups`, `population.demographic_groups`, or `population.size`** — these attributes do not exist. The manifest's `population.demographic_groups` list gives the same names and order as `Nk_names`.

## Inspecting Results

**Always read the manifest before inspecting anything else.** Bundle layouts differ by type: simulation bundles have `compartments.parquet` and `transitions.parquet`; calibration bundles have `posterior.parquet`, `trajectories.parquet`, `weights.parquet`, and `observed_data.parquet` — no `compartments.parquet`. The manifest tells you exactly which files exist, their column schemas, and what variables are available. Reading a file that isn't listed in the manifest will raise a `FileNotFoundError`.

```bash
# Do this first, every time, before any other inspect command or custom Python
epydemix inspect <bundle> manifest
```

All inspect commands follow the pattern:

```bash
epydemix inspect <bundle> <command> [options]
```

### Common flags

| Flag | Description | Example |
|---|---|---|
| `-v, --variables` | Comma-separated variable names | `-v I_total,R_total` |
| `--start` | Start date for time slice (inclusive, ISO) | `--start 2020-03-01` |
| `--end` | End date for time slice (inclusive, ISO) | `--end 2020-05-31` |
| `--resample` | Temporal resampling before output | `--resample W` (weekly), `--resample M` (monthly) |
| `-q, --quantiles` | Comma-separated quantile levels | `-q 0.05,0.5,0.95` |
| `--round` | Decimal precision (default: 6). Use `--round 2` for compact output or `--round 10` to preserve very small values. | `--round 4` |
| `--format` | Output format: `json` (default), `csv`, `tsv` | `--format csv` |

`--start` and `--end` are applied **before** resampling. When omitted, the full simulation range is used. Variables default to all `_total` columns.

### Commands

**manifest** — return the full manifest JSON.
```bash
epydemix inspect results.epx manifest
```

**quantiles** — quantiles of variable time-series across simulations. This is the most common query.
```bash
epydemix inspect results.epx quantiles -v I_total -q 0.05,0.5,0.95
# → {"dates": ["2020-01-01", ...], "I_total": {"0.05": [...], "0.5": [...], "0.95": [...]}}
```

**summary** — summary statistics: peak date/value, final value, mean across time.
```bash
epydemix inspect results.epx summary -v I_total
# → {"I_total": {"peak_date_median": "2020-04-15", "peak_value": {"0.05": ..., "0.50": ..., "0.95": ...}, ...}}
```

**peak** — peak timing and magnitude as quantiles.
```bash
epydemix inspect results.epx peak -v I_total
# → {"I_total": {"peak_date": {"0.05": "...", "0.50": "...", "0.95": "..."}, "peak_value": {...}}}
```

**posterior** — summarize calibration posteriors (calibration bundles only).
```bash
epydemix inspect calibration.epx posterior
# → {"transmission_rate": {"mean": 0.28, "std": 0.03, "ci95": [0.23, 0.33], "median": 0.27}, ...}
```

**fit** — calibration fit trajectories: simulated quantiles vs. observed data.
```bash
epydemix inspect calibration.epx fit -v I_total -q 0.05,0.5,0.95
# → {"I_total": {"0.05": [...], "0.5": [...], "0.95": [...]}, "observed": {...}}
```

## Building Custom Models

Custom models let you define any compartmental structure. The key is mapping a disease description to compartments and transitions.

### Compartments

Compartments are disease states. Any string name is valid. Every individual is in exactly one compartment at all times.

```yaml
model:
  type: custom
  compartments: ["S", "E", "I_mild", "I_severe", "H", "R", "D"]
```

Naming conventions: short, uppercase names. Use suffixes like `_mild`, `_severe`, `_asymptomatic` for sub-states.

### Transition kinds

There are exactly two kinds:

**`spontaneous`** — individuals leave the source compartment at a fixed rate, independent of other compartments. Use for: recovery, disease progression, death, waning immunity.

```yaml
# "Infected individuals recover at rate gamma"
- source: I
  target: R
  kind: spontaneous
  params: "recovery_rate"
```

The `params` field can be a parameter expression:
```yaml
# "Severe cases die at rate (mortality_rate * severity_factor)"
- source: I_severe
  target: D
  kind: spontaneous
  params: "mortality_rate * severity_factor"
```

**`mediated`** — the rate depends on contact with individuals in another compartment. Use for: infection/transmission. The force of infection is `rate × Σ(ContactMatrix × agents / population)`.

```yaml
# "Susceptibles get infected by contact with Infected at rate beta"
- source: S
  target: E
  kind: mediated
  params: ["transmission_rate", "I"]   # [rate_parameter, agent_compartment]
```

**`scheduled`** — the rate is driven by a pre-specified daily dose schedule. Use for: vaccination campaigns, prophylaxis roll-outs, or any intervention that moves individuals at a time-varying absolute rate rather than a fixed per-capita rate.

```yaml
# "Susceptibles are vaccinated according to a daily dose schedule"
- source: S
  target: SV
  kind: scheduled
  schedule: doses.csv          # path to CSV (relative to config file), or inline list
  eligible: ["S", "R"]         # optional — compartments that receive doses
                               # (only source compartment individuals are effectively moved)
```

`schedule` accepts:
- A **CSV file path** (resolved relative to the config file). The first column must be a date index; the remaining columns are daily doses per demographic group. **If the column names match the population's group names (e.g. `0-4`, `5-19`, `20-49`, `50-64`, `65+`), they are automatically reordered to match the model's group ordering** — so column order in the file does not matter as long as names match. If column names do not match group names, columns are taken positionally in the order they appear in the file. Missing dates are filled with zero.
- An **inline flat list** `[d0, d1, d2, ...]` broadcast to all groups.
- An **inline list of lists** `[[d0g0, d0g1], [d1g0, d1g1], ...]` for per-group values.

`eligible` (optional): compartments that receive doses but where only doses landing on the source compartment are effective. For example, `eligible: ["S", "R"]` models a campaign that jabs both susceptibles and recovered individuals — only doses administered to S produce vaccinations, the rest are wasted. If omitted, effective doses equal total doses.

The transition rate at each timestep is: `rate = min(doses / eligible_pop, 0.999)` (with eligible correction) or `min(doses / source_pop, 0.999)` (without).

| Natural language | Transition |
|---|---|
| "X individuals vaccinated per day from a time-series schedule" | `scheduled`, schedule: `doses.csv` |
| "doses distributed across S and R, only S benefit" | `scheduled`, schedule: `doses.csv`, eligible: `["S", "R"]` |

For models with multiple infectious classes, add one mediated transition per infectious compartment:
```yaml
- source: S
  target: E
  kind: mediated
  params: ["beta_mild", "I_mild"]
- source: S
  target: E
  kind: mediated
  params: ["beta_severe", "I_severe"]
```

### Translation rules

Use these rules to translate natural-language disease descriptions into transitions:

| Natural language | Transition |
|---|---|
| "X becomes Y at rate r" | `spontaneous`, params: `"r"` |
| "X gets infected by contact with Y at rate r" | `mediated`, params: `["r", "Y"]` |
| "X loses immunity and returns to Y" | `spontaneous` from X to Y, params: `"waning_rate"` |
| "fraction p of X go to Y, the rest go to Z" | Two `spontaneous` transitions from X. Set rates so `rate_Y / (rate_Y + rate_Z) = p` |
| "case fatality rate p over mean duration d" | `spontaneous` to D with `rate = -ln(1-p) / d` |
| "multiple infectious stages contribute to transmission" | One `mediated` transition per infectious compartment |

### Constraints

- Every parameter referenced in `transitions` must appear in the `parameters` section.
- Every compartment referenced in `transitions` must appear in `compartments`.
- `initial_conditions` must cover all compartments and fractions must sum to 1.0.
- Only `spontaneous` and `mediated` kinds are available via config.
- Do not create transitions where source == target.

### Worked example: SEIRD with hospitalization and waning immunity

Description: *"COVID-19 with exposed period, mild and severe infection, hospitalization, death, and immunity that wanes after ~1 year."*

```yaml
model:
  type: custom
  compartments: ["S", "E", "I_mild", "I_severe", "H", "R", "D"]
  transitions:
    # Transmission (both mild and severe cases are infectious)
    - {source: S, target: E, kind: mediated, params: ["beta_mild", "I_mild"]}
    - {source: S, target: E, kind: mediated, params: ["beta_severe", "I_severe"]}
    - {source: S, target: E, kind: mediated, params: ["beta_hosp", "H"]}
    # Incubation → branching into mild vs severe
    - {source: E, target: I_mild, kind: spontaneous, params: "sigma * (1 - p_severe)"}
    - {source: E, target: I_severe, kind: spontaneous, params: "sigma * p_severe"}
    # Recovery / progression
    - {source: I_mild, target: R, kind: spontaneous, params: "gamma_mild"}
    - {source: I_severe, target: H, kind: spontaneous, params: "hosp_rate"}
    - {source: H, target: R, kind: spontaneous, params: "gamma_hosp"}
    - {source: H, target: D, kind: spontaneous, params: "death_rate"}
    # Waning immunity
    - {source: R, target: S, kind: spontaneous, params: "waning_rate"}

parameters:
  beta_mild: 0.3
  beta_severe: 0.1
  beta_hosp: 0.05
  sigma: 0.33             # 1/3 day incubation
  p_severe: 0.15          # 15% of cases are severe
  gamma_mild: 0.14        # ~7 day recovery
  hosp_rate: 0.2          # ~5 day progression to hospital
  gamma_hosp: 0.1         # ~10 day hospital stay
  death_rate: 0.02
  waning_rate: 0.0027     # ~1/365 days

simulation:
  start_date: "2020-01-01"
  end_date: "2020-12-31"
  n_simulations: 200

initial_conditions:
  S: 0.999
  E: 0.0005
  I_mild: 0.0004
  I_severe: 0.0001
  H: 0.0
  R: 0.0
  D: 0.0
```

## Comparing Scenarios

The `compare` command computes standard metrics across multiple bundles in a single call, replacing the need to write custom Parquet-reading code for common comparisons.

```bash
epydemix compare baseline.epx early.epx late.epx
# → JSON with attack_rate, peak, peak_date, total_deaths for each scenario

epydemix compare baseline.epx early.epx late.epx \
  -n Baseline,Early,Late \
  -m attack_rate,peak,total_deaths,days_over:500 \
  -b Baseline
# → metrics + deltas vs Baseline
```

### Built-in metrics

| Metric | Description |
|---|---|
| `attack_rate` | % of population ever infected (based on susceptible depletion) |
| `peak` | Peak value of a variable (default: first I-like _total column) |
| `peak_date` | Date of peak value |
| `total_deaths` | Final value of the death compartment |
| `days_over:N` | Days the median of a variable exceeds threshold N |
| `final_value` | Final value of a variable |

### Flags

| Flag | Description |
|---|---|
| `-n, --names` | Comma-separated scenario names (same order as bundle arguments) |
| `-m, --metrics` | Comma-separated metrics to compute |
| `-v, --variables` | Variable name for variable-specific metrics |
| `-b, --baseline` | Scenario name for delta computation |
| `--round` | Decimal precision (default: 6) |

When `--baseline` is provided, the output includes a `_deltas_vs_<name>` key with the difference (scenario minus baseline) for each numeric metric.

### Typical sweep workflow

```bash
# 1. Write one base config with the full model
# 2. Write N small overlays (one per scenario)
# 3. Validate and run all of them
for f in scenario_*.yaml; do
  epydemix validate "$f"
  epydemix run "$f" -o "${f%.yaml}.epx"
done
# 4. Compare in one shot
epydemix compare scenario_*.epx -n Baseline,Early,Late -b Baseline
```

## Visualization Recipes

Epydemix does not produce figures — you write Python with matplotlib against the Parquet files. Figures are stored **inside the bundle** at `<bundle>/figures/` and registered in the manifest so they travel with the data.

After saving a figure, call `add_figure_to_manifest` to record it:

```python
from epydemix.io import add_figure_to_manifest

# After plt.savefig("results.epx/figures/epidemic_curve.png", ...)
add_figure_to_manifest(
    "results.epx",
    "epidemic_curve.png",
    description="Infection time-series with 90% CI bands",
    variables=["I_total"],
)
```

Subsequent `epydemix inspect results.epx manifest` calls will include the figure metadata under a `"figures"` key.

**`figures` is a dict keyed by filename, not a list.** Iterate with `.items()`:

```python
import json, subprocess
manifest = json.loads(subprocess.check_output(
    ["epydemix", "inspect", "results.epx", "manifest"]))
for filename, meta in manifest.get("figures", {}).items():
    print(filename, "-", meta["description"])
```

Do not iterate as a list (`for f in figures`). Do not slice the dict (`figures[:3]`).

### Recipe 1: Epidemic curve with uncertainty bands

The standard visualization — median trajectory with a shaded credible interval.

```python
import pandas as pd
import matplotlib.pyplot as plt

BUNDLE = "results.epx"
VARS = ["I_total"]  # change to plot multiple variables

comp = pd.read_parquet(f"{BUNDLE}/compartments.parquet",
                       columns=["sim_id", "date"] + VARS)
comp["date"] = pd.to_datetime(comp["date"])

fig, ax = plt.subplots(figsize=(10, 5))
for var in VARS:
    piv = comp.pivot(index="date", columns="sim_id", values=var)
    median = piv.quantile(0.5, axis=1)
    lo = piv.quantile(0.05, axis=1)
    hi = piv.quantile(0.95, axis=1)
    ax.plot(median.index, median.values, label=var, linewidth=2)
    ax.fill_between(median.index, lo.values, hi.values, alpha=0.2)

ax.set_ylabel("Count")
ax.legend()
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{BUNDLE}/figures/epidemic_curve.png", dpi=150)

from epydemix.io import add_figure_to_manifest
add_figure_to_manifest(BUNDLE, "epidemic_curve.png",
                       "Epidemic curve with 90% CI bands", VARS)
```

### Recipe 2: Scenario comparison

Overlay multiple bundles to compare intervention strategies.

```python
import pandas as pd
import matplotlib.pyplot as plt

scenarios = {
    "Baseline": "baseline.epx",
    "Early intervention": "early.epx",
    "Late intervention": "late.epx",
}
VAR = "I_total"

fig, ax = plt.subplots(figsize=(10, 5))
for name, bundle in scenarios.items():
    comp = pd.read_parquet(f"{bundle}/compartments.parquet",
                           columns=["sim_id", "date", VAR])
    comp["date"] = pd.to_datetime(comp["date"])
    piv = comp.pivot(index="date", columns="sim_id", values=VAR)
    median = piv.quantile(0.5, axis=1)
    lo, hi = piv.quantile(0.05, axis=1), piv.quantile(0.95, axis=1)
    ax.plot(median.index, median, label=name, linewidth=2)
    ax.fill_between(median.index, lo, hi, alpha=0.15)

ax.set_ylabel(VAR)
ax.legend()
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("baseline.epx/figures/scenario_comparison.png", dpi=150)

from epydemix.io import add_figure_to_manifest
add_figure_to_manifest("baseline.epx", "scenario_comparison.png",
                       f"Scenario comparison: {VAR}", [VAR])
```

### Recipe 3: Hospital capacity analysis

Plot hospitalizations against a capacity threshold.

```python
import pandas as pd
import matplotlib.pyplot as plt

BUNDLE = "results.epx"
CAPACITY = 500  # bed threshold

comp = pd.read_parquet(f"{BUNDLE}/compartments.parquet",
                       columns=["sim_id", "date", "H_total"])
comp["date"] = pd.to_datetime(comp["date"])
piv = comp.pivot(index="date", columns="sim_id", values="H_total")
median = piv.quantile(0.5, axis=1)
lo, hi = piv.quantile(0.05, axis=1), piv.quantile(0.95, axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(median.index, median, color="steelblue", linewidth=2, label="Median")
ax.fill_between(median.index, lo, hi, alpha=0.2, color="steelblue", label="90% CI")
ax.axhline(CAPACITY, color="red", linestyle="--", linewidth=1, label=f"Capacity ({CAPACITY})")
days_over = (median > CAPACITY).sum()
ax.set_title(f"Hospital census — {days_over} days over capacity")
ax.set_ylabel("Hospital census")
ax.legend()
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{BUNDLE}/figures/hospital_capacity.png", dpi=150)

from epydemix.io import add_figure_to_manifest
add_figure_to_manifest(BUNDLE, "hospital_capacity.png",
                       f"Hospital census vs {CAPACITY}-bed capacity",
                       ["H_total"])
```

### Recipe 4: Calibration fit vs. observed data

Compare simulated quantiles against the data the model was calibrated to.

```python
import json
import subprocess
import matplotlib.pyplot as plt

BUNDLE = "calibration.epx"
VAR = "Infected_total"   # must match target_variable in the calibration config

# Use the CLI to get fit data as JSON
result = subprocess.run(
    ["epydemix", "inspect", BUNDLE, "fit", "-v", VAR, "-q", "0.05,0.5,0.95"],
    capture_output=True, text=True,
)
fit = json.loads(result.stdout)
# fit structure: {VAR: {"0.05": [...], "0.5": [...], "0.95": [...]},
#                 "observed": {VAR: [...]}}

ts = range(len(fit[VAR]["0.5"]))
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ts, fit[VAR]["0.5"], color="steelblue", linewidth=2, label="Median fit")
ax.fill_between(ts, fit[VAR]["0.05"], fit[VAR]["0.95"],
                alpha=0.2, color="steelblue", label="90% CI")
if "observed" in fit and VAR in fit["observed"]:
    obs_vals = fit["observed"][VAR]
    ax.scatter(range(len(obs_vals)), obs_vals,
               color="black", s=20, zorder=5, label="Observed")
ax.set_ylabel(VAR)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{BUNDLE}/figures/calibration_fit.png", dpi=150)

from epydemix.io import add_figure_to_manifest
add_figure_to_manifest(BUNDLE, "calibration_fit.png",
                       f"Calibration fit vs observed: {VAR}", [VAR])
```

## Calibration

Calibration fits model parameters to observed data using Approximate Bayesian Computation (ABC). The `calibrate` command works like `run`: you write a config, validate it, and submit it. The config is the same as a simulation config, plus a `calibration` section.

### Calibration config

```yaml
model:
  type: SEIR

parameters:
  incubation_rate: 0.2          # fixed — not calibrated

simulation:
  start_date: "2024-01-01"
  end_date: "2024-06-30"

initial_conditions:
  Susceptible: 0.999
  Exposed: 0.0005
  Infected: 0.0005
  Recovered: 0.0

calibration:
  strategy: smc                 # smc | rejection | top_fraction
  priors:
    transmission_rate:
      distribution: uniform
      low: 0.1
      high: 0.8
    recovery_rate:
      distribution: uniform
      low: 0.05
      high: 0.3
  observed_data: observed.csv   # path to CSV, or inline list
  observed_column: cases        # column name in CSV (auto-detected if 2-column CSV)
  target_variable: Infected_total  # model variable to compare against observed data
  distance: rmse                # rmse | mae | wmape | mape
  # Strategy-specific settings:
  num_particles: 500
  num_generations: 10
```

Parameters listed in `calibration.priors` are sampled from the specified distributions during calibration. Parameters in the `parameters` section that are NOT in `priors` are held fixed. Every parameter the model needs must appear in one place or the other.

### Observed data

The `observed_data` field accepts either a file path or an inline list:

```yaml
# File path (CSV, resolved relative to config file)
observed_data: data/weekly_cases.csv
observed_column: cases

# Inline (for quick testing)
observed_data: [100, 95, 90, 86, 82, 78, 74, 70, 67, 64]
```

The observed data array must have the same length as the simulation time-series for the target variable.

### Prior distributions

| Distribution | Parameters | Example |
|---|---|---|
| `uniform` | `low`, `high` | `{distribution: uniform, low: 0.1, high: 0.8}` |
| `normal` | `mean`, `std` | `{distribution: normal, mean: 0.3, std: 0.05}` |
| `truncnorm` | `mean`, `std`, `low`, `high` | `{distribution: truncnorm, mean: 0.3, std: 0.1, low: 0.1, high: 0.5}` |
| `beta` | `a`, `b` | `{distribution: beta, a: 2, b: 5}` |
| `gamma` | `a`, `scale` | `{distribution: gamma, a: 2.0, scale: 0.1}` |
| `lognormal` | `shape`, `scale` | `{distribution: lognormal, shape: 0.5, scale: 1.0}` |
| `expon` | `scale` | `{distribution: expon, scale: 0.1}` |

### Strategies

**`smc`** (Sequential Monte Carlo) — the default and recommended strategy. Iteratively refines the posterior through multiple generations. Settings: `num_particles` (default 1000), `num_generations` (default 10), `epsilon_quantile_level` (default 0.5), `minimum_epsilon`, `total_simulations_budget`.

**`rejection`** — simple ABC rejection sampling. Accepts particles whose distance to observed data is below a threshold. Settings: `epsilon` (default 0.1), `num_particles` (default 1000), `total_simulations_budget`.

**`top_fraction`** — runs a fixed number of simulations and keeps the best-fitting fraction. Settings: `top_fraction` (default 0.05), `n_simulations` (default 100).

### Running calibration

```bash
# Validate the calibration config
epydemix validate calibrate_config.yaml

# Run calibration
epydemix calibrate calibrate_config.yaml -o calibration.epx
# → manifest JSON to stdout; calibration.epx/ directory on disk

# Inspect the posterior
epydemix inspect calibration.epx posterior
# → {"transmission_rate": {"mean": 0.28, "std": 0.03, "ci95": [0.23, 0.33], "median": 0.27}, ...}

# Inspect calibration fit vs. observed data
epydemix inspect calibration.epx fit -v Infected_total -q 0.05,0.5,0.95
```

### Calibration bundle structure

```
calibration.epx/
  manifest.json
  posterior.parquet     # parameter samples per generation
  distances.parquet     # distance values per generation
  trajectories.parquet  # selected trajectories (last generation)
  config.yaml           # config that produced this run
  figures/              # for agent-produced visualizations
```

### Calibration workflow

```bash
# 1. Write a simulation config for your model
# 2. Add a calibration section with priors and observed data
# 3. Validate
epydemix validate cal_config.yaml
# 4. Run calibration (may take minutes depending on strategy settings)
epydemix calibrate cal_config.yaml -o calibration.epx
# 5. Check posterior parameter estimates
epydemix inspect calibration.epx posterior
# 6. Check fit quality
epydemix inspect calibration.epx fit -v Infected_total -q 0.05,0.5,0.95
# 7. Visualize (see Recipe 4 in Visualization Recipes)
# 8. Project forward (see Projections section)
#    Write a projection overlay (only the deltas), then run directly — no validate step:
#    epydemix project calibration.epx -c projection.yaml -o projection.epx
```

Config inheritance works with calibration configs: write a base simulation config, then an overlay that adds only the `calibration` section. This keeps the model definition and the calibration setup separate and reusable.

## Projections from Calibration Posteriors

### What projections do

After calibrating a model, you often want to run forward scenarios using the estimated parameters — for example, extending the simulation period, adding interventions, or exploring "what if" questions while respecting the calibrated uncertainty. The `project` command samples parameter sets from the calibration posterior (weighted by particle importance), runs forward simulations, and saves results as a standard simulation bundle that works with all existing `inspect` and `compare` commands.

### Running a projection

The simplest form replays the calibration over the same period, sampling from the posterior:

```bash
epydemix project calibration.epx -o projection.epx
```

This uses the calibration bundle's stored config and samples 200 simulations (default) from the last-generation posterior.

**CLI signature:** `epydemix project CALIBRATION_BUNDLE [-c CONFIG] -o OUTPUT`

The calibration bundle is the only positional argument. The overlay config is always a **named** option (`-c` / `--config`). Do **not** pass it as a second positional argument — that will fail with `Got unexpected extra argument`.

### Projection config overlay

To change the simulation period, add interventions, or override parameters, write a projection overlay. **The calibration bundle's stored config is the automatic base — do not add a `base:` key.** Only specify what changes:

```yaml
# projection.yaml  — only the deltas from the calibration config
simulation:
  end_date: "2025-06-30"                # extend beyond calibration period

overrides:                               # hypothetical intervention
  - parameter: transmission_rate
    start_date: "2025-03-01"
    end_date: "2025-06-30"
    value: 0.15

projection:
  n_simulations: 200                     # how many posterior samples (default: 200)
  generation: -1                         # which posterior generation (-1 = last)
```

Then run:

```bash
epydemix project calibration.epx --config projection.yaml -o projection.epx
```

> **Do not validate projection overlays standalone.** `epydemix validate` treats its input as a complete config and will error on a minimal overlay that omits `model`, `parameters`, or `initial_conditions`. Projection overlays are not standalone configs — they are diffs that `epydemix project` merges at runtime against the calibration bundle's stored config. Validation happens internally before the simulation starts. Skip the `epydemix validate` step for overlay files passed via `-c`.

### Projection config reference

The `projection` section is optional. When omitted, defaults are used:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_simulations` | int | 200 | Number of posterior samples to simulate |
| `generation` | int | -1 | Posterior generation to sample from (-1 = last) |

All other config sections (`model`, `simulation`, `parameters`, `initial_conditions`, `overrides`, `interventions`) follow the same format as simulation configs and can be overridden via config inheritance.

### Projection output

The output is a standard simulation bundle (type `SimulationResults`), which means all existing inspection commands work:

```bash
# Quantiles of projected epidemic curve
epydemix inspect projection.epx quantiles -v Infected_total -q 0.05,0.5,0.95
# Peak timing
epydemix inspect projection.epx peak -v Infected_total
# Summary statistics
epydemix inspect projection.epx summary -v Infected_total
# Compare projection scenarios
epydemix compare baseline_proj.epx intervention_proj.epx
```

### Projection workflow

```bash
# 1. Run calibration first (if not already done)
epydemix calibrate cal_config.yaml -o calibration.epx
# 2. Write projection overlay (or skip for same-period replay)
# 3. Run projection
epydemix project calibration.epx --config projection.yaml -o projection.epx
# 4. Inspect results
epydemix inspect projection.epx quantiles -v Infected_total -q 0.05,0.5,0.95
# 5. Compare multiple projection scenarios
epydemix compare baseline_proj.epx intervention_proj.epx -n Baseline,Intervention
```

## Bundle Naming

Choose descriptive names for output bundles — the bundle filename is the primary identifier in downstream commands. Generic names like `results.epx` become unreadable when you have several scenarios, and re-running a command with the same `-o` path silently overwrites the previous bundle.

Good naming practice: encode the scenario's distinguishing feature in the bundle name.

```bash
# Instead of this:
epydemix run config1.yaml -o results.epx
epydemix run config2.yaml -o results2.epx

# Do this:
epydemix run baseline.yaml        -o baseline_no_intervention.epx
epydemix run early_closure.yaml   -o early_school_closure.epx
epydemix run late_closure.yaml    -o late_school_closure.epx
```

This pays off immediately with `epydemix compare`, which defaults to the bundle directory stem as the scenario label:

```bash
# Readable output with descriptive names — no --names flag needed:
epydemix compare baseline_no_intervention.epx early_school_closure.epx late_school_closure.epx
```

For calibration and projection workflows, a useful convention is `{model}_{purpose}.epx`:

```bash
epydemix calibrate cal.yaml   -o sir_calibration.epx
epydemix project sir_calibration.epx -o sir_proj_baseline.epx
epydemix project sir_calibration.epx -c intervention.yaml -o sir_proj_school_closure.epx
```

## Shell and Path Conventions

**Always use absolute paths with the CLI.** The working directory persists across Bash tool calls (a `cd` in one call affects subsequent calls), but the CLI resolves relative config paths from the current working directory. To avoid confusion:

- Use absolute paths for config files, output bundles, and observed data: `epydemix run /abs/path/to/config.yaml -o /abs/path/to/output.epx`
- Never `cd` into a project directory and then use relative paths — the next Bash call will still be in that directory, causing double-prefix errors (e.g. `calib_demo/calib_demo/config.yaml`).
- The `base:` field in config inheritance is resolved relative to the config file itself, regardless of the shell working directory.

## Error Handling

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Validation or input error |

### Structured errors

Errors are printed as JSON to stderr:

```json
{
  "error": true,
  "code": "INVALID_CONFIG",
  "message": "Config validation failed.",
  "details": ["simulation.start_date is required", "simulation.end_date is required"]
}
```

Error codes: `CONFIG_LOAD_ERROR`, `INVALID_CONFIG`, `RUNTIME_ERROR`, `INSPECT_ERROR`, `UNKNOWN_MODEL`, `UNKNOWN_DEFAULTS`, `POPULATION_ERROR`.

## Complete Workflow Examples

### Example 1: Basic SIR outbreak

```bash
# Check the SIR parameter space
epydemix schema SIR
# Write a minimal config
cat > sir.yaml << 'EOF'
model:
  type: SIR
parameters:
  transmission_rate: 0.4
  recovery_rate: 0.1
simulation:
  start_date: "2024-01-01"
  end_date: "2024-06-30"
  n_simulations: 50
initial_conditions:
  Susceptible: 0.999
  Infected: 0.001
  Recovered: 0.0
EOF
# Validate and run
epydemix validate sir.yaml
epydemix run sir.yaml -o sir_results.epx
# When did the epidemic peak?
epydemix inspect sir_results.epx peak -v Infected_total
# Weekly median infections
epydemix inspect sir_results.epx quantiles -v Infected_total -q 0.5 --resample W
```

### Example 2: COVID-19 with defaults and school closure

```bash
# Start from literature defaults
epydemix defaults covid19
# Write config using those values
cat > covid.yaml << 'EOF'
model:
  type: SEIR
parameters:
  transmission_rate: 0.4
  incubation_rate: 0.33
  recovery_rate: 0.14
population:
  name: "Italy"
  contact_layers: ["home", "work", "school", "community"]
simulation:
  start_date: "2020-01-15"
  end_date: "2020-07-31"
  n_simulations: 100
initial_conditions:
  Susceptible: 0.9999
  Exposed: 0.00005
  Infected: 0.00005
  Recovered: 0.0
interventions:
  - layer: school
    start_date: "2020-03-10"
    end_date: "2020-06-01"
    reduction: 0.9
overrides:
  - parameter: transmission_rate
    start_date: "2020-03-22"
    end_date: "2020-05-04"
    value: 0.15
EOF
epydemix run covid.yaml -o covid_results.epx
# Compare pre- and post-lockdown
epydemix inspect covid_results.epx quantiles -v Infected_total -q 0.05,0.5,0.95 --end 2020-03-21
epydemix inspect covid_results.epx quantiles -v Infected_total -q 0.05,0.5,0.95 --start 2020-03-22
```

### Example 3: Analyze results with custom Python

When the canned inspect commands aren't enough, use the manifest to understand the Parquet schema, then write Python:

```bash
# Get the manifest to see column names and dtypes
epydemix inspect results.epx manifest
```

```python
import pandas as pd

# Compute attack rate: fraction ever infected
comp = pd.read_parquet("results.epx/compartments.parquet",
                       columns=["sim_id", "date", "Recovered_total"])
final = comp.groupby("sim_id")["Recovered_total"].last()
population = 10000  # from manifest
attack_rate = final / population
print(f"Attack rate: {attack_rate.median():.1%} "
      f"[{attack_rate.quantile(0.05):.1%}, {attack_rate.quantile(0.95):.1%}]")
```

### Example 4: Calibrate SIR to observed data

```bash
# Write calibration config
cat > calibrate_sir.yaml << 'EOF'
model:
  type: SIR
parameters:
  recovery_rate: 0.1           # fixed
simulation:
  start_date: "2024-01-01"
  end_date: "2024-03-31"
initial_conditions:
  Susceptible: 0.999
  Infected: 0.001
  Recovered: 0.0
calibration:
  strategy: smc
  priors:
    transmission_rate:
      distribution: uniform
      low: 0.1
      high: 0.8
  observed_data: weekly_cases.csv
  observed_column: cases
  target_variable: Infected_total
  distance: rmse
  num_particles: 200
  num_generations: 5
EOF
# Run calibration
epydemix calibrate calibrate_sir.yaml -o sir_calibration.epx
# What did we learn about the transmission rate?
epydemix inspect sir_calibration.epx posterior
# How well does the calibrated model fit the data?
epydemix inspect sir_calibration.epx fit -v Infected_total -q 0.05,0.5,0.95
```

### Example 5: Project from calibration posterior

```bash
# After calibrating (Example 4), project forward with an intervention.
# No ``base:`` needed — the calibration bundle's config is the automatic base.
cat > projection.yaml << 'EOF'
simulation:
  end_date: "2024-06-30"                # extend 3 months beyond calibration
overrides:
  - parameter: transmission_rate
    start_date: "2024-04-01"
    end_date: "2024-06-30"
    value: 0.15                          # hypothetical intervention
projection:
  n_simulations: 200
EOF
epydemix project sir_calibration.epx --config projection.yaml -o sir_projection.epx
# Inspect the projection
epydemix inspect sir_projection.epx quantiles -v Infected_total -q 0.05,0.5,0.95
epydemix inspect sir_projection.epx peak -v Infected_total
```

## Installation

```bash
# Core library only (Python API, no CLI)
pip install epydemix

# With CLI support (adds click, pyyaml, pyarrow)
pip install epydemix[cli]

# Full agent support (same as cli, future-proofed for additional agent tooling)
pip install epydemix[agent]
```

## Python API (alternative to CLI)

The CLI is the recommended agent interface, but the same functionality is available in Python:

```python
from epydemix import load_predefined_model
from epydemix.parameters import load_defaults
from epydemix.io.bundle import save_bundle, load_bundle
from epydemix.io.inspect import inspect_bundle

# Load disease defaults
defaults = load_defaults("covid19")
model = load_predefined_model("SEIR", **defaults.as_params())

# Introspect parameters
schema = model.parameter_registry.to_json_schema()
model.parameter_registry.describe()  # human-readable

# Run and save
results = model.run_simulations(...)
save_bundle(results, "results.epx")

# Inspect programmatically
summary = inspect_bundle("results.epx", "summary", variables=["I_total"])
quantiles = inspect_bundle("results.epx", "quantiles",
                           variables=["I_total"], quantiles=[0.05, 0.5, 0.95])
```
