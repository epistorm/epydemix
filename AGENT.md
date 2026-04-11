# Driving Epydemix Programmatically

Epydemix is a Python library for compartmental epidemic modeling (SIR, SEIR, SIS, and custom models) with age-structured populations, contact matrices, and ABC calibration. This document is the contract for LLM agents (and any automation pipeline) that drive epydemix through its CLI.

All CLI commands print structured JSON to **stdout** and diagnostics to **stderr**. Parse stdout only.

## Quick Start

```bash
# 1. Discover available models
epydemix models
# → {"models": ["SIR", "SEIR", "SIS"]}

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
```

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
  # If omitted, a default single-group population is used.

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

**Optional sections:** `parameters` (defaults used if omitted), `population`, `initial_conditions`, `interventions`, `overrides`.

## Output Reference

### Bundle structure

Running `epydemix run` produces a `.epx` directory:

```
results.epx/
  manifest.json         # metadata + full Parquet schemas (see below)
  compartments.parquet  # (sim_id, date, S_total, I_total, ..., S_0, S_1, ...)
  transitions.parquet   # (sim_id, date, S_to_I_total, ..., S_to_I_0, ...)
  parameters.parquet    # (sim_id, param1, param2, ...)
```

Column naming convention: `{Compartment}_{group}` for age-group-specific columns, `{Compartment}_total` for the sum across all groups. Transition columns follow the same pattern: `{Source}_to_{Target}_{group}` and `{Source}_to_{Target}_total`.

### Manifest

The `manifest.json` is the bridge between the opaque Parquet files and the agent. It contains:

- `model`: compartments, transitions
- `population`: name, size, demographic groups
- `simulation`: n_simulations, start/end dates, timesteps
- `parameters_used`: scalar parameter values
- `files`: for each Parquet file, the full column schema with dtypes and descriptions
- `usage_hints`: instructions for CLI inspection and custom Python access

**The `files` section is your schema reference.** Use it to write correct `pd.read_parquet()` calls when the canned inspect commands are insufficient.

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

## Inspecting Results

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
| `--round` | Decimal precision (default: 2) | `--round 0` |
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
import matplotlib.pyplot as plt

BUNDLE = "calibration.epx"
VAR = "I_total"

# Use the CLI to get fit data as JSON
import subprocess
result = subprocess.run(
    ["python", "-m", "epydemix.cli.main", "inspect", BUNDLE, "fit",
     "-v", VAR, "-q", "0.05,0.5,0.95"],
    capture_output=True, text=True,
)
fit = json.loads(result.stdout)

fig, ax = plt.subplots(figsize=(10, 5))
ts = range(len(fit[VAR]["0.5"]))
ax.plot(ts, fit[VAR]["0.5"], color="steelblue", linewidth=2, label="Median fit")
ax.fill_between(ts, fit[VAR]["0.05"], fit[VAR]["0.95"],
                alpha=0.2, color="steelblue", label="90% CI")
if "observed" in fit:
    obs = fit["observed"]
    ax.scatter(range(len(obs[VAR])), obs[VAR],
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
