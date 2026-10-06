# Epydemix — Recipes and Workflow Examples

**Agent Framework v1**

This file contains visualization recipes and end-to-end workflow examples for the epydemix CLI. For the full reference (config format, output structure, CLI commands, custom models, calibration, projections), see [AGENT.md](AGENT.md).

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

Plot hospitalizations against a capacity threshold. The column name depends on how the hospital compartment was introduced: predefined backbones with `outcome: hospitalization` produce `Hospitalized_total`; custom models use whatever short name you defined (e.g. `H_total`). Verify with `epydemix inspect <bundle> manifest` first.

```python
import pandas as pd
import matplotlib.pyplot as plt

BUNDLE = "results.epx"
HOSP_COL = "Hospitalized_total"   # or "H_total" for custom models — check the manifest
CAPACITY = 500  # bed threshold

comp = pd.read_parquet(f"{BUNDLE}/compartments.parquet",
                       columns=["sim_id", "date", HOSP_COL])
comp["date"] = pd.to_datetime(comp["date"])
piv = comp.pivot(index="date", columns="sim_id", values=HOSP_COL)
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
                       [HOSP_COL])
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

### Example 6: SEIAR with asymptomatic transmission

The SEIAR backbone adds an `Asymptomatic` infectious compartment that branches off `Exposed`. A fraction (`asymptomatic_fraction`) of exposed individuals go to `Asymptomatic` and transmit at a reduced rate (`asymptomatic_relative_infectivity`), while the rest become symptomatic `Infected`.

```bash
# Discover the SEIAR parameter set
epydemix schema SEIAR

cat > seiar.yaml << 'EOF'
model:
  type: SEIAR
parameters:
  transmission_rate: 0.4
  incubation_rate: 0.25
  recovery_rate: 0.14
  asymptomatic_fraction: 0.4              # 40% of cases are asymptomatic
  asymptomatic_recovery_rate: 0.14
  asymptomatic_relative_infectivity: 0.5  # asymptomatics transmit at half rate
simulation:
  start_date: "2024-01-01"
  end_date: "2024-06-30"
  n_simulations: 100
initial_conditions:
  Susceptible: 0.999
  Exposed: 0.0005
  Infected: 0.0003
  Asymptomatic: 0.0002
  Recovered: 0.0
EOF
epydemix validate seiar.yaml
epydemix run seiar.yaml -o seiar.epx

# Compare symptomatic vs asymptomatic prevalence at peak
epydemix inspect seiar.epx peak -v Infected_total,Asymptomatic_total

# Compute attack rate from S depletion. Do NOT sum
# Susceptible_to_Exposed_total here — SEIAR has two S→E mediated transitions
# (one via Infected, one via Asymptomatic) and the framework sums them into
# a single column, double-counting the actual flow. S depletion is unambiguous
# since no S→V vaccination transition is present.
python - << 'PY'
import pandas as pd
comp = pd.read_parquet("seiar.epx/compartments.parquet",
                       columns=["sim_id", "date", "Susceptible_total"])
N = 100000
S0 = comp.groupby("sim_id")["Susceptible_total"].first()
Sf = comp.groupby("sim_id")["Susceptible_total"].last()
attack = (S0 - Sf) / N * 100
print(f"Attack rate: median={attack.median():.1f}%  "
      f"[{attack.quantile(0.05):.1f}%, {attack.quantile(0.95):.1f}%]")
PY
```

### Example 7: Modular composition — endemic disease with hospitalization

Compose two modules on a SEIR backbone: `outcome: hospitalization` adds a `Hospitalized` compartment for capacity planning, and `waning_immunity` makes the disease endemic by recycling recovered individuals back to susceptible after ~1 year. Then compare scenarios with and without waning to see how the long-run dynamics differ.

```bash
# Discover the extended parameter set
epydemix schema SEIR --waning-immunity --outcome hospitalization

# Baseline: lifelong immunity, single wave
cat > seir_outbreak.yaml << 'EOF'
model:
  type: SEIR
  outcome: hospitalization
parameters:
  transmission_rate: 0.35
  incubation_rate: 0.2
  recovery_rate: 0.1
  hospitalization_rate: 0.01
  hospitalization_recovery_rate: 0.1
population:
  size: 500000
simulation:
  start_date: "2024-01-01"
  end_date: "2026-12-31"           # 3 years
  n_simulations: 50
initial_conditions:
  Susceptible: 0.9999
  Exposed: 0.00005
  Infected: 0.00005
  Recovered: 0.0
  Hospitalized: 0.0
EOF

# Endemic variant: same model + waning immunity
cat > seir_endemic.yaml << 'EOF'
base: seir_outbreak.yaml
model:
  type: SEIR
  outcome: hospitalization
  waning_immunity: true
parameters:
  transmission_rate: 0.35
  incubation_rate: 0.2
  recovery_rate: 0.1
  hospitalization_rate: 0.01
  hospitalization_recovery_rate: 0.1
  waning_rate: 0.00274              # 1/365 → ~1-year immunity
EOF

for f in seir_outbreak.yaml seir_endemic.yaml; do
  epydemix validate "$f"
  epydemix run "$f" -o "${f%.yaml}.epx"
done

# Compare peak hospitalization timing and magnitude
epydemix compare seir_outbreak.epx seir_endemic.epx \
  -n Outbreak,Endemic \
  -m peak,peak_date,days_over:500 \
  -v Hospitalized_total \
  -b Outbreak
```

Plot the hospital census of both scenarios overlaid using Recipe 2 (with `HOSP_COL = "Hospitalized_total"`) — the endemic variant should show repeated waves while the outbreak case has a single peak followed by herd-immunity decline.
