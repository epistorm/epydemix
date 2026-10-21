# Running simulations

Simulations are stochastic: each run draws the number of people making each transition
at each time step, so two runs give different epidemics. You normally run many and look
at their distribution.

## Running

```python
from epydemix import load_predefined_model

model = load_predefined_model("SIR")
results = model.run_simulations(
    start_date="2024-01-01",
    end_date="2024-04-10",
    Nsim=100,
)
```

{meth}`~epydemix.model.epimodel.EpiModel.run_simulations` returns a
{class}`~epydemix.model.simulation_results.SimulationResults` holding all the runs.

## Getting the results out

Every variable is named `{compartment}_{group}` or `{source}_to_{target}_{group}`, where
the group is a demographic group or `total`. Compartments are the number of people in a
state; transitions are the number of people who moved during each step (for example,
`Susceptible_to_Infected_total` is the number of new infections per day).

**Quantiles across runs**, one row per date and quantile:

```python
results.get_quantiles_compartments(quantiles=[0.05, 0.5, 0.95]).head(3)
```

```text
        date  quantile  Susceptible_0  Susceptible_total  Infected_0  Infected_total  Recovered_0  Recovered_total
0 2024-01-01      0.05       99929.95           99929.95       52.00           52.00         2.00             2.00
1 2024-01-02      0.05       99907.00           99907.00       60.95           60.95         6.00             6.00
2 2024-01-03      0.05       99879.00           99879.00       70.95           70.95        10.95            10.95
```

`results.get_quantiles_transitions()` does the same for transitions.

**Every run in one tidy table**, one row per run, date and variable:

```python
results.to_dataframe(variables=["Infected_total", "Susceptible_to_Infected_total"]).head(3)
```

```text
   sim_id       date        variable  value
0       0 2024-01-01  Infected_total   61.0
1       0 2024-01-02  Infected_total   70.0
2       0 2024-01-03  Infected_total   90.0
```

This format works directly with pandas `groupby` and with seaborn. Without `variables`
you get every compartment and transition, which can be a large table.

**Raw arrays**, of shape `(Nsim, time steps)`:

```python
results.get_stacked_compartments()["Infected_total"].shape
# (100, 101)
```

`results.get_stacked_transitions()` is the equivalent for transitions, and
`results.trajectories` is the list of individual runs.

## Reproducible runs

Pass `rng` (an integer seed or a `numpy.random.Generator`) to get the same runs every
time:

```python
a = model.run_simulations(start_date="2024-01-01", end_date="2024-04-10", Nsim=10, rng=7)
b = model.run_simulations(start_date="2024-01-01", end_date="2024-04-10", Nsim=10, rng=7)
# a and b contain identical trajectories
```

## Initial conditions

By default, 0.05% of each group starts in the infectious compartments (those that drive
a mediated transition) and everyone else in the susceptible one. Change the share with
`percentage_in_agents` (a fraction: `0.0005` is 0.05%), or give the starting population
of each compartment explicitly, one value per demographic group:

```python
import numpy as np

Nk = model.population.Nk          # size of each demographic group
initial_conditions = {
    "Susceptible": Nk - 10,
    "Infected": np.full_like(Nk, 10),
    "Recovered": np.zeros_like(Nk),
}
results = model.run_simulations(
    start_date="2024-01-01",
    end_date="2024-04-10",
    initial_conditions_dict=initial_conditions,
)
```

Models without a mediated transition have no obvious default, so they need
`initial_conditions_dict`.

## Time step and output frequency

- **`dt`**: the length of a simulation step in days (default `1`). Use a smaller value,
  e.g. `dt=0.25`, when rates are high.
- **`resample_frequency`**: the frequency of the output (default `"D"`, daily). Use
  `"W"` for weekly output, or `None` to keep every simulation step. When resampling,
  compartments take the last value of each period and transitions are summed.

```python
weekly = model.run_simulations(
    start_date="2024-01-01", end_date="2024-04-10", Nsim=100, resample_frequency="W"
)
```

## A single run

{func}`~epydemix.model.epimodel.simulate` runs the model once and returns a single
{class}`~epydemix.model.simulation_output.Trajectory`. Extra keyword arguments replace
model parameters for that run only, which is what {doc}`calibration <calibration>`
builds on.

```python
from epydemix import simulate

trajectory = simulate(
    model, start_date="2024-01-01", end_date="2024-04-10", transmission_rate=0.4
)
trajectory.compartments["Infected_total"]   # array, one value per date
```

For running many simulations in parallel, see the
{doc}`multiprocessing tutorial <../tutorials/10_Multiprocessing>`.
