# Calibration

Calibration finds the parameter values that make the model reproduce observed data.
epydemix uses **Approximate Bayesian Computation (ABC)**: draw parameters from a prior,
simulate, and keep the draws whose simulation is close to the data. The kept draws
approximate the posterior distribution. No likelihood is needed, only a way to simulate
and a distance between a simulation and the data.

## The ingredients

An {class}`~epydemix.calibration.abc.ABCSampler` needs four things.

```python
import pandas as pd
from scipy import stats

from epydemix import load_predefined_model, simulate
from epydemix.calibration import ABCSampler, rmse

model = load_predefined_model("SIR")

# 1. Observed data: a DataFrame with a "date" and a "data" column. To keep the example
#    self-contained we simulate it: daily new infections from one run with known parameters.
truth = simulate(
    model, start_date="2024-01-01", end_date="2024-03-31",
    transmission_rate=0.35, recovery_rate=0.12, rng=0,
)
data = pd.DataFrame({
    "date": pd.to_datetime(truth.dates),
    "data": truth.transitions["Susceptible_to_Infected_total"],
})

# 2. A function that runs the model with a set of parameters and returns the
#    quantity to compare with the data, under the key "data"
def simulate_wrapper(parameters):
    trajectory = simulate(**parameters)
    return {"data": trajectory.transitions["Susceptible_to_Infected_total"]}

# 3. Priors for the parameters to estimate (any scipy.stats distribution)
priors = {
    "transmission_rate": stats.uniform(0.15, 0.35),   # uniform on [0.15, 0.50]
    "recovery_rate": stats.uniform(0.05, 0.15),       # uniform on [0.05, 0.20]
}

# 4. Everything else the simulation function needs, kept fixed
parameters = {"epimodel": model, "start_date": "2024-01-01", "end_date": "2024-03-31"}

sampler = ABCSampler(
    simulation_function=simulate_wrapper,
    priors=priors,
    parameters=parameters,
    observed_data=data["data"].values,
    distance_function=rmse,
    rng=42,
)
```

At each draw, the sampled values are merged into `parameters` and passed to the
simulation function. Because {func}`~epydemix.model.epimodel.simulate` accepts model
parameters as keyword arguments, a prior named `transmission_rate` replaces the model's
`transmission_rate` for that run. The simulated array must have the same length as the
observed one.

Note that `scipy.stats.uniform(loc, scale)` covers `[loc, loc + scale]`. Discrete priors
(for example `stats.randint`) are supported too.

## Running

```python
results = sampler.calibrate(strategy="smc", num_particles=200, num_generations=5)
```

Three strategies are available:

| `strategy` | How it works | Main arguments |
|---|---|---|
| `"smc"` | Sequential Monte Carlo. Several generations, each with a tighter tolerance and proposals drawn around the previous generation's particles. The most efficient in general. | `num_particles`, `num_generations`, `epsilon_quantile_level`, `minimum_epsilon` |
| `"rejection"` | Draw from the prior and keep simulations whose distance is below `epsilon`, until `num_particles` are accepted. | `epsilon`, `num_particles` |
| `"top_fraction"` | Run `Nsim` simulations from the prior and keep the closest `top_fraction` of them. Fixed cost, no tolerance to choose. | `Nsim`, `top_fraction` |

`"smc"` and `"rejection"` also accept `max_time` and `total_simulations_budget` to stop
early. Pass `verbose=False` to silence the progress messages.

## Looking at the results

`calibrate` returns a
{class}`~epydemix.calibration.calibration_results.CalibrationResults`.

```python
posterior = results.get_posterior_distribution()   # DataFrame, one row per particle
posterior.describe().loc[["mean", "std"]]
```

```text
      transmission_rate  recovery_rate
mean              0.348          0.120
std               0.032          0.037
```

The posterior means recover the values used to simulate the data (0.35 and 0.12).

```python
from epydemix.visualization import plot_posterior_distribution, plot_quantiles

plot_posterior_distribution(posterior, "transmission_rate", kind="kde", prior=priors["transmission_rate"])
```

```{image} ../_static/posterior.png
:alt: Posterior distributions of the two parameters, with the true values marked
:width: 100%
```

The simulations that were accepted show how well the calibrated model fits:

```python
quantiles = results.get_calibration_quantiles(dates=data["date"])
plot_quantiles(quantiles, columns="data", data=data, show_data=True, ylabel="New infections")
```

```{image} ../_static/calibration_fit.png
:alt: Median and 90% interval of the calibrated model against the observed data
:width: 650px
```

Other accessors: `get_weights()`, `get_distances()` and
`get_calibration_trajectories()`. With `"smc"` each takes a `generation` argument; the
default is the last generation.

## Projections

{meth}`~epydemix.calibration.abc.ABCSampler.run_projections` runs the model forward with
parameters drawn from the posterior. Give it the fixed parameters to use, typically the
same as for calibration with a later end date.

```python
projection_parameters = {"epimodel": model, "start_date": "2024-01-01", "end_date": "2024-05-31"}
results = sampler.run_projections(projection_parameters, iterations=100)

quantiles = results.get_projection_quantiles(dates=pd.date_range("2024-01-01", "2024-05-31"))
```

Use `scenario_id` to keep several projections side by side, for example with and without
an {doc}`intervention <interventions>`.

## Distance functions

`distance_function` takes the observed and the simulated dictionaries and returns a
number. The built-in ones are in {mod}`epydemix.calibration.metrics`:

| Function | Distance |
|---|---|
| {func}`~epydemix.calibration.metrics.rmse` | root mean square error (the default) |
| {func}`~epydemix.calibration.metrics.mae` | mean absolute error |
| {func}`~epydemix.calibration.metrics.wmape` | weighted mean absolute percentage error |
| {func}`~epydemix.calibration.metrics.mape` | mean absolute percentage error; points where the data is zero are left out |

You can pass any function with the same signature, `f(data, simulation) -> float`.

## Reproducibility

Passing `rng` to `ABCSampler` (a seed or a `numpy.random.Generator`) makes the whole
calibration reproducible: prior draws, proposals, resampling, the simulations, and the
projections that follow.

The calibration tutorials ({doc}`part 1 <../tutorials/04_Model_Calibration_part1>`,
{doc}`part 2 <../tutorials/05_Model_Calibration_part2>`) and the
{doc}`COVID-19 case study <../tutorials/07_Covid-19_Example>` show complete workflows on
real data.
