# Plotting

The plotting functions live in `epydemix.visualization`. They are built on matplotlib:
each one accepts an optional `ax` to draw on, and returns the axes so you can keep
customising the figure.

```python
from epydemix.visualization import plot_quantiles, plot_trajectories
```

| Function | Shows |
|---|---|
| {func}`~epydemix.visualization.plotting.plot_quantiles` | median and interval over time, optionally with observed data |
| {func}`~epydemix.visualization.plotting.plot_trajectories` | every individual run |
| {func}`~epydemix.visualization.plotting.plot_posterior_distribution` | posterior of one parameter |
| {func}`~epydemix.visualization.plotting.plot_posterior_distribution_2d` | joint posterior of two parameters |
| {func}`~epydemix.visualization.plotting.plot_distance_distribution` | distances of the accepted simulations |
| {func}`~epydemix.visualization.plotting.plot_population` | population by demographic group |
| {func}`~epydemix.visualization.plotting.plot_contact_matrix` | contact matrix of one layer |
| {func}`~epydemix.visualization.plotting.plot_spectral_radius` | intensity of contacts over time, with interventions |

## Simulation results

```python
from epydemix import load_predefined_model

model = load_predefined_model("SIR")
results = model.run_simulations(start_date="2024-01-01", end_date="2024-04-10", Nsim=100)

# Median and 90% interval
plot_quantiles(
    results.get_quantiles_compartments(),
    columns=["Infected_total", "Recovered_total"],
    labels=["Infected", "Recovered"],
    ylabel="People",
)

# Every run as a thin line
plot_trajectories(
    results.get_stacked_compartments(),
    columns="Infected_total",
    dates=results.dates,
)
```

The interval drawn by `plot_quantiles` is set by `lower_q` and `upper_q` (5% and 95% by
default), which must be among the quantiles in the table you pass. To overlay observed
data, pass a DataFrame as `data` with a `date` column and a column named like the
variable you plot, and set `show_data=True`, as in {doc}`calibration`.

## Several plots in one figure

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_quantiles(results.get_quantiles_compartments(), columns="Infected_total", ax=axes[0], title="Infected")
plot_quantiles(results.get_quantiles_transitions(), columns="Susceptible_to_Infected_total", ax=axes[1], title="New infections")
```

## Other plots

- Populations and contact matrices: see {doc}`populations`.
- Contact intensity over time: see {doc}`interventions`.
- Posterior distributions and model fit: see {doc}`calibration`.

For plots the built-in functions don't cover, get the data out as a table with
`results.to_dataframe()` (see {doc}`simulations`) and use matplotlib or seaborn directly.
