# Interventions

There are two ways to change what happens during part of a simulation:

- **reduce contacts** in one setting (school closure, remote work) with
  {meth}`~epydemix.model.epimodel.EpiModel.add_intervention`;
- **change a parameter** for a period with
  {meth}`~epydemix.model.epimodel.EpiModel.override_parameter`.

Both take a start and an end date and apply on those dates inclusive.

## Reducing contacts

An intervention acts on one contact layer of the {doc}`population <populations>`.

```python
from epydemix import load_predefined_model
from epydemix.population import load_epydemix_population

model = load_predefined_model("SEIR", transmission_rate=0.02)
model.set_population(load_epydemix_population("Italy"))

baseline = model.copy()   # keep a copy without interventions to compare against

model.add_intervention(
    layer_name="school",
    start_date="2024-03-01",
    end_date="2024-05-31",
    reduction_factor=0.2,
    name="school closure",
)
model.add_intervention(
    layer_name="work",
    start_date="2024-03-01",
    end_date="2024-05-31",
    reduction_factor=0.5,
    name="remote work",
)
```

:::{important}
`reduction_factor` is the factor the contacts are **multiplied by**: `0.2` leaves 20% of
the usual contacts (an 80% reduction), and `1.0` changes nothing.
:::

Instead of a factor you can pass `new_matrix`, a full contact matrix to use for that
layer during the intervention.

Running both models with the same settings shows the effect:

```python
from epydemix.visualization import plot_quantiles

settings = dict(start_date="2024-01-01", end_date="2024-09-30", Nsim=50)
ax = plot_quantiles(
    baseline.run_simulations(**settings).get_quantiles_compartments(),
    columns="Infected_total", colors="grey", labels="no intervention",
)
plot_quantiles(
    model.run_simulations(**settings).get_quantiles_compartments(),
    columns="Infected_total", colors="teal", labels="school closure + remote work", ax=ax,
)
ax.legend(loc="upper right")
```

```{image} ../_static/interventions.png
:alt: Infected over time with and without the interventions
:width: 650px
```

{func}`~epydemix.visualization.plotting.plot_spectral_radius` shows how the overall
intensity of contacts changes over time, with the interventions shaded:

```python
from epydemix.visualization import plot_spectral_radius

plot_spectral_radius(model)
```

```{image} ../_static/spectral_radius.png
:alt: Spectral radius of the contact matrix over time, dropping during the interventions
:width: 650px
```

`model.clear_interventions()` removes them all.

## Changing a parameter for a period

```python
model.override_parameter(
    start_date="2024-03-01",
    end_date="2024-05-01",
    parameter_name="transmission_rate",
    value=0.012,
)
```

Between those dates the parameter takes the new value, and goes back to its usual value
afterwards. The value can be a number or an array, with the same shapes as any other
{doc}`parameter <models>`. Overrides work on named parameters, so the transition must
refer to the parameter by name (alone or inside an expression), not use a bare number.

`model.clear_overrides()` removes all overrides, and
`model.delete_override("transmission_rate")` those of one parameter.

The {doc}`interventions tutorial <../tutorials/03_Modeling_Interventions>` goes through
a complete example.
