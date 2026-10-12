# Building a model

A model is a set of **compartments** (the states a person can be in), **transitions**
between them, and **parameters** that set how fast the transitions happen. All of it
lives in an {class}`~epydemix.model.epimodel.EpiModel`.

## Compartments, transitions and parameters

```python
from epydemix import EpiModel

model = EpiModel(
    compartments=["S", "I", "R"],
    parameters={"beta": 0.3, "mu": 0.1},
)
model.add_transition(source="S", target="I", params=("beta", "I"), kind="mediated")
model.add_transition(source="I", target="R", params="mu", kind="spontaneous")
```

There are two built-in kinds of transition:

| Kind | `params` | Meaning |
|---|---|---|
| `"spontaneous"` | a rate, e.g. `"mu"` | Each person in `source` moves to `target` at that rate per day. Recovery, incubation and waning are spontaneous. |
| `"mediated"` | `(rate, agent)`, e.g. `("beta", "I")` | People in `source` move to `target` through contact with people in the `agent` compartment. Infection is mediated. |

For a mediated transition, the rate for age group $i$ is
$\beta \sum_j C_{ij} \, I_j / N_j$, where $C$ is the contact matrix of the
{doc}`population <populations>`.

`print(model)` gives a summary of the compartments, transitions, parameters and
population.

## Ways to write a rate

The rate part of `params` can be:

- **a number**: `params=0.1`
- **the name of a parameter**: `params="mu"`. This is what you need for
  {doc}`calibration <calibration>` and parameter overrides, which work on named
  parameters.
- **an expression of parameters**: any combination using `+`, `-`, `*`, `/`, `**` and
  parentheses.

```python
model = EpiModel(
    compartments=["S", "V", "I", "R"],
    parameters={"beta": 0.3, "mu": 0.1, "VE": 0.8, "nu": 0.005},
)
model.add_transition("S", "I", params=("beta", "I"), kind="mediated")
model.add_transition("V", "I", params=("beta * (1 - VE)", "I"), kind="mediated")
model.add_transition("S", "V", params="nu", kind="spontaneous")
model.add_transition("I", "R", params="mu", kind="spontaneous")
```

## Parameters that change in time or by age group

A parameter value can be a single number or an array. With `T` time steps and `G`
demographic groups:

| Value | Meaning |
|---|---|
| a number | constant in time, same for every group |
| array of length `T` | one value per time step, same for every group |
| array of shape `(1, G)` | constant in time, one value per group |
| array of shape `(T, G)` | one value per time step and group |

```python
import numpy as np
import pandas as pd

# A seasonal transmission rate: one value per simulated day
T = len(pd.date_range("2024-01-01", "2024-04-10"))
model.add_parameter("beta", 0.3 * (1 + 0.2 * np.cos(2 * np.pi * np.arange(T) / 365)))
```

To change a parameter only during a given period, use
{meth}`~epydemix.model.epimodel.EpiModel.override_parameter`, described in
{doc}`interventions`.

## Predefined models

{func}`~epydemix.model.predefined_models.load_predefined_model` returns ready-made
models: `"SIR"`, `"SEIR"`, `"SIS"` and `"SEIAR"` (with an asymptomatic compartment).

```python
from epydemix import load_predefined_model

model = load_predefined_model("SEIR", transmission_rate=0.3, incubation_rate=0.2, recovery_rate=0.1)
model.compartments
# ['Susceptible', 'Exposed', 'Infected', 'Recovered']
```

Optional modules can be combined on top of any of them:

| Argument | Adds |
|---|---|
| `waning_immunity=True` | Recovered → Susceptible at `waning_rate` |
| `vaccination=True` | a Vaccinated compartment, with `vaccination_rate` and `vaccine_efficacy` |
| `outcome="deaths"` | a Dead compartment; `mortality_rate` is the fraction of infections that die |
| `outcome="hospitalization"` | a Hospitalized compartment; `hospitalization_rate` is the fraction hospitalized |

```python
model = load_predefined_model("SIR", waning_immunity=True, vaccination=True)
model.compartments
# ['Susceptible', 'Infected', 'Recovered', 'Vaccinated']
```

See the {doc}`predefined models tutorial <../tutorials/12_Predefined_Models>` for all
combinations.

## Custom transition rules

If neither built-in kind fits, register your own. A rate function receives the
transition's `params` and a `data` dictionary describing the current state, and returns
the rate (a number, or one value per demographic group).

```python
import numpy as np

def behavioral_rate(params, data):
    """People adopt protective behaviour faster when there are more infected."""
    rate = data["parameters"][params[0]][data["t"]]
    infected = data["pop"][data["comp_indices"][params[1]]]
    return rate * (1 - np.exp(-np.sum(infected) / 1000))

model = EpiModel(
    compartments=["S", "SB", "I", "R"],
    parameters={"beta": 0.3, "mu": 0.1, "r": 0.3, "beta_B": 0.05},
)
model.register_transition_kind(kind="behavioral", function=behavioral_rate)

model.add_transition("S", "I", params=("beta", "I"), kind="mediated")
model.add_transition("SB", "I", params=("r * beta", "I"), kind="mediated")
model.add_transition("S", "SB", params=("beta_B", "I"), kind="behavioral")
model.add_transition("I", "R", params="mu", kind="spontaneous")
```

The `data` dictionary contains:

| Key | Content |
|---|---|
| `parameters` | every parameter as an array of shape `(T, G)` |
| `t` | index of the current time step |
| `pop` | current population, shape `(compartments, G)` |
| `comp_indices` | compartment name → row of `pop` |
| `pop_sizes` | total population of each group |
| `contact_matrix` | contact matrices of the current date, by layer, plus `"overall"` |
| `dt` | length of the time step, in days |

A custom kind must be registered before a transition uses it; an unknown `kind` in
`add_transition` raises an error. A rate function should only read `data`, not modify
it. The {doc}`advanced features tutorial <../tutorials/06_Advanced_Modeling_Features>`
has a full example.

## Editing a model

```python
model = load_predefined_model("SEIAR")

# Susceptible -> Exposed exists twice in SEIAR: driven by Infected and by Asymptomatic.
# Remove only the one driven by Asymptomatic.
model.remove_transition("Susceptible", "Exposed", agent="Asymptomatic")

# Remove a compartment together with every transition that involves it
model = load_predefined_model("SEIR", vaccination=True)
model.remove_compartment("Vaccinated")
```

{meth}`~epydemix.model.epimodel.EpiModel.remove_transition` without `agent` or `kind`
removes every transition between the two compartments.

To compare scenarios, copy a model and change the copy. The two are fully independent.

```python
baseline = load_predefined_model("SEIR")
scenario = baseline.copy(name="higher transmission")
scenario.add_parameter("transmission_rate", 0.5)
```
