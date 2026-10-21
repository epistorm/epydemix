# Populations and contacts

Every model has a {class}`~epydemix.population.population.Population`: how many people
are in each demographic group, and how much the groups are in contact with each other.
Contacts are split into **layers** (home, school, work, community), so that
{doc}`interventions <interventions>` can act on one setting at a time.

A model created without a population gets a default one: 100,000 people in a single
group.

## Loading a real population

{func}`~epydemix.population.population.load_epydemix_population` downloads the age
structure and contact matrices of a location from
[epydemix-data](https://github.com/epistorm/epydemix-data/).

```python
from epydemix import load_predefined_model
from epydemix.population import load_epydemix_population

population = load_epydemix_population("Italy")
print(population)
```

```text
Population(name='Italy')
Demographic groups: 5 groups
Population distribution:
  - 0-4: 2042699 individuals
  - 5-19: 7997107 individuals
  - 20-49: 20936936 individuals
  - 50-64: 13953616 individuals
  - 65+: 14504782 individuals
Contact matrices: 4 layers
Available layers:
  - school
  - work
  - home
  - community
```

Attach it to a model with {meth}`~epydemix.model.epimodel.EpiModel.set_population`:

```python
model = load_predefined_model("SEIR")
model.set_population(population)
```

or load and attach in one step with `model.import_epydemix_population("Italy")`.

Once the population has several groups, every result is available per group as well as
in total: `Infected_0-4`, `Infected_5-19`, ..., `Infected_total`.

## Which locations are available

```python
from epydemix.population import get_available_locations

locations = get_available_locations()
locations.head(3)
```

```text
      location  level iso_code primary_contact_source      contact_sources               population_source
0  Afghanistan      0       AF              prem_2021            prem_2021  https://population.un.org/wpp/
1      Albania      0       AL              prem_2021  prem_2021|prem_2017  https://population.un.org/wpp/
2      Algeria      0       DZ              prem_2021  prem_2021|prem_2017  https://population.un.org/wpp/
```

`level` is 0 for countries, 1 for states or provinces and 2 for counties; pass
`level=0` to list only countries.

## Options when loading

```python
population = load_epydemix_population(
    "Italy",
    contacts_source="mistry_2021",       # which contact study to use
    layers=["home", "school"],           # load only some layers
    age_group_mapping={                  # custom age groups
        "0-19": [str(age) for age in range(0, 20)],
        "20-64": [str(age) for age in range(20, 65)],
        "65+": [str(age) for age in range(65, 84)] + ["84+"],
    },
)
population.Nk_names
# ['0-19' '20-64' '65+']
```

- **`contacts_source`**: `"prem_2017"`, `"prem_2021"`, `"mistry_2021"` or
  `"litvinova_2025"`. Each location has a default (the `primary_contact_source` column
  above), and not every source covers every location. The ages you can group depend on
  the source: `mistry_2021` has single years of age up to `"84+"`.
- **`path_to_data`**: a local copy of epydemix-data, to work offline.
- **`data_version`**: the epydemix-data release to use.
- **`attribute`**: `"age"` by default; `"sex"` and `"race_ethnicity"` are available for
  some locations, see the
  {doc}`Epistorm-Mix tutorial <../tutorials/11_Epistorm_Mix_Matrices>`.

## Building your own population

```python
import numpy as np
from epydemix.population import Population

population = Population(name="two groups")
population.add_population(Nk=[30_000, 70_000], Nk_names=["children", "adults"])
population.add_contact_matrix(np.array([[8.0, 3.0], [1.5, 6.0]]), layer_name="all")
```

Entry `[i, j]` of a contact matrix is the average number of daily contacts a person in
group `i` has with people in group `j`. Add one matrix per layer.

## Looking at a population

```python
from epydemix.visualization import plot_contact_matrix, plot_population

plot_population(population)
plot_contact_matrix(population, "school")
```

```{image} ../_static/population.png
:alt: Population of Italy by age group, and the school contact matrix
:width: 100%
```

`population.Nk` holds the group sizes, `population.contact_matrices` the matrices by
layer, and `population.mean_contacts` the average number of contacts in each layer.
