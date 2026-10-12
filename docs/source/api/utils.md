# Lower level

These functions are used internally. They are documented for people extending epydemix
and are not needed for normal use.

## Simulation engine

```{eval-rst}
.. autofunction:: epydemix.model.epimodel.stochastic_simulation
.. autofunction:: epydemix.model.epimodel.compute_spontaneous_transition_rate
.. autofunction:: epydemix.model.epimodel.compute_mediated_transition_rate
.. autofunction:: epydemix.model.epimodel.evaluate_rate_expression
.. autofunction:: epydemix.model.epimodel.validate_transition_function
```

## Building predefined models

```{eval-rst}
.. autofunction:: epydemix.model.predefined_models.create_sir
.. autofunction:: epydemix.model.predefined_models.create_seir
.. autofunction:: epydemix.model.predefined_models.create_sis
.. autofunction:: epydemix.model.predefined_models.create_seiar
.. autofunction:: epydemix.model.predefined_models.add_waning_immunity
.. autofunction:: epydemix.model.predefined_models.add_vaccination
.. autofunction:: epydemix.model.predefined_models.add_outcome
```

## Population helpers

```{eval-rst}
.. autofunction:: epydemix.population.population.aggregate_matrix
.. autofunction:: epydemix.population.population.aggregate_demographic
.. autofunction:: epydemix.population.population.map_age_groups_to_idx
```

## ABC-SMC building blocks

```{eval-rst}
.. automodule:: epydemix.utils.abc_smc_utils
   :members:
```

## General utilities

```{eval-rst}
.. automodule:: epydemix.utils.utils
   :members:
   :exclude-members: njit
```
