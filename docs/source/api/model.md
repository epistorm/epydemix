# Models and simulations

```python
from epydemix import EpiModel, load_predefined_model, simulate
from epydemix.model import SimulationResults, Transition
```

## EpiModel

```{eval-rst}
.. autoclass:: epydemix.model.epimodel.EpiModel
   :members:
```

## Predefined models

```{eval-rst}
.. autofunction:: epydemix.model.predefined_models.load_predefined_model
```

## Running a single simulation

```{eval-rst}
.. autofunction:: epydemix.model.epimodel.simulate
```

## Results

```{eval-rst}
.. autoclass:: epydemix.model.simulation_results.SimulationResults
   :members:

.. autoclass:: epydemix.model.simulation_output.Trajectory
   :members:
```

## Transition

```{eval-rst}
.. autoclass:: epydemix.model.transition.Transition
   :members:
```
