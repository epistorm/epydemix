# Bundles (I/O)

```python
from epydemix.io import (
    add_figure_to_manifest, compare_bundles, inspect_bundle,
    load_bundle, load_bundle_dataframe, save_bundle,
)
```

Reading and writing `.epx` bundles, the result format of the
{doc}`command-line interface <../guide/cli>`. Simulation and calibration results can also
be saved as bundles from Python with
{meth}`SimulationResults.save <epydemix.model.simulation_results.SimulationResults.save>`
and
{meth}`CalibrationResults.save <epydemix.calibration.calibration_results.CalibrationResults.save>`.

## Saving and loading

```{eval-rst}
.. automodule:: epydemix.io.bundle
   :members: save_bundle, load_bundle, load_bundle_dataframe, add_figure_to_manifest
```

## Querying

```{eval-rst}
.. automodule:: epydemix.io.inspect
   :members: inspect_bundle, compare_bundles
```
