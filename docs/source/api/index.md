# API reference

The public API, grouped by task. The most used entry points can be imported from the
top-level package:

```python
from epydemix import EpiModel, load_predefined_model, simulate
```

| Page | Contents |
|---|---|
| {doc}`model` | `EpiModel`, predefined models, `simulate`, simulation results |
| {doc}`population` | `Population` and loading population data |
| {doc}`calibration` | `ABCSampler`, calibration results, distance functions |
| {doc}`visualization` | plotting functions |
| {doc}`io` | reading and writing result bundles, the format of the command-line interface |
| {doc}`utils` | lower-level functions, not needed for normal use |

```{toctree}
:maxdepth: 1
:hidden:

model
population
calibration
visualization
io
utils
```
