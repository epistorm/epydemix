# epydemix

**epydemix** is a Python package for epidemic modeling. Define a compartmental model in a
few lines, run stochastic simulations on real-world populations and contact patterns,
and calibrate the model to data with Approximate Bayesian Computation (ABC).

[Website](https://www.epydemix.org/) ·
[GitHub](https://github.com/epistorm/epydemix) ·
[Paper](https://doi.org/10.1371/journal.pcbi.1013735) ·
[PyPI](https://pypi.org/project/epydemix/)

- **Any compartmental model**: SIR, SEIR and friends are built in, and you can define
  your own compartments, transitions and transition rules.
- **Real populations**: age-structured populations and contact matrices (home, school,
  work, community) for more than 400 locations, from
  [epydemix-data](https://github.com/epistorm/epydemix-data/).
- **Interventions**: reduce contacts in a setting or change a parameter over a period.
- **Calibration**: ABC-SMC, rejection and top-fraction sampling, then projections from
  the posterior.
- **Plots**: quantile bands, trajectories, posteriors, contact matrices.

## Installation

```bash
pip install epydemix
```

or, with conda:

```bash
conda install -c conda-forge epydemix
```

Requires Python 3.8 or later. For faster simulations, `pip install "epydemix[numba]"`
adds an optional JIT-compiled speedup; epydemix runs fine without it, just slower.

## Quickstart

```python
from epydemix import EpiModel
from epydemix.visualization import plot_quantiles

# Define a basic SIR model
model = EpiModel(name="SIR Model", compartments=["S", "I", "R"])
model.add_transition(source="S", target="I", params=(0.3, "I"), kind="mediated")
model.add_transition(source="I", target="R", params=0.1, kind="spontaneous")

# Run 100 stochastic simulations
results = model.run_simulations(start_date="2024-01-01", end_date="2024-04-10", Nsim=100)

# Median and 90% interval of each compartment over time
df_quantiles = results.get_quantiles_compartments()
plot_quantiles(df_quantiles, columns=["I_total", "S_total", "R_total"])
```

```{image} _static/quickstart.png
:alt: Median and 90% interval of S, I and R over time
:width: 650px
```

To try it without installing anything, open the
[first tutorial in Colab](https://colab.research.google.com/github/epistorm/epydemix/blob/main/tutorials/01_Model_Definition_and_Simulation.ipynb).

```{toctree}
:maxdepth: 2
:hidden:
:caption: User guide

guide/models
guide/populations
guide/simulations
guide/interventions
guide/calibration
guide/plotting
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Learn

tutorials/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

api/index
changelog
citing
```

## Contents

**User guide**

- {doc}`guide/models`: compartments, transitions, parameters that vary in time or by age
  group, predefined models, custom transition rules
- {doc}`guide/populations`: loading a real population and its contact matrices, or
  building your own
- {doc}`guide/simulations`: running simulations, initial conditions, reproducibility and
  getting the results out
- {doc}`guide/interventions`: reducing contacts and changing parameters over a period
- {doc}`guide/calibration`: fitting a model to data with ABC and projecting forward
- {doc}`guide/plotting`: the built-in plots

**Learn**

- {doc}`tutorials/index`: twelve worked notebooks, from a first model to a COVID-19 case
  study

**Reference**

- {doc}`api/index`: every public class and function
- {doc}`changelog`
- {doc}`citing`
