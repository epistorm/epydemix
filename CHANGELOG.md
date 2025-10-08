# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.0.2] – 2025-10-08

### [Unreleased]
### Added

* Custom multinomial sampling implementation: replaces `numpy.random.multinomial()` to improve the calculation of transition probabilities when hazard correction is applied.

  * Users can enable or disable hazard correction using the `use_hazard_correction` argument in `simulate()` and `EpiModel.run_simulations()`.
  * Transition probability functions now return *rates* instead of *risks*, as the conversion is automatically handled during multinomial sampling.
* Support for reproducible random generation: both `simulate()` and `EpiModel.run_simulations()` now include an `rng` argument accepting a `numpy.random.Generator` object.

  * By default, it is set to `None`, in which case `numpy.random.default_rng()` is used.
  * Users can supply a custom generator to ensure reproducibility.
* Expanded `simulate()` arguments to optimize the execution time of `EpiModel.run_simulations()`.
* `ABCSampler.run_projections()` improvement: now incorporates ABC weights when sampling parameter sets from the approximate posterior distribution.
* Improved `epydemix.visualization.plot_quantiles()`: added the `data_date_column` argument (default: `"date"`) to allow users to specify the name of the date column in the provided data frame.
* Added a new utils function to create initial conditions dictionary (`utils.get_initial_conditions_dict`).

### Changed

* Internal handling of transition probability functions adjusted to reflect the new multinomial sampling mechanism (e.g., `compute_mediated_transition_probability()` and `compute_spontaneous_transition_probability()`). Advanced users defining custom transition types should review their probability functions accordingly.

### Tutorials

* Added three new tutorials:

  * Modeling of multiple pathogen strains using a two-virus SIR-like model.
  * Implementation of a realistic vaccination campaign rollout, including age-specific dose administration and a new transition type for vaccinations.
  * Speeding up simulations and calibration using `multiprocess`.

### Data

* (TODO) Updated U.S. national and state population data in the **epydemix_data** repository using the latest ACS Community Survey estimates.

### Compatibility

* These updates do **not** introduce breaking changes. Existing code written for previous versions remains compatible.
