# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

* Added per-simulation time/budget checks to ABC-SMC (`run_smc`): `_initialize_particles` and `_run_smc_generation` now accept `start_time`, `max_time`, `total_simulations_budget`, and `n_simulations` parameters, returning `None` when interrupted mid-generation. `run_smc` handles `None` by discarding the incomplete generation and keeping the last fully completed one, preventing indefinite overshooting when the acceptance rate drops near zero.
* Added `verbose` parameter to `_check_stopping_conditions` to suppress duplicate log messages from inner loops.
* Added tests for ABC-SMC with time limit, budget limit, generation-0 interruption, verbose output, `minimum_epsilon` stopping, and no-limits backward compatibility.
* Added `ignore_nan` parameter to quantile computation methods in `CalibrationResults` (`_compute_quantiles()`, `get_calibration_quantiles()`, `get_projection_quantiles()`) and `SimulationResults` (`get_quantiles()`, `get_quantiles_transitions()`, `get_quantiles_compartments()`) to handle NaN values from epidemic start date priors. Uses `np.nanquantile` when enabled, with warnings for variables exceeding 50% NaN values.
* Comprehensive test coverage for the new `ignore_nan` functionality.
* Added `variables` parameter to trajectory and quantile methods in `CalibrationResults` (`get_calibration_trajectories()`, `get_projection_trajectories()`, `get_calibration_quantiles()`, `get_projection_quantiles()`) and `SimulationResults` (`get_quantiles()`, `get_quantiles_transitions()`, `get_quantiles_compartments()`) to filter variables before array stacking, reducing memory usage.

### Changed

* Migrated linting and formatting tooling to [Ruff](https://docs.astral.sh/ruff/), replacing the previous linting setup.
* Added a `CONTRIBUTING.md` guide for new contributors.
* Added a CI workflow (`.github/workflows/ci.yml`) and pre-commit configuration (`.pre-commit-config.yaml`) for automated linting checks.
* Added `dev-requirements.txt` with development dependencies.

### Fixed

* Fixed `TypeError: ufunc 'isnan' not supported` when `_compute_quantiles` encounters non-numeric arrays (e.g., dates) with `ignore_nan=True`. The method now skips non-numeric arrays in NaN checks and quantile computation loops.

---

## [1.0.2] – 2025-10-30

### Added

* Custom multinomial sampling implementation: replaces `numpy.random.multinomial()` to improve the calculation of transition probabilities.

  * Transition rate functions now return *rates* instead of *risks*, as the conversion is automatically handled during multinomial sampling.
  * Users can enable linear approximation to the probabilities using the `apply_linear_approximation` argument in `simulate()` and `EpiModel.run_simulations()`.
  
* Support for reproducible random generation: both `simulate()` and `EpiModel.run_simulations()` now include an `rng` argument accepting a `numpy.random.Generator` object.

  * By default, it is set to `None`, in which case `numpy.random.default_rng()` is used.
  * Users can supply a custom generator to ensure reproducibility.
* Expanded `simulate()` arguments to optimize the execution time of `EpiModel.run_simulations()`.
* `ABCSampler.run_projections()` improvement: now incorporates ABC weights when sampling parameter sets from the approximate posterior distribution.
* Improved `epydemix.visualization.plot_quantiles()`: added the `data_date_column` argument (default: `"date"`) to allow users to specify the name of the date column in the provided data frame.
* Added a new utils function to create initial conditions dictionary (`utils.get_initial_conditions_dict`).

### Changed

* Internal handling of transition rate functions adjusted to reflect the new multinomial sampling mechanism (e.g., `compute_mediated_transition_rate()` and `compute_spontaneous_transition_rate()`). Advanced users defining custom transition types should review their rate functions accordingly.

### Tutorials

* Added three new tutorials:

  * Modeling of multiple pathogen strains using a two-virus SIR-like model.
  * Implementation of a realistic vaccination campaign rollout, including age-specific dose administration and a new transition type for vaccinations.
  * Speeding up simulations and calibration using `multiprocess`.

### Data

* Updated U.S. national and state population data in the **epydemix_data** repository using the more recent estimates from the U.S. Census Bureau.

### Compatibility

* These updates do **not** introduce breaking changes. Existing code written for previous versions remains compatible.
