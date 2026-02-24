# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Changed

* Added support for two new demographic attributes: `"sex"` and `"race_ethnicity"`, with the same folder structure as `"age"`.
* Added `"litvinova_2025"` as a new contact source. For `"age"`, it uses the same mapping as `"prem"`. For `"sex"` and `"race_ethnicity"`, it is the only available contact source.
* Updated default `data_version` from `"vtest"` to `"vtest-epistorm-mix"` across `load_epydemix_population()`, `get_available_locations()`, and `EpiModel`.
* Updated `supported_contacts_sources` defaults to include `"sex"` and `"race_ethnicity"` keys with `["litvinova_2025"]`, and added `"litvinova_2025"` to the `"age"` sources list.
* Updated data paths to match restructured `epydemix-data` repository: data now lives under `data/{attribute}/`, demographic file renamed from `age_distribution.csv` to `population.csv`, contact matrix files no longer use `contacts_matrix_` prefix, and `locations.csv` moved to `data/{attribute}/locations.csv`.
* Added `attribute` parameter (default `"age"`) to `load_epydemix_population()`, `get_available_locations()`, and `EpiModel` to support the new attribute layer in the data directory structure.
* Added `data_version` parameter (default `"vtest"`) to `load_epydemix_population()`, `get_available_locations()`, and `EpiModel` to allow pinning the `epydemix-data` repository to a specific git tag. Replaces the old `path_to_data_github` URL parameter.
* Changed `supported_contacts_sources` from `List[str]` to `Dict[str, List[str]]` (keyed by attribute) in `load_epydemix_population()` and `EpiModel`, so each attribute can define its own set of valid contact sources.
* Demographic and contact matrix aggregation logic is now only applied when `attribute == "age"`. Non-age attributes use raw data without aggregation.
* Migrated linting and formatting tooling to [Ruff](https://docs.astral.sh/ruff/), replacing the previous linting setup.
* Simplified `get_available_locations()` to always fetch from remote GitHub URL, removing the `path_to_data` parameter. Now only accepts `attribute` and `data_version` parameters.

### Added

* Added `default_population_size` parameter (default `100000`) to `EpiModel` to allow configuring the size of the default population when `use_default_population=True`.
* Added per-simulation time/budget checks to ABC-SMC (`run_smc`): `_initialize_particles` and `_run_smc_generation` now accept `start_time`, `max_time`, `total_simulations_budget`, and `n_simulations` parameters, returning `None` when interrupted mid-generation. `run_smc` handles `None` by discarding the incomplete generation and keeping the last fully completed one, preventing indefinite overshooting when the acceptance rate drops near zero.
* Added `verbose` parameter to `_check_stopping_conditions` to suppress duplicate log messages from inner loops.
* Added tests for ABC-SMC with time limit, budget limit, generation-0 interruption, verbose output, `minimum_epsilon` stopping, and no-limits backward compatibility.
* Added `ignore_nan` parameter to quantile computation methods in `CalibrationResults` (`_compute_quantiles()`, `get_calibration_quantiles()`, `get_projection_quantiles()`) and `SimulationResults` (`get_quantiles()`, `get_quantiles_transitions()`, `get_quantiles_compartments()`) to handle NaN values from epidemic start date priors. Uses `np.nanquantile` when enabled, with warnings for variables exceeding 50% NaN values.
* Comprehensive test coverage for the new `ignore_nan` functionality.
* Added `variables` parameter to trajectory and quantile methods in `CalibrationResults` (`get_calibration_trajectories()`, `get_projection_trajectories()`, `get_calibration_quantiles()`, `get_projection_quantiles()`) and `SimulationResults` (`get_quantiles()`, `get_quantiles_transitions()`, `get_quantiles_compartments()`) to filter variables before array stacking, reducing memory usage.
* Added a `CONTRIBUTING.md` guide for new contributors.
* Added a CI workflow (`.github/workflows/ci.yml`) and pre-commit configuration (`.pre-commit-config.yaml`) for automated linting checks.
* Added `dev-requirements.txt` with development dependencies.
* Improved `plot_population()`: bar labels now use human-readable suffixes (K/M/B) for absolute numbers and append `%` for percentages by default. Changed default `xlabel` from `"Age group"` to `"Demographic group"`.

### Tutorials

* Added Tutorial 11: Using [Epistorm-Mix](https://www.epistorm.org/data/epistorm-mix) Contact Matrices, demonstrating the new `sex` and `race_ethnicity` demographic attributes based on [Litvinova et al. (2025)](https://www.medrxiv.org/content/10.1101/2025.11.20.25340662v1), with visualization of contact matrices, comparison of mean contacts across demographic groups, and SIR simulations with attack rate analysis.

### Data

* Added two new demographic attributes in **epydemix-data**: `"sex"` and `"race_ethnicity"`, with population and contact matrix data for the United States.
* Added `"litvinova_2025"` contact matrices for all three attributes (`age`, `sex`, `race_ethnicity`).

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
