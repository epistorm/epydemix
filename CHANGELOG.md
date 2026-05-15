# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.2.1] - 2026-05-15

### Added

* New `SEIAR` backbone model in `load_predefined_model`: adds an **Asymptomatic** infectious compartment branching from Exposed. New parameters: `asymptomatic_fraction`, `asymptomatic_recovery_rate`, `asymptomatic_relative_infectivity`.
* Three orthogonal modular extensions that can be composed on top of any backbone via keyword arguments to `load_predefined_model`:
  * `waning_immunity=True` — adds an **R → S** spontaneous transition (`waning_rate`, default `1/365`). Not compatible with `SIS`.
  * `vaccination=True` — adds a **Vaccinated** compartment with `S → Vaccinated` (rate `vaccination_rate`) and `Vaccinated → Infected` at reduced rate `transmission_rate * (1 - vaccine_efficacy)`.
  * `outcome="deaths"` — adds a **Dead** compartment with an `Infected → Dead` spontaneous transition (`mortality_rate`).
  * `outcome="hospitalization"` — adds a **Hospitalized** compartment with `Infected → Hospitalized` (`hospitalization_rate`) and `Hospitalized → Recovered` (`hospitalization_recovery_rate`). Not compatible with `SIS`.
* `SUPPORTED_MODELS` updated to `["SIR", "SEIR", "SIS", "SEIAR"]`.
* All new rate parameters accept scalars, 1D time-varying arrays of shape `(T,)`, or 2D age-stratified arrays of shape `(T, G)`, consistent with the existing parameter system.
* Tests for all new backbones and modules in `tests/test_predefined_models.py`, bringing `predefined_models.py` to 100% coverage.

### Fixed

* `create_default_initial_conditions` in `epimodel.py` now correctly handles models with duplicate mediated-transition sources (e.g. SEIAR, where `S` appears twice) and models where module compartments like `Vaccinated` or `Exposed` are transition targets. The method now uses a three-level strategy: (1) seed residual population into sources with no inflow at all; (2) if all sources have inflow (e.g. SIRS where waning makes `S` a target), fall back to the source with the most outgoing mediated transitions, preferring non-mediated-targets. This ensures `Susceptible` always receives the bulk of the population and accumulator compartments (`Vaccinated`, `Exposed`, `Hospitalized`, `Dead`) always start at zero when no explicit initial conditions are provided.

### Tutorials

* Added Tutorial 12: Predefined Epidemic Models — demonstrates all four backbone models and the three modular extensions (waning immunity, vaccination, outcome tracking), with side-by-side comparisons and an example of time-varying parameter overrides post-construction.

---

## [1.2.0] - 2026-05-12

### Added

* [Numba](https://numba.pydata.org/) JIT compilation for the multinomial probability computation (`_multinomial_probs` in `utils.py`). The probability kernel is compiled at import time via `@njit`, eliminating interpreter overhead on the hot simulation path.
* Added `numba>=0.57.0` as a dependency in `requirements.txt`, `setup.py`, and `pyproject.toml`.
* Support for US county-level geographies (~3,000 locations) from the `epydemix-data` repository (now at `v1.2.0`). Counties are stored using folder names following the `Country__State__County_Name` convention (e.g., `United_States__Alabama__Autauga_County`).
* `locations.csv` now includes two new columns: `level` (integer: 0=country, 1=state/province/region, 2=US county) and `iso_code` (ISO 3166-1 alpha-2 for countries such as `US`; ISO 3166-2 for states such as `US-AL`; 5-digit FIPS code for US counties such as `01001`).
* Optional `level` parameter to `get_available_locations()` to filter the returned DataFrame to a specific geographic level (0, 1, or 2). Silently ignored when the loaded `locations.csv` lacks a `level` column, preserving backward compatibility with `data_version="v1.1.0"`.

### Changed

* Optimized `compute_spontaneous_transition_rate()` and `compute_mediated_transition_rate()` in `epimodel.py`: when the rate expression is a plain parameter name that already exists in the parameters dictionary, the value is looked up directly instead of triggering a full `evaluate()` call with a `deepcopy` of the parameter environment. This avoids unnecessary copying on the hot simulation path.
* Location names in `epydemix-data` now use `_` for spaces within a single geographic name and `__` as a separator between hierarchy levels (e.g., `United_States__Alabama__Autauga_County`). Country and state names present in `v1.1.0` have been renamed consistently (spaces replaced by `_`).
* Default `data_version` bumped from `"v1.1.0"` to `"v1.2.0"` across `load_epydemix_population()`, `get_available_locations()`, `EpiModel.__init__()`, `EpiModel._load_or_create_population()`, and `EpiModel.import_epydemix_population()`.
* `validate_population_name()` error message now includes a hint about the `_` / `__` naming convention and directs users to call `get_available_locations()` to browse valid names.

---

## [1.1.0] - 2026-02-24

### Changed

* Added support for two new demographic attributes: `"sex"` and `"race_ethnicity"`, with the same folder structure as `"age"`.
* Added `"litvinova_2025"` as a new contact source. For `"age"`, it uses the same mapping as `"prem"`. For `"sex"` and `"race_ethnicity"`, it is the only available contact source.
* Updated default `data_version` from `"vtest"` to `"v1.1.0"` across `load_epydemix_population()`, `get_available_locations()`, and `EpiModel`.
* Updated `supported_contacts_sources` defaults to include `"sex"` and `"race_ethnicity"` keys with `["litvinova_2025"]`, and added `"litvinova_2025"` to the `"age"` sources list.
* Updated data paths to match restructured `epydemix-data` repository: data now lives under `data/{attribute}/`, demographic file renamed from `age_distribution.csv` to `population.csv`, contact matrix files no longer use `contacts_matrix_` prefix, and `locations.csv` moved to `data/{attribute}/locations.csv`.
* Added `attribute` parameter (default `"age"`) to `load_epydemix_population()`, `get_available_locations()`, and `EpiModel` to support the new attribute layer in the data directory structure.
* Added `data_version` parameter (default `"v1.1.0"`) to `load_epydemix_population()`, `get_available_locations()`, and `EpiModel` to allow pinning the `epydemix-data` repository to a specific git tag. Replaces the old `path_to_data_github` URL parameter.
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
