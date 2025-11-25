import copy
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.abc_smc_utils import (
    DefaultPerturbationContinuous,
    DefaultPerturbationDiscrete,
    sample_prior,
)
from .calibration_results import CalibrationResults
from .metrics import rmse


class ABCSampler:
    """
    Approximate Bayesian Computation (ABC) class implementing different ABC strategies.
    """

    def __init__(
        self,
        simulation_function: Callable,
        priors: Dict[str, Any],
        parameters: Dict[str, Any],
        observed_data: Any,
        distance_function: Callable = rmse,
    ):
        """Initialize ABC calibration."""
        self.simulation_function = simulation_function
        self.priors = priors
        self.parameters = parameters.copy()
        self.observed_data = {"data": observed_data}
        self.distance_function = distance_function
        self.param_names = list(priors.keys())
        self.results = None

        # Separate continuous and discrete parameters
        self.continuous_params = [
            name for name in self.param_names if hasattr(priors[name], "pdf")
        ]
        self.discrete_params = [
            name for name in self.param_names if name not in self.continuous_params
        ]

    def calibrate(self, strategy: str = "smc", **kwargs) -> CalibrationResults:
        """Run calibration using the specified strategy.

        This function allows the user to run Approximate Bayesian Computation (ABC) calibration
        using one of the available strategies: Sequential Monte Carlo (SMC), rejection sampling,
        or top fraction selection. The appropriate method is selected based on the `strategy`
        argument, and additional keyword arguments (`**kwargs`) are passed to the corresponding
        function.

        ### Available Strategies:
        - `"smc"`: Uses Sequential Monte Carlo (ABC-SMC) for calibration.
        - `"rejection"`: Uses ABC rejection sampling.
        - `"top_fraction"`: Selects the best fraction of simulated results.

        ### Arguments:
        - **strategy** (`str`, default: `"smc"`):
        Specifies the calibration strategy. Must be one of `{"smc", "rejection", "top_fraction"}`.
        - **kwargs**: Additional parameters depending on the chosen strategy.

        ### Strategy-Specific Arguments:

        #### `"smc"` (Sequential Monte Carlo)
        - `num_particles` (`int`, default: `1000`): Number of particles (samples) per generation.
        - `num_generations` (`int`, default: `10`): Number of generations for the ABC-SMC process.
        - `epsilon_schedule` (`Optional[List[float]]`, default: `None`): Predefined schedule for epsilon values.
        - `epsilon_quantile_level` (`float`, default: `0.5`): Quantile level to adapt epsilon if no schedule is provided.
        - `minimum_epsilon` (`Optional[float]`, default: `None`): Minimum allowable epsilon value.
        - `max_time` (`Optional[timedelta]`, default: `None`): Maximum allowed runtime.
        - `total_simulations_budget` (`Optional[int]`, default: `None`): Maximum number of allowed simulations.
        - `perturbations` (`Optional[Dict[str, Any]]`, default: `None`): Perturbation kernels for parameters.
        - `verbose` (`bool`, default: `True`): Whether to print progress updates.

        #### `"rejection"` (ABC Rejection Sampling)
        - `epsilon` (`float`, default: `0.1`): Distance threshold for accepting samples.
        - `num_particles` (`int`, default: `1000`): Number of accepted samples.
        - `max_time` (`Optional[timedelta]`, default: `None`): Maximum allowed runtime.
        - `total_simulations_budget` (`Optional[int]`, default: `None`): Maximum number of allowed simulations.
        - `verbose` (`bool`, default: `True`): Whether to print progress updates.
        - `progress_update_interval` (`int`, default: `1000`): Interval at which progress updates are printed.

        #### `"top_fraction"` (ABC Top-Fraction Selection)
        - `top_fraction` (`float`, default: `0.05`): Fraction of best-fitting simulations to keep.
        - `Nsim` (`int`, default: `100`): Total number of simulations to run.
        - `verbose` (`bool`, default: `True`): Whether to print progress updates.

        ### Returns:
        - `CalibrationResults`: A deep copy of the results from the chosen calibration strategy.

        ### Raises:
        - `ValueError`: If an unknown strategy is specified.

        Example Usage:
        ```python
        # Run SMC calibration with custom parameters
        results = model.calibrate(strategy="smc", num_particles=500, num_generations=15)

        # Run rejection sampling with a different epsilon threshold
        results = model.calibrate(strategy="rejection", epsilon=0.05, num_particles=2000)

        # Run top fraction selection with a different fraction
        results = model.calibrate(strategy="top_fraction", top_fraction=0.1, Nsim=500)
        ```
        """
        strategies = {
            "smc": self.run_smc,
            "rejection": self.run_rejection,
            "top_fraction": self.run_top_fraction,
        }

        if strategy not in strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be one of {list(strategies.keys())}"
            )

        self.results = strategies[strategy](**kwargs)
        return copy.deepcopy(self.results)

    def run_smc(
        self,
        num_particles: int = 1000,
        num_generations: int = 10,
        epsilon_schedule: Optional[List[float]] = None,
        epsilon_quantile_level: float = 0.5,
        minimum_epsilon: Optional[float] = None,
        max_time: Optional[timedelta] = None,
        total_simulations_budget: Optional[int] = None,
        perturbations: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> CalibrationResults:
        """Run ABC-SMC calibration."""
        # Initialize perturbations if not provided
        if perturbations is None:
            perturbations = {
                param: (
                    DefaultPerturbationContinuous(param)
                    if param in self.continuous_params
                    else DefaultPerturbationDiscrete(param, self.priors[param])
                )
                for param in self.param_names
            }

        if verbose:
            print(
                f"Starting ABC-SMC with {num_particles} particles and {num_generations} generations"
            )

        # Run generations
        start_time = datetime.now()
        n_simulations = 0

        for gen in range(num_generations):
            start_generation_time = datetime.now()

            if gen == 0:
                epsilon = (
                    epsilon_schedule[0]
                    if epsilon_schedule is not None
                    else float("inf")
                )
                if verbose:
                    print(
                        f"\nGeneration {gen + 1}/{num_generations} (epsilon: {epsilon:.6f})"
                    )
                # Initialize particles, weights, distances, simulations
                new_gen = self._initialize_particles(num_particles, epsilon)
                n_simulations += new_gen["n_simulations"]

                # Store results for generation 0
                results = self._create_results(
                    "smc",
                    pd.DataFrame(
                        data={
                            self.param_names[i]: new_gen["particles"][:, i]
                            for i in range(len(self.param_names))
                        }
                    ),
                    new_gen["weights"],
                    new_gen["distances"],
                    new_gen["simulations"],
                )

                particles = new_gen["particles"]
                weights = new_gen["weights"]
                distances = new_gen["distances"]

            else:
                # Compute epsilon for this generation
                epsilon = (
                    epsilon_schedule[gen]
                    if epsilon_schedule is not None
                    else np.quantile(distances, epsilon_quantile_level)
                )

                if verbose:
                    print(
                        f"\nGeneration {gen + 1}/{num_generations} (epsilon: {epsilon:.6f})"
                    )

                # Update perturbations
                for perturbation in perturbations.values():
                    perturbation.update(particles, weights, self.param_names)

                # Run generation
                new_gen = self._run_smc_generation(
                    particles, weights, epsilon, num_particles, perturbations
                )
                n_simulations += new_gen["n_simulations"]

                # Store results
                results.posterior_distributions[gen] = pd.DataFrame(
                    data={
                        self.param_names[i]: new_gen["particles"][:, i]
                        for i in range(len(self.param_names))
                    }
                )
                results.distances[gen] = new_gen["distances"]
                results.weights[gen] = new_gen["weights"]
                results.selected_trajectories[gen] = new_gen["simulations"]

                # Update current generation
                particles = new_gen["particles"]
                weights = new_gen["weights"]
                distances = new_gen["distances"]

            if verbose:
                # Print generation information
                end_generation_time = datetime.now()
                elapsed_time = end_generation_time - start_generation_time
                formatted_time = f"{elapsed_time.seconds // 3600:02}:{(elapsed_time.seconds % 3600) // 60:02}:{elapsed_time.seconds % 60:02}"
                acceptance_rate = (
                    len(new_gen["particles"]) / new_gen["n_simulations"] * 100
                )
                print(
                    f"\tAccepted {len(new_gen['particles'])}/{new_gen['n_simulations']} (acceptance rate: {acceptance_rate:.2f}%)"
                )
                print(f"\tElapsed time: {formatted_time}")

            # Check stopping conditions
            if self._check_stopping_conditions(
                epsilon,
                minimum_epsilon,
                start_time,
                max_time,
                n_simulations,
                total_simulations_budget,
            ):
                break

        return results

    def run_rejection(
        self,
        epsilon: float = 0.1,
        num_particles: int = 1000,
        max_time: Optional[timedelta] = None,
        total_simulations_budget: Optional[int] = None,
        verbose: bool = True,
        progress_update_interval: int = 1000,
    ) -> CalibrationResults:
        """Run ABC rejection sampling."""
        simulations, distances = [], []
        sampled_params = {p: [] for p in self.param_names}

        start_time = datetime.now()
        n_simulations = 0
        last_print = 0  # For tracking progress updates

        if verbose:
            print(
                f"Starting ABC rejection sampling with {num_particles} particles and epsilon threshold {epsilon}"
            )

        while len(distances) < num_particles:
            # Check stopping conditions
            if self._check_stopping_conditions(
                None,
                None,
                start_time,
                max_time,
                n_simulations,
                total_simulations_budget,
            ):
                break

            # Sample and simulate
            params = self._sample_parameters()
            simulation = self._run_simulation(params)
            distance = self.distance_function(self.observed_data, simulation)
            n_simulations += 1

            if distance < epsilon:
                simulations.append(simulation)
                distances.append(distance)
                for i, p in enumerate(self.param_names):
                    sampled_params[p].append(params[i])

            # Print progress every progress_update_interval simulations if verbose
            if (
                verbose
                and n_simulations % progress_update_interval == 0
                and n_simulations != last_print
            ):
                last_print = n_simulations
                acceptance_rate = len(distances) / n_simulations * 100
                print(
                    f"\tSimulations: {n_simulations}, Accepted: {len(distances)}, "
                    f"Acceptance rate: {acceptance_rate:.2f}%"
                )

        if verbose:
            print(
                f"\tFinal: {len(distances)} particles accepted from {n_simulations} simulations "
                f"({len(distances) / n_simulations * 100:.2f}% acceptance rate)"
            )

        return self._create_results(
            "rejection",
            pd.DataFrame(sampled_params),
            np.ones(len(distances)) / len(distances),
            np.array(distances),
            simulations,
        )

    def run_top_fraction(
        self, top_fraction: float = 0.05, Nsim: int = 100, verbose: bool = True
    ) -> CalibrationResults:
        """Run ABC top fraction selection."""
        simulations, distances = [], []
        sampled_params = {p: [] for p in self.param_names}

        if verbose:
            print(
                f"Starting ABC top fraction selection with {Nsim} simulations and top {top_fraction * 100:.1f}% selected"
            )

        for n in range(Nsim):
            params = self._sample_parameters()
            simulation = self._run_simulation(params)
            distance = self.distance_function(self.observed_data, simulation)

            simulations.append(simulation)
            distances.append(distance)
            for i, p in enumerate(self.param_names):
                sampled_params[p].append(params[i])

            # Print progress every 10% if verbose
            if verbose and (n + 1) % max(1, Nsim // 10) == 0:
                print(
                    f"\tProgress: {n + 1}/{Nsim} simulations completed ({(n + 1) / Nsim * 100:.1f}%)"
                )

        # Select top fraction
        threshold = np.quantile(distances, top_fraction)
        mask = np.array(distances) <= threshold
        n_selected = sum(mask)

        if verbose:
            print(
                f"\tSelected {n_selected} particles (top {top_fraction * 100:.1f}%) "
                f"with distance threshold {threshold:.6f}"
            )

        return self._create_results(
            "top_fraction",
            pd.DataFrame(sampled_params)[mask],
            np.ones(sum(mask)) / sum(mask),
            np.array(distances)[mask],
            np.array(simulations)[mask],
        )

    def _run_simulation(self, params: List[float]) -> Dict[str, Any]:
        """Run a single simulation with given parameters."""
        full_params = {**self.parameters, **dict(zip(self.param_names, params))}
        simulation = self.simulation_function(full_params)
        return self._validate_simulation(simulation)

    def _validate_simulation(self, simulation: Any) -> Dict[str, Any]:
        """Ensure simulation output is in correct format."""
        if not isinstance(simulation, dict):
            raise ValueError(
                f"Simulation must return dictionary, got {type(simulation)}"
            )
        return simulation

    def _sample_parameters(self) -> List[float]:
        """Sample parameters from priors."""
        return sample_prior(self.priors, self.param_names)

    def _create_results(
        self,
        strategy: str,
        particles: pd.DataFrame,
        weights: np.ndarray,
        distances: np.ndarray,
        simulations: List[Dict],
    ) -> CalibrationResults:
        """Create CalibrationResults object."""
        return CalibrationResults(
            calibration_strategy=strategy,
            posterior_distributions={0: particles},
            selected_trajectories={0: simulations},
            distances={0: distances},
            weights={0: weights},
            observed_data=self.observed_data,
            priors=self.priors,
        )

    def _check_stopping_conditions(
        self,
        epsilon: Optional[float],
        minimum_epsilon: Optional[float],
        start_time: datetime,
        max_time: Optional[timedelta],
        n_simulations: int,
        total_simulations_budget: Optional[int],
    ) -> bool:
        """Check if any stopping condition is met."""
        if minimum_epsilon and epsilon and epsilon < minimum_epsilon:
            print("Minimum epsilon reached")
            return True
        if max_time and datetime.now() - start_time > max_time:
            print("Maximum time reached")
            return True
        if total_simulations_budget and n_simulations > total_simulations_budget:
            print("Total simulations budget reached")
            return True
        return False

    def _initialize_particles(self, num_particles, epsilon):
        """
        Initialize the first generation of particles by sampling from priors.

        Args:
            num_particles (int): Number of particles to generate
            epsilon (float): Epsilon threshold for initial generation

        Returns:
            tuple: (particles, weights, distances, simulations) where
                - particles: numpy array of shape (num_particles, num_parameters)
                - weights: numpy array of uniform weights
                - distances: numpy array of distances between simulations and observed data
                - simulations: list of simulation results
        """
        particles, weights, distances, simulations = [], [], [], []

        n_simulations = 0
        # Sample from priors and run simulations
        while len(particles) < num_particles:
            params = sample_prior(self.priors, self.param_names)
            full_params = {**self.parameters, **dict(zip(self.param_names, params))}
            simulated_data = self.simulation_function(full_params)
            dist = self.distance_function(
                data=self.observed_data, simulation=simulated_data
            )
            n_simulations += 1

            if dist <= epsilon:
                particles.append(params)
                weights.append(1.0 / num_particles)  # Uniform weights initially
                distances.append(dist)
                simulations.append(simulated_data)

        return {
            "particles": np.array(particles),
            "weights": np.array(weights),
            "distances": np.array(distances),
            "simulations": simulations,
            "n_simulations": n_simulations,
        }

    def _run_smc_generation(
        self,
        particles: np.ndarray,
        weights: np.ndarray,
        epsilon: float,
        num_particles: int,
        perturbations: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single generation of ABC-SMC."""
        new_particles, new_weights, new_distances, new_simulations = [], [], [], []

        n_simulations = 0
        for _ in range(num_particles):
            while True:
                # Resample a particle based on weights
                index = np.random.choice(len(particles), p=weights / weights.sum())
                candidate_params = particles[index]

                # Propose new parameters (perturbation kernel)
                perturbed_params = [
                    perturbations[self.param_names[i]].propose(candidate_params[i])
                    for i in range(len(self.param_names))
                ]

                # Check if perturbed parameters have prior probability > 0
                prior_probabilities = [
                    self.priors[param].pdf(perturbed_params[i])
                    if param in self.continuous_params
                    else self.priors[param].pmf(perturbed_params[i])
                    for i, param in enumerate(self.param_names)
                ]
                if all(prob > 0 for prob in prior_probabilities):
                    simulation = self._run_simulation(perturbed_params)
                    distance = self.distance_function(self.observed_data, simulation)
                    n_simulations += 1

                    if distance < epsilon:
                        new_particles.append(perturbed_params)
                        weight_numerator = np.prod(
                            [
                                self.priors[param].pdf(perturbed_params[i])
                                if param in self.continuous_params
                                else self.priors[param].pmf(perturbed_params[i])
                                for i, param in enumerate(self.param_names)
                            ]
                        )
                        weight_denominator = np.sum(
                            [
                                weights[j]
                                * np.prod(
                                    [
                                        perturbations[self.param_names[i]].pdf(
                                            perturbed_params[i], particles[j][i]
                                        )
                                        for i in range(len(self.param_names))
                                    ]
                                )
                                for j in range(len(particles))
                            ]
                        )
                        new_weights.append(weight_numerator / weight_denominator)
                        new_distances.append(distance)
                        new_simulations.append(simulation)
                        break

        # Normalize weights
        new_weights = np.array(new_weights)
        new_weights /= new_weights.sum()

        return {
            "particles": np.array(new_particles),
            "weights": np.array(new_weights),
            "distances": np.array(new_distances),
            "simulations": new_simulations,
            "n_simulations": n_simulations,
        }

    def run_projections(
        self,
        parameters: Dict[str, Any],
        iterations: int = 100,
        generation: Optional[int] = None,
        scenario_id: str = "baseline",
    ) -> CalibrationResults:
        """
        Run projections using parameters sampled from the posterior distribution.

        Args:
            parameters: Dictionary of parameters for the projections
            iterations: Number of projection iterations to run. Default is 100.
            generation: Which generation to use for posterior. If None, the last generation is used.
            scenario_id: Identifier for this projection scenario. Default is "baseline".

        Returns:
            CalibrationResults: A new CalibrationResults object containing the original results plus the new projections
        """

        # Get posterior distribution and weights from specified generation
        posterior = self.results.get_posterior_distribution(generation)
        weights = self.results.get_weights(generation)

        # Run projections and store results
        projections, posterior_samples = [], {}
        for _ in range(iterations):
            # Sample from posterior according to weights
            idx = np.random.choice(len(posterior), p=weights / weights.sum())
            posterior_sample = posterior.iloc[idx]

            for k in posterior_sample.keys():
                if k not in posterior_samples:
                    posterior_samples[k] = []
                posterior_samples[k].append(posterior_sample[k])

            # Prepare and run simulation
            proj_params = parameters.copy()
            proj_params.update(posterior_sample)
            result = self.simulation_function(proj_params)
            projections.append(result)

        self.results.projections[scenario_id] = projections
        self.results.projection_parameters[scenario_id] = pd.DataFrame(
            posterior_samples
        )

        return copy.deepcopy(self.results)
