from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import datetime

@dataclass
class CalibrationResults:
    """
    Class to store and manage the results of a calibration process.

    Attributes:
        calibration_strategy: The strategy used for calibration
        posterior_distributions: Dictionary of posterior distributions per generation
        selected_trajectories: Dictionary of selected trajectories per generation
        observed_data: Observed data used for calibration
        priors: Dictionary of prior distributions for parameters
        calibration_params: Dictionary of parameters used in calibration
        distances: Dictionary of distances per generation
        weights: Dictionary of weights per generation
        projections: Dictionary of projections 
        projection_parameters: Dictionary of projection parameters 
    """
    calibration_strategy: Optional[str] = None
    posterior_distributions: Dict[int, pd.DataFrame] = field(default_factory=dict)
    selected_trajectories: Dict[int, List[Any]] = field(default_factory=dict)
    observed_data: Optional[Any] = None
    priors: Dict[str, Any] = field(default_factory=dict)
    calibration_params: Dict[str, Any] = field(default_factory=dict)
    distances: Dict[int, List[Any]] = field(default_factory=dict)
    weights: Dict[int, List[Any]] = field(default_factory=dict)
    projections: Dict[str, List[Any]]= field(default_factory=dict)
    projection_parameters: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def _get_generation(self, generation: Optional[int], data_dict: Dict[int, Any]) -> Any:
        """Helper method to get data for a specific generation."""
        generations = list(data_dict.keys())
        if not generations:
            return None
            
        if generation is not None:
            if generation not in generations:
                raise ValueError(f"Generation {generation} not found, possible generations are {generations}.")
            return data_dict[generation]
        return data_dict[max(generations)]

    def get_posterior_distribution(self, generation: Optional[int] = None) -> pd.DataFrame:
        """Gets the posterior distribution DataFrame for a specific generation."""
        return self._get_generation(generation, self.posterior_distributions)

    def get_selected_trajectories(self, generation: Optional[int] = None) -> List[Any]:
        """Gets the selected trajectories for a specific generation."""
        return self._get_generation(generation, self.selected_trajectories)

    def get_weights(self, generation: Optional[int] = None) -> List[Any]:
        """Gets the weights for a specific generation."""
        return self._get_generation(generation, self.weights)

    def get_distances(self, generation: Optional[int] = None) -> List[Any]:
        """Gets the distances for a specific generation."""
        return self._get_generation(generation, self.distances)

    def get_calibration_trajectories(
        self,
        generation: Optional[int] = None,
        variables: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Get stacked trajectories from calibration results.

        Args:
            generation: The generation to get trajectories from. If None, uses the last generation.
            variables: Optional list of variable names to include. If None, all variables are included.
        """
        simulations = self.get_selected_trajectories(generation)
        if simulations is None or len(simulations) == 0:  # Better check for empty simulations
            return {}

        # Use user-provided variables or all keys from the first simulation
        keys = variables if variables else simulations[0].keys()
        return {
            key: np.stack([sim[key] for sim in simulations], axis=0)
            for key in keys if key in simulations[0]
        }

    def get_projection_trajectories(
        self,
        scenario_id: str = "baseline",
        variables: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Get stacked trajectories from projection results.

        Args:
            scenario_id: The scenario identifier to get projections for.
            variables: Optional list of variable names to include. If None, all variables are included.
        """
        if scenario_id not in self.projections:
            raise ValueError(f"No projections found for id {scenario_id}")

        simulations = self.projections[scenario_id]
        if simulations is None or len(simulations) == 0:  # Better check for empty simulations
            return {}

        # Use user-provided variables or all keys from the first simulation
        keys = variables if variables else simulations[0].keys()
        return {
            key: np.stack([sim[key] for sim in simulations], axis=0)
            for key in keys if key in simulations[0]
        }

    def get_calibration_quantiles(self,
                                dates: Optional[List[datetime.date]] = None,
                                quantiles: List[float] = [0.05, 0.5, 0.95],
                                generation: Optional[int] = None,
                                variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute quantiles from calibration results."""
        trajectories = self.get_calibration_trajectories(generation, variables=variables)
        return self._compute_quantiles(trajectories, dates, quantiles, variables)

    def get_projection_quantiles(self,
                               dates: Optional[List[datetime.date]] = None,
                               quantiles: List[float] = [0.05, 0.5, 0.95],
                               scenario_id: str = "baseline",
                               variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute quantiles from projection results."""
        trajectories = self.get_projection_trajectories(scenario_id, variables=variables)
        return self._compute_quantiles(trajectories, dates, quantiles, variables)

    def _compute_quantiles(self,
                         trajectories: Dict[str, np.ndarray],
                         dates: Optional[List[datetime.date]],
                         quantiles: List[float],
                         variables: Optional[List[str]]) -> pd.DataFrame:
        """Helper method to compute quantiles from trajectories."""
        if variables:
            trajectories = {k: v for k, v in trajectories.items() if k in variables}
        
        if dates is None:
            dates = np.arange(trajectories[list(trajectories.keys())[0]].shape[1])

        simulation_dates = []
        quantile_values = []
        for q in quantiles:
            simulation_dates.extend(dates)
            quantile_values.extend([q] * len(dates))

        data = {
            "date": simulation_dates,
            "quantile": quantile_values
        }
        
        for key, vals in trajectories.items():
            data[key] = [
                val for q in quantiles 
                for val in np.quantile(vals, q, axis=0)
            ]

        return pd.DataFrame(data)
