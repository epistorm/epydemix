from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from .simulation_output import Trajectory

@dataclass
class SimulationResults:
    """
    Class to store and manage multiple simulation results.
    
    Attributes:
        trajectories (List[Trajectory]): List of simulation trajectories
        parameters (Dict[str, Any]): Dictionary of parameters used in the simulations
    """
    trajectories: List[Trajectory]
    parameters: Dict[str, Any]

    @property
    def Nsim(self) -> int:
        """Number of simulations."""
        return len(self.trajectories)
    
    @property
    def dates(self) -> List[pd.Timestamp]:
        """Simulation dates."""
        return self.trajectories[0].dates if self.trajectories else []
    
    @property
    def compartment_idx(self) -> Dict[str, int]:
        """Compartment indices."""
        return self.trajectories[0].compartment_idx if self.trajectories else {}

    def get_stacked_compartments(self) -> Dict[str, np.ndarray]:
        """
        Get trajectories stacked into arrays of shape (Nsim, timesteps).
        """
        if not self.trajectories:
            return {}
        
        return {
            comp_name: np.stack([t.compartments[comp_name] for t in self.trajectories], axis=0)
            for comp_name in self.trajectories[0].compartments.keys()
        }
    
    def get_stacked_transitions(self) -> Dict[str, np.ndarray]:
        """
        Get trajectories stacked into arrays of shape (Nsim, timesteps).
        """
        if not self.trajectories:
            return {}
        
        return {
            trans_name: np.stack([t.transitions[trans_name] for t in self.trajectories], axis=0)
            for trans_name in self.trajectories[0].transitions.keys()
        }

    def get_quantiles(self, stacked: Dict[str, np.ndarray], quantiles: Optional[List[float]] = None, ignore_nan: bool = False) -> pd.DataFrame:
        """
        Compute quantiles across all trajectories.

        Args:
            stacked: Dictionary of stacked trajectory arrays
            quantiles: List of quantile values to compute. If None, defaults to [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
            ignore_nan: If True, use np.nanquantile to ignore NaN values. Defaults to False.
                When enabled, a warning is issued if any time point has >50% NaN values,
                as quantiles may be unreliable with small sample sizes.
        """
        if quantiles is None:
            quantiles = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]

        # Create dates and quantiles first (these will be the same for all compartments)
        dates = []
        quantile_values = []
        for q in quantiles:
            dates.extend(self.dates)
            quantile_values.extend([q] * len(self.dates))

        # Initialize data dictionary with dates and quantiles
        data = {
            "date": dates,
            "quantile": quantile_values
        }

        # Add data
        quantile_func = np.nanquantile if ignore_nan else np.quantile

        # Check for high NaN proportions when ignore_nan is enabled
        if ignore_nan:
            import warnings
            for comp_name, comp_data in stacked.items():
                nan_prop = np.isnan(comp_data).mean(axis=0)
                max_nan_prop = np.max(nan_prop)
                if max_nan_prop > 0.5:
                    warnings.warn(
                        f"Variable '{comp_name}' has time points with up to {max_nan_prop:.1%} NaN values. "
                        f"Quantiles at these time points may be unreliable due to small sample size."
                    )

        for comp_name, comp_data in stacked.items():
            comp_quantiles = []
            for q in quantiles:
                quant_values = quantile_func(comp_data, q, axis=0)
                comp_quantiles.extend(quant_values)
            data[comp_name] = comp_quantiles

        return pd.DataFrame(data)
    
    def get_quantiles_transitions(self, quantiles: Optional[List[float]] = None, ignore_nan: bool = False) -> pd.DataFrame:
        """
        Compute quantiles across all trajectories for transitions.
        The name of the columns are the transitions names and the demographic groups, in the following format: `{source_compartment_name}_to_{target_compartment_name}_{demographic_group}`.
        For example, the column `S_to_I_total` contains the quantiles of the number of individuals transitioning from susceptible ("S") to infected ("I") individuals across all demographic groups ("total").

        Args:
            quantiles: List of quantile values to compute. If None, defaults to [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
            ignore_nan: If True, use np.nanquantile to ignore NaN values. Defaults to False.
        """
        stacked = self.get_stacked_transitions()
        return self.get_quantiles(stacked, quantiles, ignore_nan)
    
    def get_quantiles_compartments(self, quantiles: Optional[List[float]] = None, ignore_nan: bool = False) -> pd.DataFrame:
        """
        Compute quantiles across all trajectories for compartments.
        The name of the columns are the compartments names and the demographic groups, in the following format: `{compartment_name}_{demographic_group}`.
        For example, the column `I_total` contains the quantiles of the number of infected ("I") individuals across all demographic groups ("total").

        Args:
            quantiles: List of quantile values to compute. If None, defaults to [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
            ignore_nan: If True, use np.nanquantile to ignore NaN values. Defaults to False.
        """
        stacked = self.get_stacked_compartments()
        return self.get_quantiles(stacked, quantiles, ignore_nan)

    def resample(self, freq: str, method: str = 'last', fill_method: str = 'ffill') -> 'SimulationResults':
        """Resample all trajectories to new frequency."""
        return SimulationResults(
            trajectories=[t.resample(freq, method, fill_method) for t in self.trajectories],
            parameters=self.parameters
        )
