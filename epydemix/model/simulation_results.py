from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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
            comp_name: np.stack(
                [t.compartments[comp_name] for t in self.trajectories], axis=0
            )
            for comp_name in self.trajectories[0].compartments.keys()
        }

    def get_stacked_transitions(self) -> Dict[str, np.ndarray]:
        """
        Get trajectories stacked into arrays of shape (Nsim, timesteps).
        """
        if not self.trajectories:
            return {}

        return {
            trans_name: np.stack(
                [t.transitions[trans_name] for t in self.trajectories], axis=0
            )
            for trans_name in self.trajectories[0].transitions.keys()
        }

    def get_quantiles(
        self, stacked: Dict[str, np.ndarray], quantiles: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Compute quantiles across all trajectories.
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
        data = {"date": dates, "quantile": quantile_values}

        # Add data
        for comp_name, comp_data in stacked.items():
            comp_quantiles = []
            for q in quantiles:
                quant_values = np.quantile(comp_data, q, axis=0)
                comp_quantiles.extend(quant_values)
            data[comp_name] = comp_quantiles

        return pd.DataFrame(data)

    def get_quantiles_transitions(
        self, quantiles: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Compute quantiles across all trajectories for transitions.
        The name of the columns are the transitions names and the demographic groups, in the following format: `{source_compartment_name}_to_{target_compartment_name}_{demographic_group}`.
        For example, the column `S_to_I_total` contains the quantiles of the number of individuals transitioning from susceptible ("S") to infected ("I") individuals across all demographic groups ("total").
        """
        stacked = self.get_stacked_transitions()
        return self.get_quantiles(stacked, quantiles)

    def get_quantiles_compartments(
        self, quantiles: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Compute quantiles across all trajectories for compartments.
        The name of the columns are the compartments names and the demographic groups, in the following format: `{compartment_name}_{demographic_group}`.
        For example, the column `I_total` contains the quantiles of the number of infected ("I") individuals across all demographic groups ("total").
        """
        stacked = self.get_stacked_compartments()
        return self.get_quantiles(stacked, quantiles)

    def resample(
        self, freq: str, method: str = "last", fill_method: str = "ffill"
    ) -> "SimulationResults":
        """Resample all trajectories to new frequency."""
        return SimulationResults(
            trajectories=[
                t.resample(freq, method, fill_method) for t in self.trajectories
            ],
            parameters=self.parameters,
        )
