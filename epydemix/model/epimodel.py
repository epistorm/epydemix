import copy
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..population.population import Population, load_epydemix_population
from ..utils.utils import (
    apply_initial_conditions,
    apply_overrides,
    compute_simulation_dates,
    create_definitions,
    evaluate,
    format_simulation_output,
    multinomial,
)
from .simulation_output import Trajectory
from .simulation_results import SimulationResults
from .transition import Transition


class EpiModel:
    """
    EpiModel: A compartmental epidemic model simulator

    Example:
        >>> model = EpiModel(compartments=["S", "I", "R"], parameters={"beta": 0.3, "gamma": 0.1})
        >>> model.add_transition(source="S", target="I", params=("beta", "I"))
        >>> model.add_transition(source="I", target="R", params="gamma")
        >>> results = model.run_simulations(
        ...     start_date="2020-01-01",
        ...     end_date="2020-12-31",
        ...     Nsim=100
        ... )
    """

    def __init__(
        self,
        name: str = "EpiModel",
        compartments: Optional[List] = None,
        parameters: Optional[Dict] = None,
        population_name: str = "epydemix_population",
        population_data_path: Optional[str] = None,
        contact_layers: Optional[List] = None,
        contacts_source: Optional[str] = None,
        age_group_mapping: Optional[Dict] = None,
        supported_contacts_sources: Optional[Dict] = None,
        use_default_population: bool = True,
        default_population_size: int = 100000,
        data_version: str = "vtest-epistorm-mix",
        attribute: str = "age",
    ) -> None:
        """
        Initializes the EpiModel instance.

        Args:
            name (str, optional): The name of the epidemic model. Defaults to "EpiModel".
            compartments (list, optional): A list of compartments for the model. Defaults to an empty list if None.
            parameters (dict, optional): A dictionary of parameters for the model. Defaults to an empty dictionary if None.
            population_name (str, optional): The name of the population to load or create. Defaults to 'epydemix_population'.
            population_data_path (str or None, optional): The path to the population data.
            contact_layers (list or None, optional): A list of contact layers to load for the population. Defaults to
                ["school", "work", "home", "community"] if None.
            contacts_source (str or None, optional): The source of contact data. If None, the function will automatically detect the source.
            age_group_mapping (dict or None, optional): A mapping of age groups for the population.
            supported_contacts_sources (list or None, optional): A list of supported contact data sources. Defaults to
                ["prem_2017", "prem_2021", "mistry_2021", "litvinova_2025"] for age,
                ["litvinova_2025"] for sex and race_ethnicity if None.
            use_default_population (bool, optional): If True, creates a default population; if False, tries to load the population from the provided path and population name. Defaults to True.
            default_population_size (int, optional): The size of the default population when use_default_population is True. Defaults to 100000.
            data_version (str, optional): The git tag/version of the epydemix-data repository. Defaults to "vtest-epistorm-mix".
            attribute (str, optional): The demographic attribute layer. Defaults to "age".
        Returns:
            None
        """
        # Initialize core attributes
        self.name = name
        self.transitions = {}
        self.transitions_list = []
        self.interventions = []
        self.compartments = []
        self.compartments_idx = {}
        self.transitions_idx = {}
        self.transition_functions = {}
        self.parameters = {}
        self.definitions = {}
        self.overrides = {}
        self.Cs = {}

        # Handle default empty lists for compartments and contact layers
        if compartments is None:
            compartments = []
        self.add_compartments(compartments)

        if contact_layers is None:
            contact_layers = ["school", "work", "home", "community"]

        # Add parameters if provided
        if parameters:
            self.add_parameter(parameters_dict=parameters)

        # Handle default contact sources if not provided
        if supported_contacts_sources is None:
            supported_contacts_sources = {
                "age": ["prem_2017", "prem_2021", "mistry_2021", "litvinova_2025"],
                "sex": ["litvinova_2025"],
                "race_ethnicity": ["litvinova_2025"],
            }

        # Load or create population based on use_default_population flag
        self.population = self._load_or_create_population(
            population_name,
            population_data_path,
            contact_layers,
            contacts_source,
            age_group_mapping,
            supported_contacts_sources,
            use_default_population,
            default_population_size,
            data_version,
            attribute,
        )

        # Initalize functions to compute transition rates
        self.register_transition_kind(
            kind="spontaneous", function=compute_spontaneous_transition_rate
        )
        self.register_transition_kind(
            kind="mediated", function=compute_mediated_transition_rate
        )

    def __repr__(self) -> str:
        """
        Returns a string representation of the EpiModel object,
        summarizing its name, compartments, transitions, parameters, and population.

        Returns:
            str: String representation of the EpiModel object.
        """
        repr_str = f"EpiModel(name='{self.name}')\n"

        # Compartments summary
        repr_str += f"Compartments: {len(self.compartments)}\n"
        if len(self.compartments) > 0:
            repr_str += "  " + ", ".join(self.compartments) + "\n"
        else:
            repr_str += "  No compartments defined\n"

        # Transitions summary
        repr_str += f"Transitions: {len(self.transitions_list)}\n"
        if len(self.transitions_list) > 0:
            repr_str += "  Transitions between compartments:\n"
            for tr in self.transitions_list:
                repr_str += f"    {tr.source} -> {tr.target}, params: {tr.params} (kind: {tr.kind})\n"
        else:
            repr_str += "  No transitions defined\n"

        # Parameters summary
        repr_str += f"Parameters: {len(self.parameters)}\n"
        if len(self.parameters) > 0:
            repr_str += "  Model parameters:\n"
            for param_name, param_value in self.parameters.items():
                repr_str += f"    {param_name}: {param_value}\n"
        else:
            repr_str += "  No parameters defined\n"

        # Population summary
        repr_str += f"Population: {self.population.name}\n"
        repr_str += f"  Population size: {sum(self.population.Nk)} individuals\n"
        repr_str += f"  Demographic groups: {len(self.population.Nk_names)}\n"
        if len(self.population.Nk_names) > 0:
            repr_str += (
                "    "
                + ", ".join([str(name) for name in self.population.Nk_names])
                + "\n"
            )

        return repr_str

    def _load_or_create_population(
        self,
        population_name: str,
        population_data_path: Optional[str],
        contact_layers: List,
        contacts_source: Optional[str],
        age_group_mapping: Optional[Dict],
        supported_contacts_sources: Dict,
        use_default_population: bool,
        default_population_size: int = 100000,
        data_version: str = "vtest-epistorm-mix",
        attribute: str = "age",
    ) -> Population:
        """
        Loads a population from a file or creates a default population if specified.

        Args:
            population_name (str): The name of the population.
            population_data_path (str or None): The path to the population data file. If None, data may be fetched online.
            contact_layers (list): A list of contact layers to load for the population (e.g., "home", "work", "school").
            contacts_source (str or None): The source of contact data. If None, load_population will auto-determine the source.
            age_group_mapping (dict or None): A mapping for age groups. If None, a default mapping is used.
            supported_contacts_sources (list): A list of supported contact data sources (e.g., "prem_2017", "prem_2021").
            use_default_population (bool): If True, creates a default population with a single contact matrix and population size of default_population_size.
                If False, attempts to load population data from a local or online source.
            default_population_size (int): The size of the default population when use_default_population is True. Defaults to 100000.
            data_version (str): The git tag/version of the epydemix-data repository. Defaults to "vtest".
            attribute (str): The demographic attribute layer. Defaults to "age".

        Returns:
            Population: A population object, either created manually or loaded from data.
        """
        if use_default_population:
            # Create a default population manually
            population = Population(name=population_name)
            population.add_contact_matrix(np.array([[1.0]]))  # Default contact matrix
            population.add_population(
                np.array([default_population_size])
            )  # Default population size
            return population
        else:
            # Load population using load_population, which handles both online and local sources
            return load_epydemix_population(
                population_name,
                contacts_source=contacts_source,
                path_to_data=population_data_path,
                layers=contact_layers,
                age_group_mapping=age_group_mapping,
                supported_contacts_sources=supported_contacts_sources,
                data_version=data_version,
                attribute=attribute,
            )

    def import_epydemix_population(
        self,
        population_name: str,
        population_data_path: Optional[str] = None,
        contact_layers: Optional[List] = None,
        contacts_source: Optional[str] = None,
        age_group_mapping: Optional[Dict] = None,
        supported_contacts_sources: Optional[Dict] = None,
        data_version: str = "vtest-epistorm-mix",
        attribute: str = "age",
    ) -> None:
        """
        Sets or updates the population for the model.

        Args:
            population_name (str): The name of the population to set.
            population_data_path (str or None, optional): The path to the population data file. If None, data may be fetched online.
            contact_layers (list or None, optional): A list of contact layers to load for the population. Defaults to
                ["school", "work", "home", "community"] if None.
            contacts_source (str or None, optional): The source of contact data. If None, the function will attempt to determine the source automatically.
            age_group_mapping (dict or None, optional): A mapping for age groups. If None, a default mapping is used.
            supported_contacts_sources (list or None, optional): A list of supported contact data sources (e.g., "prem_2017", "prem_2021").
                Defaults to ["prem_2017", "prem_2021", "mistry_2021", "litvinova_2025"] for age,
                ["litvinova_2025"] for sex and race_ethnicity if None.
            data_version (str, optional): The git tag/version of the epydemix-data repository. Defaults to "vtest-epistorm-mix".
            attribute (str, optional): The demographic attribute layer. Defaults to "age".

        Returns:
            None
        """
        # Handle default contact layers if not provided
        if contact_layers is None:
            contact_layers = ["school", "work", "home", "community"]

        # Handle default contact sources if not provided
        if supported_contacts_sources is None:
            supported_contacts_sources = {
                "age": ["prem_2017", "prem_2021", "mistry_2021", "litvinova_2025"],
                "sex": ["litvinova_2025"],
                "race_ethnicity": ["litvinova_2025"],
            }

        # Load a new population using the same logic as in the constructor
        self.population = load_epydemix_population(
            population_name,
            contacts_source=contacts_source,
            path_to_data=population_data_path,
            layers=contact_layers,
            age_group_mapping=age_group_mapping,
            supported_contacts_sources=supported_contacts_sources,
            data_version=data_version,
            attribute=attribute,
        )

    def add_compartments(self, compartments: Union[List, object]) -> None:
        """
        Adds compartments to the epidemic model.

        Args:
            compartments (list or object): A list of compartments or a single compartment to be added.

        Returns:
            None
        """
        # Ensure we are always working with a list, even if a single compartment is passed
        if not isinstance(compartments, (list, tuple)):
            compartments = [compartments]

        # Add compartments to the model
        self.compartments.extend(compartments)

        # Determine the current maximum index in compartments_idx or set to -1 if empty
        max_idx = max(self.compartments_idx.values(), default=-1)

        # Add each compartment with corresponding transitions and index
        for i, compartment in enumerate(compartments, start=1):
            self.transitions[
                compartment
            ] = []  # Initialize empty transitions for each compartment
            self.compartments_idx[compartment] = (
                max_idx + i
            )  # Assign a unique index to each compartment

    @property
    def n_compartments(self) -> int:
        """Get total number of compartments."""
        return len(self.compartments)

    def clear_compartments(self) -> None:
        """
        This method resets the `compartments` list and clears the `compartments_idx` dictionary.

        Returns:
            None
        """
        self.compartments = []
        self.compartments_idx = {}

    def add_parameter(
        self,
        parameter_name: Optional[str] = None,
        value: Any = None,
        parameters_dict: Optional[Dict] = None,
    ) -> None:
        """
        Adds new parameters to the model.

        Args:
            parameter_name (str, optional): The name of the parameter to add. Ignored if `parameters_dict` is provided.
            value (any, optional): The value of the parameter to add. Ignored if `parameters_dict` is provided.
            parameters_dict (dict, optional): A dictionary of parameter names and values. If provided,
                `parameter_name` and `value` are ignored and the parameters are updated using this dictionary.

        Raises:
            ValueError: If neither `parameter_name`/`value` nor `parameters_dict` is provided.

        Returns:
            None
        """
        if parameters_dict:
            self.parameters.update(parameters_dict)
        elif parameter_name is not None and value is not None:
            self.parameters.update({parameter_name: value})
        else:
            raise ValueError(
                "Either parameter_name and value or parameters_dict must be provided."
            )

    def get_parameter(self, parameter_name: str) -> Any:
        """
        Retrieves the value of a specified parameter.

        Args:
            parameter_name (str): The name of the parameter to retrieve.

        Returns:
            any: The value of the specified parameter.

        Raises:
            KeyError: If the parameter with the given name is not found.
        """
        return self.parameters[parameter_name]

    def delete_parameter(self, parameter_name: str) -> Any:
        """
        Deletes a parameter from the dictionary of the model's parameters.

        Args:
            parameter_name (str): The name of the parameter to delete.

        Returns:
            any: The value of the deleted parameter.

        Raises:
            KeyError: If the parameter with the given name is not found.
        """
        return self.parameters.pop(parameter_name)

    def clear_parameters(self) -> None:
        """
        Clears all parameters from the model's parameters dictionary.

        Returns:
            None
        """
        self.parameters = {}

    def override_parameter(
        self, start_date: str, end_date: str, parameter_name: str, value: Any
    ) -> None:
        """
        Adds an override for a parameter with the specified name.

        Args:
            start_date (str): The start date for the override period.
            end_date (str): The end date for the override period.
            parameter_name (str): The name of the parameter to override.
            value (any): The value to override the parameter with.

        Returns:
            None
        """
        override_dict = {"start_date": start_date, "end_date": end_date, "value": value}

        if parameter_name in self.overrides:
            self.overrides[parameter_name].append(override_dict)
        else:
            self.overrides[parameter_name] = [override_dict]

    def delete_override(self, parameter_name: str) -> None:
        """
        Deletes overrides for a specific parameter.

        Args:
            parameter_name (str): The name of the parameter whose overrides are to be deleted.

        Returns:
            None
        """
        self.overrides.pop(parameter_name, None)

    def clear_overrides(self) -> None:
        """
        Clears all parameter overrides from the model.

        Returns:
            None
        """
        self.overrides = {}

    def add_transition(self, source: str, target: str, kind: str, params: Any) -> None:
        """
        Adds a transition to the epidemic model.

        Args:
            source (str): The source compartment of the transition.
            target (str): The target compartment of the transition.
            kind (str): The kind of transition (e.g., spontaneous or mediated).
            params (Any): The parameters involved in the transition.

        Raises:
            ValueError: If the source or target is not in the compartments list.

        Returns:
            None
        """
        missing_compartments = [
            comp for comp in [source, target] if comp and comp not in self.compartments
        ]
        if missing_compartments:
            raise ValueError(
                f"These compartments are not in the compartments list: {', '.join(missing_compartments)}"
            )

        transition = Transition(source=source, target=target, kind=kind, params=params)
        self.transitions_list.append(transition)
        self.transitions[source].append(transition)

        transition_name = f"{source}_to_{target}"
        if transition_name not in self.transitions_idx:
            self.transitions_idx[transition_name] = len(self.transitions_idx)

    def register_transition_kind(self, kind: str, function: Callable):
        """
        Registers a new transition kind providing a function to compute the rate of the transition.

        Args:
            kind (str): The kind of transition (e.g., spontaneous or mediated).
            function (Callable): The function to register.

        Returns:
            None
        """
        validate_transition_function(function)
        self.transition_functions[kind] = function

    @property
    def n_transitions(self) -> int:
        """Get total number of transitions."""
        return len(self.transitions_list)

    def clear_transitions(self) -> None:
        """
        Clears all transitions from the model.

        This method resets the `transitions_list` and reinitializes the `transitions` dictionary,
        clearing all existing transitions while keeping the structure intact for each compartment.

        Returns:
            None
        """
        self.transitions_list = []
        self.transitions = {comp: [] for comp in self.compartments}

    def add_intervention(
        self,
        layer_name: str,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        reduction_factor: Optional[float] = None,
        new_matrix: Optional[np.ndarray] = None,
        name: str = "",
    ) -> None:
        """
        Adds an intervention to the epidemic model.

        Args:
            layer_name (str): The name of the layer to which the intervention applies.
            start_date (str or datetime): The start date of the intervention.
            end_date (str or datetime): The end date of the intervention.
            reduction_factor (float, optional): The factor by which to reduce the contact matrix. Default is None.
            new_matrix (np.ndarray, optional): A new contact matrix to use during the intervention. Default is None.
            name (str, optional): The name of the intervention. Default is an empty string.

        Raises:
            ValueError: If neither reduction_factor nor new_matrix is provided.

        Returns:
            None
        """
        # Convert start_date and end_date to pandas datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Check if both reduction_factor and new_matrix are None
        if reduction_factor is None and new_matrix is None:
            raise ValueError("Either reduction_factor or new_matrix must be provided")

        # Append the intervention to the interventions list
        self.interventions.append(
            {
                "layer": layer_name,
                "start_date": start_date,
                "end_date": end_date,
                "reduction_factor": reduction_factor,
                "new_matrix": new_matrix,
                "name": name,
            }
        )

    def clear_interventions(self) -> None:
        """
        Clears all interventions from the model.

        This method resets the `interventions` list, removing all existing interventions.

        Returns:
            None
        """
        self.interventions = []

    def apply_intervention(
        self, intervention: Dict, simulation_dates: List[pd.Timestamp]
    ) -> None:
        """
        Applies an intervention to the contact matrices for specified simulation dates.

        Args:
            intervention (dict): A dictionary containing intervention details with the following keys:
                - "layer" (str): The name of the layer to which the intervention applies.
                - "start_date" (pd.Timestamp): The start date of the intervention.
                - "end_date" (pd.Timestamp): The end date of the intervention.
                - "reduction_factor" (float, optional): The factor by which to reduce the contact matrix.
                - "new_matrix" (np.ndarray, optional): A new contact matrix to use during the intervention.
            simulation_dates (list of pd.Timestamp): A list of dates for which the simulation is run.

        Raises:
            ValueError: If neither reduction_factor nor new_matrix is provided in the intervention.

        Returns:
            None
        """
        # Early validation of the intervention inputs
        reduction_factor = intervention.get("reduction_factor")
        new_matrix = intervention.get("new_matrix")

        if reduction_factor is None and new_matrix is None:
            raise ValueError(
                "Intervention must have either a reduction_factor or a new_matrix"
            )

        # Extract commonly used values
        layer = intervention["layer"]
        start_date = intervention["start_date"]
        end_date = intervention["end_date"]

        # Apply the intervention to the relevant dates
        for date in filter(lambda d: start_date <= d <= end_date, simulation_dates):
            if reduction_factor is not None:
                self.Cs[date][layer] *= reduction_factor
            else:  # If reduction_factor is None, we assume new_matrix is provided
                self.Cs[date][layer] = new_matrix

    def compute_contact_reductions(self, simulation_dates: List[pd.Timestamp]) -> None:
        """
        Computes the contact reductions for a population over the given simulation dates.

        This function applies interventions to the contact matrices and computes the overall contact matrix
        for each date in the simulation period.

        Args:
            simulation_dates (list of pd.Timestamp): A list of dates over which the simulation is run.

        Returns:
            None: The function updates the instance variable `self.Cs` with the contact matrices for each date,
                including the overall contact matrix after applying interventions.
        """
        # Initialize contact matrices for each simulation date, copying the population's initial contact matrices
        self.Cs = {
            date: {
                layer: np.copy(matrix)
                for layer, matrix in self.population.contact_matrices.items()
            }
            for date in simulation_dates
        }

        # Apply interventions to the contact matrices
        for intervention in self.interventions:
            self.apply_intervention(intervention, simulation_dates)

        # Compute the overall contact matrix for each date by summing up the contact matrices across layers
        for date in self.Cs:
            self.Cs[date]["overall"] = np.sum(
                np.array(list(self.Cs[date].values())), axis=0
            )

    def create_default_initial_conditions(
        self, percentage_in_agents: float = 0.0005
    ) -> Dict[str, np.ndarray]:
        """
        Creates default initial conditions for the epidemic model. If initial conditions are not provided,
        assigns a percentage of the population to agent compartments and the rest to source compartments,
        considering different age groups.

        Args:
            percentage_in_agents (float): Percentage of population to place in agent compartments. Defaults to 0.05%.
            initial_conditions (dict, optional): A dictionary specifying initial conditions. If None, this function creates default conditions.

        Returns:
            dict: A dictionary with initial conditions for each compartment, with values as arrays representing different age groups.
        """

        population = self.population.Nk

        # Initialize initial conditions dictionary
        initial_conditions_dict = {}

        # Total population for each age group
        total_population_per_age_group = np.array(population)

        # Get compartments that are agents in transitions
        agent_compartments = [
            tr.params[1] for tr in self.transitions_list if tr.kind == "mediated"
        ]

        # Get compartments that are sources in transitions with agents
        source_compartments = [
            tr.source for tr in self.transitions_list if tr.kind == "mediated"
        ]

        # Total number of agent compartments
        num_agent_compartments = len(agent_compartments)

        # Distribute population into agent compartments
        if num_agent_compartments > 0:
            population_in_agents_per_age_group = (
                total_population_per_age_group * percentage_in_agents
            ).astype(int)
            population_per_agent_per_age_group = (
                population_in_agents_per_age_group // num_agent_compartments
            )

            for comp in agent_compartments:
                initial_conditions_dict[comp] = (
                    population_per_agent_per_age_group.copy()
                )

        # Assign the rest of the population to source compartments
        remaining_population_per_age_group = total_population_per_age_group - np.sum(
            list(initial_conditions_dict.values()), axis=0
        )
        num_source_compartments = len(source_compartments)

        if num_source_compartments > 0:
            population_per_source_per_age_group = (
                remaining_population_per_age_group // num_source_compartments
            )

            for comp in source_compartments:
                initial_conditions_dict[comp] = (
                    population_per_source_per_age_group.copy()
                )

        # If some compartments are missing, ensure they get zero population
        for comp in self.compartments:
            if comp not in initial_conditions_dict:
                initial_conditions_dict[comp] = np.zeros_like(
                    total_population_per_age_group
                )

        return initial_conditions_dict

    def set_population(self, population: Population) -> None:
        """
        Sets the population for the epidemic model.

        Args:
            population (Population): The population object to set.

        Returns:
            None
        """
        self.population = population

    def run_simulations(
        self,
        start_date: Union[str, pd.Timestamp] = "2020-01-01",
        end_date: Union[str, pd.Timestamp] = "2020-12-31",
        initial_conditions_dict: Optional[Dict[str, np.ndarray]] = None,
        Nsim: int = 100,
        percentage_in_agents: float = 0.0005,
        dt: Optional[float] = 1.0,
        resample_frequency: Optional[str] = "D",
        resample_aggregation_compartments: Optional[Union[str, dict]] = "last",
        resample_aggregation_transitions: Optional[Union[str, dict]] = "sum",
        fill_method: Optional[str] = "ffill",
        apply_linear_approximation: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> SimulationResults:
        """
        Simulates the epidemic model multiple times over the given time period.

        Args:
            start_date (str or pd.Timestamp): The start date of the simulation. Default is "2020-01-01".
            end_date (str or pd.Timestamp): The end date of the simulation. Default is "2020-12-31".
            initial_conditions_dict (dict, optional): A dictionary of initial conditions for the simulation.
            Nsim (int, optional): The number of simulation runs to perform (default is 100).
            percentage_in_agents (float, optional): The percentage of the population to initialize in the agents compartment.
            dt (float, optional): The time step for the simulation, expressed in days. Default is 1 (day).
            resample_frequency (str, optional): The frequency at which to resample the simulation results. Default is "D" (daily).
            resample_aggregation_compartments (str, optional): The aggregation method to use when resampling the compartments. Default is "last".
            resample_aggregation_transitions (str, optional): The aggregation method to use when resampling the transitions. Default is "sum".
            fill_method (str, optional): Method to fill NaN values after resampling. Default is "ffill".
            apply_linear_approximation (bool, optional): Whether to use linear approximation to the probabilities. Default is False.
            rng (np.random.Generator, optional): Random number generator. Default is None.

        Returns:
            SimulationResults: An object containing all simulation trajectories.

        Raises:
            RuntimeError: If the simulation fails.
        """

        if rng is None:
            rng = np.random.default_rng()

        # Run multiple simulations and collect trajectories
        try:
            # Pre-compute simulation dates, initial conditions, and contact matrices to speed up the simulation
            if initial_conditions_dict is None:
                initial_conditions_dict = self.create_default_initial_conditions(
                    percentage_in_agents=percentage_in_agents
                )
            simulation_dates = compute_simulation_dates(start_date, end_date, dt=dt)
            self.compute_contact_reductions(simulation_dates)
            contact_matrices = [self.Cs[date] for date in simulation_dates]

            trajectories = []
            for _ in range(Nsim):
                trajectory = simulate(
                    self,
                    start_date=start_date,
                    end_date=end_date,
                    dt=dt,
                    initial_conditions_dict=initial_conditions_dict,
                    percentage_in_agents=percentage_in_agents,
                    resample_frequency=resample_frequency,
                    resample_aggregation_compartments=resample_aggregation_compartments,
                    resample_aggregation_transitions=resample_aggregation_transitions,
                    fill_method=fill_method,
                    apply_linear_approximation=apply_linear_approximation,
                    rng=rng,
                    simulation_dates=simulation_dates,
                    contact_matrices=contact_matrices,
                )
                trajectories.append(trajectory)
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {str(e)}") from e

        # Return SimulationResults with all trajectories
        return SimulationResults(trajectories=trajectories, parameters=self.parameters)


def simulate(
    epimodel,
    start_date: Union[str, pd.Timestamp] = "2020-01-01",
    end_date: Union[str, pd.Timestamp] = "2020-12-31",
    initial_conditions_dict: Optional[Dict[str, np.ndarray]] = None,
    percentage_in_agents: float = 0.0005,
    dt: Optional[float] = 1.0,
    resample_frequency: Optional[str] = "D",
    resample_aggregation_compartments: Optional[Union[str, dict]] = "last",
    resample_aggregation_transitions: Optional[Union[str, dict]] = "sum",
    fill_method: Optional[str] = "ffill",
    apply_linear_approximation: bool = False,
    rng: Optional[np.random.Generator] = None,
    contact_matrices: Optional[List[Dict[str, np.ndarray]]] = None,
    simulation_dates: Optional[List[pd.Timestamp]] = None,
    **kwargs,
) -> Trajectory:
    """
    Runs a simulation of the epidemic model over the specified simulation dates.

    Args:
        epimodel (EpiModel): The epidemic model instance to simulate.
        start_date (str or pd.Timestamp): The start date of the simulation. Default is "2020-01-01".
        end_date (str or pd.Timestamp): The end date of the simulation. Default is "2020-12-31".
        initial_conditions_dict (dict, optional): A dictionary of initial conditions for the simulation.
        percentage_in_agents (float, optional): The percentage of the population to initialize in the agents compartment.
        dt (float, optional): The time step for the simulation, expressed in days. Default is 1 (day).
        resample_frequency (str, optional): The frequency at which to resample the simulation results. Default is "D" (daily).
        resample_aggregation_compartments (str, optional): The aggregation method to use when resampling the compartments. Default is "last".
        resample_aggregation_transitions (str, optional): The aggregation method to use when resampling the transitions. Default is "sum".
        fill_method (str, optional): The method to use when filling NaN values after resampling. Default is "ffill".
        apply_linear_approximation (bool, optional): Whether to use linear approximation to the probabilities. Default is False.
        rng (np.random.Generator, optional): Random number generator. Default is None.
        contact_matrices (list, optional): A list of contact matrices for the simulation. Default is None.
        simulation_dates (list, optional): A list of simulation dates. Default is None.
        **kwargs: Additional parameters to overwrite model parameters during the simulation.

    Returns:
        Trajectory: The trajectory of the simulation

    Raises:
        ValueError: If the model has no transitions defined.
    """
    if rng is None:
        rng = np.random.default_rng()

    # check that the model has transitions
    if len(epimodel.transitions_list) == 0:
        raise ValueError(
            "The model has no transitions defined. Please add transitions before running simulations."
        )

    # Compute the simulation dates if not provided
    if simulation_dates is None:
        simulation_dates = compute_simulation_dates(start_date, end_date, dt=dt)

    # Compute initial conditions if not provided
    if initial_conditions_dict is None:
        initial_conditions_dict = epimodel.create_default_initial_conditions(
            percentage_in_agents=percentage_in_agents
        )

    # Compute the contact reductions based on the interventions if not provided
    if contact_matrices is None:
        epimodel.compute_contact_reductions(simulation_dates)
        contact_matrices = [epimodel.Cs[date] for date in simulation_dates]

    # Update parameters if any are provided via kwargs (needed for calibration purposes)
    parameters = epimodel.parameters.copy()
    parameters.update(kwargs)

    # Compute the definitions and apply overrides
    epimodel.definitions = create_definitions(
        parameters, len(simulation_dates), epimodel.population.Nk.shape[0]
    )
    epimodel.definitions = apply_overrides(
        epimodel.definitions, epimodel.overrides, simulation_dates
    )

    # Initialize population in different compartments and demographic groups
    initial_conditions = apply_initial_conditions(epimodel, initial_conditions_dict)

    # Run simulation with pre-computed contacts
    compartments_evolution, transitions_evolution = stochastic_simulation(
        T=len(simulation_dates),
        contact_matrices=contact_matrices,
        epimodel=epimodel,
        parameters=epimodel.definitions,
        initial_conditions=initial_conditions,
        dt=dt,
        apply_linear_approximation=apply_linear_approximation,
        rng=rng,
    )

    # Format the simulation output
    results = format_simulation_output(
        compartments_evolution,
        transitions_evolution,
        epimodel.compartments_idx,
        epimodel.transitions_idx,
        epimodel.population.Nk_names,
    )
    trajectory = Trajectory(
        compartments=results["compartments"],
        transitions=results["transitions"],
        dates=simulation_dates,
        compartment_idx=epimodel.compartments_idx,
        transitions_idx=epimodel.transitions_idx,
        parameters=epimodel.definitions,
    )

    # Only resample if necessary
    if resample_frequency is not None:
        # Check if resampling is needed (simulation dates frequency != requested frequency)
        sim_freq = pd.infer_freq(simulation_dates)
        if sim_freq != resample_frequency:
            trajectory.resample(
                resample_frequency,
                resample_aggregation_compartments,
                resample_aggregation_transitions,
                fill_method,
            )
    return trajectory


def stochastic_simulation(
    T: int,
    contact_matrices: List[Dict[str, np.ndarray]],
    epimodel,
    parameters: Dict,
    initial_conditions: np.ndarray,
    dt: float,
    apply_linear_approximation: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Run a stochastic simulation of the epidemic model.

    Args:
        T: Number of time steps
        contact_matrices: Pre-computed list of contact matrices dictionaries (key is the layer, value is the contact matrix)
        epimodel: The epidemic model
        parameters: Model parameters
        initial_conditions: Initial population distribution
        dt: Time step size
        apply_linear_approximation (bool, optional): Whether to use linear approximation to the probabilities. Default is False.
        rng (np.random.Generator, optional): Random number generator. Default is None.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pre-allocate arrays
    N = len(epimodel.population.Nk)
    C = len(epimodel.compartments)

    compartments_evolution = np.zeros((T + 1, C, N), dtype=np.float64)
    transitions_evolution = np.zeros((T, epimodel.n_transitions, N), dtype=np.float64)
    compartments_evolution[0] = initial_conditions

    # Pre-compute population sizes and create views for better performance
    pop_sizes = epimodel.population.Nk
    comp_indices = epimodel.compartments_idx

    # Pre-allocate arrays for rates and transitions
    rates = np.zeros((C, N), dtype=np.float64)
    new_pop = np.zeros((C, N), dtype=np.float64)
    probs_out = np.empty(C, dtype=np.float64)

    # create a dictionary to store the data needed for the transitions
    system_data = {
        "parameters": parameters,
        "t": 0,
        "comp_indices": comp_indices,
        "contact_matrix": None,
        "pop": None,
        "pop_sizes": pop_sizes,
        "dt": dt,
    }

    # Simulate each time step
    for t in range(T):
        # Update system data with current state
        system_data.update(
            {
                "t": t,
                "contact_matrix": contact_matrices[t],
                "pop": compartments_evolution[t],
            }
        )

        new_pop[:] = compartments_evolution[t]

        for comp in epimodel.compartments:
            transitions = epimodel.transitions[comp]
            if not transitions:
                continue

            rates.fill(0)
            source_idx = comp_indices[comp]
            current_pop = compartments_evolution[t, source_idx]
            mask = np.arange(rates.shape[0]) != source_idx

            if not np.any(current_pop):
                continue

            for tr in transitions:
                target_idx = comp_indices[tr.target]
                trans_rate = epimodel.transition_functions[tr.kind](
                    tr.params, system_data
                )
                rates[target_idx] += trans_rate

            delta = np.array(
                [
                    multinomial(
                        n,
                        r,
                        source_idx,
                        mask,
                        dt,
                        apply_linear_approximation=apply_linear_approximation,
                        rng=rng,
                        probs_out=probs_out,
                    )
                    if n > 0
                    else np.zeros(C)
                    for n, r in zip(current_pop, rates.T)
                ]
            )

            # Store transition counts
            for tr in transitions:
                tr_name = f"{tr.source}_to_{tr.target}"
                tr_idx = epimodel.transitions_idx[tr_name]
                transitions_evolution[t, tr_idx] += delta[
                    :, epimodel.compartments_idx[tr.target]
                ]

            # Update populations
            np.subtract(
                new_pop[source_idx], np.sum(delta, axis=1), out=new_pop[source_idx]
            )
            new_pop += delta.T

        compartments_evolution[t + 1] = new_pop

    return compartments_evolution[1:], transitions_evolution


def compute_spontaneous_transition_rate(params, data):
    """
    Compute the rate of a spontaneous transition.

    Args:
        params: The parameters of the transition provided by the user.
        data: A dictionary containing the data needed for the transition.
            - parameters: The model parameters
            - t: The current time step
    Returns:
        The rate of the transition
    """
    if isinstance(params, str):
        env_copy = copy.deepcopy(data["parameters"])
        rate_eval = evaluate(expr=params, env=env_copy)[data["t"]]
        return rate_eval
    else:
        return params


def compute_mediated_transition_rate(params, data):
    """
    Compute the rate of a mediated transition.

    Args:
        params: The parameters of the transition provided by the user.
        data: A dictionary containing the data needed for the transition.
            - parameters: The model parameters
            - t: The current time step
            - comp_indices: The indices of the compartments
            - contact_matrix: The contact matrix
            - pop: The population in different compartments
            - pop_sizes: The population sizes
    """
    if isinstance(params[0], str):
        env_copy = copy.deepcopy(data["parameters"])
        rate_eval = evaluate(expr=params[0], env=env_copy)[data["t"]]
    else:
        rate_eval = params[0]
    agent_idx = data["comp_indices"][params[1]]
    interaction = np.sum(
        data["contact_matrix"]["overall"] * data["pop"][agent_idx] / data["pop_sizes"],
        axis=1,
    )
    return rate_eval * interaction


def validate_transition_function(func: Callable) -> None:
    """
    Validates that a transition rate function has the correct signature and parameters.

    Args:
        func: The transition rate function to validate

    Raises:
        ValueError: If the function signature doesn't match requirements
    """
    # Get function signature
    sig = inspect.signature(func)
    params = sig.parameters

    # Check number of parameters
    if len(params) != 2:
        raise ValueError(
            f"Transition function must take exactly 2 parameters. Got {len(params)}: {list(params.keys())}"
        )

    # Check parameter names
    expected_params = ["params", "data"]
    actual_params = list(params.keys())

    if actual_params != expected_params:
        raise ValueError(
            f"Transition function must have parameters named {expected_params}. Got {actual_params}"
        )
