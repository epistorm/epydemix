import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EPYDEMIX_DATA_BASE_URL = "https://raw.githubusercontent.com/epistorm/epydemix-data"

demographic_grouping_prem = OrderedDict(
    {
        "0-4": np.arange(0, 5).astype(str),
        "5-9": np.arange(5, 10).astype(str),
        "10-14": np.arange(10, 15).astype(str),
        "15-19": np.arange(15, 20).astype(str),
        "20-24": np.arange(20, 25).astype(str),
        "25-29": np.arange(25, 30).astype(str),
        "30-34": np.arange(30, 35).astype(str),
        "35-39": np.arange(35, 40).astype(str),
        "40-44": np.arange(40, 45).astype(str),
        "45-49": np.arange(45, 50).astype(str),
        "50-54": np.arange(50, 55).astype(str),
        "55-59": np.arange(55, 60).astype(str),
        "60-64": np.arange(60, 65).astype(str),
        "65-69": np.arange(65, 70).astype(str),
        "70-74": np.arange(70, 75).astype(str),
        "75+": np.concatenate((np.arange(75, 84), ["84+"])).astype(str),
    }
)

contacts_age_group_mapping_prem = {
    "0-4": ["0-4"],
    "5-19": ["5-9", "10-14", "15-19"],
    "20-49": ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49"],
    "50-64": ["50-54", "55-59", "60-64"],
    "65+": ["65-69", "70-74", "75+"],
}

contacts_age_group_mapping_mistry = {
    "0-4": np.arange(0, 5).astype(str),
    "5-19": np.arange(5, 20).astype(str),
    "20-49": np.arange(20, 50).astype(str),
    "50-64": np.arange(50, 65).astype(str),
    "65+": np.concatenate((np.arange(65, 84).astype(str), ["84+"])),
}


class Population:
    """
     Represents a population for epidemiological modeling, including demographic data and contact matrices.

     The `Population` class manages and stores population data, including demographic distributions and contact matrices
     for various layers (e.g., school, work, home, community). It provides methods to add and retrieve this data for use
     in simulations and analysis.

     Attributes:
         name (str): The name of the population.
         Nk (List): List representing population data for different demographic groups.
         Nk_names (List[str]): List of demographic group names.
         contact_matrices (Dict[str, np.ndarray]): Dictionary mapping layer names to their corresponding contact matrices
             (aggregated by age groups).

    Example 1: Online import (data will be fetched from GitHub)
     population_online = load_epydemix_population(
         population_name="United_States",
         # Specify the preferred contact data source (needed only if you want to override the default primary source)
         contacts_source="mistry_2021",
         layers=["home", "work", "school", "community"]  # Load contact layers (by default all layers are imported)
     )

     Example 2: Offline import (data will be loaded from a local directory)
     # Ensure that the folder is downloaded locally before running this
     population_offline = load_epydemix_population(
         population_name="United_States",
         path_to_data="path/to/local/epydemix_data/",  # Path to the local data folder
         # Specify the preferred contact data source (needed only if you want to override the default primary source)
         contacts_source="mistry_2021",
         layers=["home", "work", "school", "community"]  # Load contact layers (by default all layers are imported)
     )
    """

    def __init__(self, name: str = "population") -> None:
        """
        Initializes the Population object.

        Args:
            name (str, optional): Name of the population object. Defaults to "population".

        Attributes:
            name (str): Name of the population.
            contact_matrices (Dict[str, np.ndarray]): Dictionary to hold contact matrices for different layers.
            Nk (List[float]): List representing population data for different demographic groups.
            Nk_names (List[str]): List of demographic group names.
        """
        self.name = name
        self.contact_matrices = {}  # Dictionary of contact matrices for different layers
        self.Nk = []  # Population data
        self.Nk_names = []  # List of demographic group names

    def __repr__(self) -> str:
        """
        Returns a string representation of the Population object,
        summarizing its key attributes such as the name, number of demographic groups,
        and number of contact matrices.

        Returns:
            str: String representation of the Population object.
        """
        # General population info
        repr_str = f"Population(name='{self.name}')\n"
        repr_str += f"Demographic groups: {len(self.Nk)} groups\n"

        # Population group names and sizes if available
        if len(self.Nk) > 0 and len(self.Nk_names) > 0:
            repr_str += "Population distribution:\n"
            for name, size in zip(self.Nk_names, self.Nk):
                repr_str += f"  - {name}: {size} individuals\n"
        else:
            repr_str += "Population data not available\n"

        # Contact matrices summary
        repr_str += f"Contact matrices: {len(self.contact_matrices)} layers\n"
        if len(self.contact_matrices) > 0:
            repr_str += "Available layers:\n"
            for layer in self.contact_matrices.keys():
                repr_str += f"  - {layer}\n"
        else:
            repr_str += "No contact matrices available\n"

        return repr_str

    def add_contact_matrix(
        self, contact_matrix: np.ndarray, layer_name: str = "all"
    ) -> None:
        """
        Adds a contact matrix for a specified layer.

        Args:
            contact_matrix (np.ndarray): The contact matrix to be added, representing contact patterns
                between different demographic groups.
            layer_name (str, optional): The name of the contact layer (e.g., "home", "work").
                Defaults to "all". Cannot be "overall" as it's reserved.

        Raises:
            ValueError: If contact_matrix is not a 2D square array or if layer_name is "overall"

        Returns:
            None
        """
        # Validate layer name
        if layer_name == "overall":
            raise ValueError(
                '"overall" is a reserved layer name used for total contacts. '
                "Please use a different name for this layer."
            )

        # Cast contact_matrix to a numpy array
        contact_matrix = np.array(contact_matrix)

        # Check that contact_matrix is a 2D square numpy array
        if (
            len(contact_matrix.shape) != 2
            or contact_matrix.shape[0] != contact_matrix.shape[1]
        ):
            raise ValueError("Contact matrix must be a 2D square numpy array.")

        self.contact_matrices[layer_name] = contact_matrix

    def add_population(
        self, Nk: List[float], Nk_names: Optional[List[str]] = None
    ) -> None:
        """
        Adds population data for different demographic groups.

        Args:
            Nk (List[float]): A list representing the population size for each demographic group.
            Nk_names (Optional[List[str]], optional): A list of demographic group names. If not provided,
                                                      a default list of indices is generated. Defaults to None.

        Returns:
            None
        """

        # Cast Nk to a numpy array
        Nk = np.array(Nk)

        # Check that Nk is a 1d array
        if len(Nk.shape) != 1:
            raise ValueError("Nk must be a 1-dimensional array.")

        # If demographic group names are not provided, generate default names
        if Nk_names is None:
            Nk_names = np.array(range(len(Nk)))
        else:
            Nk_names = np.array(Nk_names)

        # check that Nk and Nk_names have the same length
        if len(Nk) != len(Nk_names):
            raise ValueError("Nk and Nk_names must have the same length.")

        self.Nk_names = Nk_names
        self.Nk = Nk

    @property
    def total_population(self) -> float:
        """
        Total population across all demographic groups.

        Returns:
            float: Sum of population in all demographic groups
        """
        return float(np.sum(self.Nk))

    @property
    def num_groups(self) -> int:
        """
        Number of demographic groups.

        Returns:
            int: Number of demographic groups in the population
        """
        return len(self.Nk)

    @property
    def layers(self) -> List[str]:
        """
        Available contact matrix layers.

        Returns:
            List[str]: Names of available contact layers (e.g., ['home', 'work', 'school'])
        """
        return list(self.contact_matrices.keys())

    @property
    def total_contacts(self) -> Dict[str, float]:
        """
        Total number of contacts per layer.

        Returns:
            Dict[str, float]: Dictionary mapping layer names to total contacts
        """
        return {
            layer: float(np.sum(matrix * self.Nk[:, np.newaxis]))
            for layer, matrix in self.contact_matrices.items()
        }

    @property
    def mean_contacts(self) -> Dict[str, float]:
        """
        Mean number of contacts per person per layer.

        Returns:
            Dict[str, float]: Dictionary mapping layer names to mean contacts per person
        """
        return {
            layer: total / self.total_population
            for layer, total in self.total_contacts.items()
        }

    def validate(self) -> None:
        """
        Validate all aspects of population data consistency.
        Raises ValueError if any validation fails.
        """
        self._validate_population_data()
        self._validate_contact_matrices()
        self._validate_demographic_names()

    def _validate_population_data(self) -> None:
        """
        Validate population size data.
        """
        if len(self.Nk) == 0:
            raise ValueError("No population data has been added")

        if len(self.Nk) != len(self.Nk_names):
            raise ValueError(
                f"Mismatch between population sizes ({len(self.Nk)}) "
                f"and names ({len(self.Nk_names)})"
            )

        if np.any(self.Nk < 0):
            raise ValueError("Population sizes cannot be negative")

        if np.any(~np.isfinite(self.Nk)):
            raise ValueError("Population sizes must be finite")

    def _validate_contact_matrices(self) -> None:
        """
        Validate contact matrices for all layers.
        """

        for layer, matrix in self.contact_matrices.items():
            # Check for negative values
            if np.any(matrix < 0):
                raise ValueError(f"Contact matrix '{layer}' contains negative values")

            # Check for non-finite values
            if np.any(~np.isfinite(matrix)):
                raise ValueError(f"Contact matrix '{layer}' contains non-finite values")

    def _validate_demographic_names(self) -> None:
        """
        Validate demographic group names.
        """
        if len(set(self.Nk_names)) != len(self.Nk_names):
            raise ValueError("Duplicate demographic group names found")


def map_age_groups_to_idx(
    age_group_mapping: Dict[str, List[str]],
    old_age_groups_idx: Dict[str, int],
    new_age_group_idx: Dict[str, int],
) -> Dict[int, int]:
    """
    Maps old age groups to new age groups using index mappings.

    Args:
        age_group_mapping (Dict[str, List[str]]): A dictionary where keys are new age groups,
                                                  and values are lists of old age groups.
        old_age_groups_idx (Dict[str, int]): A dictionary mapping old age group names to their respective indices.
        new_age_group_idx (Dict[str, int]): A dictionary mapping new age group names to their respective indices.

    Returns:
        Dict[int, int]: A dictionary mapping old age group indices to new age group indices.
    """

    # Initialize the result dictionary
    age_group_mapping_idx = {}

    # Iterate through each key-value pair in the first dictionary
    for new_group, old_groups in age_group_mapping.items():
        # Get the corresponding integer for this new group
        related_int = new_age_group_idx[new_group]

        # Map each group in the list to the related index
        for grp in old_groups:
            # Get the integer for the list item
            item_int = old_age_groups_idx[grp]
            # Set the mapping in the result dictionary
            age_group_mapping_idx[item_int] = related_int

    return age_group_mapping_idx


def aggregate_matrix(
    initial_matrix: np.ndarray,
    old_population: np.ndarray,
    new_population: np.ndarray,
    age_group_mapping: Dict[str, list],
    old_age_groups_idx: Dict[str, int],
    new_age_group_idx: Dict[str, int],
) -> np.ndarray:
    """
    Aggregates a contact matrix based on new demographic groupings.

    Args:
        initial_matrix (np.ndarray): The initial contact matrix (rates) between old demographic groups.
        old_population (np.ndarray): The population sizes of the old demographic groups.
        new_population (np.ndarray): The population sizes of the new aggregated demographic groups.
        age_group_mapping (Dict[str, list]): A dictionary mapping new demographic group names to lists of old group names.
        old_age_groups_idx (Dict[str, int]): A dictionary mapping old age group names to their indices in the contact matrix.
        new_age_group_idx (Dict[str, int]): A dictionary mapping new age group names to their indices in the aggregated matrix.

    Returns:
        np.ndarray: The aggregated contact matrix (rates) for the new demographic groups.
    """

    # Turn matrix of rates into contacts
    real_contacts = initial_matrix.copy()
    for i in range(real_contacts.shape[0]):
        real_contacts[i] = real_contacts[i] * old_population[i]

    # compute age group mapping
    age_group_mapping_idxs = map_age_groups_to_idx(
        age_group_mapping, old_age_groups_idx, new_age_group_idx
    )

    # Determine the number of aggregated groups
    num_aggregated_groups = max(age_group_mapping_idxs.values()) + 1

    # Initialize the aggregated matrix
    aggregated_matrix = np.zeros((num_aggregated_groups, num_aggregated_groups))

    # Fill the aggregated matrix
    for i in range(real_contacts.shape[0]):
        for j in range(real_contacts.shape[1]):
            aggregated_i = age_group_mapping_idxs[i]
            aggregated_j = age_group_mapping_idxs[j]
            aggregated_matrix[aggregated_i, aggregated_j] += real_contacts[i, j]

    # Turn into rates
    aggregated_matrix_rate = aggregated_matrix.copy()
    for i in range(aggregated_matrix_rate.shape[0]):
        aggregated_matrix_rate[i] = aggregated_matrix_rate[i] / new_population[i]

    return aggregated_matrix_rate


def aggregate_demographic(
    data: pd.DataFrame, grouping: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Aggregates demographic data based on a grouping dictionary.

    Args:
        data (pd.DataFrame): A DataFrame containing demographic data with columns 'group_name' and 'value'.
        grouping (Dict[str, List[str]]): A dictionary where keys are new group names and values are lists of original group names to aggregate.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'group_name' and 'value', where 'value' is the sum of the 'value' column from the original DataFrame for each new group.
    """
    Nk_new, Nk_names_new = [], []

    for new_group in grouping.keys():
        Nk_names_new.append(new_group)
        sum_value = data.loc[data.group_name.isin(grouping[new_group])]["value"].sum()
        Nk_new.append(sum_value)

    df_Nk_new = pd.DataFrame({"group_name": Nk_names_new, "value": Nk_new})

    return df_Nk_new


def validate_population_name(
    population_name: str, path_to_data: str, attribute: str = "age"
) -> None:
    """
    Validates if a given population name exists in the locations data.

    Args:
        population_name (str): The name of the population to validate.
        path_to_data (str): The path to the directory containing the data.
        attribute (str): The demographic attribute layer. Defaults to "age".

    Raises:
        ValueError: If the population_name is not found in the list of locations.
    """
    # Construct the full path to the locations CSV file
    locations_file = os.path.join(path_to_data, "data", attribute, "locations.csv")

    # Load the locations data and extract the list of locations
    locations_list = pd.read_csv(locations_file)["location"].values

    # Check if the population name is in the list of locations
    if population_name not in locations_list:
        raise ValueError(
            f"Location {population_name} not found in the list of supported locations. See {path_to_data}/locations.csv"
        )


def get_primary_contacts_source(
    population_name: str, path_to_data: str, attribute: str = "age"
) -> Optional[str]:
    """
    Retrieves the primary contact source for a given population name from the locations data.

    Args:
        population_name (str): The name of the population whose primary contact source is to be retrieved.
        path_to_data (str): The path to the directory containing the data.
        attribute (str): The demographic attribute layer. Defaults to "age".

    Returns:
        Optional[str]: The primary contact source for the given population name.
                       Returns None if the population name is not found.

    Raises:
        ValueError: If the population name is not found in the locations data.
    """
    # Construct the full path to the locations CSV file
    locations_file = os.path.join(path_to_data, "data", attribute, "locations.csv")

    # Load the contact matrices sources data
    contact_matrices_sources = pd.read_csv(locations_file)

    # Filter the data for the specified population name
    source_location = contact_matrices_sources.loc[
        contact_matrices_sources["location"] == population_name,
        "primary_contact_source",
    ]

    # Check if the population name was found
    if source_location.empty:
        raise ValueError(
            f"Population name '{population_name}' not found in {locations_file}."
        )

    # Retrieve and return the primary contact source
    return source_location.iloc[0]


def validate_contacts_source(
    contacts_source: str, supported_contacts_sources: List[str]
) -> None:
    """
    Validates if a given contacts source is in the list of supported contact sources.

    Args:
        contacts_source (str): The contact source to validate.
        supported_contacts_sources (List[str]): A list of supported contact sources.

    Raises:
        ValueError: If the contacts_source is not found in the list of supported sources.
    """
    if contacts_source not in supported_contacts_sources:
        raise ValueError(
            f"Source {contacts_source} not found in the list of supported sources. Supported sources are {supported_contacts_sources}"
        )


def validate_age_group_mapping(
    age_group_mapping: Dict[str, List[str]], allowed_values: List[str]
) -> None:
    """
    Validates that all age group mapping values are within the allowed values.

    Args:
        age_group_mapping (Dict[str, List[str]]): A dictionary where keys are age group names and values are lists of values for each age group.
        allowed_values (List[str]): A list of allowed values that the age group mapping values should be within.

    Raises:
        ValueError: If any value in the age group mapping is not in the list of allowed values.
    """
    values = np.concatenate(list(age_group_mapping.values()))
    if not np.all(np.isin(values, allowed_values)):
        raise ValueError(f"Age group mapping values must be in {allowed_values}")


def load_epydemix_population(
    population_name: str,
    contacts_source: Optional[str] = None,
    path_to_data: Optional[str] = None,
    layers: List[str] = ["school", "work", "home", "community"],
    age_group_mapping: Optional[Dict[str, List[str]]] = None,
    supported_contacts_sources: Dict[str, List[str]] = {
        "age": ["prem_2017", "prem_2021", "mistry_2021"]
    },
    data_version: str = "vtest",
    attribute: str = "age",
) -> "Population":
    """
    Loads population and contact matrix data for a specified population.

    Args:
        population_name (str): The name of the population to load.
        contacts_source (Optional[str]): The source of contact matrices. If None, the default source is retrieved.
        path_to_data (Optional[str]): The local path to the data directory. If None, data is fetched from GitHub.
        layers (List[str]): The layers of contact matrices to load.
        age_group_mapping (Optional[Dict[str, List[str]]]): Mapping of age groups. If None, defaults based on contacts_source.
        supported_contacts_sources (Dict[str, List[str]]): Dict mapping attribute names to their supported contact sources.
        data_version (str): The git tag/version of the epydemix-data repository. Defaults to "vtest".
        attribute (str): The demographic attribute layer. Defaults to "age".

    Returns:
        Population: An instance of the Population class with the loaded data.

    Raises:
        ValueError: If any provided value is not valid or if there are issues with the data files.
    """

    population = Population(name=population_name)

    # If path_to_data is None, use the GitHub URL
    is_remote = False
    if path_to_data is None:
        path_to_data = f"{EPYDEMIX_DATA_BASE_URL}/{data_version}/"
        is_remote = True  # Mark as remote URL

    # Validate population name
    validate_population_name(population_name, path_to_data, attribute=attribute)

    # Check if contacts_source is supported
    if contacts_source is None:
        contacts_source = get_primary_contacts_source(
            population_name, path_to_data, attribute=attribute
        )
    attribute_sources = supported_contacts_sources.get(attribute, [])
    if attribute_sources:
        validate_contacts_source(contacts_source, attribute_sources)

    # Load demographic data
    demographic_file = f"data/{attribute}/{population_name}/demographic/population.csv"

    if is_remote:
        df = pd.read_csv(path_to_data + demographic_file)  # Fetch from URL
    else:
        demographic_path = (
            Path(path_to_data)
            / "data"
            / attribute
            / population_name
            / "demographic"
            / "population.csv"
        )
        df = pd.read_csv(demographic_path)

    Nk = df  # Assign the loaded DataFrame

    if attribute == "age":
        # Handle contact matrices aggregation (age-specific)
        if contacts_source in ["prem_2017", "prem_2021"]:
            Nk = aggregate_demographic(Nk, demographic_grouping_prem)

        # Determine age group mapping
        if age_group_mapping is None:
            age_group_mapping = (
                contacts_age_group_mapping_prem
                if contacts_source in ["prem_2017", "prem_2021"]
                else contacts_age_group_mapping_mistry
            )

        validate_age_group_mapping(age_group_mapping, Nk.group_name.values)

        # Aggregate population data
        Nk_new = aggregate_demographic(Nk, age_group_mapping)
        population.add_population(
            Nk=Nk_new["value"].values, Nk_names=Nk_new["group_name"].values
        )
    else:
        # No aggregation for non-age attributes
        population.add_population(
            Nk=Nk["value"].values, Nk_names=Nk["group_name"].values
        )

    # Load contact matrices
    for layer_name in layers:
        contact_matrix_file = f"data/{attribute}/{population_name}/contact_matrices/{contacts_source}/{layer_name}.csv"

        if is_remote:
            C = pd.read_csv(
                path_to_data + contact_matrix_file, header=None
            ).values  # Load from URL
        else:
            contact_matrix_path = (
                Path(path_to_data)
                / "data"
                / attribute
                / population_name
                / "contact_matrices"
                / contacts_source
                / f"{layer_name}.csv"
            )
            C = pd.read_csv(
                contact_matrix_path, header=None
            ).values  # Load from local file

        if attribute == "age":
            # Aggregate contact matrices (age-specific)
            C_aggr = aggregate_matrix(
                C,
                old_population=Nk["value"].values,
                new_population=Nk_new["value"].values,
                age_group_mapping=age_group_mapping,
                old_age_groups_idx={
                    name: idx for idx, name in enumerate(Nk.group_name.values)
                },
                new_age_group_idx={
                    name: idx for idx, name in enumerate(age_group_mapping.keys())
                },
            )
            population.add_contact_matrix(C_aggr, layer_name=layer_name)
        else:
            population.add_contact_matrix(C, layer_name=layer_name)

    return population


def get_available_locations(
    path_to_data: Optional[str] = None,
    data_version: str = "vtest",
    attribute: str = "age",
) -> pd.DataFrame:
    """
    Returns a list of available locations.

    Args:
        path_to_data (Optional[str]): The local path to the data directory. If None, data is fetched from GitHub.
        data_version (str): The git tag/version of the epydemix-data repository. Defaults to "vtest".
        attribute (str): The demographic attribute layer. Defaults to "age".

    Returns:
        pd.DataFrame: A DataFrame containing the list of available locations.
    """

    # If path_to_data is None, use the GitHub URL
    is_remote = False
    if path_to_data is None:
        path_to_data = f"{EPYDEMIX_DATA_BASE_URL}/{data_version}/"
        is_remote = True  # Mark as remote URL

    if is_remote:
        locations_file = path_to_data + f"data/{attribute}/locations.csv"
    else:
        locations_file = Path(path_to_data) / "data" / attribute / "locations.csv"
    return pd.read_csv(locations_file)
