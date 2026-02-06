from typing import Dict, Tuple

import numpy as np


def validate_data(
    data: Dict, simulation: Dict, shape_check: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates that the input Dictionaries contain the required key 'data' and that the shapes of the data arrays match.

    Args:
        data (Dict): A Dictionary containing the observed data with a key "data" pointing to an array of observations.
        simulation (Dict): A Dictionary containing the simulated data with a key "data" pointing to an array of simulated values.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays: (observed_data, simulated_data).

    Raises:
        ValueError: If the required key 'data' is missing in either Dictionary or if the shapes of the data arrays do not match.
    """
    if "data" not in data or "data" not in simulation:
        raise ValueError("Both input Dictionaries must contain the key 'data'.")

    observed = np.array(data["data"])
    simulated = np.array(simulation["data"])

    if shape_check and observed.shape != simulated.shape:
        raise ValueError("The shapes of observed and simulated data arrays must match.")

    return observed, simulated


def rmse(data: Dict, simulation: Dict) -> float:
    """
    Computes the Root Mean Square Error (RMSE) between the observed data and the simulated data.

    Args:
        data (Dict): A Dictionary containing the observed data with a key "data" pointing to an array of observations.
        simulation (Dict): A Dictionary containing the simulated data with a key "data" pointing to an array of simulated values.

    Returns:
        float: The RMSE value indicating the average magnitude of the error between the observed and simulated data.
    """
    observed, simulated = validate_data(data, simulation)
    return np.sqrt(np.mean((observed - simulated) ** 2))


def wmape(data: Dict, simulation: Dict) -> float:
    """
    Computes the Weighted Mean Absolute Percentage Error (wMAPE) between the observed data and the simulated data.

    Args:
        data (Dict): A Dictionary containing the observed data with a key "data" pointing to an array of observations.
        simulation (Dict): A Dictionary containing the simulated data with a key "data" pointing to an array of simulated values.

    Returns:
        float: The wMAPE value indicating the weighted average of the absolute percentage errors between the observed and simulated data.
    """
    observed, simulated = validate_data(data, simulation)
    return np.sum(np.abs(observed - simulated)) / np.sum(np.abs(observed))


def ae(data: Dict, simulation: Dict) -> np.ndarray:
    """
    Computes the Absolute Error (AE) between the observed data and the simulated data.

    Args:
        data (Dict): A Dictionary containing the observed data with a key "data" pointing to an array of observations.
        simulation (Dict): A Dictionary containing the simulated data with a key "data" pointing to an array of simulated values.

    Returns:
        np.ndarray: An array of absolute errors between the observed and simulated data.
    """
    observed, simulated = validate_data(data, simulation)
    return np.abs(observed - simulated)


def mae(data: Dict, simulation: Dict) -> float:
    """
    Computes the Mean Absolute Error (MAE) between the observed data and the simulated data.

    Args:
        data (Dict): A Dictionary containing the observed data with a key "data" pointing to an array of observations.
        simulation (Dict): A Dictionary containing the simulated data with a key "data" pointing to an array of simulated values.

    Returns:
        float: The MAE value indicating the average of the absolute errors between the observed and simulated data.
    """
    observed, simulated = validate_data(data, simulation)
    return np.mean(np.abs(observed - simulated))


def mape(data: Dict, simulation: Dict) -> float:
    """
    Computes the Mean Absolute Percentage Error (MAPE) between the observed data and the simulated data.

    Args:
        data (Dict): A Dictionary containing the observed data with a key "data" pointing to an array of observations.
        simulation (Dict): A Dictionary containing the simulated data with a key "data" pointing to an array of simulated values.

    Returns:
        float: The MAPE value indicating the average of the absolute percentage errors between the observed and simulated data.
    """
    observed, simulated = validate_data(data, simulation)
    return np.mean(np.abs((observed - simulated) / observed))
