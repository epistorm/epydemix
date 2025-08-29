import numpy as np 
import pandas as pd
from collections.abc import Iterable
import datetime
import random
import string
from evalidate import Expr, base_eval_model
from typing import Union, Dict, List, Any, Optional, Tuple

def is_scalar(value):
    return np.isscalar(value) and not isinstance(value, (str, bytes))


def is_iterable(value):
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def validate_parameter_shape(
        key: str,
        value: Any,
        T: int,
        n_age: int
    ) -> None:
    """
    Validates the shape of the input value based on its type and expected dimensions.

    Args:
        key (str): The key associated with the value in the parameters dictionary.
        value (Union[np.ndarray, Iterable]): The value to be validated, which can be a NumPy array or an iterable.
        T (int): The expected length of the first dimension.
        n_age (int): The expected length of the second dimension.

    Raises:
        ValueError: If the value does not meet the required shape criteria or if the type is unsupported.
    """
    #if isinstance(value, (int, float)):
    if is_scalar(value):
        return  # Scalars don't need validation 

    elif is_iterable(value):
        value = np.array(value)

        if value.ndim == 1:  # 1D array
            if len(value) < T:
                raise ValueError(f"The length of the 1D iterable for parameter '{key}' is smaller than simulation length ({T}).")
        
        elif value.ndim == 2:  # 2D array
            if value.shape[0] != T and value.shape[0] != 1:
                raise ValueError(f"The first dimension of the 2D iterable for parameter '{key}' must be 1 or match simulation length ({T}).")
            if value.shape[1] != n_age:
                raise ValueError(f"The second dimension of the 2D iterable for parameter '{key}' must match number of age groups ({n_age}).")
        
        else:
            raise ValueError(f"Unsupported number of dimensions for parameter '{key}': {value.ndim}")
        
    else:
        raise ValueError(f"Unsupported type for parameter '{key}': {type(value)}")


def resize_parameter(
        value: Any,
        T: int,
        n_age: int
    ) -> np.ndarray:
    """
    Resizes the input value to have the shape (T, n_age).

    Args:
        value (Union[np.ndarray, int, float]): The value to be resized, which can be a NumPy array or a scalar.
        T (int): The length of the first dimension.
        n_age (int): The length of the second dimension.

    Returns:
        np.ndarray: A 2D array with shape (T, n_age).
    """ 
    if is_scalar(value): # Scalar value
        return np.full((T, n_age), value)

    value = np.array(value)

    if value.ndim == 1:  # 1D array
        return np.tile(value, (n_age, 1)).T

    elif value.ndim == 2:  # 2D array
        if value.shape[0] == 1:  # If the first dimension is 1, repeat it to match T
            return np.tile(value, (T, 1))
        return value
    

def create_definitions(
        parameters: Dict[str, Any],
        T: int,
        n_age: int
    ) -> Dict[str, np.ndarray]:
    """
    Generates a dictionary where each value is a 2D array based on the input dictionary and the parameters provided.

    Args:
        parameters (Dict[str, Union[np.ndarray, int, float, Iterable]]): A dictionary where values can be either scalars or iterables.
        T (int): The length of the first dimension of the arrays to be created.
        n_age (int): The length of the second dimension of the arrays to be created.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are the same as in `parameters` and values are 2D arrays of shape `(T, n_age)`.

    Raises:
        ValueError: If any parameter value does not meet the required shape criteria.
    """
    definitions = {}
    for key, value in parameters.items():
        validate_parameter_shape(key, value, T, n_age)
        definitions[key] = resize_parameter(value, T, n_age)
    
    return definitions


def format_simulation_output(
        compartments_evolution: np.ndarray,
        transitions_evolution: np.ndarray,
        compartments_idx: Dict[str, int],
        transitions_idx: Dict[str, int], 
        demographics: List[str],
    ) -> Dict[str, np.ndarray]:
    """
    Format simulation output into dictionary with proper naming.
    
    Args:
        compartments_evolution: Array of shape (timesteps, n_compartments, n_demographics)
        transitions_evolution: Array of shape (timesteps, n_transitions, n_demographics)
        compartments_idx: Dictionary mapping compartment names to their indices
        transitions_idx: Dictionary mapping transition names to their indices
        demographics: List of demographic group names
    """
    formatted_output = {
        "compartments": {},
        "transitions": {}
    }
    for comp, pos in compartments_idx.items(): 
        for i, dem in enumerate(demographics): 
            key = f"{comp}_{dem}"
            formatted_output["compartments"][key] = compartments_evolution[:, pos, i]
        formatted_output["compartments"][f"{comp}_total"] = np.sum(compartments_evolution[:, pos, :], axis=1)
    
    for trans, pos in transitions_idx.items(): 
        for i, dem in enumerate(demographics): 
            key = f"{trans}_{dem}"
            formatted_output["transitions"][key] = transitions_evolution[:, pos, i]
        formatted_output["transitions"][f"{trans}_total"] = np.sum(transitions_evolution[:, pos, :], axis=1)
    
    return formatted_output


def str_to_date(date_str: str) -> datetime.date:
    """
    Converts a date string in the format 'YYYY-MM-DD' to a `datetime.date` object.

    Args:
        date_str (str): A string representing a date in the format 'YYYY-MM-DD'.

    Returns:
        datetime.date: A `datetime.date` object corresponding to the input string.

    Raises:
        ValueError: If the input string is not in the correct format.
    """
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()


def apply_overrides(
        definitions: Dict[str, np.ndarray],
        overrides: Dict[str, List[Dict[str, Any]]],
        dates: List[datetime.date]
    ) -> Dict[str, np.ndarray]:
    """
    Applies parameter overrides to the definitions based on the specified date ranges.

    Args:
        definitions (dict): A dictionary where keys are parameter names and values are 2D arrays
                             representing the parameter values over time and demographics.
        overrides (dict): A dictionary where keys are parameter names and values are lists of override
                          specifications. Each override specification is a dictionary containing:
                          - 'start_date' (str): The start date of the override period in 'YYYY-MM-DD' format.
                          - 'end_date' (str): The end date of the override period in 'YYYY-MM-DD' format.
                          - 'value' (np.ndarray or scalar): The value to override within the specified date range.
        dates (list): A list of `datetime.date` objects corresponding to the time steps in the definitions arrays.

    Returns:
        dict: A dictionary with the same keys as `definitions`, but with values updated according to the overrides.

    Raises:
        ValueError: If the `override` values do not match the expected shape for the specified date ranges.
    """
    if not overrides:
        return definitions
    
    result = definitions.copy()
    
    for name, overrides in overrides.items():
        if name not in definitions:
            continue
        #values = definitions[name]
        for override in overrides:
            #start_date = str_to_date(override["start_date"])
            #end_date = str_to_date(override["end_date"])
            start_date = pd.Timestamp(override["start_date"])
            end_date = pd.Timestamp(override["end_date"])
            override_value = override["value"]

            # Convert dates to pandas timestamps for comparison
            dates_pd = pd.DatetimeIndex(dates)

            # validate override value
            T = sum((dates_pd >= start_date) & (dates_pd <= end_date))
            n_age = definitions[name].shape[1]
            validate_parameter_shape(name, override_value, T=T, n_age=n_age)

            # resize override value
            override_array = resize_parameter(override_value, T=T, n_age=n_age)

            # override
            #override_idxs = [i for i, date in enumerate(dates) if start_date <= date.date() <= end_date]
            #values[override_idxs] = override_value
            # Apply override
            mask = (dates_pd >= start_date) & (dates_pd <= end_date)
            result[name][mask] = override_array

    return result


def generate_unique_string(length: int = 12) -> str:
    """
    Generates a random unique string containing only letters.

    Args:
        length (int, optional): The length of the generated string. Defaults to 12.

    Returns:
        str: A random unique string containing both uppercase and lowercase letters.
    """
    letters = string.ascii_letters  # Contains both lowercase and uppercase letters
    return ''.join(random.choice(letters) for _ in range(length))


def compute_days(start_date: Union[str, pd.Timestamp], end_date: Union[str, pd.Timestamp]) -> int:
    """
    Computes the number of days between two dates.

    Args:
        start_date (Union[str, pd.Timestamp]): The start date in string format (e.g., "YYYY-MM-DD") or as a pandas Timestamp.
        end_date (Union[str, pd.Timestamp]): The end date in string format (e.g., "YYYY-MM-DD") or as a pandas Timestamp.

    Returns:
        int: The number of days between the start date and end date, inclusive.
    """
    return pd.date_range(start_date, end_date).shape[0]


def evaluate(expr: str, env: dict) -> any:
    """
    Evaluates the expression with the given environment, allowing only whitelisted operations.

    This function extends the base evaluation model to whitelist the 'Mult' (multiplication) 
    and 'Pow' (power) operations, ensuring that only these operations are permitted during 
    the evaluation.

    Args:
        expr (str): The expression to evaluate. It is expected to be a string containing 
                    the mathematical expression to be evaluated.
        env (dict): The environment containing variable values. Keys should be variable names 
                    and values should be their corresponding numeric values.

    Returns:
        any: The result of evaluating the expression. The result type depends on the expression 
             and its evaluation.

    Raises:
        EvalException: If there is an error in evaluating the expression, such as an invalid 
                       operation or an undefined variable.
    """
    eval_model = base_eval_model
    eval_model.nodes.extend(['Mult', 'Pow'])
    return Expr(expr, model=eval_model).eval(env)


def compute_simulation_dates(start_date: Union[str, datetime.date, np.datetime64],
                           end_date: Union[str, datetime.date, np.datetime64],
                           steps: Optional[int] = None,
                           dt: float = 1.0) -> np.ndarray:
    """
    Compute simulation dates including sub-day intervals.
    
    Args:
        start_date: Start date
        end_date: End date
        steps: Number of steps (if None, computed from dt)
        dt: Time step in days (e.g., 1/3 for 8-hour intervals)
    
    Returns:
        numpy array of datetime64 objects for each simulation step
    """
    # Convert inputs to pandas Timestamp for consistent handling
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    elif isinstance(start_date, np.datetime64):
        start_date = pd.Timestamp(start_date)
    
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    elif isinstance(end_date, np.datetime64):
        end_date = pd.Timestamp(end_date)
    
    # Calculate total duration in days
    total_days = (end_date - start_date).total_seconds() / (24 * 3600)
    
    # Calculate number of steps if not provided
    if steps is None:
        steps = int(total_days / dt)
    
    # Create timestamps with proper sub-day intervals
    timestamps = pd.date_range(
        start=start_date,
        periods=steps + 1,
        freq=pd.Timedelta(days=dt)
    ).values
    
    return timestamps


def apply_initial_conditions(epimodel, initial_conditions_dict) -> np.ndarray:
    """
    Applies initial conditions to the compartments of an epidemiological model.

    Args:
        epimodel (EpiModel): An instance of an epidemiological model containing compartments and population information.
        **kwargs: Keyword arguments where each key is a compartment name and each value is an array or list 
                  specifying the initial population for that compartment.

    Returns:
        np.ndarray: A 2D array where rows correspond to compartments and columns correspond to demographic groups, 
                    representing the initial conditions of the model. The shape of the array is 
                    `(number_of_compartments, number_of_demographic_groups)`.
    """
    # initialize population in different compartments and demographic groups
    initial_conditions = np.zeros((len(epimodel.compartments), len(epimodel.population.Nk)), dtype='int')
    for comp in epimodel.compartments:
        if comp in initial_conditions_dict: 
            if comp in epimodel.compartments:
                initial_conditions[epimodel.compartments_idx[comp]] = initial_conditions_dict[comp]

    return initial_conditions


def convert_to_2Darray(lst: List[Any]) -> np.ndarray:
    """
    Converts a list into a 2D NumPy array.

    Args:
        lst (List[Any]): A list of elements to be converted into a 2D array. Elements can be of any type.

    Returns:
        np.ndarray: A 2D NumPy array with shape `(1, len(lst))`, where `len(lst)` is the length of the input list.
    """
    arr = np.array(lst)
    arr = arr.reshape(1, len(lst))
    return arr


def combine_simulation_outputs(
        simulation_outputs_list: List[Dict[str, np.ndarray]]
        ) -> Dict[str, List[np.ndarray]]:
    """
    Combines multiple simulation outputs into a single dictionary by appending new outputs to existing keys.

    Args:
        simulation_outputs_list (List[Dict[str, np.ndarray]]): A list of dictionaries containing the latest simulation output to be combined.
                                                              Each dictionary contains keys corresponding to compartment-demographic names and values are arrays of simulation results.
    Returns:
        Dict[str, List[np.ndarray]]: A dictionary where keys are compartment-demographic names and values are lists of arrays.
                                     Each list contains simulation results accumulated from multiple runs.
    """
    combined_simulation_outputs = {}    
    for simulation_outputs in simulation_outputs_list:
        if not combined_simulation_outputs:
            # If combined_dict is empty, initialize it with the new dictionary
            for key in simulation_outputs:
                combined_simulation_outputs[key] = [simulation_outputs[key]]
        else:
            # If combined_dict already has keys, append the new dictionary's values
            for key in simulation_outputs:
                if key in combined_simulation_outputs:
                    combined_simulation_outputs[key].append(simulation_outputs[key])
                else:
                    combined_simulation_outputs[key] = [simulation_outputs[key]]
    # cast lists to arrays
    for key in combined_simulation_outputs:
        combined_simulation_outputs[key] = np.array(combined_simulation_outputs[key])
    return combined_simulation_outputs


def chain_multinomial(n, p, stay_idx, rng=None, check_tol=1e-12):
    """
    Competing-risks multinomial with an explicit 'stay' category. 
    The function assumes that the probability of staying is 0 (i.e., p does not include the probability of staying).

    Args:
        n (int): Number of individuals in the source compartment.
        p (array-like): Length-K vector. For k != stay_idx, p[k] = r_k * dt.
        stay_idx (int): Index of the 'stay' category.
        rng : np.random.Generator or seed or None. If None, uses default_rng().

    Returns:
        counts (np.ndarray of int, shape (K,)): Realized counts for each category (including 'stay'), summing to n.
    """
    p = np.asarray(p, dtype=float)
    K = p.size
    if not (0 <= stay_idx < K):
        raise IndexError("stay_idx out of bounds.")
    if np.any(p < -1e-16):
        raise ValueError("All entries in p must be non-negative.")

    # Total hazard H = sum over all entries (stay entry should be 0)
    H = float(p.sum())

    rng = np.random.default_rng(rng)
    counts = np.zeros(K, dtype=int)

    if H <= check_tol:
        # No hazard -> everyone stays
        counts[stay_idx] = n
        return counts

    # Probability of leaving given total hazard H
    p_leave = -np.expm1(-H)         
    p_leave = min(max(p_leave, 0.0), 1.0)  # clamp for floating-point safety

    # Draw total exits
    exits = rng.binomial(n, p_leave)
    stays = n - exits

    if exits == 0:
        counts[stay_idx] = n
        return counts

    # Split exits proportionally to hazards (stay has zero weight, so gets 0)
    q = p / H
    alloc = rng.multinomial(exits, q)  # alloc.sum() == exits, alloc[stay_idx] == 0

    # Fill output: allocated exits + residual stays
    counts[:] = alloc
    counts[stay_idx] = stays  # overwrite (alloc[stay_idx] is 0 anyway)

    return counts
