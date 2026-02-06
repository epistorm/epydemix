from abc import ABC, abstractmethod
from typing import Union

import numpy as np


def fast_normal_pdf(x, mean, std):
    var = std * std
    denom = np.sqrt(2 * np.pi * var)
    num = np.exp(-0.5 * ((x - mean) ** 2) / var)
    return num / denom


class Perturbation(ABC):
    def __init__(self, param_name):
        self.param_name = param_name

    @abstractmethod
    def propose(self, x):
        """Propose a new value based on the current value."""
        pass

    @abstractmethod
    def pdf(self, x, center):
        """Evaluate the PDF of the kernel."""
        pass

    @abstractmethod
    def update(self, particles, weights, param_names):
        """Update the kernel parameters based on particles and weights."""
        pass


class DefaultPerturbationContinuous(Perturbation):
    """
    Componenent-wise normal perturbation kernel with adaptive standard deviation (Beaumont et al. (2009)).
    """

    def __init__(self, param_name):
        super().__init__(param_name)
        self.std = 0.1

    def propose(self, x):
        """Propose a new value based on the current value."""
        return np.random.normal(x, self.std)

    def pdf(self, x, center):
        """Evaluate the PDF of the kernel."""
        return fast_normal_pdf(x, center, self.std)

    def update(self, particles, weights, param_names):
        """Update the standard deviation based on previous generation variance."""
        index = param_names.index(self.param_name)
        values = particles[:, index]
        std = np.std(values)
        self.std = std * np.sqrt(2)


class DefaultPerturbationDiscrete(Perturbation):
    def __init__(self, param_name, prior, jump_probability=0.3):
        super().__init__(param_name)
        self.prior = prior
        self.support = np.arange(self.prior.support()[0], self.prior.support()[1] + 1)
        self.jump_probability = jump_probability
        self.rest_prob = jump_probability / (len(self.support) - 1)

    def propose(self, x):
        """Propose a new value for the discrete parameter."""
        if np.random.rand() < self.jump_probability:
            proposed = x
            while proposed == x:
                proposed = np.random.choice(self.support)
            return proposed
        return x

    def pdf(self, x, center):
        """Transition probability for the discrete parameter."""
        if x == center:
            return 1 - self.jump_probability
        elif x in self.support:
            return self.rest_prob
        return 0

    def update(self, particles, weights, param_names):
        """Update jump_probability or other characteristics if needed."""
        pass


def sample_prior(priors, param_names):
    """Samples a parameter set from the given prior distributions.
    priors: dictionary mapping parameter names to scipy.stats distributions
    param_names: list of parameter names to maintain consistent order
    Returns: list of sampled parameter values in the order of param_names
    """
    return [priors[param].rvs() for param in param_names]


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    Computes the effective sample size (ESS) of a set of weights.

    Args:
        weights (np.ndarray): A NumPy array of weights for the particles.

    Returns:
        float: The effective sample size, calculated as the reciprocal of the sum of squared weights.
    """
    ess = 1 / np.sum(weights**2)
    return ess


def weighted_quantile(
    values: Union[np.ndarray, list], weights: Union[np.ndarray, list], quantile: float
) -> float:
    """
    Compute the weighted quantile of a dataset.

    Args:
        values (Union[np.ndarray, list]): Data values for which the quantile is to be computed.
        weights (Union[np.ndarray, list]): Weights associated with the data values.
        quantile (float): The quantile to compute, must be between 0 and 1.

    Returns:
        float: The weighted quantile value.

    Raises:
        ValueError: If `quantile` is not between 0 and 1.
    """
    # Ensure quantile is between 0 and 1
    if not (0 <= quantile <= 1):
        raise ValueError("Quantile must be between 0 and 1.")

    values = np.asarray(values)
    weights = np.asarray(weights)

    # Ensure that weights sum to 1
    weights /= np.sum(weights)

    # Sort values and weights by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute the cumulative sum of weights
    cumulative_weights = np.cumsum(sorted_weights)

    # Find the index where the cumulative weight equals or exceeds the quantile
    quantile_index = np.searchsorted(cumulative_weights, quantile)

    return sorted_values[quantile_index]
