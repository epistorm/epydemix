import numpy as np


class StochasticSEIRAgeGroups:
    def __init__(
        self, S0, E0, I0, R0, beta, sigma, gamma, contact_matrix, population, time_steps
    ):
        """
        Initializes the stochastic SEIR model with age groups and a contact matrix.

        Args:
            S0 (np.ndarray): Initial number of susceptible individuals in each age group.
            E0 (np.ndarray): Initial number of exposed individuals in each age group.
            I0 (np.ndarray): Initial number of infected individuals in each age group.
            R0 (np.ndarray): Initial number of recovered individuals in each age group.
            beta (float): Transmission rate.
            sigma (float): Incubation rate (1/incubation period).
            gamma (float): Recovery rate.
            contact_matrix (np.ndarray): Contact matrix representing contact rates between age groups.
            population (np.ndarray): Total population in each age group.
            time_steps (int): Number of time steps for the simulation.
        """
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.contact_matrix = contact_matrix
        self.N = population
        self.time_steps = time_steps
        self.num_groups = len(S0)  # Number of age groups

    def simulate(self):
        """
        Runs a single simulation of the SEIR model for the defined number of time steps with age groups.

        Returns:
            results (dict): A dictionary with keys 'S', 'E', 'I', 'R' containing lists of population sizes
                            at each time step for each compartment and age group.
        """
        # Initialize the compartments for each age group
        S = self.S0.copy()
        E = self.E0.copy()
        I = self.I0.copy()
        R = self.R0.copy()

        results = {"S": [], "E": [], "I": [], "R": []}

        for t in range(self.time_steps):
            results["S"].append(S.copy())
            results["E"].append(E.copy())
            results["I"].append(I.copy())
            results["R"].append(R.copy())

            if (
                np.sum(I) == 0 and np.sum(E) == 0
            ):  # Stop if no more infected or exposed individuals
                break

            # Compute new exposures for each age group
            new_exposures = np.zeros(self.num_groups, dtype=int)
            for i in range(self.num_groups):
                # The force of infection for group i depends on contacts with all other groups
                force_of_infection = np.sum(self.contact_matrix[i, :] * I / self.N)
                infection_prob = 1 - np.exp(-self.beta * force_of_infection)
                new_exposures[i] = np.random.binomial(S[i], infection_prob)

            # Compute new infections and recoveries
            new_infections = np.random.binomial(E, 1 - np.exp(-self.sigma))
            new_recoveries = np.random.binomial(I, 1 - np.exp(-self.gamma))

            # Update compartments
            S -= new_exposures
            E += new_exposures - new_infections
            I += new_infections - new_recoveries
            R += new_recoveries

        return results

    def run_simulations(self, Nsim, quantiles=[0.25, 0.5, 0.75]):
        """
        Runs the SEIR model simulation Nsim times and computes the specified quantiles.

        Args:
            Nsim (int): The number of simulations to run.
            quantiles (list of float): A list of quantiles to compute (e.g., [0.25, 0.5, 0.75]).

        Returns:
            quantile_results (dict of dict): A dictionary of dictionaries where the outer key is the compartment ('S', 'E', 'I', 'R')
                                             and the inner key is the quantile, each containing an array of shape (num_groups, time_steps).
        """
        # Initialize lists to store all simulation results
        all_S = []
        all_E = []
        all_I = []
        all_R = []

        # Run Nsim simulations and collect results
        for _ in range(Nsim):
            results = self.simulate()
            all_S.append(np.array(results["S"]))
            all_E.append(np.array(results["E"]))
            all_I.append(np.array(results["I"]))
            all_R.append(np.array(results["R"]))

        # Find the longest simulation
        max_len = max([result.shape[0] for result in all_S])

        # Pad the trajectories with the last value to make them all the same length
        for i in range(Nsim):
            if all_S[i].shape[0] < max_len:
                padding = [
                    (0, max_len - all_S[i].shape[0]),
                    (0, 0),
                ]  # Only pad the time dimension
                all_S[i] = np.pad(all_S[i], padding, mode="edge")
                all_E[i] = np.pad(all_E[i], padding, mode="edge")
                all_I[i] = np.pad(all_I[i], padding, mode="edge")
                all_R[i] = np.pad(all_R[i], padding, mode="edge")

        # Convert lists to arrays to compute the quantiles
        all_S = np.array(all_S)
        all_E = np.array(all_E)
        all_I = np.array(all_I)
        all_R = np.array(all_R)

        # Compute the quantiles across the simulations and store them in a dict of dicts
        quantile_results = {
            "S": {q: np.quantile(all_S, q, axis=0) for q in quantiles},
            "E": {q: np.quantile(all_E, q, axis=0) for q in quantiles},
            "I": {q: np.quantile(all_I, q, axis=0) for q in quantiles},
            "R": {q: np.quantile(all_R, q, axis=0) for q in quantiles},
        }

        return quantile_results
