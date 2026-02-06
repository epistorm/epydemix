import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

from epydemix import EpiModel, load_predefined_model
from epydemix.population import Population
from epydemix.utils import compute_simulation_dates, convert_to_2Darray
from epydemix.visualization import plot_quantiles


@pytest.fixture
def mock_population():
    population = Population()
    population.add_population(
        np.random.randint(0, 100000, size=5), ["0-9", "10-19", "20-29", "30-39", "40+"]
    )
    population.add_contact_matrix(np.random.random(size=(5, 5)), "school")
    population.add_contact_matrix(np.random.random(size=(5, 5)), "work")
    population.add_contact_matrix(np.random.random(size=(5, 5)), "home")
    population.add_contact_matrix(np.random.random(size=(5, 5)), "community")
    return population


def test_new_transitions():
    def compute_behavioral_transition_probability(params, data):
        """
        Compute the probability of a behavioral transition.

        Args:
            params: The parameters of the transition.
            data: A dictionary containing information about the current state of the system.
                - parameters: The model parameters. This is a dictionary of arrays, where the key is the name of the parameter and the value is a
                            2D array of shape (n_time_steps, n_groups). The first dimension is the time step, the second is the demographic group.
                - t: The current time step
                - comp_indices: The indices of the compartments. This is a dictionary where the key is the name of the compartment and the
                                value is the index of the compartment in the system.
                - contact_matrix: The contact matrix
                - pop: The population in different compartments. This is a 2D array of shape (n_compartments, n_groups). The first dimension is
                    the compartment, the second is the demographic group.
                - pop_sizes: The population sizes. This is a 1D array of shape (n_groups,).
                - dt: The time step size
        """
        beta_B, gamma = (
            data["parameters"][params[0]][data["t"]],
            data["parameters"][params[1]][data["t"]],
        )
        agent_idx = data["comp_indices"][params[2]]
        interaction = beta_B * (1 - np.exp(-gamma * np.sum(data["pop"][agent_idx])))
        return 1 - np.exp(-interaction * data["dt"])

    model = EpiModel(
        compartments=["S", "SB", "I", "R"],
        parameters={
            "beta": 0.3,
            "mu": 0.1,
            "r": 0.3,
            "beta_B": 0.05,
            "gamma": 1.0 / 10000.0,
        },
    )

    model.register_transition_kind(
        kind="behavioral", function=compute_behavioral_transition_probability
    )

    model.add_transition(source="S", target="I", params=("beta", "I"), kind="mediated")
    model.add_transition(
        source="SB", target="I", params=("r*beta", "I"), kind="mediated"
    )
    model.add_transition(
        source="S", target="SB", params=("beta_B", "gamma", "I"), kind="behavioral"
    )
    model.add_transition(source="I", target="R", params="mu", kind="spontaneous")

    # Initial conditions
    initial_conditions = {"S": 100000 - 10, "SB": 0, "I": 10, "R": 0}

    # running the simulations
    results = model.run_simulations(
        start_date="2024-01-01",
        end_date="2024-04-10",
        initial_conditions_dict=initial_conditions,
        Nsim=5,
    )

    # plot
    df_quantiles = results.get_quantiles_compartments()
    plot_quantiles(
        df_quantiles,
        columns=["S_total", "SB_total", "I_total", "R_total"],
        legend_loc="upper right",
    )
    plt.close()


def test_varying_params(mock_population):
    start_date, end_date = "2024-01-01", "2024-04-10"

    # define SIR model
    model = load_predefined_model("SIR")

    # get simulation dates
    simulation_dates = compute_simulation_dates(
        start_date=start_date, end_date=end_date
    )

    def create_seasonal_parameter(
        n_points: int, min_val: float, max_val: float, period: int
    ) -> np.ndarray:
        """
        Create a sinusoidal parameter with specified inputs.
        """
        amplitude = (max_val - min_val) / 2
        offset = min_val + amplitude
        t = np.arange(n_points)
        frequency = 2 * np.pi / period
        return amplitude * np.sin(frequency * t) + offset

    time_varying_transmission_rate = create_seasonal_parameter(
        len(simulation_dates), 0.15, 0.35, 14
    )
    plt.plot(time_varying_transmission_rate)
    plt.close()

    # add time-varying transmission rate to the model
    model.add_parameter(
        parameter_name="transmission_rate", value=time_varying_transmission_rate
    )

    # run simulations
    results = model.run_simulations(start_date=start_date, end_date=end_date, Nsim=5)

    # plot results
    df_quantiles = results.get_quantiles_compartments()
    plot_quantiles(
        df_quantiles,
        columns=["Susceptible_total", "Infected_total", "Recovered_total"],
        legend_loc="upper right",
    )
    plt.close()
    # create model and add population
    model = load_predefined_model("SIR")
    model.set_population(mock_population)

    print(model)

    # Populations has 5 age groups: 0-4, 5-19, 20-49, 50-64, 65+
    age_varying_transmission_rate = convert_to_2Darray([0.01, 0.02, 0.02, 0.02, 0.02])

    # add age-varying transmission rate to the model
    model.add_parameter(
        parameter_name="transmission_rate", value=age_varying_transmission_rate
    )

    # run simulations
    results = model.run_simulations(start_date=start_date, end_date=end_date, Nsim=5)

    # plot results
    df_quantiles = results.get_quantiles_compartments()
    plot_quantiles(
        df_quantiles,
        columns=[
            "Infected_0-9",
            "Infected_10-19",
            "Infected_20-29",
            "Infected_30-39",
            "Infected_40+",
        ],
        legend_loc="upper right",
    )
    plt.close()

    varying_transmission_rate = np.zeros((len(simulation_dates), 5))
    for i in range(5):
        if i == 0:
            varying_transmission_rate[:, i] = create_seasonal_parameter(
                len(simulation_dates), 0.01, 0.02, 14
            )
        else:
            varying_transmission_rate[:, i] = create_seasonal_parameter(
                len(simulation_dates), 0.015, 0.025, 14
            )

    # add time-varying and age-varying transmission rate to the model
    model.add_parameter(
        parameter_name="transmission_rate", value=varying_transmission_rate
    )

    # run simulations
    results = model.run_simulations(start_date=start_date, end_date=end_date, Nsim=5)

    # plot results
    df_quantiles = results.get_quantiles_compartments()
    plot_quantiles(
        df_quantiles,
        columns=[
            "Infected_0-9",
            "Infected_10-19",
            "Infected_20-29",
            "Infected_30-39",
            "Infected_40+",
        ],
        legend_loc="upper right",
    )
    plt.close()


def test_shorter_dt():
    start_date, end_date = "2024-01-01", "2024-04-10"
    model = load_predefined_model("SIR")

    # run the models with 1/3 day time steps (by default, the data is resampled at daily frequency)
    results_shorter_dt = model.run_simulations(
        start_date=start_date, end_date=end_date, dt=1 / 3, Nsim=5
    )

    # run the models with 1 day time steps and 1 week resampling frequency
    results_resampled = model.run_simulations(
        start_date=start_date, end_date=end_date, dt=1, resample_frequency="W", Nsim=5
    )

    # plot results
    plot_quantiles(
        results_shorter_dt.get_quantiles_compartments(),
        columns=["Susceptible_total", "Infected_total", "Recovered_total"],
        legend_loc="upper right",
    )
    plot_quantiles(
        results_resampled.get_quantiles_compartments(),
        columns=["Susceptible_total", "Infected_total", "Recovered_total"],
        legend_loc="upper right",
    )
    plt.close()
    # plot total weekly new infections
    plot_quantiles(
        results_resampled.get_quantiles_transitions(),
        columns=["Susceptible_to_Infected_total"],
        legend_loc="upper right",
    )
    plt.close()
    # run the models with 1 day time steps but resample hourly
    results_resampled = model.run_simulations(
        start_date=start_date, end_date=end_date, dt=1, resample_frequency="1h", Nsim=5
    )
    plot_quantiles(
        results_resampled.get_quantiles_compartments(),
        columns=["Susceptible_total", "Infected_total", "Recovered_total"],
        legend_loc="center right",
    )
    plt.close()
