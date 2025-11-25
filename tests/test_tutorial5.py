import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from epydemix import load_predefined_model, simulate
from epydemix.calibration import ABCSampler
from epydemix.population import Population
from epydemix.utils import Perturbation, compute_simulation_dates
from epydemix.visualization import (
    plot_posterior_distribution,
    plot_quantiles,
)

colors = sns.color_palette("Dark2")


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


def test_calibration_advanced(mock_population):
    # import data and divide data into calibration and projection periods
    data = pd.read_csv(
        "https://raw.githubusercontent.com/epistorm/epydemix/refs/heads/main/tutorials/data/incidence_data.csv"
    )
    data["date"] = pd.to_datetime(data["date"])
    data_calibration = data.iloc[:-40]
    data_projection = data.iloc[-40:]

    # model
    model = load_predefined_model("SIR")
    model.set_population(mock_population)

    # initial conditions (we assume fully S population except for 0.05% infected individual across age groups)
    initial_conditions = {
        "Susceptible": model.population.Nk
        - (model.population.Nk * 0.05 / 100).astype(int),
        "Infected": (model.population.Nk * 0.05 / 100).astype(int),
        "Recovered": np.zeros(len(model.population.Nk)),
    }

    # simulation dates
    simulation_dates_calibration = compute_simulation_dates(
        start_date=data_calibration.date.values[0],
        end_date=data_calibration.date.values[-1],
    )
    simulation_dates_projection = compute_simulation_dates(
        start_date=data_calibration.date.values[0],
        end_date=data_projection.date.values[-1],
    )

    # simulation parameters
    parameters = {
        "initial_conditions_dict": initial_conditions,
        "epimodel": model,
        "start_date": data_calibration.date.values[0],
        "end_date": data_calibration.date.values[-1],
    }

    # priors
    priors = {
        "transmission_rate": stats.uniform(0.010, 0.020),
        "recovery_rate": stats.uniform(0.15, 0.1),
    }

    # wrapper function
    def simulate_wrapper(parameters):
        results = simulate(**parameters)
        return {"data": results.transitions["Susceptible_to_Infected_total"]}

    # initialize the ABCSampler object
    abc_sampler = ABCSampler(
        simulation_function=simulate_wrapper,
        priors=priors,
        parameters=parameters,
        observed_data=data_calibration["data"].values,
    )

    class UniformPerturbation(Perturbation):
        def __init__(self, param_name, scale=0.1):
            super().__init__(param_name)
            self.scale = scale

        def propose(self, x):
            """Propose a new value by adding scaled noise (with positive constraint)."""
            return np.random.uniform(x - self.scale, x + self.scale)

        def pdf(self, x, center):
            """Evaluate the PDF of the kernel."""
            return 1 / (2 * self.scale)

        def update(self, particles, weights, param_names):
            """Update the scale of the kernel based on the previous generation of particles."""
            index = param_names.index(self.param_name)
            values = particles[:, index]
            self.scale = 0.5 * (np.max(values) - np.min(values))

    results_uniform = abc_sampler.calibrate(
        strategy="smc",
        num_particles=10,
        num_generations=2,
        perturbations={
            "transmission_rate": UniformPerturbation("transmission_rate"),
            "recovery_rate": UniformPerturbation("recovery_rate"),
        },
    )

    results_abc_smc = abc_sampler.calibrate(
        strategy="smc", num_particles=10, num_generations=2
    )

    ax = plot_posterior_distribution(
        results_uniform.get_posterior_distribution(),
        "transmission_rate",
        kind="kde",
        title="Transmission rate",
        color=colors[0],
        label="Uniform Perturbation",
    )
    ax = plot_posterior_distribution(
        results_abc_smc.get_posterior_distribution(),
        "transmission_rate",
        kind="kde",
        title="Transmission rate",
        color=colors[1],
        label="Default Perturbation",
        ax=ax,
    )
    ax.legend()

    # create projection parameters
    projection_parameters = parameters.copy()
    projection_parameters["end_date"] = data_projection.date.values[-1]

    # run projections
    results_abc_smc = abc_sampler.run_projections(projection_parameters, iterations=5)

    # plot the projections
    df_quantiles_calibration = results_abc_smc.get_calibration_quantiles(
        simulation_dates_calibration
    )
    df_quantiles_projections = results_abc_smc.get_projection_quantiles(
        simulation_dates_projection
    )

    fig, ax = plt.subplots(dpi=300, figsize=(10, 3))
    plot_quantiles(
        df_quantiles_projections,
        columns="data",
        data=data,
        ax=ax,
        colors=colors[0],
        show_legend=True,
        show_data=True,
        labels=["New Infections, Projections"],
    )
    plot_quantiles(
        df_quantiles_calibration,
        columns="data",
        data=data_calibration,
        ax=ax,
        colors=colors[1],
        show_legend=False,
        show_data=True,
    )
