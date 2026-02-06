import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from epydemix import load_predefined_model, simulate
from epydemix.calibration import ABCSampler, rmse
from epydemix.population import Population
from epydemix.utils import compute_simulation_dates
from epydemix.visualization import (
    plot_distance_distribution,
    plot_posterior_distribution,
    plot_posterior_distribution_2d,
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


def test_calibration(mock_population):
    data = pd.read_csv(
        "https://raw.githubusercontent.com/epistorm/epydemix/refs/heads/main/tutorials/data/incidence_data.csv"
    )
    data["date"] = pd.to_datetime(data["date"])

    fig, ax = plt.subplots(dpi=300, figsize=(10, 3))
    plt.plot(
        data["date"], data["data"], label="I", color="k", linestyle="None", marker="o"
    )
    plt.xlabel("Date")
    plt.ylabel("New Infections")

    model = load_predefined_model("SIR")
    model.set_population(mock_population)
    print(model)

    # initial conditions (we assume fully S population except for 0.05% infected individual across age groups)
    initial_conditions = {
        "Susceptible": model.population.Nk
        - (model.population.Nk * 0.05 / 100).astype(int),
        "Infected": (model.population.Nk * 0.05 / 100).astype(int),
        "Recovered": np.zeros(len(model.population.Nk)),
    }

    # simulation dates
    simulation_dates = compute_simulation_dates(
        start_date=data.date.values[0], end_date=data.date.values[-1]
    )

    # simulation parameters
    parameters = {
        "initial_conditions_dict": initial_conditions,
        "epimodel": model,
        "start_date": data.date.values[0],
        "end_date": data.date.values[-1],
    }

    priors = {
        "transmission_rate": stats.uniform(0.010, 0.020),
        "recovery_rate": stats.uniform(0.15, 0.1),
    }

    def simulate_wrapper(parameters):
        results = simulate(**parameters)
        return {"data": results.transitions["Susceptible_to_Infected_total"]}

    # initialize the ABCSampler object
    abc_sampler = ABCSampler(
        simulation_function=simulate_wrapper,
        priors=priors,
        parameters=parameters,
        observed_data=data["data"].values,
        distance_function=rmse,
    )

    results_abc_smc = abc_sampler.calibrate(
        strategy="smc", num_particles=5, num_generations=2
    )

    results_abc_rejection = abc_sampler.calibrate(
        strategy="rejection", num_particles=5, epsilon=5500000000
    )

    results_top_perc = abc_sampler.calibrate(
        strategy="top_fraction", Nsim=20, top_fraction=0.1
    )

    # compute quantiles
    df_quantiles_abc_smc = results_abc_smc.get_calibration_quantiles(
        dates=simulation_dates
    )
    df_quantiles_abc_rejection = results_abc_rejection.get_calibration_quantiles(
        dates=simulation_dates
    )
    df_quantiles_top_perc = results_top_perc.get_calibration_quantiles(
        dates=simulation_dates
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), dpi=300)
    plot_quantiles(
        df_quantiles_abc_smc,
        columns="data",
        data=data,
        ax=axes[0],
        title="ABC-SMC",
        colors=colors[0],
        show_data=True,
        labels=["New Infections"],
    )
    plot_quantiles(
        df_quantiles_abc_rejection,
        columns="data",
        data=data,
        ax=axes[1],
        show_legend=False,
        title="ABC Rejection",
        colors=colors[1],
        show_data=True,
    )
    plot_quantiles(
        df_quantiles_top_perc,
        columns="data",
        data=data,
        ax=axes[2],
        show_legend=False,
        title="Top X%",
        colors=colors[2],
        show_data=True,
    )
    plt.tight_layout()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300, sharex=True, sharey=True)

    plot_posterior_distribution_2d(
        results_abc_smc.get_posterior_distribution(),
        "transmission_rate",
        "recovery_rate",
        ax=axes[0],
        kind="kde",
        title="ABC-SMC",
        prior_range=False,
    )
    plot_posterior_distribution_2d(
        results_abc_rejection.get_posterior_distribution(),
        "transmission_rate",
        "recovery_rate",
        ax=axes[1],
        kind="kde",
        title="ABC Rejection",
        prior_range=False,
    )
    plot_posterior_distribution_2d(
        results_top_perc.get_posterior_distribution(),
        "transmission_rate",
        "recovery_rate",
        ax=axes[2],
        kind="kde",
        title="Top X%",
        prior_range=False,
    )
    plt.tight_layout()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3), dpi=300)
    plot_posterior_distribution(
        results_abc_smc.get_posterior_distribution(),
        "transmission_rate",
        ax=axes[0],
        kind="kde",
        title="Transmission rate",
        prior_range=False,
        color=colors[0],
        label="ABC-SMC",
    )
    plot_posterior_distribution(
        results_abc_rejection.get_posterior_distribution(),
        "transmission_rate",
        ax=axes[0],
        kind="kde",
        title="Transmission rate",
        prior_range=False,
        color=colors[1],
        label="ABC Rejection",
    )
    plot_posterior_distribution(
        results_top_perc.get_posterior_distribution(),
        "transmission_rate",
        ax=axes[0],
        kind="kde",
        title="Transmission rate",
        prior_range=False,
        color=colors[2],
        label="Top X%",
    )
    axes[0].legend()

    plot_posterior_distribution(
        results_abc_smc.get_posterior_distribution(),
        "recovery_rate",
        ax=axes[1],
        kind="kde",
        title="Recovery rate",
        prior_range=False,
        color=colors[0],
        label="ABC-SMC",
    )
    plot_posterior_distribution(
        results_abc_rejection.get_posterior_distribution(),
        "recovery_rate",
        ax=axes[1],
        kind="kde",
        title="Recovery rate",
        prior_range=False,
        color=colors[1],
        label="ABC Rejection",
    )
    plot_posterior_distribution(
        results_top_perc.get_posterior_distribution(),
        "recovery_rate",
        ax=axes[1],
        kind="kde",
        title="Recovery rate",
        prior_range=False,
        color=colors[2],
        label="Top X%",
    )
    axes[1].legend()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    plot_distance_distribution(
        results_abc_smc.get_distances(),
        ax=ax,
        kind="kde",
        color=colors[0],
        label="ABC-SMC",
        xlabel="RMSE",
    )
    plot_distance_distribution(
        results_abc_rejection.get_distances(),
        ax=ax,
        kind="kde",
        color=colors[1],
        label="ABC Rejection",
        xlabel="RMSE",
    )
    plot_distance_distribution(
        results_top_perc.get_distances(),
        ax=ax,
        kind="kde",
        color=colors[2],
        label="Top X%",
        xlabel="RMSE",
    )
    ax.legend()
