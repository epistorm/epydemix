import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from epydemix import load_predefined_model
from epydemix.population import Population
from epydemix.utils import compute_simulation_dates
from epydemix.visualization import plot_quantiles, plot_spectral_radius

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


def test_modeling_interventions(mock_population):
    # import population and set simulation dates
    simulation_dates = compute_simulation_dates("2024-01-01", "2024-08-31")

    # create model with interventions and set population
    model_interventions = load_predefined_model("SIR", transmission_rate=0.035)
    model_interventions.set_population(mock_population)

    # create model without interventions and set population
    model_nointerventions = load_predefined_model("SIR", transmission_rate=0.035)
    model_nointerventions.set_population(mock_population)

    # Define the interventions by using a multplying factor
    model_interventions.add_intervention(
        layer_name="work",
        start_date="2024-01-15",
        end_date="2024-02-15",
        reduction_factor=0.3,
        name="workplace closure",
    )
    model_interventions.add_intervention(
        layer_name="school",
        start_date="2024-03-01",
        end_date="2024-05-01",
        reduction_factor=0.35,
        name="school closure",
    )

    # compute contact reductions (in order to plot them)
    model_interventions.compute_contact_reductions(simulation_dates)

    # plot percentage change in spectral radius with respect to the initial value
    plot_spectral_radius(model_interventions, legend_loc="center right", show_perc=True)
    plt.close()

    model_interventions.override_parameter(
        start_date="2024-02-01",
        end_date="2024-08-31",
        parameter_name="transmission_rate",
        value=0.02,
    )

    # simulate with 10 infected individuals
    results_interventions = model_interventions.run_simulations(
        start_date="2024-01-01",
        end_date="2024-08-31",
        Nsim=5,
        percentage_in_agents=10 / model_interventions.population.Nk.sum(),
    )

    results_nointerventions = model_nointerventions.run_simulations(
        start_date="2024-01-01",
        end_date="2024-08-31",
        Nsim=5,
        percentage_in_agents=10 / model_nointerventions.population.Nk.sum(),
    )

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

    df_interventions = results_interventions.get_quantiles_compartments()
    ax = plot_quantiles(
        df_interventions,
        columns=["Infected_total"],
        colors=colors[0],
        labels="I(t), Interventions",
        ax=ax,
    )

    df_no_interventions = results_nointerventions.get_quantiles_compartments()
    ax = plot_quantiles(
        df_no_interventions,
        columns=["Infected_total"],
        colors=colors[1],
        labels="I(t), No interventions",
        ax=ax,
    )
    plt.close()
