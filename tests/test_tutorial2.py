import matplotlib.pyplot as plt
import numpy as np
import pytest

from epydemix import load_predefined_model
from epydemix.population import Population
from epydemix.visualization import plot_contact_matrix, plot_population, plot_quantiles


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


def test_model_with_population(mock_population):
    fig, axes = plt.subplots(nrows=2, ncols=2, dpi=300)
    plot_contact_matrix(
        mock_population, "school", ax=axes[0, 0], fontsize=7, show_values=True
    )
    plot_contact_matrix(
        mock_population, "work", ax=axes[0, 1], fontsize=7, show_values=True
    )
    plot_contact_matrix(
        mock_population, "home", ax=axes[1, 0], fontsize=7, show_values=True
    )
    plot_contact_matrix(
        mock_population, "community", ax=axes[1, 1], fontsize=7, show_values=True
    )
    plt.tight_layout()
    plt.close()

    fig, axes = plt.subplots(ncols=2, dpi=300, figsize=(10, 5))
    plot_population(
        mock_population, ax=axes[0], title="Population Distribution (absolute numbers)"
    )
    plot_population(
        mock_population,
        ax=axes[1],
        title="Population Distribution (percentages)",
        show_perc=True,
    )
    plt.close()

    my_population = Population(name="My Population")
    my_population.add_population(Nk=[100, 100], Nk_names=["A", "B"])
    my_population.add_contact_matrix(
        contact_matrix=[[0.2, 0.3], [0.3, 0.2]], layer_name="all"
    )

    print(my_population)

    # create a simple SIR model
    model = load_predefined_model("SIR", transmission_rate=0.04)

    # set the population (alternatively you can import it using model.import_epydemix_population('Kenya'))
    model.set_population(mock_population)

    print(model)

    # simulate
    results = model.run_simulations(
        start_date="2019-12-01",
        end_date="2020-04-01",
        percentage_in_agents=10 / model.population.Nk.sum(),
        Nsim=5,
    )

    # plot results
    df_quantiles_comps = results.get_quantiles_compartments()
    plot_quantiles(
        df_quantiles_comps,
        columns=["Infected_total", "Susceptible_total", "Recovered_total"],
        legend_loc="center right",
    )
    plot_quantiles(
        df_quantiles_comps,
        columns=[
            "Infected_0-9",
            "Infected_10-19",
            "Infected_20-29",
            "Infected_30-39",
            "Infected_40+",
        ],
        legend_loc="center right",
    )
    plt.close()
