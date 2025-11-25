import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt

from epydemix import EpiModel
from epydemix.visualization import plot_quantiles, plot_trajectories


def test_model_definition_and_simulation():
    # Defining a basic SIR model
    sir_model = EpiModel(
        name="SIR Model",
        compartments=["S", "I", "R"],  # Susceptible, Infected, Recovered
    )

    # Defining the transitions
    sir_model.add_transition(source="S", target="I", params=(0.3, "I"), kind="mediated")
    sir_model.add_transition(source="I", target="R", params=0.1, kind="spontaneous")

    print(sir_model)

    sir_results = sir_model.run_simulations(
        start_date="2024-01-01", end_date="2024-04-10", Nsim=5
    )

    df_quantiles_comps = sir_results.get_quantiles_compartments()
    plot_quantiles(
        df_quantiles_comps,
        columns=["I_total", "S_total", "R_total"],
        title="SIR Model Simulation (Compartments, Quantiles)",
    )
    plt.close()

    df_quantiles_tr = sir_results.get_quantiles_transitions()
    plot_quantiles(
        df_quantiles_tr,
        columns=["S_to_I_total", "I_to_R_total"],
        title="SIR Model Simulation (Transitions, Quantiles)",
    )
    plt.close()

    trajectories_comp = sir_results.get_stacked_compartments()
    plot_trajectories(
        trajectories_comp,
        columns=["I_total", "S_total", "R_total"],
        title="SIR Model Simulation (Compartments, Trajectories)",
    )
    plt.close()
