import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-GUI backend before importing pyplot

import numpy as np

from epydemix.model.epimodel import EpiModel, stochastic_simulation
from epydemix.population import Population

# filepath: epydemix/tests/test_epimodel.py


@pytest.fixture
def mock_epimodel():
    model = EpiModel(
        compartments=["Susceptible", "Infected", "Recovered"],
        parameters={"transmission_rate": 0.3, "recovery_rate": 0.1},
    )
    model.add_transition(
        "Susceptible", "Infected", "mediated", ("transmission_rate", "Infected")
    )
    model.add_transition("Infected", "Recovered", "spontaneous", "recovery_rate")

    population = Population()
    population.add_population([1000, 1000, 1000])
    population.add_contact_matrix(np.ones((3, 3)))
    model.set_population(population)
    return model


@pytest.fixture
def basic_model():
    """Basic model fixture with minimal setup"""
    return EpiModel(name="Test Model")


def test_model_initialization():
    """Test model initialization with different parameters"""
    # Test default initialization
    model = EpiModel()
    assert model.name == "EpiModel"
    assert len(model.compartments) == 0
    assert len(model.parameters) == 0

    # Test initialization with parameters
    model = EpiModel(
        name="Custom Model",
        compartments=["S", "I", "R"],
        parameters={"beta": 0.3, "gamma": 0.1},
    )
    assert model.name == "Custom Model"
    assert len(model.compartments) == 3
    assert len(model.parameters) == 2


def test_compartment_management(basic_model):
    """Test adding and removing compartments"""
    # Test adding single compartment
    basic_model.add_compartments("S")
    assert "S" in basic_model.compartments
    assert len(basic_model.compartments) == 1

    # Test adding multiple compartments
    basic_model.add_compartments(["I", "R"])
    assert len(basic_model.compartments) == 3
    assert all(c in basic_model.compartments for c in ["S", "I", "R"])

    # Test clearing compartments
    basic_model.clear_compartments()
    assert len(basic_model.compartments) == 0


def test_transition_management(basic_model):
    """Test adding and removing transitions"""
    basic_model.add_compartments(["S", "I", "R"])

    # Test adding transitions
    basic_model.add_transition("S", "I", "mediated", ("beta", "I"))
    assert len(basic_model.transitions_list) == 1

    basic_model.add_transition("I", "R", "spontaneous", "gamma")
    assert len(basic_model.transitions_list) == 2

    # Test invalid transition
    with pytest.raises(ValueError):
        basic_model.add_transition("X", "Y", "spontaneous", "alpha")

    # Test clearing transitions
    basic_model.clear_transitions()
    assert len(basic_model.transitions_list) == 0


def test_parameter_management(basic_model):
    """Test parameter management"""
    # Test adding single parameter
    basic_model.add_parameter(parameter_name="beta", value=0.3)
    assert "beta" in basic_model.parameters
    assert basic_model.parameters["beta"] == 0.3

    # Test adding multiple parameters
    basic_model.add_parameter(parameters_dict={"gamma": 0.1, "R0": 3.0})
    assert len(basic_model.parameters) == 3

    # Test parameter deletion
    basic_model.delete_parameter("R0")
    assert "R0" not in basic_model.parameters


def test_intervention_management(basic_model):
    """Test intervention management"""
    basic_model.add_intervention(
        layer_name="overall",
        start_date="2023-01-01",
        end_date="2023-02-01",
        reduction_factor=0.5,
    )
    assert len(basic_model.interventions) == 1

    # Test invalid intervention
    with pytest.raises(ValueError):
        basic_model.add_intervention(
            layer_name="overall", start_date="2023-01-01", end_date="2023-02-01"
        )


def test_stochastic_simulation(mock_epimodel):
    """Test stochastic simulation with conservation laws"""
    T = 10
    N = 3

    contact_matrices = [
        {"overall": mock_epimodel.population.contact_matrices["all"]} for _ in range(T)
    ]
    initial_conditions = np.array([[9990, 10, 0], [9990, 10, 0], [9990, 10, 0]])
    parameters = {
        "transmission_rate": np.full(T, 0.3),
        "recovery_rate": np.full(T, 0.1),
    }

    compartments_evolution, transitions_evolution = stochastic_simulation(
        T=T,
        contact_matrices=contact_matrices,
        epimodel=mock_epimodel,
        parameters=parameters,
        initial_conditions=initial_conditions,
        dt=1.0,
    )

    # Test shape of outputs
    assert compartments_evolution.shape == (T, 3, N)
    assert transitions_evolution.shape == (T, 2, N)

    # Test population conservation
    for t in range(T):
        assert np.isclose(np.sum(compartments_evolution[t]), np.sum(initial_conditions))

    # Test non-negative populations
    assert np.all(compartments_evolution >= 0)
    assert np.all(transitions_evolution >= 0)


def test_stochastic_simulation_invalid_initial_conditions(mock_epimodel):
    T = 10
    contact_matrices = [
        {"overall": mock_epimodel.population.contact_matrices["all"]} for _ in range(T)
    ]
    initial_conditions = np.array([[1000, 0], [0, 10], [0, 0]])  # Invalid shape
    parameters = {
        "transmission_rate": np.full(T, 0.3),
        "recovery_rate": np.full(T, 0.1),
    }
    dt = 1.0

    with pytest.raises(ValueError):
        stochastic_simulation(
            T=T,
            contact_matrices=contact_matrices,
            epimodel=mock_epimodel,
            parameters=parameters,
            initial_conditions=initial_conditions,
            dt=dt,
        )


# ---------------------------------------------------------------------------
# Scheduled transition kind
# ---------------------------------------------------------------------------

def test_scheduled_transition_vaccinees_leave_S():
    """Scheduled transition moves individuals from S to V at the dose rate."""
    T = 30
    pop_size = 10_000
    daily_doses = 100  # flat dose schedule

    model = EpiModel(
        compartments=["S", "V", "I", "R"],
        parameters={"transmission_rate": 0.1, "recovery_rate": 0.05},
    )
    # Vaccination: S → V at a scheduled dose rate
    dose_array = np.full((T, 1), daily_doses, dtype=float)
    model.add_transition("S", "V", kind="scheduled", params=(dose_array,))
    # Standard SIR dynamics on the unvaccinated
    model.add_transition("S", "I", kind="mediated", params=("transmission_rate", "I"))
    model.add_transition("I", "R", kind="spontaneous", params="recovery_rate")

    pop = Population()
    pop.add_population([pop_size])
    pop.add_contact_matrix(np.ones((1, 1)))
    model.set_population(pop)

    results = model.run_simulations(
        start_date="2026-01-01",
        end_date="2026-01-30",
        Nsim=5,
        initial_conditions_dict={
            "S": np.array([pop_size - 10], dtype=float),
            "I": np.array([10], dtype=float),
            "V": np.array([0], dtype=float),
            "R": np.array([0], dtype=float),
        },
    )

    df = results.get_quantiles_compartments()
    median = df[df["quantile"] == 0.5]
    final_V = median["V_total"].iloc[-1]
    final_S = median["S_total"].iloc[-1]

    # V should have grown substantially (at least 10 days * ~100 doses each)
    assert final_V > 500, f"Expected V > 500 after 30 days of vaccination, got {final_V}"
    # S should be lower than initial
    assert final_S < pop_size - 10, "S should have decreased due to vaccination"


def test_scheduled_transition_eligible_correction():
    """With eligible correction, dose-wasting on R reduces effective rate."""
    T = 20
    pop_size = 1_000
    daily_doses = 50

    model_no_correction = EpiModel(
        compartments=["S", "V", "I", "R"],
        parameters={"transmission_rate": 0.0, "recovery_rate": 0.0},
    )
    model_with_correction = EpiModel(
        compartments=["S", "V", "I", "R"],
        parameters={"transmission_rate": 0.0, "recovery_rate": 0.0},
    )

    dose_array = np.full((T, 1), daily_doses, dtype=float)

    # No correction: rate = doses / S
    model_no_correction.add_transition("S", "V", kind="scheduled", params=(dose_array,))
    # With correction: rate = doses / (S + R) → fewer effective doses when R > 0
    model_with_correction.add_transition(
        "S", "V", kind="scheduled", params=(dose_array, ["S", "R"])
    )

    for m in (model_no_correction, model_with_correction):
        pop = Population()
        pop.add_population([pop_size])
        pop.add_contact_matrix(np.ones((1, 1)))
        m.set_population(pop)

    ic = {
        "S": np.array([600.0]),
        "V": np.array([0.0]),
        "I": np.array([0.0]),
        "R": np.array([400.0]),  # large recovered pool absorbs wasted doses
    }

    res_no = model_no_correction.run_simulations(
        "2026-01-01", "2026-01-20", Nsim=1, initial_conditions_dict=ic
    )
    res_with = model_with_correction.run_simulations(
        "2026-01-01", "2026-01-20", Nsim=1, initial_conditions_dict=ic
    )

    df_no = res_no.get_quantiles_compartments()
    df_with = res_with.get_quantiles_compartments()

    V_no   = df_no[df_no["quantile"] == 0.5]["V_total"].iloc[-1]
    V_with = df_with[df_with["quantile"] == 0.5]["V_total"].iloc[-1]

    # Without correction ignores the R pool → more S vaccinated
    # With correction only effective doses (fraction S/(S+R) = 0.6) reach S
    assert V_no > V_with, (
        f"No-correction model should vaccinate more (got {V_no:.1f}) "
        f"than with-correction model (got {V_with:.1f})"
    )
