import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-GUI backend before importing pyplot

import numpy as np

from epydemix.model.epimodel import EpiModel, stochastic_simulation
from epydemix.population import Population
from epydemix.utils.utils import apply_initial_conditions

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


def test_apply_initial_conditions_unknown_compartment(mock_epimodel):
    """Test apply_initial_conditions raises ValueError for a mistyped/unknown compartment name"""
    with pytest.raises(ValueError, match="Susceptibl"):
        apply_initial_conditions(
            mock_epimodel,
            {"Susceptibl": np.array([990, 990, 990])},  # typo, missing trailing "e"
        )

    with pytest.raises(ValueError, match="Bogus"):
        apply_initial_conditions(mock_epimodel, {"Bogus": np.array([990, 990, 990])})


def test_apply_initial_conditions_partial_dict(mock_epimodel):
    """Test apply_initial_conditions accepts a valid partial dict, defaulting unset compartments to 0"""
    initial_conditions = apply_initial_conditions(
        mock_epimodel, {"Infected": np.array([10, 10, 10])}
    )

    assert initial_conditions.shape == (3, 3)
    infected_idx = mock_epimodel.compartments_idx["Infected"]
    susceptible_idx = mock_epimodel.compartments_idx["Susceptible"]
    assert np.array_equal(initial_conditions[infected_idx], np.array([10, 10, 10]))
    assert np.array_equal(initial_conditions[susceptible_idx], np.array([0, 0, 0]))


def test_run_simulations_unknown_compartment_in_initial_conditions(mock_epimodel):
    """Test run_simulations surfaces a clear error for a mistyped initial_conditions_dict key"""
    with pytest.raises(RuntimeError, match="Susceptibl"):
        mock_epimodel.run_simulations(
            start_date="2023-01-01",
            end_date="2023-01-10",
            initial_conditions_dict={
                "Susceptibl": np.array([990, 990, 990]),
                "Infected": np.array([10, 10, 10]),
                "Recovered": np.array([0, 0, 0]),
            },
            Nsim=1,
        )


@pytest.fixture
def duplicate_pair_epimodel():
    """Model with two mediated transitions sharing the same (source, target) pair,
    mirroring SEIAR's Susceptible -> Exposed via Infected and via Asymptomatic."""
    model = EpiModel(
        compartments=["Susceptible", "Exposed", "Infected", "Asymptomatic"],
        parameters={
            "transmission_rate": 0.3,
            "transmission_rate_asym": 0.15,
            "progression_rate": 0.2,
        },
    )
    model.add_transition(
        "Susceptible", "Exposed", "mediated", ("transmission_rate", "Infected")
    )
    model.add_transition(
        "Susceptible", "Exposed", "mediated", ("transmission_rate_asym", "Asymptomatic")
    )
    model.add_transition("Exposed", "Infected", "spontaneous", "progression_rate")

    population = Population()
    population.add_population([1000, 1000, 1000])
    population.add_contact_matrix(np.ones((3, 3)))
    model.set_population(population)
    return model


def test_stochastic_simulation_duplicate_transition_pair_not_double_counted(
    duplicate_pair_epimodel,
):
    """Test that transitions_evolution isn't double-counted when two Transition objects
    share the same (source, target) pair (e.g. two mediated agents for the same edge)"""
    T = 20
    contact_matrices = [
        {"overall": duplicate_pair_epimodel.population.contact_matrices["all"]}
        for _ in range(T)
    ]
    # shape (n_compartments, n_age_groups) = (4, 3): Susceptible, Exposed, Infected, Asymptomatic
    initial_conditions = np.array(
        [[800, 800, 800], [0, 0, 0], [100, 100, 100], [100, 100, 100]]
    )
    parameters = {
        "transmission_rate": np.full(T, 0.3),
        "transmission_rate_asym": np.full(T, 0.15),
        "progression_rate": np.full(T, 0.2),
    }

    compartments_evolution, transitions_evolution = stochastic_simulation(
        T=T,
        contact_matrices=contact_matrices,
        epimodel=duplicate_pair_epimodel,
        parameters=parameters,
        initial_conditions=initial_conditions,
        dt=1.0,
    )

    susceptible_idx = duplicate_pair_epimodel.compartments_idx["Susceptible"]
    exposed_idx = duplicate_pair_epimodel.compartments_idx["Exposed"]
    progression_idx = duplicate_pair_epimodel.transitions_idx["Exposed_to_Infected"]
    se_idx = duplicate_pair_epimodel.transitions_idx["Susceptible_to_Exposed"]

    # Net change in Exposed equals inflow from Susceptible minus outflow to Infected.
    # If Susceptible_to_Exposed were double-counted, this identity would fail.
    exposed_start = initial_conditions[exposed_idx, :].sum()
    exposed_end = compartments_evolution[-1, exposed_idx].sum()
    total_inflow = transitions_evolution[:, se_idx].sum()
    total_outflow = transitions_evolution[:, progression_idx].sum()
    assert np.isclose(exposed_end - exposed_start, total_inflow - total_outflow)

    # Total Susceptible_to_Exposed recorded flow must equal actual Susceptible depletion
    # (net of any inflow to Susceptible, of which there is none in this model).
    susceptible_start = initial_conditions[susceptible_idx, :].sum()
    susceptible_end = compartments_evolution[-1, susceptible_idx].sum()
    assert np.isclose(susceptible_start - susceptible_end, total_inflow)
