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
