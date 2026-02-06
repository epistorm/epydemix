import numpy as np
import pytest

from epydemix.model.predefined_models import (
    SUPPORTED_MODELS,
    create_seir,
    create_sir,
    create_sis,
    load_predefined_model,
)
from epydemix.population import Population


@pytest.fixture
def basic_population():
    """Fixture providing a basic population setup"""
    population = Population()
    population.add_population([10000])  # Single age group with 10000 people
    population.add_contact_matrix(np.array([[1.0]]))  # Simple contact matrix
    return population


def test_load_predefined_model():
    """Test loading different predefined models"""
    # Test loading each supported model
    for model_name in SUPPORTED_MODELS:
        model = load_predefined_model(model_name)
        assert model is not None

    # Test invalid model name
    with pytest.raises(ValueError, match="Unknown predefined model"):
        load_predefined_model("INVALID_MODEL")


def test_sir_model(basic_population):
    """Test SIR model creation and basic properties"""
    # Create model with custom rates
    beta, gamma = 0.3, 0.1
    model = create_sir(transmission_rate=beta, recovery_rate=gamma)

    # Test structure
    assert set(model.compartments) == {"Susceptible", "Infected", "Recovered"}
    assert len(model.transitions_list) == 2
    assert model.parameters["transmission_rate"] == beta
    assert model.parameters["recovery_rate"] == gamma

    # Test transitions
    transitions = {(t.source, t.target): t for t in model.transitions_list}
    assert ("Susceptible", "Infected") in transitions
    assert ("Infected", "Recovered") in transitions

    # Test simulation
    model.set_population(basic_population)
    initial_conditions = {
        "Susceptible": np.array([9900]),
        "Infected": np.array([100]),
        "Recovered": np.array([0]),
    }

    trajectory = model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict=initial_conditions,
        Nsim=10,
    )

    # Check population conservation
    total_population = [
        np.array([v.compartments[c] for c in v.compartments if "total" in c]).sum(
            axis=0
        )
        for v in trajectory.trajectories
    ]
    assert np.allclose(total_population, 10000)


def test_seir_model(basic_population):
    """Test SEIR model creation and basic properties"""
    # Create model with custom rates
    beta, sigma, gamma = 0.3, 0.2, 0.1
    model = create_seir(
        transmission_rate=beta, incubation_rate=sigma, recovery_rate=gamma
    )

    # Test structure
    assert set(model.compartments) == {
        "Susceptible",
        "Exposed",
        "Infected",
        "Recovered",
    }
    assert len(model.transitions_list) == 3
    assert model.parameters["transmission_rate"] == beta
    assert model.parameters["incubation_rate"] == sigma
    assert model.parameters["recovery_rate"] == gamma

    # Test transitions
    transitions = {(t.source, t.target): t for t in model.transitions_list}
    assert ("Susceptible", "Exposed") in transitions
    assert ("Exposed", "Infected") in transitions
    assert ("Infected", "Recovered") in transitions

    # Test simulation
    model.set_population(basic_population)
    initial_conditions = {
        "Susceptible": np.array([9800]),
        "Exposed": np.array([100]),
        "Infected": np.array([100]),
        "Recovered": np.array([0]),
    }

    trajectory = model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict=initial_conditions,
        Nsim=10,
    )

    # Check population conservation
    total_population = [
        np.array([v.compartments[c] for c in v.compartments if "total" in c]).sum(
            axis=0
        )
        for v in trajectory.trajectories
    ]
    assert np.allclose(total_population, 10000)


def test_sis_model(basic_population):
    """Test SIS model creation and basic properties"""
    # Create model with custom rates
    beta, gamma = 0.3, 0.1
    model = create_sis(transmission_rate=beta, recovery_rate=gamma)

    # Test structure
    assert set(model.compartments) == {"Susceptible", "Infected"}
    assert len(model.transitions_list) == 2
    assert model.parameters["transmission_rate"] == beta
    assert model.parameters["recovery_rate"] == gamma

    # Test transitions
    transitions = {(t.source, t.target): t for t in model.transitions_list}
    assert ("Susceptible", "Infected") in transitions
    assert ("Infected", "Susceptible") in transitions  # Note: returns to Susceptible

    # Test simulation
    model.set_population(basic_population)
    initial_conditions = {"Susceptible": np.array([9900]), "Infected": np.array([100])}

    trajectory = model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict=initial_conditions,
        Nsim=10,
    )

    # Check population conservation
    total_population = [
        np.array([v.compartments[c] for c in v.compartments if "total" in c]).sum(
            axis=0
        )
        for v in trajectory.trajectories
    ]
    assert np.allclose(total_population, 10000)
