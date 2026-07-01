import numpy as np
import pytest

from epydemix.model.predefined_models import (
    SUPPORTED_MODELS,
    add_outcome,
    add_vaccination,
    add_waning_immunity,
    create_seiar,
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


def test_seiar_model(basic_population):
    """Test SEIAR model creation and basic properties"""
    beta, sigma, gamma = 0.3, 0.2, 0.1
    model = create_seiar(
        transmission_rate=beta,
        incubation_rate=sigma,
        recovery_rate=gamma,
        asymptomatic_fraction=0.4,
        asymptomatic_recovery_rate=0.14,
        asymptomatic_relative_infectivity=0.5,
    )

    assert set(model.compartments) == {
        "Susceptible",
        "Exposed",
        "Infected",
        "Asymptomatic",
        "Recovered",
    }
    assert len(model.transitions_list) == 6
    assert model.parameters["transmission_rate"] == beta
    assert model.parameters["incubation_rate"] == sigma
    assert model.parameters["recovery_rate"] == gamma
    assert model.parameters["asymptomatic_fraction"] == 0.4
    assert model.parameters["asymptomatic_relative_infectivity"] == 0.5

    transitions = [(t.source, t.target) for t in model.transitions_list]
    assert (
        transitions.count(("Susceptible", "Exposed")) == 2
    )  # one per infectious agent
    assert ("Exposed", "Infected") in transitions
    assert ("Exposed", "Asymptomatic") in transitions
    assert ("Infected", "Recovered") in transitions
    assert ("Asymptomatic", "Recovered") in transitions

    model.set_population(basic_population)
    initial_conditions = {
        "Susceptible": np.array([9800]),
        "Exposed": np.array([100]),
        "Infected": np.array([100]),
        "Asymptomatic": np.array([0]),
        "Recovered": np.array([0]),
    }
    trajectory = model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict=initial_conditions,
        Nsim=10,
    )
    total_population = [
        np.array([v.compartments[c] for c in v.compartments if "total" in c]).sum(
            axis=0
        )
        for v in trajectory.trajectories
    ]
    assert np.allclose(total_population, 10000)


def test_waning_immunity_module(basic_population):
    """Test add_waning_immunity adds R → S transition and respects guard rails"""
    model = create_sir(transmission_rate=0.3, recovery_rate=0.1)
    model = add_waning_immunity(model, waning_rate=1.0 / 365)

    assert model.parameters["waning_rate"] == 1.0 / 365
    transitions = {(t.source, t.target) for t in model.transitions_list}
    assert ("Recovered", "Susceptible") in transitions

    with pytest.raises(ValueError, match="'Recovered'"):
        add_waning_immunity(create_sis(0.3, 0.1), waning_rate=0.01)

    model.set_population(basic_population)
    model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict={
            "Susceptible": np.array([9900]),
            "Infected": np.array([100]),
            "Recovered": np.array([0]),
        },
        Nsim=5,
    )


def test_vaccination_module(basic_population):
    """Test add_vaccination adds Vaccinated compartment and correct transitions"""
    model = create_sir(transmission_rate=0.3, recovery_rate=0.1)
    model = add_vaccination(model, vaccination_rate=0.01, vaccine_efficacy=0.9)

    assert "Vaccinated" in model.compartments
    assert model.parameters["vaccination_rate"] == 0.01
    assert model.parameters["vaccine_efficacy"] == 0.9

    transitions = {(t.source, t.target) for t in model.transitions_list}
    assert ("Susceptible", "Vaccinated") in transitions
    assert ("Vaccinated", "Infected") in transitions

    model.set_population(basic_population)
    model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict={
            "Susceptible": np.array([9900]),
            "Infected": np.array([100]),
            "Recovered": np.array([0]),
            "Vaccinated": np.array([0]),
        },
        Nsim=5,
    )


def test_vaccination_module_with_exposed_backbone(basic_population):
    """Test add_vaccination routes breakthrough infections to Exposed when the backbone has one"""
    seir_model = create_seir(
        transmission_rate=0.3, incubation_rate=0.2, recovery_rate=0.1
    )
    seir_model = add_vaccination(
        seir_model, vaccination_rate=0.01, vaccine_efficacy=0.9
    )

    seir_transitions = {(t.source, t.target) for t in seir_model.transitions_list}
    assert ("Vaccinated", "Exposed") in seir_transitions
    assert ("Vaccinated", "Infected") not in seir_transitions

    seiar_model = create_seiar(
        transmission_rate=0.3,
        incubation_rate=0.2,
        recovery_rate=0.1,
        asymptomatic_fraction=0.4,
        asymptomatic_recovery_rate=0.14,
        asymptomatic_relative_infectivity=0.5,
    )
    seiar_model = add_vaccination(
        seiar_model, vaccination_rate=0.01, vaccine_efficacy=0.9
    )

    seiar_transitions = {(t.source, t.target) for t in seiar_model.transitions_list}
    assert ("Vaccinated", "Exposed") in seiar_transitions
    assert ("Vaccinated", "Infected") not in seiar_transitions

    seir_model.set_population(basic_population)
    seir_model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict={
            "Susceptible": np.array([9900]),
            "Exposed": np.array([0]),
            "Infected": np.array([100]),
            "Recovered": np.array([0]),
            "Vaccinated": np.array([0]),
        },
        Nsim=5,
    )


def test_outcome_deaths_module(basic_population):
    """Test add_outcome with deaths adds Dead compartment and I → Dead transition"""
    model = create_sir(transmission_rate=0.3, recovery_rate=0.1)
    model = add_outcome(
        model,
        "deaths",
        mortality_rate=0.01,
        hospitalization_rate=0.01,
        hospitalization_recovery_rate=0.1,
    )

    assert "Dead" in model.compartments
    assert model.parameters["mortality_rate"] == 0.01
    transitions = {(t.source, t.target) for t in model.transitions_list}
    assert ("Infected", "Dead") in transitions

    model.set_population(basic_population)
    model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict={
            "Susceptible": np.array([9900]),
            "Infected": np.array([100]),
            "Recovered": np.array([0]),
            "Dead": np.array([0]),
        },
        Nsim=5,
    )


def test_outcome_hospitalization_module(basic_population):
    """Test add_outcome with hospitalization adds Hospitalized compartment and correct transitions"""
    model = create_sir(transmission_rate=0.3, recovery_rate=0.1)
    model = add_outcome(
        model,
        "hospitalization",
        mortality_rate=0.01,
        hospitalization_rate=0.05,
        hospitalization_recovery_rate=0.1,
    )

    assert "Hospitalized" in model.compartments
    assert model.parameters["hospitalization_rate"] == 0.05
    assert model.parameters["hospitalization_recovery_rate"] == 0.1
    transitions = {(t.source, t.target) for t in model.transitions_list}
    assert ("Infected", "Hospitalized") in transitions
    assert ("Hospitalized", "Recovered") in transitions

    with pytest.raises(ValueError, match="'Recovered'"):
        add_outcome(
            create_sis(0.3, 0.1),
            "hospitalization",
            mortality_rate=0.01,
            hospitalization_rate=0.05,
            hospitalization_recovery_rate=0.1,
        )

    model.set_population(basic_population)
    model.run_simulations(
        start_date="2023-01-01",
        end_date="2023-01-10",
        initial_conditions_dict={
            "Susceptible": np.array([9900]),
            "Infected": np.array([100]),
            "Recovered": np.array([0]),
            "Hospitalized": np.array([0]),
        },
        Nsim=5,
    )


def test_outcome_invalid():
    """Test add_outcome raises ValueError for unknown outcome strings"""
    with pytest.raises(ValueError, match="Unknown outcome"):
        add_outcome(
            create_sir(0.3, 0.1),
            "flying",
            mortality_rate=0.01,
            hospitalization_rate=0.01,
            hospitalization_recovery_rate=0.1,
        )


def test_load_predefined_model_combinations():
    """Test load_predefined_model with module flags produces correct compartment sets"""
    cases = [
        (
            {"model_name": "SEIAR"},
            {"Susceptible", "Exposed", "Infected", "Asymptomatic", "Recovered"},
        ),
        (
            {"model_name": "SIR", "waning_immunity": True},
            {"Susceptible", "Infected", "Recovered"},
        ),
        (
            {"model_name": "SEIR", "waning_immunity": True},
            {"Susceptible", "Exposed", "Infected", "Recovered"},
        ),
        (
            {"model_name": "SIR", "vaccination": True},
            {"Susceptible", "Infected", "Recovered", "Vaccinated"},
        ),
        (
            {"model_name": "SIR", "outcome": "deaths"},
            {"Susceptible", "Infected", "Recovered", "Dead"},
        ),
        (
            {"model_name": "SEIR", "outcome": "deaths"},
            {"Susceptible", "Exposed", "Infected", "Recovered", "Dead"},
        ),
        (
            {"model_name": "SIR", "outcome": "hospitalization"},
            {"Susceptible", "Infected", "Recovered", "Hospitalized"},
        ),
        (
            {
                "model_name": "SEIAR",
                "waning_immunity": True,
                "vaccination": True,
                "outcome": "deaths",
            },
            {
                "Susceptible",
                "Exposed",
                "Infected",
                "Asymptomatic",
                "Recovered",
                "Vaccinated",
                "Dead",
            },
        ),
    ]
    for kwargs, expected in cases:
        model = load_predefined_model(**kwargs)
        assert set(model.compartments) == expected, f"Failed for {kwargs}"


def test_load_predefined_model_guard_rails():
    """Test load_predefined_model raises ValueError for incompatible combinations"""
    with pytest.raises(ValueError):
        load_predefined_model("SIS", waning_immunity=True)
    with pytest.raises(ValueError):
        load_predefined_model("SIS", outcome="hospitalization")
    with pytest.raises(ValueError):
        load_predefined_model("SIR", outcome="unknown")
    with pytest.raises(ValueError):
        load_predefined_model("SEIRX")
