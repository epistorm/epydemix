from .epimodel import EpiModel

SUPPORTED_MODELS = ["SIR", "SEIR", "SIS"]


def load_predefined_model(
    model_name: str,
    transmission_rate: float = 0.3,
    recovery_rate: float = 0.1,
    incubation_rate: float = 0.2,
) -> None:
    """
    Loads a predefined epidemic model (e.g., SIR, SEIR, SIS) with predefined compartments and transitions.

    Args:
        model_name (str): The name of the predefined model to load (e.g., "SIR").
        transmission_rate (float, optional): The transmission rate for the SIR model. Default is 0.3.
        recovery_rate (float, optional): The recovery rate for the SIR model. Default is 0.1.
        incubation_rate (float, optional): The incubation rate for the SEIR model. Default is 0.2.

    Raises:
        ValueError: If the model_name is not recognized.
    """
    if model_name == "SIR":
        return create_sir(transmission_rate, recovery_rate)

    elif model_name == "SEIR":
        return create_seir(transmission_rate, incubation_rate, recovery_rate)

    elif model_name == "SIS":
        return create_sis(transmission_rate, recovery_rate)

    else:
        raise ValueError(
            f"Unknown predefined model: {model_name}, supported models are: {SUPPORTED_MODELS}"
        )


def create_sir(transmission_rate: float, recovery_rate: float) -> EpiModel:
    """Create a SIR model with the given transmission rate and recovery rate."""
    model = EpiModel(
        compartments=["Susceptible", "Infected", "Recovered"],
        parameters={
            "transmission_rate": transmission_rate,
            "recovery_rate": recovery_rate,
        },
    )
    model.add_transition(
        source="Susceptible",
        target="Infected",
        params=("transmission_rate", "Infected"),
        kind="mediated",
    )
    model.add_transition(
        source="Infected",
        target="Recovered",
        params="recovery_rate",
        kind="spontaneous",
    )
    return model


def create_seir(
    transmission_rate: float, incubation_rate: float, recovery_rate: float
) -> EpiModel:
    """Create a SEIR model with the given transmission rate, incubation rate, and recovery rate."""
    model = EpiModel(
        compartments=["Susceptible", "Exposed", "Infected", "Recovered"],
        parameters={
            "transmission_rate": transmission_rate,
            "incubation_rate": incubation_rate,
            "recovery_rate": recovery_rate,
        },
    )
    model.add_transition(
        source="Susceptible",
        target="Exposed",
        params=("transmission_rate", "Infected"),
        kind="mediated",
    )
    model.add_transition(
        source="Exposed",
        target="Infected",
        params="incubation_rate",
        kind="spontaneous",
    )
    model.add_transition(
        source="Infected",
        target="Recovered",
        params="recovery_rate",
        kind="spontaneous",
    )
    return model


def create_sis(transmission_rate: float, recovery_rate: float) -> EpiModel:
    """Create a SIS model with the given transmission rate and recovery rate."""
    model = EpiModel(
        compartments=["Susceptible", "Infected"],
        parameters={
            "transmission_rate": transmission_rate,
            "recovery_rate": recovery_rate,
        },
    )
    model.add_transition(
        source="Susceptible",
        target="Infected",
        params=("transmission_rate", "Infected"),
        kind="mediated",
    )
    model.add_transition(
        source="Infected",
        target="Susceptible",
        params="recovery_rate",
        kind="spontaneous",
    )
    return model
