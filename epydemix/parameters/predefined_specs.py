"""Canonical parameter specifications for predefined epidemic models.

Each function returns a list of :class:`ParameterSpec` objects describing
the parameters of a predefined model (SIR, SEIR, SIS). These are registered
automatically when a predefined model is created.
"""

from .spec import ParameterSpec


def sir_specs(
    transmission_rate: float = 0.3,
    recovery_rate: float = 0.1,
) -> list:
    """Parameter specs for the SIR model."""
    return [
        ParameterSpec(
            name="transmission_rate",
            description=(
                "Rate at which susceptible individuals become infected through "
                "contact with infected individuals. Controls the speed of epidemic "
                "growth. Higher values produce faster, larger outbreaks."
            ),
            kind="rate",
            default=transmission_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["transmission", "SIR"],
        ),
        ParameterSpec(
            name="recovery_rate",
            description=(
                "Rate at which infected individuals recover and become immune. "
                "The inverse (1/recovery_rate) gives the mean infectious period "
                "in days."
            ),
            kind="rate",
            default=recovery_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["recovery", "SIR"],
        ),
    ]


def seir_specs(
    transmission_rate: float = 0.3,
    incubation_rate: float = 0.2,
    recovery_rate: float = 0.1,
) -> list:
    """Parameter specs for the SEIR model."""
    return [
        ParameterSpec(
            name="transmission_rate",
            description=(
                "Rate at which susceptible individuals become exposed through "
                "contact with infected individuals. Controls the force of "
                "infection. Higher values produce faster epidemic growth."
            ),
            kind="rate",
            default=transmission_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["transmission", "SEIR"],
        ),
        ParameterSpec(
            name="incubation_rate",
            description=(
                "Rate at which exposed individuals become infectious. "
                "The inverse (1/incubation_rate) gives the mean latent "
                "(incubation) period in days."
            ),
            kind="rate",
            default=incubation_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["incubation", "SEIR"],
        ),
        ParameterSpec(
            name="recovery_rate",
            description=(
                "Rate at which infected individuals recover and become immune. "
                "The inverse (1/recovery_rate) gives the mean infectious period "
                "in days."
            ),
            kind="rate",
            default=recovery_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["recovery", "SEIR"],
        ),
    ]


def sis_specs(
    transmission_rate: float = 0.3,
    recovery_rate: float = 0.1,
) -> list:
    """Parameter specs for the SIS model."""
    return [
        ParameterSpec(
            name="transmission_rate",
            description=(
                "Rate at which susceptible individuals become infected through "
                "contact with infected individuals. In the SIS model, recovered "
                "individuals return to susceptible, so this rate also governs "
                "reinfection dynamics."
            ),
            kind="rate",
            default=transmission_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["transmission", "SIS"],
        ),
        ParameterSpec(
            name="recovery_rate",
            description=(
                "Rate at which infected individuals recover and return to the "
                "susceptible state (no lasting immunity). The inverse "
                "(1/recovery_rate) gives the mean infectious period in days."
            ),
            kind="rate",
            default=recovery_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["recovery", "SIS"],
        ),
    ]


# Map of model name → spec factory function
MODEL_SPEC_FACTORIES = {
    "SIR": sir_specs,
    "SEIR": seir_specs,
    "SIS": sis_specs,
}
