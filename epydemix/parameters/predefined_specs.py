"""Canonical parameter specifications for predefined epidemic models.

Each function returns a list of :class:`ParameterSpec` objects describing
the parameters of a predefined model (SIR, SEIR, SIS, SEIAR) or a modular
extension (waning immunity, vaccination, outcome). These are registered
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


def seiar_specs(
    transmission_rate: float = 0.3,
    incubation_rate: float = 0.2,
    recovery_rate: float = 0.1,
    asymptomatic_fraction: float = 0.4,
    asymptomatic_recovery_rate: float = 0.14,
    asymptomatic_relative_infectivity: float = 0.5,
) -> list:
    """Parameter specs for the SEIAR model (SEIR with an asymptomatic branch)."""
    return [
        ParameterSpec(
            name="transmission_rate",
            description=(
                "Rate at which susceptible individuals become exposed through "
                "contact with symptomatic infected individuals. Asymptomatic "
                "contacts contribute at a reduced rate scaled by "
                "asymptomatic_relative_infectivity."
            ),
            kind="rate",
            default=transmission_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["transmission", "SEIAR"],
        ),
        ParameterSpec(
            name="incubation_rate",
            description=(
                "Rate at which exposed individuals leave the Exposed compartment. "
                "A fraction asymptomatic_fraction progresses to Asymptomatic; the "
                "rest progresses to Infected. The inverse (1/incubation_rate) is "
                "the mean latent period in days."
            ),
            kind="rate",
            default=incubation_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["incubation", "SEIAR"],
        ),
        ParameterSpec(
            name="recovery_rate",
            description=(
                "Recovery rate for symptomatic Infected individuals. The inverse "
                "(1/recovery_rate) is the mean symptomatic infectious period."
            ),
            kind="rate",
            default=recovery_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["recovery", "SEIAR"],
        ),
        ParameterSpec(
            name="asymptomatic_fraction",
            description=(
                "Fraction of Exposed individuals who progress to the "
                "Asymptomatic compartment rather than Infected. The remaining "
                "(1 - asymptomatic_fraction) develops symptoms."
            ),
            kind="proportion",
            default=asymptomatic_fraction,
            min=0,
            max=1,
            units="dimensionless",
            tags=["SEIAR"],
        ),
        ParameterSpec(
            name="asymptomatic_recovery_rate",
            description=(
                "Recovery rate for Asymptomatic individuals. The inverse "
                "(1/asymptomatic_recovery_rate) is the mean asymptomatic "
                "infectious period in days."
            ),
            kind="rate",
            default=asymptomatic_recovery_rate,
            min=0,
            max=10,
            units="1/days",
            tags=["recovery", "SEIAR"],
        ),
        ParameterSpec(
            name="asymptomatic_relative_infectivity",
            description=(
                "Infectivity of Asymptomatic individuals relative to symptomatic "
                "Infected. Multiplies transmission_rate on the S → E flow driven "
                "by Asymptomatic. 0 means asymptomatics do not transmit; 1 means "
                "they transmit as much as symptomatic cases."
            ),
            kind="proportion",
            default=asymptomatic_relative_infectivity,
            min=0,
            max=1,
            units="dimensionless",
            tags=["transmission", "SEIAR"],
        ),
    ]


def waning_immunity_specs(waning_rate: float = 1.0 / 365) -> list:
    """Parameter specs for the waning-immunity module (R → S)."""
    return [
        ParameterSpec(
            name="waning_rate",
            description=(
                "Rate at which recovered individuals lose immunity and return to "
                "Susceptible. The inverse (1/waning_rate) is the mean immune "
                "duration in days."
            ),
            kind="rate",
            default=waning_rate,
            min=0,
            max=1,
            units="1/days",
            tags=["waning", "module"],
        ),
    ]


def vaccination_specs(
    vaccination_rate: float = 0.01,
    vaccine_efficacy: float = 0.9,
) -> list:
    """Parameter specs for the vaccination module (S → V, leaky V → I breakthrough)."""
    return [
        ParameterSpec(
            name="vaccination_rate",
            description=(
                "Per-capita rate at which Susceptible individuals are vaccinated "
                "and move to the Vaccinated compartment."
            ),
            kind="rate",
            default=vaccination_rate,
            min=0,
            max=1,
            units="1/days",
            tags=["vaccination", "module"],
        ),
        ParameterSpec(
            name="vaccine_efficacy",
            description=(
                "Fractional reduction in transmission for Vaccinated individuals. "
                "Breakthrough transmission on V → I uses "
                "transmission_rate * (1 - vaccine_efficacy). 0 means no protection; "
                "1 means perfect protection."
            ),
            kind="proportion",
            default=vaccine_efficacy,
            min=0,
            max=1,
            units="dimensionless",
            tags=["vaccination", "module"],
        ),
    ]


def outcome_specs(
    outcome: str,
    mortality_rate: float = 0.01,
    hospitalization_rate: float = 0.01,
    hospitalization_recovery_rate: float = 0.1,
) -> list:
    """Parameter specs for the outcome module ('deaths' or 'hospitalization')."""
    if outcome == "deaths":
        return [
            ParameterSpec(
                name="mortality_rate",
                description=(
                    "Per-capita death rate for Infected individuals. Adds an "
                    "Infected → Dead spontaneous transition."
                ),
                kind="rate",
                default=mortality_rate,
                min=0,
                max=1,
                units="1/days",
                tags=["outcome", "deaths", "module"],
            ),
        ]
    if outcome == "hospitalization":
        return [
            ParameterSpec(
                name="hospitalization_rate",
                description=(
                    "Per-capita rate at which Infected individuals are hospitalized. "
                    "Adds an Infected → Hospitalized spontaneous transition."
                ),
                kind="rate",
                default=hospitalization_rate,
                min=0,
                max=1,
                units="1/days",
                tags=["outcome", "hospitalization", "module"],
            ),
            ParameterSpec(
                name="hospitalization_recovery_rate",
                description=(
                    "Rate at which Hospitalized individuals recover. The inverse "
                    "(1/hospitalization_recovery_rate) is the mean length of stay."
                ),
                kind="rate",
                default=hospitalization_recovery_rate,
                min=0,
                max=10,
                units="1/days",
                tags=["outcome", "hospitalization", "module"],
            ),
        ]
    raise ValueError(
        f"Unknown outcome '{outcome}'. Supported values: 'deaths', 'hospitalization'."
    )


# Map of model name → spec factory function
MODEL_SPEC_FACTORIES = {
    "SIR": sir_specs,
    "SEIR": seir_specs,
    "SIS": sis_specs,
    "SEIAR": seiar_specs,
}
