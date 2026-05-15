from .epimodel import EpiModel

SUPPORTED_MODELS = ["SIR", "SEIR", "SIS", "SEIAR"]


def load_predefined_model(
    model_name: str,
    transmission_rate: float = 0.3,
    recovery_rate: float = 0.1,
    incubation_rate: float = 0.2,
    # SEIAR-specific
    asymptomatic_fraction: float = 0.4,
    asymptomatic_recovery_rate: float = 0.14,
    asymptomatic_relative_infectivity: float = 0.5,
    # Waning immunity module
    waning_immunity: bool = False,
    waning_rate: float = 1.0 / 365,
    # Vaccination module
    vaccination: bool = False,
    vaccination_rate: float = 0.01,
    vaccine_efficacy: float = 0.9,
    # Outcome module
    outcome: str = None,
    mortality_rate: float = 0.01,
    hospitalization_rate: float = 0.01,
    hospitalization_recovery_rate: float = 0.1,
) -> EpiModel:
    """
    Load a predefined epidemic model with optional modular extensions.

    Args:
        model_name (str): Backbone model. One of "SIR", "SEIR", "SIS", "SEIAR".
        transmission_rate (float): Rate of transmission. Default 0.3.
        recovery_rate (float): Rate of recovery from Infected. Default 0.1.
        incubation_rate (float): Rate of progression from Exposed to Infected (SEIR, SEIAR). Default 0.2.
        asymptomatic_fraction (float): Fraction of exposed who become asymptomatic (SEIAR only). Default 0.4.
        asymptomatic_recovery_rate (float): Recovery rate for asymptomatic individuals (SEIAR only). Default 0.14.
        asymptomatic_relative_infectivity (float): Infectivity of asymptomatics relative to symptomatics (SEIAR only). Default 0.5.
        waning_immunity (bool): Add waning immunity (R → S). Not compatible with SIS. Default False.
        waning_rate (float): Rate of immunity waning. Default 1/365 (~1 year). Interpreted as the inverse
            of the average immunity duration in days.
        vaccination (bool): Add vaccination compartment (S → Vaccinated → Infected at reduced rate). Default False.
        vaccination_rate (float): Rate of vaccination from Susceptible. Default 0.01.
        vaccine_efficacy (float): Fraction by which vaccine reduces transmission. Default 0.9.
        outcome (str or None): Track a disease outcome. One of None, "deaths", "hospitalization". Default None.
        mortality_rate (float): Rate of death from Infected. Used when outcome="deaths". Default 0.01.
        hospitalization_rate (float): Rate of hospitalization from Infected. Used when outcome="hospitalization". Default 0.01.
        hospitalization_recovery_rate (float): Rate of recovery from Hospitalized. Used when outcome="hospitalization". Default 0.1.

    All rate parameters accept scalars, 1D arrays of shape (T,) for time-varying values, or 2D arrays
    of shape (T, G) for age-stratified values, consistent with the rest of the epydemix parameter system.

    Returns:
        EpiModel: Configured epidemic model.

    Raises:
        ValueError: If model_name is not recognised, or a module is incompatible with the chosen backbone.

    Examples:
        SIRS:   load_predefined_model("SIR", waning_immunity=True)
        SEIRS:  load_predefined_model("SEIR", waning_immunity=True)
        SEIR-V: load_predefined_model("SEIR", vaccination=True)
        SIRD:   load_predefined_model("SIR", outcome="deaths")
        SEIRD:  load_predefined_model("SEIR", outcome="deaths")
        SEIRH:  load_predefined_model("SEIR", outcome="hospitalization")
    """
    if model_name == "SIR":
        model = create_sir(transmission_rate, recovery_rate)
    elif model_name == "SEIR":
        model = create_seir(transmission_rate, incubation_rate, recovery_rate)
    elif model_name == "SIS":
        model = create_sis(transmission_rate, recovery_rate)
    elif model_name == "SEIAR":
        model = create_seiar(
            transmission_rate,
            incubation_rate,
            recovery_rate,
            asymptomatic_fraction,
            asymptomatic_recovery_rate,
            asymptomatic_relative_infectivity,
        )
    else:
        raise ValueError(
            f"Unknown predefined model: {model_name}. Supported models are: {SUPPORTED_MODELS}"
        )

    if waning_immunity:
        model = add_waning_immunity(model, waning_rate)
    if vaccination:
        model = add_vaccination(model, vaccination_rate, vaccine_efficacy)
    if outcome is not None:
        model = add_outcome(
            model,
            outcome,
            mortality_rate,
            hospitalization_rate,
            hospitalization_recovery_rate,
        )

    return model


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


def create_seiar(
    transmission_rate: float,
    incubation_rate: float,
    recovery_rate: float,
    asymptomatic_fraction: float,
    asymptomatic_recovery_rate: float,
    asymptomatic_relative_infectivity: float,
) -> EpiModel:
    """Create a SEIAR model with symptomatic and asymptomatic infectious compartments."""
    model = EpiModel(
        compartments=["Susceptible", "Exposed", "Infected", "Asymptomatic", "Recovered"],
        parameters={
            "transmission_rate": transmission_rate,
            "incubation_rate": incubation_rate,
            "recovery_rate": recovery_rate,
            "asymptomatic_fraction": asymptomatic_fraction,
            "asymptomatic_recovery_rate": asymptomatic_recovery_rate,
            "asymptomatic_relative_infectivity": asymptomatic_relative_infectivity,
        },
    )
    model.add_transition(
        source="Susceptible",
        target="Exposed",
        params=("transmission_rate", "Infected"),
        kind="mediated",
    )
    model.add_transition(
        source="Susceptible",
        target="Exposed",
        params=("transmission_rate * asymptomatic_relative_infectivity", "Asymptomatic"),
        kind="mediated",
    )
    model.add_transition(
        source="Exposed",
        target="Infected",
        params="incubation_rate * (1 - asymptomatic_fraction)",
        kind="spontaneous",
    )
    model.add_transition(
        source="Exposed",
        target="Asymptomatic",
        params="incubation_rate * asymptomatic_fraction",
        kind="spontaneous",
    )
    model.add_transition(
        source="Infected",
        target="Recovered",
        params="recovery_rate",
        kind="spontaneous",
    )
    model.add_transition(
        source="Asymptomatic",
        target="Recovered",
        params="asymptomatic_recovery_rate",
        kind="spontaneous",
    )
    return model


def add_waning_immunity(model: EpiModel, waning_rate: float) -> EpiModel:
    """Add waning immunity (Recovered → Susceptible) to an existing model."""
    if "Recovered" not in model.compartments:
        raise ValueError(
            "waning_immunity requires a 'Recovered' compartment, which is not present in the chosen backbone."
        )
    model.add_parameter("waning_rate", waning_rate)
    model.add_transition(
        source="Recovered",
        target="Susceptible",
        params="waning_rate",
        kind="spontaneous",
    )
    return model


def add_vaccination(
    model: EpiModel, vaccination_rate: float, vaccine_efficacy: float
) -> EpiModel:
    """Add a vaccination compartment (Susceptible → Vaccinated → Infected at reduced rate)."""
    model.add_compartments(["Vaccinated"])
    model.add_parameter("vaccination_rate", vaccination_rate)
    model.add_parameter("vaccine_efficacy", vaccine_efficacy)
    model.add_transition(
        source="Susceptible",
        target="Vaccinated",
        params="vaccination_rate",
        kind="spontaneous",
    )
    model.add_transition(
        source="Vaccinated",
        target="Infected",
        params=("transmission_rate * (1 - vaccine_efficacy)", "Infected"),
        kind="mediated",
    )
    return model


def add_outcome(
    model: EpiModel,
    outcome: str,
    mortality_rate: float,
    hospitalization_rate: float,
    hospitalization_recovery_rate: float,
) -> EpiModel:
    """Add an outcome compartment (Dead or Hospitalized) branching from the Infected compartment."""
    if outcome == "deaths":
        model.add_compartments(["Dead"])
        model.add_parameter("mortality_rate", mortality_rate)
        model.add_transition(
            source="Infected",
            target="Dead",
            params="mortality_rate",
            kind="spontaneous",
        )
    elif outcome == "hospitalization":
        if "Recovered" not in model.compartments:
            raise ValueError(
                "outcome='hospitalization' requires a 'Recovered' compartment for the Hospitalized → Recovered transition, "
                "which is not present in the chosen backbone."
            )
        model.add_compartments(["Hospitalized"])
        model.add_parameter("hospitalization_rate", hospitalization_rate)
        model.add_parameter("hospitalization_recovery_rate", hospitalization_recovery_rate)
        model.add_transition(
            source="Infected",
            target="Hospitalized",
            params="hospitalization_rate",
            kind="spontaneous",
        )
        model.add_transition(
            source="Hospitalized",
            target="Recovered",
            params="hospitalization_recovery_rate",
            kind="spontaneous",
        )
    else:
        raise ValueError(
            f"Unknown outcome: '{outcome}'. Supported values are: 'deaths', 'hospitalization'."
        )
    return model
