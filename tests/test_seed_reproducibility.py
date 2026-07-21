"""Seed-reproducibility / paired-trajectory tests for the calibration pipeline."""

import numpy as np
import pytest
from scipy import stats

from epydemix.calibration.abc import ABCSampler
from epydemix.model import simulate
from epydemix.model.predefined_models import create_sir
from epydemix.population import Population

START_DATE = "2023-01-01"
END_DATE = "2023-01-20"
INITIAL_CONDITIONS = {
    "Susceptible": np.array([9900]),
    "Infected": np.array([100]),
    "Recovered": np.array([0]),
}


def _make_sir_model():
    """A single-group SIR model with a hand-built population (no network)."""
    model = create_sir(transmission_rate=0.3, recovery_rate=0.1)
    pop = Population()
    pop.add_population([10000])
    pop.add_contact_matrix(np.array([[1.0]]))
    model.set_population(pop)
    return model


def _simulate_wrapper(parameters):
    """ABCSampler-compatible wrapper returning the calibration target series."""
    return {"data": simulate(**parameters).transitions["Susceptible_to_Infected_total"]}


def _base_parameters(rng):
    return dict(
        epimodel=_make_sir_model(),
        start_date=START_DATE,
        end_date=END_DATE,
        initial_conditions_dict=INITIAL_CONDITIONS,
        rng=rng,
    )


@pytest.fixture
def observed():
    """'Truth' incidence series used as the calibration target."""
    return simulate(**_base_parameters(np.random.default_rng(0))).transitions[
        "Susceptible_to_Infected_total"
    ]


def test_simulate_is_seed_reproducible():
    """``simulate`` is reproducible under a seed and genuinely stochastic without one.

    Same seed gives identical trajectories; a different seed gives a different one.
    Confirms the model path respects rng, so any divergence in the tests below is
    attributable to the calibration code, not to ``simulate``.
    """
    key = "Susceptible_to_Infected_total"
    traj_a = simulate(**_base_parameters(np.random.default_rng(123))).transitions[key]
    traj_b = simulate(**_base_parameters(np.random.default_rng(123))).transitions[key]
    assert np.array_equal(traj_a, traj_b)

    # Negative control: a different seed must give a different trajectory
    traj_c = simulate(**_base_parameters(np.random.default_rng(456))).transitions[key]
    assert not np.array_equal(traj_a, traj_c)


def test_run_projections_paired_trajectories_are_seed_reproducible(observed):
    """Two projection runs seeded identically must produce identical trajectories.

    This is the paired-scenario case: with the same seed the posterior samples
    (and therefore the trajectories) should match exactly. Targets the
    ``np.random.choice`` posterior-sampling leak in ``ABCSampler.run_projections``.
    """
    # Calibrate once to obtain a posterior for both projection runs to sample from.
    sampler = ABCSampler(
        simulation_function=_simulate_wrapper,
        priors={
            "transmission_rate": stats.uniform(0.1, 0.4),
            "recovery_rate": stats.uniform(0.05, 0.15),
        },
        parameters=_base_parameters(np.random.default_rng(0)),
        observed_data=observed,
    )
    sampler.calibrate(
        strategy="rejection", epsilon=500.0, num_particles=20, verbose=False
    )

    posterior = sampler.results.get_posterior_distribution()
    # Guard: resampling only matters if the posterior has distinct rows.
    assert posterior.drop_duplicates().shape[0] > 1

    proj_a = sampler.run_projections(
        parameters=_base_parameters(np.random.default_rng(42)),
        iterations=10,
        scenario_id="a",
    )
    proj_b = sampler.run_projections(
        parameters=_base_parameters(np.random.default_rng(42)),
        iterations=10,
        scenario_id="b",
    )

    # Same seed -> same posterior samples drawn, in the same order
    assert proj_a.projection_parameters["a"].equals(proj_b.projection_parameters["b"])
    # therefore identical output trajectories.
    traj_a = proj_a.get_projection_trajectories(scenario_id="a")["data"]
    traj_b = proj_b.get_projection_trajectories(scenario_id="b")["data"]
    assert np.array_equal(traj_a, traj_b)


def test_smc_calibration_is_seed_reproducible(observed):
    """Two SMC calibrations seeded identically must yield identical posteriors.

    ``run_projections`` samples from a posterior produced by calibration, so the
    calibration itself must be reproducible for paired trajectories to hold. This
    exercises the leak sites the projection test does not: prior sampling
    (``sample_prior``), the perturbation kernel, and SMC particle resampling.
    """

    def _calibrate(seed):
        sampler = ABCSampler(
            simulation_function=_simulate_wrapper,
            priors={
                "transmission_rate": stats.uniform(0.1, 0.4),
                "recovery_rate": stats.uniform(0.05, 0.15),
            },
            parameters=_base_parameters(np.random.default_rng(seed)),
            observed_data=observed,
        )
        return sampler.calibrate(
            strategy="smc",
            num_particles=15,
            num_generations=3,
            verbose=False,
        )

    res_a = _calibrate(43)
    res_b = _calibrate(43)
    assert res_a.get_posterior_distribution().equals(res_b.get_posterior_distribution())

    # Negative control: a different seed yields a different posterior.
    res_c = _calibrate(44)
    assert not res_a.get_posterior_distribution().equals(
        res_c.get_posterior_distribution()
    )


def _simulate_wrapper_discrete(parameters):
    """Wrapper driven by an integer ``initial_infected`` parameter.

    A discrete prior makes SMC use ``DefaultPerturbationDiscrete``, exercising the
    ``np.random.rand`` / ``np.random.choice`` calls in that kernel.
    """
    params = dict(parameters)
    n_inf = int(params.pop("initial_infected"))
    params["initial_conditions_dict"] = {
        "Susceptible": np.array([10000 - n_inf]),
        "Infected": np.array([n_inf]),
        "Recovered": np.array([0]),
    }
    return {"data": simulate(**params).transitions["Susceptible_to_Infected_total"]}


def test_smc_discrete_prior_is_seed_reproducible():
    """Two SMC calibrations over a *discrete* prior, seeded identically, must match.

    Covers ``DefaultPerturbationDiscrete`` (lines 69/72), which the continuous SMC
    test above never reaches.
    """

    def _calibrate(seed):
        params = _base_parameters(np.random.default_rng(seed))
        # The wrapper builds initial conditions from the discrete parameter.
        params.pop("initial_conditions_dict", None)
        observed = _simulate_wrapper_discrete({**params, "initial_infected": 100})[
            "data"
        ]
        sampler = ABCSampler(
            simulation_function=_simulate_wrapper_discrete,
            priors={"initial_infected": stats.randint(50, 300)},
            parameters=params,
            observed_data=observed,
        )
        return sampler.calibrate(
            strategy="smc",
            num_particles=15,
            num_generations=3,
            verbose=False,
        )

    res_a = _calibrate(7)
    res_b = _calibrate(7)
    assert res_a.get_posterior_distribution().equals(res_b.get_posterior_distribution())

    # Negative control: a different seed yields a different posterior.
    res_c = _calibrate(8)
    assert not res_a.get_posterior_distribution().equals(
        res_c.get_posterior_distribution()
    )


def _params_without_rng():
    """Simulation parameters carrying no ``rng`` key at all."""
    params = _base_parameters(None)
    params.pop("rng")
    return params


def test_rng_argument_makes_calibration_and_projections_reproducible(observed):
    """Seeding the sampler via ``rng=`` alone must make BOTH calibration and its
    projections reproducible as a set, without any ``"rng"`` key in ``parameters``.

    This pins the ``rng=`` argument as a first-class, self-sufficient seed and guards
    against the regression where it seeded the ABC bookkeeping but left the simulation
    (and, by inheritance, the projections) unseeded.
    """

    def calibrate_and_project(seed):
        sampler = ABCSampler(
            simulation_function=_simulate_wrapper,
            priors={
                "transmission_rate": stats.uniform(0.1, 0.4),
                "recovery_rate": stats.uniform(0.05, 0.15),
            },
            parameters=_params_without_rng(),  # seeded only through rng= below
            observed_data=observed,
            rng=seed,
        )
        sampler.calibrate(
            strategy="rejection", epsilon=500.0, num_particles=20, verbose=False
        )
        posterior = sampler.results.get_posterior_distribution()
        # No rng passed here -> projections must inherit the sampler's seed.
        proj = sampler.run_projections(
            parameters=_params_without_rng(), iterations=8, scenario_id="s"
        )
        return posterior, proj.get_projection_trajectories(scenario_id="s")["data"]

    post_a, traj_a = calibrate_and_project(44)
    post_b, traj_b = calibrate_and_project(44)

    # Same seed on the sampler -> identical calibration posterior AND projections.
    assert post_a.equals(post_b)
    assert np.array_equal(traj_a, traj_b)

    # Negative control: a different sampler seed changes the results.
    post_c, _ = calibrate_and_project(4242)
    assert not post_a.equals(post_c)
