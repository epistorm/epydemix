"""Seed-reproducibility / paired-trajectory tests for the calibration pipeline."""

import numpy as np
import pytest
from scipy import stats

from epydemix.calibration.abc import ABCSampler
from epydemix.model import simulate
from epydemix.model.epimodel import stochastic_simulation
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
    assert traj_a == pytest.approx(traj_b)

    # Negative control: a different seed must give a different trajectory
    traj_c = simulate(**_base_parameters(np.random.default_rng(456))).transitions[key]
    assert traj_a != pytest.approx(traj_c)


def test_run_simulations_is_seed_reproducible():
    """``EpiModel.run_simulations`` respects the rng.

    Two calls with identically-seeded generators must produce identical trajectory
    ensembles; a different seed must produce a different one. Guards the rng plumbing
    in ``run_simulations`` (the method users actually call), which is otherwise only
    covered indirectly through the lower-level ``simulate``.
    """
    key = "Susceptible_to_Infected_total"

    def _stacked(seed):
        model = _make_sir_model()
        results = model.run_simulations(
            start_date=START_DATE,
            end_date=END_DATE,
            initial_conditions_dict=INITIAL_CONDITIONS,
            Nsim=5,
            rng=np.random.default_rng(seed),
        )
        return results.get_stacked_transitions([key])[key]

    stacked_a = _stacked(123)
    stacked_b = _stacked(123)
    assert stacked_a == pytest.approx(stacked_b)

    # Negative control: a different seed must give a different ensemble.
    stacked_c = _stacked(456)
    assert stacked_a != pytest.approx(stacked_c)


def test_run_simulations_trajectories_differ_across_nsim():
    """The ``Nsim`` trajectories share one Generator advanced sequentially.

    Because the single rng keeps advancing between simulations, the individual
    trajectories within one ``run_simulations`` call must not be identical to each other.
    """
    key = "Susceptible_to_Infected_total"
    model = _make_sir_model()
    results = model.run_simulations(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_conditions_dict=INITIAL_CONDITIONS,
        Nsim=5,
        rng=np.random.default_rng(123),
    )
    stacked = results.get_stacked_transitions([key])[key]  # shape (Nsim, timesteps)

    # At least two distinct trajectories in the ensemble.
    unique_rows = np.unique(stacked, axis=0)
    assert unique_rows.shape[0] > 1


def test_run_simulations_prefix_is_stable_across_nsim():
    """A larger ``Nsim`` run reproduces the smaller run's trajectories as a prefix.

    The trajectories are consecutive slices of one sequentially-advanced Generator, so
    with the same seed the first ``n`` trajectories of an ``Nsim=n+k`` run must match an
    ``Nsim=n`` run exactly; the larger run only appends more.
    """
    key = "Susceptible_to_Infected_total"

    def _stacked(nsim):
        results = _make_sir_model().run_simulations(
            start_date=START_DATE,
            end_date=END_DATE,
            initial_conditions_dict=INITIAL_CONDITIONS,
            Nsim=nsim,
            rng=np.random.default_rng(123),
        )
        return results.get_stacked_transitions([key])[key]

    small = _stacked(10)  # shape (10, timesteps)
    large = _stacked(15)  # shape (15, timesteps)

    assert large.shape[0] == 15
    assert small == pytest.approx(large[:10])


def test_run_simulations_accepts_integer_seed():
    """The public API accepts an int seed, not just a Generator, and normalizes it.

    Passing ``rng=0`` must behave identically to ``rng=np.random.default_rng(0)``; the
    int is normalized once at the entry point and threaded down as a Generator.
    """
    key = "Susceptible_to_Infected_total"

    def _stacked(rng):
        return (
            _make_sir_model()
            .run_simulations(
                start_date=START_DATE,
                end_date=END_DATE,
                initial_conditions_dict=INITIAL_CONDITIONS,
                Nsim=5,
                rng=rng,
            )
            .get_stacked_transitions([key])[key]
        )

    from_int = _stacked(0)
    from_generator = _stacked(np.random.default_rng(0))
    assert from_int == pytest.approx(from_generator)


def test_stochastic_simulation_is_seed_reproducible():
    """``stochastic_simulation``, where the multinomial draws happen, respects rng.

    Same seed gives byte-identical compartment/transition arrays; a different seed
    diverges. ``test_epimodel`` only checks shape/conservation on an unseeded run.
    """
    model = _make_sir_model()
    T = 15
    contact_matrices = [
        {"overall": model.population.contact_matrices["all"]} for _ in range(T)
    ]
    initial_conditions = np.array([[9900], [100], [0]])
    parameters = {
        "transmission_rate": np.full(T, 0.3),
        "recovery_rate": np.full(T, 0.1),
    }

    def _run(seed):
        return stochastic_simulation(
            T=T,
            contact_matrices=contact_matrices,
            epimodel=model,
            parameters=parameters,
            initial_conditions=initial_conditions,
            dt=1.0,
            rng=np.random.default_rng(seed),
        )

    comp_a, trans_a = _run(123)
    comp_b, trans_b = _run(123)
    assert comp_a == pytest.approx(comp_b)
    assert trans_a == pytest.approx(trans_b)

    # Negative control: a different seed must diverge somewhere.
    comp_c, trans_c = _run(456)
    assert comp_a != pytest.approx(comp_c) or trans_a != pytest.approx(trans_c)


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
    assert traj_a == pytest.approx(traj_b)


def test_run_projections_explicit_rng_overrides_and_is_reproducible(observed):
    """An explicit ``rng=`` seeds ``run_projections`` and overrides other seed sources.

    Covers the highest-precedence ``rng is not None`` branch: passing a seed directly
    must be reproducible and must take precedence over ``parameters["rng"]``. The two
    calls below use different ``parameters["rng"]`` but the same explicit ``rng``, so
    they must still produce identical posterior samples and trajectories.
    """
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
    assert sampler.results.get_posterior_distribution().drop_duplicates().shape[0] > 1

    proj_a = sampler.run_projections(
        parameters=_base_parameters(np.random.default_rng(1)),
        iterations=10,
        scenario_id="a",
        rng=7,
    )
    proj_b = sampler.run_projections(
        parameters=_base_parameters(np.random.default_rng(2)),  # different param seed
        iterations=10,
        scenario_id="b",
        rng=7,  # same explicit seed must win, making the two runs identical
    )

    assert proj_a.projection_parameters["a"].equals(proj_b.projection_parameters["b"])
    traj_a = proj_a.get_projection_trajectories(scenario_id="a")["data"]
    traj_b = proj_b.get_projection_trajectories(scenario_id="b")["data"]
    assert traj_a == pytest.approx(traj_b)


def test_rejection_calibration_is_seed_reproducible(observed):
    """Two rejection calibrations seeded identically must yield identical posteriors.

    Rejection sampling draws every candidate from the prior via the sampler's rng, so
    the posterior must be reproducible under a seed and differ under another. Pins the
    rejection strategy standalone; the projection tests only use it as a fixture.
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
            strategy="rejection", epsilon=500.0, num_particles=20, verbose=False
        )

    res_a = _calibrate(43)
    res_b = _calibrate(43)
    assert res_a.get_posterior_distribution().equals(res_b.get_posterior_distribution())

    # Negative control: a different seed yields a different posterior.
    res_c = _calibrate(44)
    assert not res_a.get_posterior_distribution().equals(
        res_c.get_posterior_distribution()
    )


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
    assert traj_a == pytest.approx(traj_b)

    # Negative control: a different sampler seed changes the results.
    post_c, _ = calibrate_and_project(4242)
    assert not post_a.equals(post_c)
