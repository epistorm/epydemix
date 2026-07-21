"""Unit tests for ``epydemix.utils.utils``."""

import numpy as np
import pytest

from epydemix.utils.utils import multinomial

# A simple 3-compartment layout: index 0 is the 'stay' compartment, indices 1 and 2
# are the two 'leave' destinations selected by the mask.
STAY_IDX = 0
MASK = np.array([False, True, True])
RATES = np.array([0.0, 0.3, 0.1])


def test_multinomial_is_seed_reproducible():
    """Same seed -> identical draw; different seed -> (very likely) different draw."""
    draw_a = multinomial(
        1000, RATES, STAY_IDX, MASK, dt=1.0, rng=np.random.default_rng(0)
    )
    draw_b = multinomial(
        1000, RATES, STAY_IDX, MASK, dt=1.0, rng=np.random.default_rng(0)
    )
    assert draw_a == pytest.approx(draw_b)

    draw_c = multinomial(
        1000, RATES, STAY_IDX, MASK, dt=1.0, rng=np.random.default_rng(1)
    )
    assert draw_a != pytest.approx(draw_c)


def test_multinomial_conserves_population():
    """The draw partitions exactly ``n`` individuals across the compartments."""
    n = 1000
    for seed in range(5):
        draw = multinomial(
            n, RATES, STAY_IDX, MASK, dt=1.0, rng=np.random.default_rng(seed)
        )
        assert draw.sum() == n
        assert np.all(draw >= 0)
        assert len(draw) == len(RATES)


def test_multinomial_zero_trials_all_stay():
    """With ``n == 0`` no draw happens: everyone is placed in the stay compartment."""
    draw = multinomial(0, RATES, STAY_IDX, MASK, dt=1.0, rng=np.random.default_rng(0))
    assert draw.sum() == 0
    assert draw == pytest.approx(np.zeros(len(RATES), dtype=int))
    # The n==0 branch must not touch the rng, so it is deterministic regardless of seed.
    other = multinomial(0, RATES, STAY_IDX, MASK, dt=1.0, rng=np.random.default_rng(99))
    assert draw == pytest.approx(other)


def test_multinomial_zero_rates_all_stay():
    """With all leave-rates zero, every individual remains in the stay compartment."""
    zero_rates = np.zeros(len(RATES))
    n = 500
    draw = multinomial(
        n, zero_rates, STAY_IDX, MASK, dt=1.0, rng=np.random.default_rng(0)
    )
    assert draw[STAY_IDX] == n
    assert draw.sum() == n


def test_multinomial_linear_approximation_branch_is_reproducible():
    """The ``apply_linear_approximation`` path is seedable and conserves population."""
    n = 1000
    draw_a = multinomial(
        n,
        RATES,
        STAY_IDX,
        MASK,
        dt=1.0,
        apply_linear_approximation=True,
        rng=np.random.default_rng(0),
    )
    draw_b = multinomial(
        n,
        RATES,
        STAY_IDX,
        MASK,
        dt=1.0,
        apply_linear_approximation=True,
        rng=np.random.default_rng(0),
    )
    assert draw_a == pytest.approx(draw_b)
    assert draw_a.sum() == n

    # The exact and linear-approximation branches build different probability vectors,
    # so with the same seed they should generally produce different draws.
    draw_exact = multinomial(
        n,
        RATES,
        STAY_IDX,
        MASK,
        dt=1.0,
        apply_linear_approximation=False,
        rng=np.random.default_rng(0),
    )
    assert draw_a != pytest.approx(draw_exact)


def test_multinomial_accepts_integer_seed():
    """An integer seed is normalized like a Generator (fixes the API inconsistency).

    ``multinomial`` previously used the rng argument verbatim, so an int reached
    ``rng.multinomial`` and crashed. It must now accept an int seed and be reproducible
    with it, matching a Generator built from the same seed.
    """
    draw_from_int = multinomial(1000, RATES, STAY_IDX, MASK, dt=1.0, rng=0)
    draw_from_gen = multinomial(
        1000, RATES, STAY_IDX, MASK, dt=1.0, rng=np.random.default_rng(0)
    )
    assert draw_from_int == pytest.approx(draw_from_gen)


def test_multinomial_only_leaves_to_masked_destinations():
    """No individuals may flow to destinations excluded by the mask.

    Here only index 1 is a valid leave destination; index 2 is masked out and must
    stay empty regardless of its rate.
    """
    mask = np.array([False, True, False])
    rates = np.array([0.0, 0.3, 0.9])  # index 2 has a high rate but is masked out
    draw = multinomial(
        1000, rates, STAY_IDX, mask, dt=1.0, rng=np.random.default_rng(0)
    )
    assert draw[2] == 0
    assert draw.sum() == 1000


def test_multinomial_without_rng_is_nondeterministic():
    """The ``rng=None`` path draws from a fresh generator, so it is not reproducible.

    Negative control for the seeded tests: without a seed two draws must (almost
    surely) differ, confirming the stochasticity is real rather than a fixed default.
    """
    draw_a = multinomial(1000, RATES, STAY_IDX, MASK, dt=1.0)
    draw_b = multinomial(1000, RATES, STAY_IDX, MASK, dt=1.0)
    assert draw_a != pytest.approx(draw_b)
