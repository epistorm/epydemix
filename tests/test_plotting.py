import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epydemix.population import Population
from epydemix.visualization.plotting import (
    get_timeseries_data,
    plot_contact_matrix,
    plot_distance_distribution,
    plot_population,
    plot_posterior_distribution,
    plot_posterior_distribution_2d,
    plot_quantiles,
    plot_trajectories,
)


@pytest.fixture
def mock_quantiles_data():
    """Fixture providing mock quantiles data"""
    dates = pd.date_range("2023-01-01", "2023-01-10")
    data = []

    for q in [0.05, 0.5, 0.95]:
        for date in dates:
            data.append(
                {
                    "date": date,
                    "I_total": 100
                    * np.exp(-0.1 * (date - dates[0]).days)
                    * (1 + 0.2 * (q - 0.5)),
                    "S_total": 900
                    - 100
                    * np.exp(-0.1 * (date - dates[0]).days)
                    * (1 + 0.2 * (q - 0.5)),
                    "quantile": q,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def mock_trajectories_data():
    """Fixture providing mock trajectories data"""
    dates = pd.date_range("2023-01-01", "2023-01-10")
    trajectories = {
        "I_total": np.random.normal(
            loc=np.exp(-0.1 * np.arange(len(dates))),
            scale=0.1,
            size=(10, len(dates)),  # 10 trajectories
        )
        * 100,
        "S_total": 1000
        - np.random.normal(
            loc=np.exp(-0.1 * np.arange(len(dates))), scale=0.1, size=(10, len(dates))
        )
        * 100,
    }
    return trajectories, dates


@pytest.fixture
def mock_observed_data():
    """Fixture providing mock observed data"""
    dates = pd.date_range("2023-01-01", "2023-01-10")
    return pd.DataFrame(
        {
            "date": dates,
            "I_total": 100 * np.exp(-0.1 * np.arange(len(dates)))
            + np.random.normal(0, 5, len(dates)),
        }
    )


@pytest.fixture
def mock_posterior_data():
    """Fixture providing mock posterior distribution data"""
    n_samples = 1000
    posterior_data = pd.DataFrame(
        {
            "transmission_rate": np.random.normal(0.3, 0.05, n_samples),
            "recovery_rate": np.random.normal(0.1, 0.02, n_samples),
            "R0": np.random.normal(3.0, 0.5, n_samples),
        }
    )
    return posterior_data


@pytest.fixture
def mock_population_data():
    """Fixture providing mock population data"""
    population = Population()
    population.add_population(
        np.array([1000, 2000, 3000, 2500, 1500]),
        ["0-4", "5-14", "15-44", "45-64", "65+"],
    )
    population.add_contact_matrix(np.random.random((5, 5)))
    return population


@pytest.fixture
def mock_distance_data():
    """Fixture providing mock distance distribution data"""
    # Create mock distance data with realistic shape
    n_points = 1000
    distances = np.random.lognormal(mean=1.0, sigma=0.5, size=n_points)
    weights = np.random.uniform(0.5, 1.5, size=n_points)
    return distances, weights


def test_get_timeseries_data(mock_quantiles_data):
    """Test timeseries data extraction"""
    data = get_timeseries_data(mock_quantiles_data, "I_total", 0.5)
    assert isinstance(data, pd.DataFrame)
    assert "date" in data.columns
    assert "I_total" in data.columns
    assert len(data) == 10  # 10 days of data


def test_plot_quantiles(mock_quantiles_data, mock_observed_data):
    """Test quantile plotting"""
    fig, ax = plt.subplots()

    # Basic plot test
    ax = plot_quantiles(
        df_quantiles=mock_quantiles_data,
        columns=["I_total"],
        data=mock_observed_data,
        ax=ax,
        title="Test Plot",
        ylabel="Cases",
        show_data=True,
    )

    # Check plot elements
    assert ax.get_title() == "Test Plot"
    assert ax.get_ylabel() == "Cases"
    assert len(ax.lines) >= 1  # At least median line
    assert len(ax.collections) >= 1  # At least one confidence interval

    plt.close()


def test_plot_trajectories(mock_trajectories_data, mock_observed_data):
    """Test trajectory plotting"""
    trajectories, dates = mock_trajectories_data
    fig, ax = plt.subplots()

    # Basic plot test
    ax = plot_trajectories(
        stacked=trajectories,
        columns=["I_total"],
        dates=dates,
        data=mock_observed_data,
        ax=ax,
        title="Test Trajectories",
        ylabel="Cases",
        show_data=True,
    )

    # Check plot elements
    assert ax.get_title() == "Test Trajectories"
    assert ax.get_ylabel() == "Cases"
    assert len(ax.lines) >= 10  # At least 10 trajectory lines

    plt.close()


def test_plot_multiple_compartments(mock_quantiles_data):
    """Test plotting multiple compartments"""
    fig, ax = plt.subplots()

    ax = plot_quantiles(
        df_quantiles=mock_quantiles_data,
        columns=["S_total", "I_total"],
        ax=ax,
        title="Multiple Compartments",
        labels=["Susceptible", "Infected"],
    )

    # Check legend
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 2

    plt.close()


def test_plot_styling(mock_quantiles_data):
    """Test plot styling options"""
    fig, ax = plt.subplots()

    ax = plot_quantiles(
        df_quantiles=mock_quantiles_data,
        columns=["I_total"],
        ax=ax,
        y_scale="log",
        show_grid=True,
        ci_alpha=0.2,
        colors="red",
        show_legend=False,
    )

    # Check styling
    assert ax.get_yscale() == "log"
    assert ax.yaxis.get_gridlines()[0].get_visible()
    assert ax.get_legend() is None

    plt.close()


def test_plot_posterior(mock_posterior_data):
    """Test posterior distribution plotting"""
    fig, ax = plt.subplots()

    # Test basic posterior plot
    ax = plot_posterior_distribution(
        posterior=mock_posterior_data,
        parameter="transmission_rate",
        ax=ax,
        title="Test Posterior",
        xlabel="Transmission Rate",
        ylabel="Density",
    )

    # Check plot elements
    assert ax.get_title() == "Test Posterior"
    assert ax.get_xlabel() == "Transmission Rate"
    assert ax.get_ylabel() == "Density"
    assert len(ax.patches) > 0  # Should have histogram bars

    plt.close()


def test_plot_posterior_2d(mock_posterior_data):
    """Test 2D posterior distribution plotting"""
    fig, ax = plt.subplots()

    # Test basic 2D posterior plot
    ax = plot_posterior_distribution_2d(
        posterior=mock_posterior_data,
        parameter_x="transmission_rate",
        parameter_y="recovery_rate",
        ax=ax,
        title="Test 2D Posterior",
        xlabel="Transmission Rate",
        ylabel="Recovery Rate",
    )

    # Check plot elements
    assert ax.get_title() == "Test 2D Posterior"
    assert ax.get_xlabel() == "Transmission Rate"
    assert ax.get_ylabel() == "Recovery Rate"
    assert len(ax.collections) > 0  # Should have scatter or contour plot

    plt.close()


def test_plot_population(mock_population_data):
    """Test population pyramid plotting"""
    fig, ax = plt.subplots()
    population = mock_population_data

    # Test basic population plot
    ax = plot_population(
        population=population,
        ax=ax,
        title="Population Structure",
        xlabel="Population",
        ylabel="Age Groups",
    )

    # Check plot elements
    assert ax.get_title() == "Population Structure"
    assert ax.get_xlabel() == "Population"
    assert ax.get_ylabel() == "Age Groups"

    # Check bars
    assert len(ax.patches) == len(population.Nk)  # Should have one bar per age group

    # Check y-ticks (age groups)
    xtick_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert all(group in xtick_labels for group in population.Nk_names)

    plt.close()


def test_plot_contact_matrix(mock_population_data):
    """Test contact matrix plotting"""
    fig, ax = plt.subplots()
    population = mock_population_data

    # Test basic contact matrix plot
    ax = plot_contact_matrix(
        population=population,
        layer="all",
        ax=ax,
        title="Contact Matrix",
        cmap="YlOrRd",
        show_colorbar=True,
    )

    # Check plot elements
    assert ax.get_title() == "Contact Matrix"
    assert len(ax.images) == 1  # Should have one heatmap

    # Check axis labels
    xtick_labels = [label.get_text() for label in ax.get_xticklabels()]
    ytick_labels = [label.get_text() for label in ax.get_yticklabels()]
    assert all(group in xtick_labels for group in population.Nk_names)
    assert all(group in ytick_labels for group in population.Nk_names)

    # Check matrix dimensions
    assert ax.images[0].get_array().shape == (len(population.Nk), len(population.Nk))

    plt.close()


def test_plot_distance_distribution(mock_distance_data):
    """Test distance distribution plotting"""
    fig, ax = plt.subplots()
    distances, weights = mock_distance_data

    # Test basic distance distribution plot
    ax = plot_distance_distribution(
        distances=distances,
        ax=ax,
        title="Distance Distribution",
        xlabel="Errors",
        ylabel="Density",
        bins=30,
    )

    # Check plot elements
    assert ax.get_title() == "Distance Distribution"
    assert ax.get_xlabel() == "Errors"
    assert ax.get_ylabel() == "Density"
    assert len(ax.patches) == 30  # Number of histogram bins

    # Check that histogram is normalized
    total_area = sum(patch.get_height() * patch.get_width() for patch in ax.patches)
    assert np.isclose(total_area, 1.0, rtol=1e-2)  # Should be normalized to 1

    plt.close()
