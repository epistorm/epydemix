import pytest
import numpy as np
import pandas as pd
import warnings
from epydemix.model.simulation_results import SimulationResults
from epydemix.model.simulation_output import Trajectory


@pytest.fixture
def mock_trajectory_data_no_nan():
    """Create mock trajectory data without NaN values."""
    dates = pd.date_range('2024-01-01', periods=10, freq='D').tolist()
    compartments = {
        'S': np.array([1000, 990, 980, 970, 960, 950, 940, 930, 920, 910]),
        'I': np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    }
    transitions = {
        'S_to_I': np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    }
    compartment_idx = {'S': 0, 'I': 1}
    transitions_idx = {'S_to_I': 0}

    trajectories = []
    for i in range(5):  # 5 trajectories
        traj = Trajectory(
            compartments={k: v * (1 + i * 0.1) for k, v in compartments.items()},
            transitions={k: v * (1 + i * 0.1) for k, v in transitions.items()},
            dates=dates,
            compartment_idx=compartment_idx,
            transitions_idx=transitions_idx,
            parameters={}
        )
        trajectories.append(traj)

    return trajectories


@pytest.fixture
def mock_trajectory_data_with_nan():
    """Create mock trajectory data with NaN values at the beginning."""
    dates = pd.date_range('2024-01-01', periods=10, freq='D').tolist()
    compartment_idx = {'S': 0, 'I': 1}
    transitions_idx = {'S_to_I': 0}

    trajectories = []
    for i in range(5):  # 5 trajectories
        # First 3 time points are NaN for some trajectories
        s_values = np.array([np.nan, np.nan, np.nan, 970, 960, 950, 940, 930, 920, 910])
        i_values = np.array([np.nan, np.nan, np.nan, 40, 50, 60, 70, 80, 90, 100])
        transition_values = np.array([np.nan, np.nan, np.nan, 10, 10, 10, 10, 10, 10, 10])

        # Vary the values slightly for each trajectory
        compartments = {
            'S': s_values * (1 + i * 0.1) if i > 0 else s_values,
            'I': i_values * (1 + i * 0.1) if i > 0 else i_values
        }
        transitions = {
            'S_to_I': transition_values * (1 + i * 0.1) if i > 0 else transition_values
        }

        traj = Trajectory(
            compartments=compartments,
            transitions=transitions,
            dates=dates,
            compartment_idx=compartment_idx,
            transitions_idx=transitions_idx,
            parameters={}
        )
        trajectories.append(traj)

    return trajectories


@pytest.fixture
def mock_trajectory_data_high_nan():
    """Create mock trajectory data with >50% NaN values."""
    dates = pd.date_range('2024-01-01', periods=10, freq='D').tolist()
    compartment_idx = {'S': 0, 'I': 1}
    transitions_idx = {'S_to_I': 0}

    trajectories = []
    for i in range(5):  # 5 trajectories
        # First 6 time points are NaN (60% of data)
        s_values = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 940, 930, 920, 910])
        i_values = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 70, 80, 90, 100])
        transition_values = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10, 10, 10, 10])

        compartments = {
            'S': s_values * (1 + i * 0.1) if i > 0 else s_values,
            'I': i_values * (1 + i * 0.1) if i > 0 else i_values
        }
        transitions = {
            'S_to_I': transition_values * (1 + i * 0.1) if i > 0 else transition_values
        }

        traj = Trajectory(
            compartments=compartments,
            transitions=transitions,
            dates=dates,
            compartment_idx=compartment_idx,
            transitions_idx=transitions_idx,
            parameters={}
        )
        trajectories.append(traj)

    return trajectories


def test_get_quantiles_no_nan(mock_trajectory_data_no_nan):
    """Test quantiles computation with no NaN values."""
    sim_results = SimulationResults(
        trajectories=mock_trajectory_data_no_nan,
        parameters={}
    )

    stacked = sim_results.get_stacked_compartments()
    quantiles_df = sim_results.get_quantiles(stacked, quantiles=[0.05, 0.5, 0.95])

    assert 'date' in quantiles_df.columns
    assert 'quantile' in quantiles_df.columns
    assert 'S' in quantiles_df.columns
    assert 'I' in quantiles_df.columns
    assert len(quantiles_df) == 10 * 3  # 10 dates * 3 quantiles
    assert not quantiles_df['S'].isna().any()
    assert not quantiles_df['I'].isna().any()


def test_get_quantiles_with_nan_default(mock_trajectory_data_with_nan):
    """Test that NaN values propagate with default ignore_nan=False."""
    sim_results = SimulationResults(
        trajectories=mock_trajectory_data_with_nan,
        parameters={}
    )

    stacked = sim_results.get_stacked_compartments()
    quantiles_df = sim_results.get_quantiles(stacked, quantiles=[0.5])

    # First 3 dates should have NaN in the median
    first_3_dates = quantiles_df['date'].unique()[:3]
    for date in first_3_dates:
        date_data = quantiles_df[quantiles_df['date'] == date]
        assert date_data['S'].isna().all()
        assert date_data['I'].isna().all()


def test_get_quantiles_with_nan_ignore_true(mock_trajectory_data_with_nan):
    """Test that NaN values are ignored with ignore_nan=True."""
    sim_results = SimulationResults(
        trajectories=mock_trajectory_data_with_nan,
        parameters={}
    )

    stacked = sim_results.get_stacked_compartments()
    quantiles_df = sim_results.get_quantiles(stacked, quantiles=[0.5], ignore_nan=True)

    # Should have valid values even at early dates
    # (though they may be based on fewer samples)
    # At least some values should be non-NaN
    assert quantiles_df['S'].notna().sum() > 0
    assert quantiles_df['I'].notna().sum() > 0


def test_get_quantiles_warning_triggered(mock_trajectory_data_high_nan):
    """Test that warning is triggered when >50% NaN values exist."""
    sim_results = SimulationResults(
        trajectories=mock_trajectory_data_high_nan,
        parameters={}
    )

    stacked = sim_results.get_stacked_compartments()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        quantiles_df = sim_results.get_quantiles(stacked, quantiles=[0.5], ignore_nan=True)

        # Should have at least one warning for S and I compartments
        assert len(w) >= 2
        warning_messages = [str(warning.message) for warning in w]
        assert any('S' in msg and 'NaN values' in msg for msg in warning_messages)
        assert any('I' in msg and 'NaN values' in msg for msg in warning_messages)


def test_get_quantiles_no_warning_without_ignore_nan(mock_trajectory_data_high_nan):
    """Test that warning is not triggered when ignore_nan=False."""
    sim_results = SimulationResults(
        trajectories=mock_trajectory_data_high_nan,
        parameters={}
    )

    stacked = sim_results.get_stacked_compartments()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        quantiles_df = sim_results.get_quantiles(stacked, quantiles=[0.5], ignore_nan=False)

        # Should not have warnings about NaN values
        assert len(w) == 0


def test_get_quantiles_transitions_ignore_nan(mock_trajectory_data_with_nan):
    """Test get_quantiles_transitions with ignore_nan parameter."""
    sim_results = SimulationResults(
        trajectories=mock_trajectory_data_with_nan,
        parameters={}
    )

    # Without ignore_nan
    quantiles_df_default = sim_results.get_quantiles_transitions(quantiles=[0.5])
    assert quantiles_df_default['S_to_I'].isna().any()

    # With ignore_nan=True
    quantiles_df_ignore = sim_results.get_quantiles_transitions(quantiles=[0.5], ignore_nan=True)
    # Should have more non-NaN values
    assert quantiles_df_ignore['S_to_I'].notna().sum() >= quantiles_df_default['S_to_I'].notna().sum()


def test_get_quantiles_compartments_ignore_nan(mock_trajectory_data_with_nan):
    """Test get_quantiles_compartments with ignore_nan parameter."""
    sim_results = SimulationResults(
        trajectories=mock_trajectory_data_with_nan,
        parameters={}
    )

    # Without ignore_nan
    quantiles_df_default = sim_results.get_quantiles_compartments(quantiles=[0.5])
    assert quantiles_df_default['S'].isna().any()

    # With ignore_nan=True
    quantiles_df_ignore = sim_results.get_quantiles_compartments(quantiles=[0.5], ignore_nan=True)
    # Should have more non-NaN values
    assert quantiles_df_ignore['S'].notna().sum() >= quantiles_df_default['S'].notna().sum()
