import datetime
import warnings

import numpy as np
import pytest

from epydemix.calibration.calibration_results import CalibrationResults


@pytest.fixture
def mock_calibration_data_no_nan():
    """Create mock calibration data without NaN values."""
    # Simulate 5 trajectories with 10 time steps each
    trajectories = []
    for i in range(5):
        traj = {
            "S": np.array([1000, 990, 980, 970, 960, 950, 940, 930, 920, 910])
            * (1 + i * 0.1),
            "I": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) * (1 + i * 0.1),
            "R": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) * (1 + i * 0.1),
        }
        trajectories.append(traj)

    calib_results = CalibrationResults()
    calib_results.selected_trajectories[0] = trajectories

    return calib_results


@pytest.fixture
def mock_calibration_data_with_nan():
    """Create mock calibration data with NaN values at the beginning."""
    trajectories = []
    for i in range(5):
        traj = {
            "S": np.array([np.nan, np.nan, np.nan, 970, 960, 950, 940, 930, 920, 910])
            * (1 + i * 0.1)
            if i > 0
            else np.array([np.nan, np.nan, np.nan, 970, 960, 950, 940, 930, 920, 910]),
            "I": np.array([np.nan, np.nan, np.nan, 40, 50, 60, 70, 80, 90, 100])
            * (1 + i * 0.1)
            if i > 0
            else np.array([np.nan, np.nan, np.nan, 40, 50, 60, 70, 80, 90, 100]),
            "R": np.array([np.nan, np.nan, np.nan, 30, 40, 50, 60, 70, 80, 90])
            * (1 + i * 0.1)
            if i > 0
            else np.array([np.nan, np.nan, np.nan, 30, 40, 50, 60, 70, 80, 90]),
        }
        trajectories.append(traj)

    calib_results = CalibrationResults()
    calib_results.selected_trajectories[0] = trajectories

    return calib_results


@pytest.fixture
def mock_calibration_data_high_nan():
    """Create mock calibration data with >50% NaN values."""
    trajectories = []
    for i in range(5):
        # First 6 time points are NaN (60% of data)
        traj = {
            "S": np.array(
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 940, 930, 920, 910]
            )
            * (1 + i * 0.1)
            if i > 0
            else np.array(
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 940, 930, 920, 910]
            ),
            "I": np.array(
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 70, 80, 90, 100]
            )
            * (1 + i * 0.1)
            if i > 0
            else np.array(
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 70, 80, 90, 100]
            ),
            "R": np.array(
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 60, 70, 80, 90]
            )
            * (1 + i * 0.1)
            if i > 0
            else np.array(
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 60, 70, 80, 90]
            ),
        }
        trajectories.append(traj)

    calib_results = CalibrationResults()
    calib_results.selected_trajectories[0] = trajectories
    calib_results.projections["test_scenario"] = trajectories

    return calib_results


def test_get_calibration_quantiles_no_nan(mock_calibration_data_no_nan):
    """Test calibration quantiles computation with no NaN values."""
    quantiles_df = mock_calibration_data_no_nan.get_calibration_quantiles(
        quantiles=[0.05, 0.5, 0.95]
    )

    assert "date" in quantiles_df.columns
    assert "quantile" in quantiles_df.columns
    assert "S" in quantiles_df.columns
    assert "I" in quantiles_df.columns
    assert "R" in quantiles_df.columns
    assert len(quantiles_df) == 10 * 3  # 10 time steps * 3 quantiles
    assert not quantiles_df["S"].isna().any()
    assert not quantiles_df["I"].isna().any()
    assert not quantiles_df["R"].isna().any()


def test_get_calibration_quantiles_with_nan_default(mock_calibration_data_with_nan):
    """Test that NaN values propagate with default ignore_nan=False."""
    quantiles_df = mock_calibration_data_with_nan.get_calibration_quantiles(
        quantiles=[0.5]
    )

    # First 3 time steps should have NaN in the median
    first_3_steps = quantiles_df["date"].unique()[:3]
    for step in first_3_steps:
        step_data = quantiles_df[quantiles_df["date"] == step]
        assert step_data["S"].isna().all()
        assert step_data["I"].isna().all()
        assert step_data["R"].isna().all()


def test_get_calibration_quantiles_with_nan_ignore_true(mock_calibration_data_with_nan):
    """Test that NaN values are ignored with ignore_nan=True."""
    quantiles_df = mock_calibration_data_with_nan.get_calibration_quantiles(
        quantiles=[0.5], ignore_nan=True
    )

    # Should have valid values even at early time steps
    assert quantiles_df["S"].notna().sum() > 0
    assert quantiles_df["I"].notna().sum() > 0
    assert quantiles_df["R"].notna().sum() > 0


def test_get_calibration_quantiles_warning_triggered(mock_calibration_data_high_nan):
    """Test that warning is triggered when >50% NaN values exist."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _quantiles_df = mock_calibration_data_high_nan.get_calibration_quantiles(
            quantiles=[0.5], ignore_nan=True
        )

        # Should have at least one warning for S, I, and R compartments
        assert len(w) >= 3
        warning_messages = [str(warning.message) for warning in w]
        assert any("S" in msg and "NaN values" in msg for msg in warning_messages)
        assert any("I" in msg and "NaN values" in msg for msg in warning_messages)
        assert any("R" in msg and "NaN values" in msg for msg in warning_messages)


def test_get_calibration_quantiles_no_warning_without_ignore_nan(
    mock_calibration_data_high_nan,
):
    """Test that warning is not triggered when ignore_nan=False."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _quantiles_df = mock_calibration_data_high_nan.get_calibration_quantiles(
            quantiles=[0.5], ignore_nan=False
        )

        # Should not have warnings about NaN values
        assert len(w) == 0


def test_get_projection_quantiles_ignore_nan(mock_calibration_data_high_nan):
    """Test get_projection_quantiles with ignore_nan parameter."""
    # Without ignore_nan
    quantiles_df_default = mock_calibration_data_high_nan.get_projection_quantiles(
        quantiles=[0.5], scenario_id="test_scenario"
    )
    assert quantiles_df_default["S"].isna().any()

    # With ignore_nan=True
    quantiles_df_ignore = mock_calibration_data_high_nan.get_projection_quantiles(
        quantiles=[0.5], scenario_id="test_scenario", ignore_nan=True
    )
    # Should have more non-NaN values
    assert (
        quantiles_df_ignore["S"].notna().sum()
        >= quantiles_df_default["S"].notna().sum()
    )


def test_calibration_quantiles_with_dates(mock_calibration_data_no_nan):
    """Test calibration quantiles with explicit dates."""
    dates = [datetime.date(2024, 1, i) for i in range(1, 11)]
    quantiles_df = mock_calibration_data_no_nan.get_calibration_quantiles(
        dates=dates, quantiles=[0.5]
    )

    assert len(quantiles_df) == 10
    assert quantiles_df["date"].iloc[0] == datetime.date(2024, 1, 1)
    assert quantiles_df["date"].iloc[-1] == datetime.date(2024, 1, 10)


def test_calibration_quantiles_with_variables_filter(mock_calibration_data_no_nan):
    """Test calibration quantiles with variables filtering."""
    quantiles_df = mock_calibration_data_no_nan.get_calibration_quantiles(
        quantiles=[0.5], variables=["S", "I"]
    )

    assert "S" in quantiles_df.columns
    assert "I" in quantiles_df.columns
    assert "R" not in quantiles_df.columns


# --- Fixture with compartment and transition keys ---


@pytest.fixture
def mock_data_with_transitions():
    """Create mock data with both compartment and transition keys."""
    n_traj = 5
    trajectories = []
    for i in range(n_traj):
        scale = 1 + i * 0.1
        traj = {
            "S_total": np.array([1000, 990, 980, 970, 960, 950, 940, 930, 920, 910])
            * scale,
            "I_total": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) * scale,
            "R_total": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) * scale,
            "S_to_I_total": np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]) * scale,
            "I_to_R_total": np.array([0, 10, 10, 10, 10, 10, 10, 10, 10, 10]) * scale,
        }
        trajectories.append(traj)

    calib_results = CalibrationResults()
    calib_results.selected_trajectories[0] = trajectories
    calib_results.projections["baseline"] = trajectories
    return calib_results


# --- get_calibration_trajectories direct tests ---


def test_get_calibration_trajectories_all_variables(mock_calibration_data_no_nan):
    """Test that all variables are returned when no variables arg is passed."""
    result = mock_calibration_data_no_nan.get_calibration_trajectories()
    assert set(result.keys()) == {"S", "I", "R"}
    for key in result:
        assert result[key].shape == (5, 10)


def test_get_calibration_trajectories_filter_subset(mock_calibration_data_no_nan):
    """Test filtering to a subset of variables."""
    result = mock_calibration_data_no_nan.get_calibration_trajectories(
        variables=["S", "I"]
    )
    assert set(result.keys()) == {"S", "I"}
    assert "R" not in result
    for key in result:
        assert result[key].shape == (5, 10)


def test_get_calibration_trajectories_nonexistent_variable(
    mock_calibration_data_no_nan,
):
    """Test that requesting a nonexistent variable returns an empty dict."""
    result = mock_calibration_data_no_nan.get_calibration_trajectories(variables=["X"])
    assert result == {}


def test_get_calibration_trajectories_empty():
    """Test that empty selected_trajectories returns an empty dict."""
    calib_results = CalibrationResults()
    result = calib_results.get_calibration_trajectories()
    assert result == {}


# --- get_projection_trajectories direct tests ---


def test_get_projection_trajectories_all_variables(mock_calibration_data_high_nan):
    """Test that all variables are returned when no variables arg is passed."""
    result = mock_calibration_data_high_nan.get_projection_trajectories(
        scenario_id="test_scenario"
    )
    assert set(result.keys()) == {"S", "I", "R"}
    for key in result:
        assert result[key].shape == (5, 10)


def test_get_projection_trajectories_filter_subset(mock_calibration_data_high_nan):
    """Test filtering projection trajectories to a single variable."""
    result = mock_calibration_data_high_nan.get_projection_trajectories(
        scenario_id="test_scenario", variables=["S"]
    )
    assert set(result.keys()) == {"S"}
    assert result["S"].shape == (5, 10)


def test_get_projection_trajectories_nonexistent_scenario():
    """Test that a nonexistent scenario raises ValueError."""
    calib_results = CalibrationResults()
    with pytest.raises(ValueError, match="No projections found"):
        calib_results.get_projection_trajectories(scenario_id="nonexistent")


# --- Compartment vs transition filtering tests ---


def test_calibration_trajectories_filter_compartments_only(
    mock_data_with_transitions,
):
    """Test filtering to only compartment keys (no transition keys)."""
    result = mock_data_with_transitions.get_calibration_trajectories(
        variables=["S_total", "I_total", "R_total"]
    )
    assert set(result.keys()) == {"S_total", "I_total", "R_total"}
    assert not any("_to_" in key for key in result)
    for key in result:
        assert result[key].shape == (5, 10)


def test_calibration_trajectories_filter_transitions_only(
    mock_data_with_transitions,
):
    """Test filtering to only transition keys (no compartment keys)."""
    result = mock_data_with_transitions.get_calibration_trajectories(
        variables=["S_to_I_total", "I_to_R_total"]
    )
    assert set(result.keys()) == {"S_to_I_total", "I_to_R_total"}
    # Verify no compartment-only keys
    for key in result:
        assert "_to_" in key
    for key in result:
        assert result[key].shape == (5, 10)


def test_projection_trajectories_filter_transitions_only(
    mock_data_with_transitions,
):
    """Test filtering projection trajectories to only transition keys."""
    result = mock_data_with_transitions.get_projection_trajectories(
        scenario_id="baseline", variables=["S_to_I_total", "I_to_R_total"]
    )
    assert set(result.keys()) == {"S_to_I_total", "I_to_R_total"}
    assert not any(key in result for key in ["S_total", "I_total", "R_total"])
    for key in result:
        assert result[key].shape == (5, 10)


def test_calibration_quantiles_filter_transitions(mock_data_with_transitions):
    """Test that calibration quantiles DataFrame only has transition columns."""
    quantiles_df = mock_data_with_transitions.get_calibration_quantiles(
        quantiles=[0.5], variables=["S_to_I_total", "I_to_R_total"]
    )
    assert "S_to_I_total" in quantiles_df.columns
    assert "I_to_R_total" in quantiles_df.columns
    assert "S_total" not in quantiles_df.columns
    assert "I_total" not in quantiles_df.columns
    assert "R_total" not in quantiles_df.columns
    assert "date" in quantiles_df.columns
    assert "quantile" in quantiles_df.columns


def test_projection_quantiles_with_variables_filter(mock_data_with_transitions):
    """Test that projection quantiles exclude filtered-out variables."""
    quantiles_df = mock_data_with_transitions.get_projection_quantiles(
        quantiles=[0.05, 0.5, 0.95],
        scenario_id="baseline",
        variables=["S_total", "I_total"],
    )
    assert "S_total" in quantiles_df.columns
    assert "I_total" in quantiles_df.columns
    assert "R_total" not in quantiles_df.columns
    assert "S_to_I_total" not in quantiles_df.columns
    assert "I_to_R_total" not in quantiles_df.columns
    assert len(quantiles_df) == 10 * 3  # 10 timesteps * 3 quantiles


# --- Non-numeric array skipping tests ---


@pytest.fixture
def mock_data_with_dates():
    """Create mock data where trajectories include a non-numeric 'dates' key."""
    import pandas as pd

    trajectories = []
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    for i in range(5):
        scale = 1 + i * 0.1
        traj = {
            "S": np.array([1000, 990, 980, 970, 960, 950, 940, 930, 920, 910]) * scale,
            "I": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) * scale,
            "dates": dates,
        }
        trajectories.append(traj)

    calib_results = CalibrationResults()
    calib_results.selected_trajectories[0] = trajectories
    calib_results.projections["baseline"] = trajectories
    return calib_results


def test_calibration_quantiles_skips_non_numeric(mock_data_with_dates):
    """Test that non-numeric arrays (e.g. dates) are skipped in quantile computation."""
    quantiles_df = mock_data_with_dates.get_calibration_quantiles(quantiles=[0.5])

    assert "S" in quantiles_df.columns
    assert "I" in quantiles_df.columns
    assert "dates" not in quantiles_df.columns


def test_projection_quantiles_skips_non_numeric(mock_data_with_dates):
    """Test that non-numeric arrays (e.g. dates) are skipped in projection quantiles."""
    quantiles_df = mock_data_with_dates.get_projection_quantiles(
        quantiles=[0.5], scenario_id="baseline"
    )

    assert "S" in quantiles_df.columns
    assert "I" in quantiles_df.columns
    assert "dates" not in quantiles_df.columns


def test_calibration_quantiles_skips_non_numeric_with_ignore_nan(mock_data_with_dates):
    """Test that the NaN check also skips non-numeric arrays."""
    quantiles_df = mock_data_with_dates.get_calibration_quantiles(
        quantiles=[0.5], ignore_nan=True
    )

    assert "S" in quantiles_df.columns
    assert "I" in quantiles_df.columns
    assert "dates" not in quantiles_df.columns
