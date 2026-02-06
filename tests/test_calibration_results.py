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
