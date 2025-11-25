import numpy as np
import pytest

from epydemix.calibration.metrics import ae, mae, mape, rmse, validate_data, wmape


@pytest.fixture
def sample_data():
    """Fixture providing sample observed and simulated data"""
    observed = {"data": np.array([10, 20, 30, 40, 50])}
    simulated = {"data": np.array([12, 18, 32, 38, 52])}
    return observed, simulated


def test_validate_data():
    """Test data validation function"""
    # Test valid data
    observed = {"data": np.array([1, 2, 3])}
    simulated = {"data": np.array([1.1, 2.2, 3.3])}
    obs, sim = validate_data(observed, simulated)
    assert np.array_equal(obs, observed["data"])
    assert np.array_equal(sim, simulated["data"])

    # Test missing key
    with pytest.raises(
        ValueError, match="Both input Dictionaries must contain the key 'data'"
    ):
        validate_data({"wrong_key": [1, 2, 3]}, simulated)

    # Test shape mismatch
    with pytest.raises(
        ValueError, match="The shapes of observed and simulated data arrays must match"
    ):
        validate_data({"data": np.array([1, 2, 3])}, {"data": np.array([1, 2])})


def test_rmse(sample_data):
    """Test Root Mean Square Error calculation"""
    observed, simulated = sample_data
    result = rmse(observed, simulated)

    # Manual calculation for verification
    expected = np.sqrt(np.mean((observed["data"] - simulated["data"]) ** 2))
    assert np.isclose(result, expected)

    # Test with perfect prediction
    perfect = {"data": observed["data"]}
    assert rmse(observed, perfect) == 0


def test_mae(sample_data):
    """Test Mean Absolute Error calculation"""
    observed, simulated = sample_data
    result = mae(observed, simulated)

    # Manual calculation for verification
    expected = np.mean(np.abs(observed["data"] - simulated["data"]))
    assert np.isclose(result, expected)

    # Test with perfect prediction
    perfect = {"data": observed["data"]}
    assert mae(observed, perfect) == 0


def test_mape(sample_data):
    """Test Mean Absolute Percentage Error calculation"""
    observed, simulated = sample_data
    result = mape(observed, simulated)

    # Manual calculation for verification
    expected = np.mean(
        np.abs((observed["data"] - simulated["data"]) / observed["data"])
    )
    assert np.isclose(result, expected)

    # Test with perfect prediction
    perfect = {"data": observed["data"]}
    assert mape(observed, perfect) == 0


def test_wmape(sample_data):
    """Test Weighted Mean Absolute Percentage Error calculation"""
    observed, simulated = sample_data
    result = wmape(observed, simulated)

    # Manual calculation for verification
    expected = np.sum(np.abs(observed["data"] - simulated["data"])) / np.sum(
        np.abs(observed["data"])
    )
    assert np.isclose(result, expected)

    # Test with perfect prediction
    perfect = {"data": observed["data"]}
    assert wmape(observed, perfect) == 0


def test_ae(sample_data):
    """Test Absolute Error calculation"""
    observed, simulated = sample_data
    result = ae(observed, simulated)

    # Manual calculation for verification
    expected = np.abs(observed["data"] - simulated["data"])
    assert np.array_equal(result, expected)

    # Test with perfect prediction
    perfect = {"data": observed["data"]}
    assert np.all(ae(observed, perfect) == 0)


def test_edge_cases():
    """Test edge cases for all metrics"""
    # Test with single value
    obs = {"data": np.array([1])}
    sim = {"data": np.array([1.1])}

    assert isinstance(rmse(obs, sim), float)
    assert isinstance(mae(obs, sim), float)
    assert isinstance(mape(obs, sim), float)
    assert isinstance(wmape(obs, sim), float)
    assert isinstance(ae(obs, sim), np.ndarray)

    # Test with large values
    obs = {"data": np.array([1e6, 2e6])}
    sim = {"data": np.array([1.1e6, 1.9e6])}

    assert np.isfinite(rmse(obs, sim))
    assert np.isfinite(mae(obs, sim))
    assert np.isfinite(mape(obs, sim))
    assert np.isfinite(wmape(obs, sim))
    assert np.all(np.isfinite(ae(obs, sim)))


def test_metric_properties(sample_data):
    """Test mathematical properties of metrics"""
    observed, simulated = sample_data

    # Test non-negativity
    assert rmse(observed, simulated) >= 0
    assert mae(observed, simulated) >= 0
    assert mape(observed, simulated) >= 0
    assert wmape(observed, simulated) >= 0
    assert np.all(ae(observed, simulated) >= 0)

    # Test triangle inequality for MAE
    third = {"data": np.array([11, 19, 31, 39, 51])}
    assert mae(observed, simulated) <= mae(observed, third) + mae(third, simulated)
