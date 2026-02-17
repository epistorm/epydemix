import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-GUI backend before importing pyplot
import numpy as np
import pandas as pd

from epydemix.population import Population
from epydemix.population.population import (
    aggregate_demographic,
    aggregate_matrix,
    get_available_locations,
    load_epydemix_population,
    validate_age_group_mapping,
    validate_contacts_source,
)


@pytest.fixture
def basic_population():
    """Fixture providing a basic population setup"""
    pop = Population(name="test_population")
    pop.add_population([1000, 2000, 3000], ["0-4", "5-19", "20+"])
    pop.add_contact_matrix(
        np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]), "home"
    )
    return pop


def test_population_initialization():
    """Test Population class initialization"""
    pop = Population(name="test")
    assert pop.name == "test"
    assert len(pop.Nk) == 0
    assert len(pop.contact_matrices) == 0


def test_add_population():
    """Test adding population data"""
    pop = Population()

    # Test adding population with names
    pop.add_population([1000, 2000], ["0-4", "5+"])
    assert np.all(np.array_equal(pop.Nk, [1000, 2000]))
    assert np.all(pop.Nk_names == ["0-4", "5+"])

    # Test adding population without names
    pop2 = Population()
    pop2.add_population([1000, 2000])
    assert np.all(np.array_equal(pop2.Nk, [1000, 2000]))
    assert np.all(len(pop2.Nk_names) == 2)

    # Test invalid inputs
    with pytest.raises(ValueError):
        pop.add_population([1000, 2000], ["0-4"])  # Mismatched lengths


def test_add_contact_matrix():
    """Test adding contact matrices"""
    pop = Population()

    # Test adding valid contact matrix
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    pop.add_contact_matrix(matrix, "home")
    assert "home" in pop.contact_matrices
    assert np.array_equal(pop.contact_matrices["home"], matrix)

    # Test adding multiple matrices
    pop.add_contact_matrix(matrix, "work")
    assert len(pop.contact_matrices) == 2

    # Test invalid inputs
    with pytest.raises(ValueError):
        pop.add_contact_matrix(np.array([1, 2, 3]))  # Not 2D

    with pytest.raises(ValueError):
        pop.add_contact_matrix(np.array([[1, 2], [3, 4], [5, 6]]))  # Not square

    with pytest.raises(ValueError):
        pop.add_contact_matrix(matrix, "overall")  # Reserved name


def test_population_properties(basic_population):
    """Test population properties"""
    # Test total population
    assert basic_population.total_population == 6000

    # Test number of groups
    assert basic_population.num_groups == 3

    # Test contact matrix layers
    assert "home" in basic_population.contact_matrices
    assert basic_population.contact_matrices["home"].shape == (3, 3)


def test_matrix_aggregation():
    """Test contact matrix aggregation"""
    # Setup test data
    old_matrix = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
    old_pop = np.array([100, 100, 100, 100])
    new_pop = np.array([200, 200])

    age_group_mapping = {"0-9": ["0-4", "5-9"], "10+": ["10-14", "15+"]}

    old_groups = {"0-4": 0, "5-9": 1, "10-14": 2, "15+": 3}
    new_groups = {"0-9": 0, "10+": 1}

    # Test aggregation
    result = aggregate_matrix(
        old_matrix, old_pop, new_pop, age_group_mapping, old_groups, new_groups
    )

    assert result.shape == (2, 2)
    assert np.all(result >= 0)


def test_demographic_aggregation():
    """Test demographic data aggregation"""
    data = pd.DataFrame(
        {"group_name": ["0-4", "5-9", "10-14", "15+"], "value": [100, 100, 150, 150]}
    )

    grouping = {"0-9": ["0-4", "5-9"], "10+": ["10-14", "15+"]}

    result = aggregate_demographic(data, grouping)
    assert len(result) == 2
    assert result.loc[result.group_name == "0-9", "value"].iloc[0] == 200
    assert result.loc[result.group_name == "10+", "value"].iloc[0] == 300


def test_validation_functions():
    """Test various validation functions"""

    # Test contacts source validation
    with pytest.raises(ValueError):
        validate_contacts_source("invalid_source", ["prem_2017", "prem_2021"])

    # Test age group mapping validation
    with pytest.raises(ValueError):
        validate_age_group_mapping({"0-9": ["0-4", "invalid"]}, ["0-4", "5-9", "10+"])


def test_population_repr(basic_population):
    """Test string representation of Population"""
    repr_str = str(basic_population)
    assert "test_population" in repr_str
    assert "Demographic groups: 3" in repr_str
    assert "Contact matrices: 1" in repr_str


def test_online_population_import():
    load_epydemix_population("Italy")
    get_available_locations()


def test_online_population_import_sex():
    pop = load_epydemix_population("United_States", attribute="sex")
    assert pop.num_groups > 0
    assert len(pop.contact_matrices) > 0


def test_online_population_import_race_ethnicity():
    pop = load_epydemix_population("United_States", attribute="race_ethnicity")
    assert pop.num_groups > 0
    assert len(pop.contact_matrices) > 0


def test_get_available_locations_sex():
    locations = get_available_locations(attribute="sex")
    assert len(locations) > 0


def test_get_available_locations_race_ethnicity():
    locations = get_available_locations(attribute="race_ethnicity")
    assert len(locations) > 0
