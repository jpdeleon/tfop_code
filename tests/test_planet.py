import pytest
import numpy as np
from tfop_analysis import Planet

target = "TOI-6449"
star_params = {
    "rstar": (0.454183, 0.0155273),
    "mstar": (0.454183, 0.0155273),
    "rhostar": (4.813903712185822, 0.5460704755765515),
    "teff": (3356.0, 157.0),
    "logg": (4.7778, 0.008642),
    "feh": (0.0, 0.1),
}
planet_params = {
    "t0": (2459984.192328, 0.0018611),
    "period": (2.4404127, 0.000413),
    "tdur": np.array([0.05641667, 0.00825]),
    "imp": (0, 0.1),
    "rprs": np.array([0.25979992, 0.07893168]),
    "a_Rs": (12.879247193048114, 0.48699053548942756),
}


@pytest.fixture
def sample_inputs():
    # Provide sample data for testing
    return {
        "name": target,
        "alias": ".01",
        "star_params": star_params,
        "source": "toi",
    }


def test_planet_initialization(sample_inputs):
    planet_instance = Planet(**sample_inputs)
    # Check that instance is created
    assert isinstance(planet_instance, Planet)
    pp = planet_instance.params_to_dict()
    assert np.all([k in planet_params.keys() for k in pp])
    # assert np.all([v==planet_params[k] for k,v in pp.items()])
    print(pp, planet_params)
    assert are_dicts_similar(pp, planet_params, tolerance=1e-3)


@pytest.mark.exclude
def are_dicts_similar(dict1, dict2, tolerance=1e-3):
    if set(dict1.keys()) != set(dict2.keys()):
        return False  # Different sets of keys, dictionaries are not similar
    for key in dict1.keys():
        value1 = dict1[key]
        value2 = dict2[key]
        if isinstance(value1, (int, float)):
            # Compare numeric values within tolerance
            if abs(value1 - value2) > tolerance:
                return False  # Values are not similar within tolerance
        elif isinstance(value1, tuple) and isinstance(value2, tuple):
            # Compare each element of the tuples within tolerance using np.all()
            if len(value1) != len(value2) or not np.all(
                np.abs(np.array(value1) - np.array(value2)) <= tolerance
            ):
                return False  # Tuples are not similar within tolerance
        elif isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            # Compare each element of the NumPy arrays within tolerance using np.all()
            if not np.all(np.abs(value1 - value2) <= tolerance):
                return False  # NumPy arrays are not similar within tolerance
        else:
            # Non-numeric values are compared for equality
            if value1 != value2:
                return False  # Non-numeric values are not equal
    return True  # All key-value pairs are similar within tolerance
