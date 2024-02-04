
import pytest
import numpy as np
from tfop_analysis import Planet 

target = "TOI-6449"
star_params = {
    'rstar': (0.454183, 0.0155273),
    'mstar': (0.454183, 0.0155273),
    'rhostar': (4.813903712185822, 0.5460704755765515),
    'teff': (3356.0, 157.0),
    'logg': (4.7778, 0.008642),
    'feh': (0.0, 0.1)
    }
planet_params = {
    't0': (2459984.192328, 0.0018611),
    'period': (2.4404127, 0.000413),
    'tdur': np.array([0.     , 0.00825]),
    'imp': (0.0, 0.1),
    'rprs': np.array([0.25979992, 0.07893168]),
    'a_Rs': (12.879247193048114, 0.48699053548942756)
    }

@pytest.fixture
def sample_inputs():
    # Provide sample data for testing
    return {
        "name": target,
        "alias": '.01',
        'star_params': star_params, 
        'source': 'toi'

    }

def test_star_initialization(sample_inputs):
    planet_instance = Planet(**sample_inputs)
    # Check that instance is created
    assert isinstance(planet_instance, Planet)
    are_dicts_equal(planet_instance.params_to_dict(), planet_params)

@pytest.mark.exclude
def are_dicts_equal(dict1, dict2):
    if len(dict1) != len(dict2):
        return False

    for key, value1 in dict1.items():
        if key not in dict2:
            return False

        value2 = dict2[key]

        if isinstance(value1, list) and isinstance(value2, list):
            if value1 != value2:
                return False
        elif isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            if not np.array_equal(value1, value2):
                return False
        elif value1 != value2:
            return False

    return True