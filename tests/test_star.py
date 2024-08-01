import pytest
import pandas as pd
from tfop_analysis import Star 

target = "TOI-6449"
star_params = {
    'ra': 125.930455146652,
    'dec': -5.18949555185015,
    'rstar': (0.454183, 0.0155273),
    'mstar': (0.454183, 0.0155273),
    'rhostar': (4.813903712185822, 0.5460704755765515),
    'teff': (3356.0, 157.0),
    'logg': (4.7778, 0.008642),
    'feh': (0.0, 0.1)
    }

@pytest.fixture
def sample_inputs():
    # Provide sample data for testing
    return {
        "name": target,
    }

def test_star_initialization(sample_inputs):
    star_instance = Star(**sample_inputs, source='tic')
    # Check that instance is created
    assert isinstance(star_instance, Star)
    assert star_instance.params_to_dict() == star_params
    gaia_sources = star_instance.get_gaia_sources(rad_arcsec=60)
    assert isinstance(gaia_sources, pd.DataFrame)
