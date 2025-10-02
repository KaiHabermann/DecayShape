"""
Pytest configuration and fixtures for DecayShape tests.
"""

import pytest
import numpy as np
from decayshape import FixedParam
from decayshape.particles import CommonParticles


@pytest.fixture
def sample_s_values():
    """Sample s values for testing lineshapes."""
    return np.linspace(0.1, 2.0, 100)


@pytest.fixture
def sample_complex_s_values():
    """Sample complex s values for testing analytic continuation."""
    real_part = np.linspace(0.1, 2.0, 20)
    imag_part = np.linspace(-0.1, 0.1, 20)
    return real_part + 1j * imag_part


@pytest.fixture
def sample_numpy_array():
    """Sample numpy array for FixedParam testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_particle():
    """Sample particle for testing."""
    return CommonParticles.PI_PLUS


@pytest.fixture
def sample_fixed_param_array(sample_numpy_array):
    """Sample FixedParam with numpy array."""
    return FixedParam(value=sample_numpy_array)  # Keep for explicit FixedParam testing


@pytest.fixture
def sample_fixed_param_particle(sample_particle):
    """Sample FixedParam with particle."""
    return FixedParam(value=sample_particle)  # Keep for explicit FixedParam testing


@pytest.fixture
def rho_parameters():
    """Standard rho(770) meson parameters."""
    return {
        'pole_mass': 0.775,
        'width': 0.15,
        'r': 1.0,
        'L': 1
    }


@pytest.fixture
def f0_980_parameters():
    """Standard f0(980) meson parameters for Flatte."""
    return {
        'mass': 0.98,
        'g_pipi': 0.2,
        'g_kk': 0.8
    }
