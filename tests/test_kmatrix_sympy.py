"""
Numeric equivalence test between the sympy K-matrix reference implementation
(decayshape._lineshapes_sympy.kmatrix_amplitude) and the production numeric
implementation (decayshape.kmatrix_advanced.KMatrixAdvanced), for up to 3 poles
and 3 channels.

This only exercises the K-matrix construction itself: KMatrixAdvanced's
background / production_background terms are not implemented in the sympy
version, so they are left unset (default) on both sides.
"""

import itertools

import numpy as np
import pytest
import sympy as sp

from decayshape import set_backend
from decayshape._lineshapes_sympy import kmatrix_amplitude
from decayshape.kmatrix_advanced import KMatrixAdvanced
from decayshape.particles import Channel, CommonParticles

set_backend("numpy")

# Three channels with distinct thresholds, mirroring benchmark/kmatrix_performance.py.
_CHANNEL_PARTICLES = [
    (CommonParticles.PI_PLUS, CommonParticles.PI_MINUS),  # pipi, threshold ~0.279
    (CommonParticles.PI_PLUS, CommonParticles.K_PLUS),  # piK, threshold ~0.633
    (CommonParticles.K_PLUS, CommonParticles.K_MINUS),  # KK, threshold ~0.987
]

# s-grid spanning below/between/above all three thresholds above.
_S_VALUES = np.array([0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.5])


def _build_channels_and_masses(n_channels):
    channels = [Channel(particle1=p1, particle2=p2) for p1, p2 in _CHANNEL_PARTICLES[:n_channels]]
    channel_masses = [(p1.mass, p2.mass) for p1, p2 in _CHANNEL_PARTICLES[:n_channels]]
    return channels, channel_masses


def _make_parameters(n_poles, n_channels):
    """Deterministic, distinct pole masses / couplings so no coupling accidentally cancels."""
    pole_masses = [0.5 + 0.4 * i for i in range(n_poles)]
    production_couplings = [0.7 + 0.3 * i for i in range(n_poles)]
    decay_couplings = [
        0.4 + 0.15 * pole_idx + 0.1 * channel_idx for pole_idx in range(n_poles) for channel_idx in range(n_channels)
    ]
    return pole_masses, production_couplings, decay_couplings


def _evaluate_sympy(pole_masses, production_couplings, decay_couplings, channel_masses, r, output_channel, angular_momentum):
    # Build (and invert) the K-matrix separately for each concrete s value rather than
    # inverting once with s left as a free symbol: with a Piecewise phase-space factor in
    # every matrix entry, a fully symbolic n x n matrix inverse (n > 1) is combinatorially
    # expensive to simplify. Plugging in a plain number for s keeps every step (including
    # the matrix inverse) numeric and fast, while kmatrix_amplitude itself stays general
    # enough to be called with a genuine free symbol elsewhere (e.g. for analytic work in
    # the coupling constants at fixed s).
    result = np.empty(len(_S_VALUES), dtype=complex)
    for idx, s_val in enumerate(_S_VALUES):
        expr = kmatrix_amplitude(
            sp.Float(s_val),
            pole_masses=pole_masses,
            production_couplings=production_couplings,
            decay_couplings=decay_couplings,
            channel_masses=channel_masses,
            r=r,
            output_channel=output_channel,
            angular_momentum=angular_momentum,
        )
        result[idx] = complex(expr.evalf())
    return result


@pytest.mark.parametrize("n_poles,n_channels", list(itertools.product([1, 2, 3], repeat=2)))
def test_kmatrix_sympy_matches_numeric(n_poles, n_channels):
    """Sympy and numeric K-matrix amplitudes must agree for every (n_poles, n_channels) up to 3x3."""
    channels, channel_masses = _build_channels_and_masses(n_channels)
    pole_masses, production_couplings, decay_couplings = _make_parameters(n_poles, n_channels)
    r = 1.0
    angular_momentum = 2  # doubled convention -> L = 1
    output_channel = n_channels - 1  # exercise a non-trivial output index when possible

    kmat = KMatrixAdvanced(
        s=_S_VALUES,
        channels=channels,
        pole_masses=pole_masses,
        production_couplings=production_couplings,
        decay_couplings=decay_couplings,
        r=r,
        output_channel=output_channel,
    )
    numeric_result = np.asarray(kmat(angular_momentum, 0))

    sympy_result = _evaluate_sympy(
        pole_masses, production_couplings, decay_couplings, channel_masses, r, output_channel, angular_momentum
    )

    np.testing.assert_allclose(numeric_result, sympy_result, rtol=1e-8, atol=1e-10)


def test_kmatrix_sympy_matches_numeric_l0():
    """Also check L=0 (angular_momentum=0), where the barrier/form factors trivialize to 1."""
    n_poles, n_channels = 2, 2
    channels, channel_masses = _build_channels_and_masses(n_channels)
    pole_masses, production_couplings, decay_couplings = _make_parameters(n_poles, n_channels)
    r = 1.0
    angular_momentum = 0
    output_channel = 0

    kmat = KMatrixAdvanced(
        s=_S_VALUES,
        channels=channels,
        pole_masses=pole_masses,
        production_couplings=production_couplings,
        decay_couplings=decay_couplings,
        r=r,
        output_channel=output_channel,
    )
    numeric_result = np.asarray(kmat(angular_momentum, 0))

    sympy_result = _evaluate_sympy(
        pole_masses, production_couplings, decay_couplings, channel_masses, r, output_channel, angular_momentum
    )

    np.testing.assert_allclose(numeric_result, sympy_result, rtol=1e-8, atol=1e-10)
