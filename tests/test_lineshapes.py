"""
Tests for lineshape classes.
"""

import numpy as np
import pytest

from decayshape import FixedParam, RelativisticBreitWigner
from decayshape.lineshapes import Flatte
from decayshape.particles import Channel, CommonParticles


class TestRelativisticBreitWigner:
    """Test RelativisticBreitWigner lineshape."""

    def test_create_breit_wigner(self, sample_s_values, rho_parameters):
        """Test creating a Breit-Wigner lineshape."""
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)

        # Remove L from rho_parameters since it's no longer a field
        params = {k: v for k, v in rho_parameters.items() if k != "L"}
        bw = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, **params)

        assert isinstance(bw.s, FixedParam)
        assert bw.pole_mass == rho_parameters["pole_mass"]
        assert bw.width == rho_parameters["width"]
        assert bw.r == rho_parameters["r"]

    def test_breit_wigner_evaluation(self, sample_s_values, rho_parameters):
        """Test evaluating Breit-Wigner lineshape."""
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)

        # Remove L from rho_parameters since it's no longer a field
        params = {k: v for k, v in rho_parameters.items() if k != "L"}
        bw = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, **params)

        # Now need to provide spin and angular_momentum as positional arguments
        result = bw(1, 2)  # spin=1 (1/2), angular_momentum=2 (L=1)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_s_values.shape
        assert np.all(np.isfinite(result))  # Should not have NaN or inf

    def test_breit_wigner_parameter_override(self, sample_s_values, rho_parameters):
        """Test parameter override in Breit-Wigner."""
        # Remove L from rho_parameters since it's no longer a field
        params = {k: v for k, v in rho_parameters.items() if k != "L"}
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, **params)

        # Evaluate with default parameters
        result_default = bw(1, 2)  # spin=1 (1/2), angular_momentum=2 (L=1)

        # Evaluate with overridden width
        new_width = 0.2
        result_override = bw(1, 2, width=new_width)

        # Results should be different
        assert not np.allclose(result_default, result_override)

        # Test positional arguments
        result_positional = bw(1, 2, rho_parameters["pole_mass"], new_width)
        assert np.allclose(result_override, result_positional)

    def test_breit_wigner_parameter_order(self, sample_s_values):
        """Test parameter order property."""
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, pole_mass=0.775, width=0.15)

        expected_order = ["pole_mass", "width", "r", "q0"]
        assert bw.parameter_order == expected_order

    def test_breit_wigner_resonance_behavior(self):
        """Test that Breit-Wigner shows resonance behavior."""
        # Create s values around the resonance
        pole_mass = 0.775
        s_values = np.linspace(0.4, 1.2, 100)

        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw = RelativisticBreitWigner(s=s_values, channel=pipi_channel, pole_mass=pole_mass, width=0.15, r=1.0)

        result = bw(1, 2)  # spin=1 (1/2), angular_momentum=2 (L=1)
        magnitude = np.abs(result)

        # Find the peak
        peak_idx = np.argmax(magnitude)
        peak_s = s_values[peak_idx]

        # Peak should be near the pole mass squared
        assert peak_s == pytest.approx(pole_mass**2, rel=0.1)

    def test_breit_wigner_width_effect(self, sample_s_values):
        """Test effect of width parameter."""
        base_params = {"pole_mass": 0.775, "r": 1.0}

        # Narrow resonance
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw_narrow = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, width=0.05, **base_params)
        result_narrow = bw_narrow(1, 2)  # spin=1 (1/2), angular_momentum=2 (L=1)

        # Wide resonance
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw_wide = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, width=0.3, **base_params)
        result_wide = bw_wide(1, 2)  # spin=1 (1/2), angular_momentum=2 (L=1)

        # Peak of narrow resonance should be higher
        assert np.max(np.abs(result_narrow)) > np.max(np.abs(result_wide))

    def test_breit_wigner_angular_momentum(self, sample_s_values):
        """Test effect of angular momentum."""
        base_params = {"pole_mass": 0.775, "width": 0.15, "r": 1.0}

        # S-wave (angular_momentum=0)
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw_s = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, **base_params)
        result_s = bw_s(0, 1)  # angular_momentum=0 (L=0), spin=1 (1/2)

        # P-wave (angular_momentum=2)
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw_p = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, **base_params)
        result_p = bw_p(2, 1)  # angular_momentum=2 (L=1), spin=1 (1/2)

        # Results should be different due to different form factors
        assert not np.allclose(result_s, result_p)

    def test_breit_wigner_q0_calculation(self, sample_s_values):
        """Test automatic q0 calculation."""
        pole_mass = 0.775
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, pole_mass=pole_mass, width=0.15)

        # q0 should be calculated using the two-body breakup momentum
        # Manual calculation for verification
        m1 = CommonParticles.PI_PLUS.mass
        m2 = CommonParticles.PI_MINUS.mass
        s_pole = pole_mass**2
        expected_q0 = np.sqrt((s_pole - (m1 + m2) ** 2) * (s_pole - (m1 - m2) ** 2)) / (2 * np.sqrt(s_pole))
        assert bw.q0 == pytest.approx(expected_q0, rel=1e-10)

    def test_breit_wigner_serialization(self, sample_s_values, rho_parameters):
        """Test Breit-Wigner serialization."""
        # Remove L from rho_parameters since it's no longer a field
        params = {k: v for k, v in rho_parameters.items() if k != "L"}
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, **params)

        # Test model dump
        data = bw.model_dump()
        assert "pole_mass" in data
        assert "width" in data

        # Test model validation
        bw_restored = RelativisticBreitWigner.model_validate(data)
        assert bw_restored.pole_mass == bw.pole_mass
        assert bw_restored.width == bw.width


class TestFlatte:
    """Test Flatte lineshape."""

    def test_create_flatte(self, sample_s_values, f0_980_parameters):
        """Test creating a Flatte lineshape."""
        # Create channels
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        kk_channel = Channel(particle1=CommonParticles.K_PLUS, particle2=CommonParticles.K_MINUS)

        flatte = Flatte(
            s=sample_s_values,
            channel1=pipi_channel,
            channel2=kk_channel,
            pole_mass=f0_980_parameters["mass"],
            width1=f0_980_parameters["g_pipi"],
            width2=f0_980_parameters["g_kk"],
            r1=1.0,
            r2=1.0,
        )

        assert isinstance(flatte.s, FixedParam)
        assert flatte.pole_mass == f0_980_parameters["mass"]
        assert flatte.width1 == f0_980_parameters["g_pipi"]
        assert flatte.width2 == f0_980_parameters["g_kk"]

    def test_flatte_evaluation(self, sample_s_values, f0_980_parameters):
        """Test evaluating Flatte lineshape."""
        # Create channels
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        kk_channel = Channel(particle1=CommonParticles.K_PLUS, particle2=CommonParticles.K_MINUS)

        flatte = Flatte(
            s=sample_s_values,
            channel1=pipi_channel,
            channel2=kk_channel,
            pole_mass=f0_980_parameters["mass"],
            width1=f0_980_parameters["g_pipi"],
            width2=f0_980_parameters["g_kk"],
            r1=1.0,
            r2=1.0,
        )

        result = flatte(1, 2)  # spin=1 (1/2), angular_momentum=2 (L=1)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_s_values.shape
        assert np.all(np.isfinite(result))

    def test_flatte_threshold_effects(self):
        """Test Flatte threshold effects."""
        # Create s values spanning both thresholds
        s_values = np.linspace(0.1, 1.5, 200)

        # Create channels
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        kk_channel = Channel(particle1=CommonParticles.K_PLUS, particle2=CommonParticles.K_MINUS)

        flatte = Flatte(
            s=s_values,
            channel1=pipi_channel,
            channel2=kk_channel,
            pole_mass=0.98,  # Near KK threshold
            width1=0.2,
            width2=0.8,
            r1=1.0,
            r2=1.0,
        )

        result = flatte(1, 2)  # spin=1 (1/2), angular_momentum=2 (L=1)

        # Should show threshold effects
        assert np.all(np.isfinite(result))

        # Imaginary part should change at thresholds
        imag_part = np.imag(result)
        assert np.any(imag_part != 0)  # Should have imaginary part above thresholds


# Note: Basic KMatrix class is not implemented, only KMatrixAdvanced
# Tests for KMatrixAdvanced are in test_kmatrix.py


class TestLineshapeBase:
    """Test base lineshape functionality."""

    def test_fixed_parameters(self, sample_s_values, rho_parameters):
        """Test fixed parameter extraction."""
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        # Remove L from rho_parameters since it's no longer a field
        params = {k: v for k, v in rho_parameters.items() if k != "L"}
        bw = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, **params)

        fixed_params = bw.get_fixed_parameters()
        assert "s" in fixed_params
        np.testing.assert_array_equal(fixed_params["s"], sample_s_values)

    def test_optimization_parameters(self, sample_s_values, rho_parameters):
        """Test optimization parameter extraction."""
        # Remove L from rho_parameters since it's no longer a field
        params = {k: v for k, v in rho_parameters.items() if k != "L"}
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, **params)

        opt_params = bw.get_optimization_parameters()
        assert "pole_mass" in opt_params
        assert "width" in opt_params
        assert "r" in opt_params
        assert "q0" in opt_params
        assert "s" not in opt_params  # s is fixed

    def test_parameter_override_validation(self, sample_s_values):
        """Test parameter override validation."""
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, pole_mass=0.775, width=0.15)

        # Valid override
        result = bw(1, 2, width=0.2)  # spin=1 (1/2), angular_momentum=2 (L=1), width=0.2
        assert isinstance(result, np.ndarray)

        # Test positional and keyword conflict
        with pytest.raises(ValueError, match="provided both positionally and as keyword"):
            bw(1, 2, 0.775, pole_mass=0.8)  # pole_mass both positional and keyword

    def test_too_many_positional_args(self, sample_s_values):
        """Test error with too many positional arguments."""
        # Create a channel (rho -> pi+ pi-)
        pipi_channel = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
        bw = RelativisticBreitWigner(s=sample_s_values, channel=pipi_channel, pole_mass=0.775, width=0.15)

        # Too many positional arguments
        with pytest.raises(ValueError, match="Too many positional arguments"):
            bw(1, 2, 0.775, 0.15, 1.0, 0.4, 999)  # 7 args, but only 4 expected (after spin, angular_momentum)
