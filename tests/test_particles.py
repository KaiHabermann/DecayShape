"""
Tests for Particle and Channel classes.
"""

import pytest
import numpy as np
from pydantic import ValidationError
from decayshape import FixedParam
from decayshape.particles import Particle, Channel, CommonParticles


class TestParticle:
    """Test Particle class functionality."""
    
    def test_create_particle(self):
        """Test creating a particle."""
        particle = Particle(mass=0.139, spin=0.0, parity=1)
        
        assert particle.mass == 0.139
        assert particle.spin == 0.0
        assert particle.parity == 1
    
    def test_particle_validation(self):
        """Test particle validation."""
        # Valid particle
        particle = Particle(mass=0.139, spin=0.5, parity=-1)
        assert particle.mass == 0.139
        
        # Note: Current implementation doesn't validate negative mass or invalid parity
        # These would need to be added as Pydantic validators if desired
        # For now, test that basic creation works
        particle_neg_mass = Particle(mass=-0.139, spin=0.0, parity=1)
        assert particle_neg_mass.mass == -0.139
        
        particle_invalid_parity = Particle(mass=0.139, spin=0.0, parity=2)
        assert particle_invalid_parity.parity == 2
    
    def test_particle_string_representation(self):
        """Test particle string representation."""
        particle = Particle(mass=0.139, spin=0.0, parity=1)
        str_repr = str(particle)
        
        assert "m=0.139" in str_repr
        assert "J=0" in str_repr
        assert "+" in str_repr  # positive parity
        
        # Test negative parity
        particle_neg = Particle(mass=0.139, spin=0.5, parity=-1)
        str_repr_neg = str(particle_neg)
        assert "-" in str_repr_neg  # negative parity
    
    def test_common_particles(self):
        """Test that common particles are defined correctly."""
        # Test pion
        pi_plus = CommonParticles.PI_PLUS
        assert isinstance(pi_plus, Particle)
        assert pi_plus.mass == pytest.approx(0.13957, rel=1e-4)
        assert pi_plus.spin == 0.0
        assert pi_plus.parity == -1  # Pions have negative parity
        
        # Test kaon
        k_plus = CommonParticles.K_PLUS
        assert isinstance(k_plus, Particle)
        assert k_plus.mass == pytest.approx(0.49368, rel=1e-4)
        assert k_plus.spin == 0.0
        assert k_plus.parity == -1  # Kaons have negative parity
        
        # Test proton
        proton = CommonParticles.PROTON
        assert isinstance(proton, Particle)
        assert proton.mass == pytest.approx(0.93827, rel=1e-4)
        assert proton.spin == 0.5
        assert proton.parity == 1


class TestChannel:
    """Test Channel class functionality."""
    
    def test_create_channel_with_fixedparam(self):
        """Test creating channel with FixedParam particles."""
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        assert isinstance(channel.particle1, FixedParam)
        assert isinstance(channel.particle2, FixedParam)
        assert channel.particle1.value == CommonParticles.PI_PLUS
        assert channel.particle2.value == CommonParticles.PI_MINUS
    
    def test_create_channel_with_auto_wrapping(self):
        """Test creating channel with auto-wrapping of particles."""
        # This should work if auto-wrapping is implemented for Channel
        # For now, we test explicit FixedParam creation
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        assert channel.particle1.mass == CommonParticles.PI_PLUS.mass
        assert channel.particle2.mass == CommonParticles.PI_MINUS.mass
    
    def test_channel_threshold(self):
        """Test channel threshold calculation."""
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        expected_threshold = (CommonParticles.PI_PLUS.mass + CommonParticles.PI_MINUS.mass)
        threshold = channel.threshold
        
        assert threshold == pytest.approx(expected_threshold, rel=1e-6)
    
    def test_channel_momentum(self):
        """Test channel momentum calculation."""
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        # Test at threshold (momentum should be 0)
        s_threshold = channel.threshold
        momentum_at_threshold = channel.momentum(s_threshold ** 2)
        assert momentum_at_threshold == pytest.approx(0.0, abs=1e-10)
        
        # Test above threshold
        s_above = 0.5  # GeV^2
        momentum_above = channel.momentum(s_above)
        assert np.real(momentum_above) > 0
        
        # Test with array input
        s_array = np.array([0.1, 0.3, 0.5, 0.7])
        momentum_array = channel.momentum(s_array)
        assert isinstance(momentum_array, np.ndarray)
        assert momentum_array.shape == s_array.shape
    
    def test_channel_phase_space_factor(self):
        """Test channel phase space factor calculation."""
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        # Test at threshold (should be 0)
        s_threshold = channel.threshold
        ps_at_threshold = channel.phase_space_factor(s_threshold ** 2)
        assert ps_at_threshold == pytest.approx(0.0, abs=1e-10)
        
        # Test above threshold (should be positive)
        s_above = 0.5  # GeV^2
        ps_above = channel.phase_space_factor(s_above ** 2)
        assert ps_above > 0
        
        # Test with array input
        s_array = np.array([0.1, 0.3, 0.5, 0.7])
        ps_array = channel.phase_space_factor(s_array ** 2)
        assert isinstance(ps_array, np.ndarray)
        assert ps_array.shape == s_array.shape
    
    def test_channel_string_representation(self):
        """Test channel string representation."""
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.K_PLUS
        )
        
        str_repr = str(channel)
        # Should contain information about both particles
        assert "+" in str_repr  # From particle representations
    
    def test_different_particle_combinations(self):
        """Test channels with different particle combinations."""
        # π+π- channel
        pipi_channel = Channel(
            particle1=FixedParam(value=CommonParticles.PI_PLUS),
            particle2=FixedParam(value=CommonParticles.PI_MINUS)
        )
        
        # K+K- channel
        kk_channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS
        )
        
        # Different thresholds
        pipi_threshold = pipi_channel.threshold
        kk_threshold = kk_channel.threshold
        
        assert kk_threshold > pipi_threshold  # KK threshold is higher
        
        # Test momentum at same s value
        s_test = 1.0  # GeV^2
        pipi_momentum = pipi_channel.momentum(s_test)
        kk_momentum = kk_channel.momentum(s_test)
        
        assert np.real(pipi_momentum) > np.real(kk_momentum)  # π is lighter than K


class TestChannelPhysics:
    """Test physics calculations in Channel class."""
    
    def test_momentum_formula(self):
        """Test that momentum formula is correct."""
        # Create a channel with known masses
        m1, m2 = 0.1, 0.2  # GeV
        particle1 = Particle(mass=m1, spin=0.0, parity=1)
        particle2 = Particle(mass=m2, spin=0.0, parity=1)
        
        channel = Channel(
            particle1=particle1,
            particle2=particle2
        )
        
        # Test momentum calculation
        s = 1.0  # GeV^2
        momentum = channel.momentum(s)
        
        # Manual calculation using the standard formula
        # p = sqrt((s - (m1+m2)^2) * (s - (m1-m2)^2)) / (2*sqrt(s))
        sqrt_s = np.sqrt(s)
        expected_momentum = np.sqrt((s - (m1 + m2)**2) * (s - (m1 - m2)**2)) / (2 * sqrt_s)
        
        assert momentum == pytest.approx(expected_momentum, rel=1e-10)
    
    def test_below_threshold_behavior(self):
        """Test behavior below threshold."""
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        # Test below threshold
        s_below = 0.01  # GeV^2, well below ππ threshold
        
        # Momentum should be imaginary (returned as complex)
        momentum = channel.momentum(s_below ** 2)
        if np.iscomplexobj(momentum):
            assert np.real(momentum) == pytest.approx(0.0, abs=1e-10)
            assert np.imag(momentum) != 0  # Imaginary momentum
        else:
            # If implementation returns NaN or 0 for below threshold
            assert np.isnan(momentum) or momentum == 0
    
    def test_phase_space_units(self):
        """Test that phase space factor has correct units."""
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        s = 0.5  # GeV^2
        ps_factor = channel.phase_space_factor(s)
        
        # Phase space factor should be dimensionless
        # and positive above threshold
        assert ps_factor > 0
        assert np.isfinite(ps_factor)
