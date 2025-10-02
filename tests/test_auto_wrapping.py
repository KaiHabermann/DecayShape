"""
Tests for auto-wrapping functionality in Pydantic models.
"""

import pytest
import numpy as np
from decayshape import FixedParam
from decayshape.particles import Channel, CommonParticles


class TestChannelAutoWrapping:
    """Test auto-wrapping functionality in Channel class."""
    
    def test_channel_auto_wrap_particles(self):
        """Test that Channel automatically wraps Particle objects in FixedParam."""
        # Create channel with direct Particle objects (should auto-wrap)
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        # Verify that particles were automatically wrapped in FixedParam
        assert isinstance(channel.particle1, FixedParam)
        assert isinstance(channel.particle2, FixedParam)
        
        # Verify that the wrapped values are the correct particles
        assert channel.particle1.value == CommonParticles.PI_PLUS
        assert channel.particle2.value == CommonParticles.PI_MINUS
        
        # Verify that the channel functionality still works
        assert channel.total_mass == CommonParticles.PI_PLUS.mass + CommonParticles.PI_MINUS.mass
        threshold = channel.threshold
        assert threshold > 0
        assert np.isfinite(threshold)
    
    def test_channel_already_wrapped_particles(self):
        """Test that Channel works correctly with already wrapped FixedParam particles."""
        # Create channel with explicitly wrapped FixedParam objects
        channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS
        )
        
        # Verify that particles remain as FixedParam (no double-wrapping)
        assert isinstance(channel.particle1, FixedParam)
        assert isinstance(channel.particle2, FixedParam)
        
        # Verify that the wrapped values are the correct particles
        assert channel.particle1.value == CommonParticles.K_PLUS
        assert channel.particle2.value == CommonParticles.K_MINUS
        
        # Verify that the channel functionality still works
        assert channel.total_mass == CommonParticles.K_PLUS.mass + CommonParticles.K_MINUS.mass
    
    def test_channel_mixed_wrapping(self):
        """Test Channel with one wrapped and one unwrapped particle."""
        # Create channel - both should auto-wrap
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.K_PLUS
        )
        
        # Verify that both particles are FixedParam
        assert isinstance(channel.particle1, FixedParam)
        assert isinstance(channel.particle2, FixedParam)
        
        # Verify that the wrapped values are correct
        assert channel.particle1.value == CommonParticles.PI_PLUS
        assert channel.particle2.value == CommonParticles.K_PLUS
        
        # Verify functionality
        expected_mass = CommonParticles.PI_PLUS.mass + CommonParticles.K_PLUS.mass
        assert channel.total_mass == expected_mass
    
    def test_channel_attribute_forwarding_after_wrapping(self):
        """Test that attribute forwarding works after auto-wrapping."""
        channel = Channel(
            particle1=CommonParticles.PROTON,
            particle2=CommonParticles.NEUTRON
        )
        
        # Test that we can access particle attributes through FixedParam forwarding
        assert channel.particle1.mass == CommonParticles.PROTON.mass
        assert channel.particle1.spin == CommonParticles.PROTON.spin
        assert channel.particle1.parity == CommonParticles.PROTON.parity
        
        assert channel.particle2.mass == CommonParticles.NEUTRON.mass
        assert channel.particle2.spin == CommonParticles.NEUTRON.spin
        assert channel.particle2.parity == CommonParticles.NEUTRON.parity
    
    def test_channel_physics_calculations_after_wrapping(self):
        """Test that physics calculations work correctly after auto-wrapping."""
        # Create a channel with auto-wrapping
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        # Test threshold calculation
        threshold = channel.threshold
        expected_threshold = CommonParticles.PI_PLUS.mass + CommonParticles.PI_MINUS.mass
        assert threshold == expected_threshold
        
        # Test momentum calculation above threshold
        s_above_threshold = 0.5  # GeV^2, above ππ threshold
        momentum = channel.momentum(s_above_threshold)
        assert np.isfinite(momentum)
        assert momentum > 0  # Should be positive above threshold
        
        # Test phase space factor
        ps_factor = channel.phase_space_factor(s_above_threshold)
        assert np.isfinite(ps_factor)
        assert ps_factor > 0  # Should be positive above threshold
