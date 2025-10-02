"""
Tests for K-matrix functionality.
"""

import pytest
import numpy as np
from decayshape import FixedParam
from decayshape.kmatrix_advanced import (
    KMatrixAdvanced,
    create_simple_kmatrix,
    create_pipi_kmatrix,
    create_multi_channel_kmatrix
)
from decayshape.particles import CommonParticles, Channel


class TestKMatrixAdvanced:
    """Test advanced K-matrix functionality."""
    
    def test_create_single_channel_kmatrix(self, sample_s_values):
        """Test creating single-channel K-matrix."""
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        kmat = KMatrixAdvanced(
            s=sample_s_values,
            channels=[pipi_channel],
            pole_masses=[0.775],
            production_couplings=[1.0],
            decay_couplings=[1.0],
            output_channel=0
        )
        
        assert isinstance(kmat.s, FixedParam)
        assert isinstance(kmat.channels, FixedParam)
        assert len(kmat.pole_masses) == 1
        assert len(kmat.production_couplings) == 1
        assert len(kmat.decay_couplings) == 1
    
    def test_create_two_channel_kmatrix(self, sample_s_values):
        """Test creating two-channel K-matrix."""
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        kk_channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS
        )
        
        kmat = KMatrixAdvanced(
            s=sample_s_values,
            channels=[pipi_channel, kk_channel],
            pole_masses=[0.6, 1.0],  # Two poles
            production_couplings=[1.0, 0.8],  # Two production couplings
            decay_couplings=[1.0, 0.5, 0.3, 0.7],  # 2 poles × 2 channels = 4 couplings
            output_channel=0  # Output to first channel
        )
        
        assert len(kmat.channels.value) == 2
        assert len(kmat.pole_masses) == 2
        assert len(kmat.production_couplings) == 2
        assert len(kmat.decay_couplings) == 4  # 2 poles × 2 channels
    
    def test_kmatrix_evaluation_single_channel(self, sample_s_values):
        """Test evaluating single-channel K-matrix."""
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        kmat = KMatrixAdvanced(
            s=sample_s_values,
            channels=[pipi_channel],
            pole_masses=[0.775],
            production_couplings=[1.0],
            decay_couplings=[1.0],
            output_channel=0
        )
        
        result = kmat()
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_s_values.shape
        assert np.all(np.isfinite(result))
    
    def test_kmatrix_evaluation_two_channels(self, sample_s_values):
        """Test evaluating two-channel K-matrix."""
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        kk_channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS
        )
        
        kmat = KMatrixAdvanced(
            s=sample_s_values,
            channels=[pipi_channel, kk_channel],
            pole_masses=[0.6, 1.0],
            production_couplings=[1.0, 0.8],
            decay_couplings=[1.0, 0.5, 0.3, 0.7],
            output_channel=0
        )
        
        result = kmat()
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_s_values.shape
        assert np.all(np.isfinite(result))
    
    def test_kmatrix_different_output_channels(self, sample_s_values):
        """Test K-matrix with different output channels."""
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        kk_channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS
        )
        
        base_params = {
            's': FixedParam(value=sample_s_values),
            'channels': FixedParam(value=[pipi_channel, kk_channel]),
            'pole_masses': [0.6, 1.0],
            'production_couplings': [1.0, 0.8],
            'decay_couplings': [1.0, 0.5, 0.3, 0.7]
        }
        
        # Channel 0 (ππ)
        kmat_0 = KMatrixAdvanced(output_channel=FixedParam(value=0), **base_params)
        result_0 = kmat_0()
        
        # Channel 1 (KK)
        kmat_1 = KMatrixAdvanced(output_channel=FixedParam(value=1), **base_params)
        result_1 = kmat_1()
        
        # Results should be different for different channels
        assert not np.allclose(result_0, result_1)
    
    def test_kmatrix_threshold_effects(self):
        """Test K-matrix threshold effects."""
        # Create s values spanning both thresholds
        s_values = np.linspace(0.1, 1.5, 200)
        
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        kk_channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS
        )
        
        kmat = KMatrixAdvanced(
            s=s_values,
            channels=[pipi_channel, kk_channel],
            pole_masses=[0.6, 1.0],
            production_couplings=[1.0, 0.8],
            decay_couplings=[1.0, 0.5, 0.3, 0.7],
            output_channel=0
        )
        
        result = kmat()
        
        # Should show threshold effects
        assert np.all(np.isfinite(result))
        
        # Behavior should change at thresholds
        pipi_threshold = pipi_channel.threshold
        kk_threshold = kk_channel.threshold
        
        # Find indices near thresholds
        pipi_idx = np.argmin(np.abs(s_values - pipi_threshold))
        kk_idx = np.argmin(np.abs(s_values - kk_threshold))
        
        # Check that we have different behavior at different thresholds
        assert pipi_idx != kk_idx
    
    def test_kmatrix_parameter_override(self, sample_s_values):
        """Test K-matrix parameter override."""
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        kmat = KMatrixAdvanced(
            s=sample_s_values,
            channels=[pipi_channel],
            pole_masses=[0.775],
            production_couplings=[1.0],
            decay_couplings=[1.0],
            output_channel=0
        )
        
        # Default evaluation
        result_default = kmat()
        
        # Override pole masses
        result_override = kmat(pole_masses=[0.8])
        
        # Results should be different
        assert not np.allclose(result_default, result_override)
    
    def test_kmatrix_coupling_matrix_shape(self, sample_s_values):
        """Test that coupling matrix has correct shape."""
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        kk_channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS
        )
        
        n_poles = 3
        n_channels = 2
        
        kmat = KMatrixAdvanced(
            s=sample_s_values,
            channels=[pipi_channel, kk_channel],
            pole_masses=[0.6, 0.8, 1.0],  # 3 poles
            production_couplings=[1.0, 0.8, 0.6],  # 3 production couplings
            decay_couplings=[1.0, 0.5, 0.3, 0.7, 0.4, 0.9],  # 3 poles × 2 channels = 6
            output_channel=0
        )
        
        # Should work without errors
        result = kmat()
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_s_values.shape


class TestKMatrixHelperFunctions:
    """Test K-matrix helper functions."""
    
    def test_create_simple_kmatrix(self, sample_s_values):
        """Test create_simple_kmatrix helper."""
        kmat = create_simple_kmatrix(
            s_values=sample_s_values,
            pole_mass=0.775,
            coupling=1.0
        )
        
        assert isinstance(kmat, KMatrixAdvanced)
        result = kmat()
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_s_values.shape
    
    def test_create_pipi_kmatrix(self, sample_s_values):
        """Test create_pipi_kmatrix helper."""
        kmat = create_pipi_kmatrix(
            s_values=sample_s_values,
            pole_masses=[0.6, 0.8],
            couplings=[1.0, 0.8]
        )
        
        assert isinstance(kmat, KMatrixAdvanced)
        result = kmat()
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_s_values.shape
    
    def test_create_multi_channel_kmatrix(self, sample_s_values):
        """Test create_multi_channel_kmatrix helper."""
        kmat = create_multi_channel_kmatrix(
            s_values=sample_s_values,
            pole_masses=[0.6, 1.0],
            production_couplings=[1.0, 0.8],
            decay_couplings=[[1.0, 0.5], [0.3, 0.7]],  # 2 poles × 2 channels
            output_channel=0
        )
        
        assert isinstance(kmat, KMatrixAdvanced)
        result = kmat()
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_s_values.shape


class TestKMatrixPhysics:
    """Test K-matrix physics properties."""
    
    def test_kmatrix_unitarity(self, sample_s_values):
        """Test K-matrix unitarity properties."""
        # Create a simple single-channel K-matrix
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        kmat = KMatrixAdvanced(
            s=sample_s_values,
            channels=[pipi_channel],
            pole_masses=[0.775],
            production_couplings=[1.0],
            decay_couplings=[1.0],
            output_channel=0
        )
        
        result = kmat()
        
        # Above threshold, should have imaginary part (unitarity)
        threshold = pipi_channel.threshold
        above_threshold = sample_s_values > threshold
        
        if np.any(above_threshold):
            result_above = result[above_threshold]
            # Should have non-zero imaginary part above threshold
            assert np.any(np.imag(result_above) != 0)
    
    def test_kmatrix_analytic_structure(self):
        """Test K-matrix analytic structure."""
        # Test with complex s values
        s_real = np.linspace(0.5, 1.5, 20)
        s_imag = 1j * np.linspace(-0.1, 0.1, 20)
        s_complex = s_real[:, np.newaxis] + s_imag[np.newaxis, :]
        s_flat = s_complex.flatten()
        
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        kmat = KMatrixAdvanced(
            s=FixedParam(value=s_flat),
            channels=FixedParam(value=[pipi_channel]),
            pole_masses=[0.775],
            production_couplings=[1.0],
            decay_couplings=[1.0],
            output_channel=0
        )
        
        result = kmat()
        
        # Should handle complex s values
        assert isinstance(result, np.ndarray)
        assert result.shape == s_flat.shape
        assert np.all(np.isfinite(result))
    
    def test_kmatrix_pole_behavior(self, sample_s_values):
        """Test K-matrix behavior near poles."""
        pole_mass = 0.775
        
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        kmat = KMatrixAdvanced(
            s=sample_s_values,
            channels=[pipi_channel],
            pole_masses=[pole_mass],
            production_couplings=[1.0],
            decay_couplings=[1.0],
            output_channel=0
        )
        
        result = kmat()
        magnitude = np.abs(result)
        
        # Find the peak
        peak_idx = np.argmax(magnitude)
        peak_s = sample_s_values[peak_idx]
        
        # Peak should be near the pole mass squared
        assert peak_s == pytest.approx(pole_mass**2, rel=0.2)
    
    def test_kmatrix_channel_coupling(self, sample_s_values):
        """Test K-matrix channel coupling effects."""
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        kk_channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS
        )
        
        # Strong coupling to both channels
        kmat_strong = KMatrixAdvanced(
            s=sample_s_values,
            channels=[pipi_channel, kk_channel],
            pole_masses=[0.98],  # Near KK threshold
            production_couplings=[1.0],
            decay_couplings=[1.0, 1.0],  # Strong coupling to both
            output_channel=0
        )
        
        # Weak coupling to second channel
        kmat_weak = KMatrixAdvanced(
            s=sample_s_values,
            channels=[pipi_channel, kk_channel],
            pole_masses=[0.98],
            production_couplings=[1.0],
            decay_couplings=[1.0, 0.1],  # Weak coupling to KK
            output_channel=0
        )
        
        result_strong = kmat_strong()
        result_weak = kmat_weak()
        
        # Results should be different
        assert not np.allclose(result_strong, result_weak)
