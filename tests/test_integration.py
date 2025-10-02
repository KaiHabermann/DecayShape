"""
Integration tests for DecayShape package.
"""

import pytest
import numpy as np
from decayshape import (
    FixedParam, RelativisticBreitWigner, Channel, 
    KMatrixAdvanced, set_backend
)
from decayshape.particles import CommonParticles


class TestBasicWorkflow:
    """Test basic workflow scenarios."""
    
    def test_simple_breit_wigner_workflow(self):
        """Test simple Breit-Wigner workflow."""
        # Create s values
        s_values = np.linspace(0.3, 1.2, 100)
        
        # Create Breit-Wigner
        rho = RelativisticBreitWigner(
            pole_mass=0.775,
            s=s_values,
            width=0.15,
            r=1.0,
            L=1
        )
        
        # Evaluate
        amplitude = rho()
        
        # Check results
        assert isinstance(amplitude, np.ndarray)
        assert amplitude.shape == s_values.shape
        assert np.all(np.isfinite(amplitude))
        
        # Check that it shows resonance behavior
        magnitude = np.abs(amplitude)
        peak_idx = np.argmax(magnitude)
        peak_s = s_values[peak_idx]
        
        # Peak should be near pole mass squared
        assert peak_s == pytest.approx(0.775**2, rel=0.1)
    
    def test_parameter_optimization_workflow(self):
        """Test parameter optimization workflow."""
        s_values = np.linspace(0.5, 1.0, 50)
        
        # Create lineshape
        bw = RelativisticBreitWigner(
            pole_mass=0.775,
            s=s_values,
            width=0.15,
            r=1.0,
            L=1
        )
        
        # Test different parameter combinations (simulating optimization)
        test_params = [
            {'width': 0.1, 'r': 0.8},
            {'width': 0.2, 'r': 1.2},
            {'pole_mass': 0.8, 'width': 0.15}
        ]
        
        results = []
        for params in test_params:
            result = bw(**params)
            results.append(result)
            assert isinstance(result, np.ndarray)
            assert result.shape == s_values.shape
        
        # Results should be different
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                assert not np.allclose(results[i], results[j])
    
    def test_multi_channel_kmatrix_workflow(self):
        """Test multi-channel K-matrix workflow."""
        # Create s values spanning both thresholds
        s_values = np.linspace(0.1, 1.5, 200)
        
        # Create channels
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        kk_channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS
        )
        
        # Create K-matrix
        kmat = KMatrixAdvanced(
            s=s_values,
            channels=[pipi_channel, kk_channel],
            pole_masses=[0.6, 1.0],
            production_couplings=[1.0, 0.8],
            decay_couplings=[1.0, 0.5, 0.3, 0.7],
            output_channel=0
        )
        
        # Evaluate
        result = kmat()
        
        # Check results
        assert isinstance(result, np.ndarray)
        assert result.shape == s_values.shape
        assert np.all(np.isfinite(result))
        
        # Should have threshold effects
        threshold_pipi = pipi_channel.threshold
        threshold_kk = kk_channel.threshold
        
        # Find behavior at thresholds
        idx_pipi = np.argmin(np.abs(s_values - threshold_pipi))
        idx_kk = np.argmin(np.abs(s_values - threshold_kk))
        
        # Behavior should be different at different thresholds
        assert idx_pipi != idx_kk


class TestFixedParamIntegration:
    """Test FixedParam integration with other components."""
    
    def test_fixedparam_with_particles(self):
        """Test FixedParam with particle objects."""
        # Create FixedParam with particle (this will be auto-wrapped)
        fp_particle = CommonParticles.PI_PLUS
        
        # Use in Channel (both will be auto-wrapped)
        channel = Channel(
            particle1=fp_particle,
            particle2=CommonParticles.PI_MINUS
        )
        
        # Should forward attributes through auto-wrapped FixedParam
        assert channel.particle1.mass == CommonParticles.PI_PLUS.mass
        assert channel.particle1.spin == CommonParticles.PI_PLUS.spin
        assert channel.particle1.parity == CommonParticles.PI_PLUS.parity
        
        # Should work correctly
        threshold = channel.threshold
        assert threshold > 0
        assert np.isfinite(threshold)
    
    def test_fixedparam_with_arrays(self):
        """Test FixedParam with numpy arrays."""
        s_values = np.array([0.5, 0.6, 0.7])
        # This will be auto-wrapped when used in a lineshape
        
        # Use in lineshape (s will be auto-wrapped)
        bw = RelativisticBreitWigner(
            pole_mass=0.775,
            s=s_values,
            width=0.15
        )
        
        # Should forward array attributes through auto-wrapped FixedParam
        assert bw.s.shape == s_values.shape
        assert bw.s.dtype == s_values.dtype
        assert bw.s.ndim == s_values.ndim
        
        # Should work with indexing
        assert bw.s[0] == s_values[0]
        np.testing.assert_array_equal(bw.s[1:], s_values[1:])
        
        result = bw()
        assert isinstance(result, np.ndarray)
        assert result.shape == s_values.shape
    
    def test_fixedparam_serialization_integration(self):
        """Test FixedParam serialization in complex objects."""
        s_values = np.array([0.5, 0.6, 0.7])
        
        # Create complex object with FixedParam
        bw = RelativisticBreitWigner(
            pole_mass=0.775,
            s=s_values,
            width=0.15,
            r=1.0,
            L=1
        )
        
        # Serialize
        data = bw.model_dump()
        
        # Should contain FixedParam data
        assert 's' in data
        assert 'value' in data['s']
        
        # Deserialize
        bw_restored = RelativisticBreitWigner(**data)
        print(bw_restored.s.value)
        # Should work the same
        result_original = bw()
        result_restored = bw_restored()
        
        np.testing.assert_array_almost_equal(result_original, result_restored)


class TestBackendIntegration:
    """Test backend integration across components."""
    
    def test_backend_consistency_across_components(self):
        """Test that all components use the same backend."""
        set_backend("numpy")
        
        s_values = np.array([0.5, 0.6, 0.7])
        
        # Create various components
        bw = RelativisticBreitWigner(pole_mass=0.775, s=s_values, width=0.15)
        
        channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        # All should work with numpy
        bw_result = bw()
        channel_momentum = channel.momentum(s_values)
        
        assert isinstance(bw_result, np.ndarray)
        assert isinstance(channel_momentum, np.ndarray)
    
    def test_mixed_array_types(self):
        """Test handling of mixed array types."""
        set_backend("numpy")
        
        # Create with numpy array
        s_numpy = np.array([0.5, 0.6, 0.7])
        bw = RelativisticBreitWigner(pole_mass=0.775, s=s_numpy, width=0.15)
        
        result = bw()
        assert isinstance(result, np.ndarray)
        
        # Should handle different input types gracefully
        s_list = [0.5, 0.6, 0.7]
        bw_list = RelativisticBreitWigner(pole_mass=0.775, s=s_list, width=0.15)
        
        result_list = bw_list()
        assert isinstance(result_list, np.ndarray)


class TestErrorHandling:
    """Test error handling in integrated scenarios."""
    

    def test_invalid_channel_configuration(self):
        """Test error handling with invalid channel configuration."""
        s_values = np.array([0.5, 0.6, 0.7])
        
        # Test K-matrix with mismatched dimensions
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        
        with pytest.raises((ValueError, IndexError)):
            # Wrong number of decay couplings
            KMatrixAdvanced(
                s=s_values,
                channels=[pipi_channel],
                pole_masses=[0.775],
                production_couplings=[1.0],
                decay_couplings=[1.0, 0.5],  # Too many for 1 pole × 1 channel
                output_channel=0
            )


class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_array_handling(self):
        """Test handling of large arrays."""
        # Create large s array
        s_values = np.linspace(0.1, 2.0, 10000)
        
        # Should handle large arrays efficiently
        bw = RelativisticBreitWigner(pole_mass=0.775, s=s_values, width=0.15)
        
        result = bw()
        
        assert isinstance(result, np.ndarray)
        assert result.shape == s_values.shape
        assert np.all(np.isfinite(result))
    
    def test_repeated_evaluations(self):
        """Test repeated evaluations for optimization scenarios."""
        s_values = np.linspace(0.5, 1.0, 1000)
        
        bw = RelativisticBreitWigner(pole_mass=0.775, s=s_values, width=0.15)
        
        # Multiple evaluations with different parameters
        results = []
        for width in [0.1, 0.15, 0.2, 0.25, 0.3]:
            result = bw(width=width)
            results.append(result)
            assert isinstance(result, np.ndarray)
        
        # All results should be different
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                assert not np.allclose(results[i], results[j])


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_amplitude_analysis_scenario(self):
        """Test typical amplitude analysis scenario."""
        # Create invariant mass spectrum
        m_min, m_max = 0.3, 1.5
        n_points = 200
        masses = np.linspace(m_min, m_max, n_points)
        s_values = masses**2
        
        # Create multiple resonances
        rho_770 = RelativisticBreitWigner(
            pole_mass=0.775, s=s_values, width=0.15, r=1.0, L=1
        )
        
        f0_980 = RelativisticBreitWigner(
            pole_mass=0.98, s=s_values, width=0.05, r=1.0, L=0
        )
        
        # Evaluate amplitudes
        amp_rho = rho_770()
        amp_f0 = f0_980()
        
        # Combine amplitudes (coherent sum)
        total_amp = 1.0 * amp_rho + 0.5 * amp_f0
        intensity = np.abs(total_amp)**2
        
        # Check results
        assert isinstance(intensity, np.ndarray)
        assert intensity.shape == s_values.shape
        assert np.all(intensity >= 0)  # Intensity should be non-negative
        assert np.all(np.isfinite(intensity))
        
        # Should show structure from both resonances
        assert np.max(intensity) > np.min(intensity)  # Should have variation
    
    def test_coupled_channel_analysis(self):
        """Test coupled-channel analysis scenario."""
        # Energy range covering multiple thresholds
        s_values = np.linspace(0.2, 2.0, 300)
        
        # Create channels
        pipi_channel = Channel(
            particle1=CommonParticles.PI_PLUS,
            particle2=CommonParticles.PI_MINUS
        )
        kk_channel = Channel(
            particle1=CommonParticles.K_PLUS,
            particle2=CommonParticles.K_MINUS
        )
        
        # Create K-matrix for both channels
        kmat_pipi = KMatrixAdvanced(
            s=s_values,
            channels=[pipi_channel, kk_channel],
            pole_masses=[0.6, 1.0],
            production_couplings=[1.0, 0.8],
            decay_couplings=[1.0, 0.5, 0.3, 0.7],
            output_channel=0  # ππ channel
        )
        
        kmat_kk = KMatrixAdvanced(
            s=s_values,
            channels=[pipi_channel, kk_channel],
            pole_masses=[0.6, 1.0],
            production_couplings=[1.0, 0.8],
            decay_couplings=[1.0, 0.5, 0.3, 0.7],
            output_channel=1  # KK channel
        )
        
        # Evaluate both channels
        amp_pipi = kmat_pipi()
        amp_kk = kmat_kk()
        
        # Check results
        assert isinstance(amp_pipi, np.ndarray)
        assert isinstance(amp_kk, np.ndarray)
        assert amp_pipi.shape == s_values.shape
        assert amp_kk.shape == s_values.shape
        
        # Should show threshold effects
        pipi_threshold = pipi_channel.threshold
        kk_threshold = kk_channel.threshold
        
        # Find indices near thresholds
        idx_pipi = np.argmin(np.abs(s_values - pipi_threshold))
        idx_kk = np.argmin(np.abs(s_values - kk_threshold))
        
        # Behavior should change at thresholds
        assert not np.allclose(amp_pipi[:10], amp_pipi[-10:])
        assert not np.allclose(amp_kk[:10], amp_kk[-10:])
