"""
Tests for utility functions.
"""

import numpy as np
import pytest

from decayshape.utils import angular_momentum_barrier_factor, blatt_weiskopf_form_factor, relativistic_breit_wigner_denominator


class TestBlattWeiskopfFormFactor:
    """Test Blatt-Weiskopf form factor calculations."""

    def test_s_wave_form_factor(self):
        """Test S-wave (L=0) form factor."""
        q = np.array([0.1, 0.2, 0.3])
        q0 = 0.2
        r = 1.0

        F = blatt_weiskopf_form_factor(q, q0, r, L=0)

        # S-wave should be 1 everywhere
        expected = np.ones_like(q)
        np.testing.assert_array_almost_equal(F, expected)

    def test_p_wave_form_factor(self):
        """Test P-wave (L=1) form factor."""
        q = np.array([0.1, 0.2, 0.3])
        q0 = 0.2
        r = 1.0

        F = blatt_weiskopf_form_factor(q, q0, r, L=1)

        # P-wave formula: sqrt((1 + (r*q0)^2) / (1 + (r*q)^2))
        expected = np.sqrt((1 + (r * q0) ** 2) / (1 + (r * q) ** 2))
        np.testing.assert_array_almost_equal(F, expected)

    def test_d_wave_form_factor(self):
        """Test D-wave (L=2) form factor."""
        q = np.array([0.1, 0.2, 0.3])
        q0 = 0.2
        r = 1.0

        F = blatt_weiskopf_form_factor(q, q0, r, L=2)

        # Should be finite and positive
        assert np.all(F > 0)
        assert np.all(np.isfinite(F))

    def test_higher_l_values(self):
        """Test higher L values."""
        q = np.array([0.1, 0.2, 0.3])
        q0 = 0.2
        r = 1.0

        for L in [3, 4]:
            F = blatt_weiskopf_form_factor(q, q0, r, L=L)
            assert np.all(F > 0)
            assert np.all(np.isfinite(F))

    def test_invalid_l_value(self):
        """Test error for invalid L value."""
        q = 0.2
        q0 = 0.2
        r = 1.0

        with pytest.raises(ValueError, match="not implemented for L="):
            blatt_weiskopf_form_factor(q, q0, r, L=10)

    def test_scalar_input(self):
        """Test with scalar input."""
        q = 0.2
        q0 = 0.2
        r = 1.0

        F = blatt_weiskopf_form_factor(q, q0, r, L=1)

        assert isinstance(F, (float, np.floating))
        assert F > 0
        assert np.isfinite(F)

    def test_zero_momentum(self):
        """Test behavior at zero momentum."""
        q = 0.0
        q0 = 0.2
        r = 1.0

        # S-wave should be 1
        F0 = blatt_weiskopf_form_factor(q, q0, r, L=0)
        assert F0 == 1.0

        # P-wave should be finite
        F1 = blatt_weiskopf_form_factor(q, q0, r, L=1)
        assert np.isfinite(F1)
        assert F1 > 0

    def test_form_factor_normalization(self):
        """Test form factor normalization at q = q0."""
        q0 = 0.2
        r = 1.0

        for L in [0, 1, 2, 3, 4]:
            F = blatt_weiskopf_form_factor(q0, q0, r, L=L)
            assert F == pytest.approx(1.0, rel=1e-10)


class TestAngularMomentumBarrierFactor:
    """Test angular momentum barrier factor calculations."""

    def test_s_wave_barrier(self):
        """Test S-wave (L=0) barrier factor."""
        q = np.array([0.1, 0.2, 0.3])
        q0 = 0.2

        B = angular_momentum_barrier_factor(q, q0, L=0)

        # S-wave should be 1 everywhere
        expected = np.ones_like(q)
        np.testing.assert_array_almost_equal(B, expected)

    def test_p_wave_barrier(self):
        """Test P-wave (L=1) barrier factor."""
        q = np.array([0.1, 0.2, 0.3])
        q0 = 0.2

        B = angular_momentum_barrier_factor(q, q0, L=1)

        # P-wave should be q/q0
        expected = q / q0
        np.testing.assert_array_almost_equal(B, expected)

    def test_higher_l_values(self):
        """Test higher L values."""
        q = np.array([0.1, 0.2, 0.3])
        q0 = 0.2

        for L in [2, 3, 4]:
            B = angular_momentum_barrier_factor(q, q0, L=L)
            expected = (q / q0) ** L
            np.testing.assert_array_almost_equal(B, expected)

    def test_barrier_at_reference(self):
        """Test barrier factor at reference momentum."""
        q0 = 0.2

        for L in [0, 1, 2, 3, 4]:
            B = angular_momentum_barrier_factor(q0, q0, L=L)
            assert B == pytest.approx(1.0, rel=1e-10)

    def test_zero_momentum(self):
        """Test barrier factor at zero momentum."""
        q = 0.0
        q0 = 0.2

        # S-wave should be 1
        B0 = angular_momentum_barrier_factor(q, q0, L=0)
        assert B0 == 1.0

        # Higher L should be 0
        for L in [1, 2, 3]:
            BL = angular_momentum_barrier_factor(q, q0, L=L)
            assert BL == 0.0

    def test_scalar_input(self):
        """Test with scalar input."""
        q = 0.3
        q0 = 0.2

        B = angular_momentum_barrier_factor(q, q0, L=2)
        expected = (q / q0) ** 2
        assert B == pytest.approx(expected, rel=1e-10)


class TestRelativisticBreitWignerDenominator:
    """Test relativistic Breit-Wigner denominator."""

    def test_denominator_calculation(self):
        """Test basic denominator calculation."""
        s = np.array([0.5, 0.6, 0.7])
        mass = 0.775
        width = 0.15

        denom = relativistic_breit_wigner_denominator(s, mass, width)

        # Should be complex
        assert np.iscomplexobj(denom)
        assert denom.shape == s.shape

        # Real part should be s - mass^2
        expected_real = s - mass**2
        np.testing.assert_array_almost_equal(np.real(denom), expected_real)

        # Imaginary part should be mass * width
        expected_imag = mass * width
        np.testing.assert_array_almost_equal(np.imag(denom), expected_imag)

    def test_denominator_at_resonance(self):
        """Test denominator at resonance."""
        mass = 0.775
        width = 0.15
        s = mass**2  # At resonance

        denom = relativistic_breit_wigner_denominator(s, mass, width)

        # Real part should be 0
        assert np.real(denom) == pytest.approx(0.0, abs=1e-10)

        # Imaginary part should be mass * width
        assert np.imag(denom) == pytest.approx(mass * width, rel=1e-10)

    def test_scalar_input(self):
        """Test with scalar input."""
        s = 0.6
        mass = 0.775
        width = 0.15

        denom = relativistic_breit_wigner_denominator(s, mass, width)

        assert isinstance(denom, complex)
        assert np.real(denom) == pytest.approx(s - mass**2, rel=1e-10)
        assert np.imag(denom) == pytest.approx(mass * width, rel=1e-10)

    def test_zero_width(self):
        """Test with zero width."""
        s = np.array([0.5, 0.6, 0.7])
        mass = 0.775
        width = 0.0

        denom = relativistic_breit_wigner_denominator(s, mass, width)

        # Should be purely real
        assert np.all(np.imag(denom) == 0)
        np.testing.assert_array_almost_equal(np.real(denom), s - mass**2)


class TestUtilityFunctionIntegration:
    """Test integration between utility functions."""

    def test_form_factor_barrier_consistency(self):
        """Test consistency between form factor and barrier factor."""
        q = np.array([0.1, 0.2, 0.3])
        q0 = 0.2
        r = 1.0

        for L in [0, 1, 2]:
            F = blatt_weiskopf_form_factor(q, q0, r, L=L)
            B = angular_momentum_barrier_factor(q, q0, L=L)

            # Both should be positive
            assert np.all(F > 0)
            assert np.all(B >= 0)

            # Both should be 1 at q = q0
            F_ref = blatt_weiskopf_form_factor(q0, q0, r, L=L)
            B_ref = angular_momentum_barrier_factor(q0, q0, L=L)

            assert F_ref == pytest.approx(1.0, rel=1e-10)
            assert B_ref == pytest.approx(1.0, rel=1e-10)

    def test_breit_wigner_denominator_properties(self):
        """Test properties of BW denominator."""
        s = 0.6
        mass = 0.775
        width = 0.15

        # BW denominator
        bw_denom = relativistic_breit_wigner_denominator(s, mass, width)

        # Should be finite and complex
        assert np.isfinite(bw_denom)
        assert np.iscomplexobj(bw_denom)

        # Should have correct real and imaginary parts
        assert np.real(bw_denom) == pytest.approx(s - mass**2, rel=1e-10)
        assert np.imag(bw_denom) == pytest.approx(mass * width, rel=1e-10)

    def test_momentum_dependent_functions(self):
        """Test functions that depend on momentum."""
        # Create momentum values
        s_values = np.array([0.3, 0.6, 0.9])
        sqrt_s = np.sqrt(s_values)
        q = sqrt_s / 2.0  # Simple momentum approximation
        q0 = 0.2
        r = 1.0

        for L in [0, 1, 2]:
            F = blatt_weiskopf_form_factor(q, q0, r, L=L)
            B = angular_momentum_barrier_factor(q, q0, L=L)

            # Product should be well-behaved
            product = F * B
            assert np.all(np.isfinite(product))
            assert np.all(product > 0)

    def test_physical_units_consistency(self):
        """Test that functions work with physical units."""
        # Use typical hadron physics values
        mass = 0.775  # GeV
        width = 0.15  # GeV
        s = 0.6  # GeV^2
        r = 1.0  # GeV^-1
        q = 0.3  # GeV
        q0 = 0.2  # GeV

        # All functions should work with these units
        bw_denom = relativistic_breit_wigner_denominator(s, mass, width)
        F = blatt_weiskopf_form_factor(q, q0, r, L=1)
        B = angular_momentum_barrier_factor(q, q0, L=1)

        # All should be finite
        assert np.isfinite(bw_denom)
        assert np.isfinite(F)
        assert np.isfinite(B)

        # Form factor and barrier should be positive
        assert F > 0
        assert B > 0
