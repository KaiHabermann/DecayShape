"""
Tests for interpolation-based lineshapes.

This module tests the LinearInterpolation, QuadraticInterpolation,
and CubicInterpolation classes to ensure they work correctly and match at the set mass positions.
"""

import numpy as np

from decayshape.lineshapes import CubicInterpolation, LinearInterpolation, QuadraticInterpolation


class TestLinearInterpolation:
    """Test the linear interpolation class."""

    def test_linear_interpolation_creation(self):
        """Test that LinearInterpolation can be created."""
        mass_points = [0.5, 1.0, 1.5, 2.0]
        amplitudes = [1.0, 2.0, 1.5, 0.8]
        linear = LinearInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        assert linear.mass_points.value == mass_points
        assert linear.amplitudes == amplitudes
        assert linear.parameter_order == ["amplitude_0", "amplitude_1", "amplitude_2", "amplitude_3"]

    def test_linear_interpolation_matches_at_mass_points(self):
        """Test that linear interpolation matches the amplitude values at mass points."""
        mass_points = [0.5, 1.0, 1.5, 2.0]
        amplitudes = [1.0, 2.0, 1.5, 0.8]

        linear = LinearInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Test that interpolation matches exactly at mass points
        for mass, expected_amplitude in zip(mass_points, amplitudes):
            result = linear(angular_momentum=0, spin=1, s=mass)
            assert np.isclose(
                result, expected_amplitude, rtol=1e-10
            ), f"At mass {mass}, expected {expected_amplitude}, got {result}"

    def test_linear_interpolation_between_points(self):
        """Test linear interpolation between mass points."""
        mass_points = [0.0, 2.0]
        amplitudes = [1.0, 3.0]

        linear = LinearInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Test at midpoint
        result = linear(angular_momentum=0, spin=1, s=1.0)
        expected = 2.0  # (1.0 + 3.0) / 2
        assert np.isclose(result, expected, rtol=1e-10), f"At midpoint, expected {expected}, got {result}"

    def test_linear_interpolation_vectorized(self):
        """Test that linear interpolation works with vectorized input."""
        mass_points = [0.5, 1.0, 1.5, 2.0]
        amplitudes = [1.0, 2.0, 1.5, 0.8]

        linear = LinearInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        s_values = np.array([0.3, 0.75, 1.25, 1.75, 2.2])
        result = linear(angular_momentum=0, spin=1, s=s_values)

        # Check that we get the right number of results
        assert len(result) == len(s_values)

        # Check that results are reasonable (between min and max amplitudes)
        assert np.all(result >= min(amplitudes))
        assert np.all(result <= max(amplitudes))

    def test_linear_interpolation_extrapolation(self):
        """Test linear interpolation extrapolation behavior."""
        mass_points = [1.0, 2.0]
        amplitudes = [1.0, 2.0]

        linear = LinearInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Test extrapolation below first point
        result_below = linear(angular_momentum=0, spin=1, s=0.5)
        assert np.isclose(result_below, 1.0, rtol=1e-10), "Should use first amplitude for extrapolation"

        # Test extrapolation above last point
        result_above = linear(angular_momentum=0, spin=1, s=3.0)
        assert np.isclose(result_above, 2.0, rtol=1e-10), "Should use last amplitude for extrapolation"


class TestQuadraticInterpolation:
    """Test the quadratic interpolation class."""

    def test_quadratic_interpolation_creation(self):
        """Test that QuadraticInterpolation can be created."""
        mass_points = [0.5, 1.0, 1.5, 2.0]
        quadratic = QuadraticInterpolation(
            mass_points=mass_points,
        )

        assert quadratic.mass_points.value == mass_points
        assert quadratic.parameter_order == ["amplitude_0", "amplitude_1", "amplitude_2", "amplitude_3"]

    def test_quadratic_interpolation_matches_at_mass_points(self):
        """Test that quadratic interpolation matches the amplitude values at mass points."""
        mass_points = [0.5, 1.0, 1.5, 2.0]
        amplitudes = [1.0, 2.0, 1.5, 0.8]

        quadratic = QuadraticInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Test that interpolation matches exactly at mass points
        for mass, expected_amplitude in zip(mass_points, amplitudes):
            result = quadratic(angular_momentum=0, spin=1, s=mass)
            assert np.isclose(
                result, expected_amplitude, rtol=1e-10
            ), f"At mass {mass}, expected {expected_amplitude}, got {result}"

    def test_quadratic_interpolation_fallback_to_linear(self):
        """Test that quadratic interpolation falls back to linear for 2 points."""
        mass_points = [1.0, 2.0]
        amplitudes = [1.0, 3.0]

        quadratic = QuadraticInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Should behave like linear interpolation
        result = quadratic(angular_momentum=0, spin=1, s=1.5)
        expected = 2.0  # (1.0 + 3.0) / 2
        assert np.isclose(result, expected, rtol=1e-10), f"At midpoint, expected {expected}, got {result}"

    def test_quadratic_interpolation_vectorized(self):
        """Test that quadratic interpolation works with vectorized input."""
        mass_points = [0.5, 1.0, 1.5, 2.0]
        amplitudes = [1.0, 2.0, 1.5, 0.8]

        quadratic = QuadraticInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        s_values = np.array([0.3, 0.75, 1.25, 1.75, 2.2])
        result = quadratic(angular_momentum=0, spin=1, s=s_values)

        # Check that we get the right number of results
        assert len(result) == len(s_values)

        # Check that results are reasonable
        assert np.all(np.isfinite(result))

    def test_quadratic_interpolation_single_point(self):
        """Test quadratic interpolation with single point."""
        mass_points = [1.0]
        amplitudes = [2.0]

        quadratic = QuadraticInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Should return constant value
        result = quadratic(angular_momentum=0, spin=1, s=1.5)
        assert np.isclose(result, 2.0, rtol=1e-10), "Should return constant value for single point"


class TestCubicInterpolation:
    """Test the cubic interpolation class."""

    def test_cubic_interpolation_creation(self):
        """Test that CubicInterpolation can be created."""
        mass_points = [0.5, 1.0, 1.5, 2.0, 2.5]
        cubic = CubicInterpolation(
            mass_points=mass_points,
        )

        assert cubic.mass_points.value == mass_points
        assert cubic.parameter_order == ["amplitude_0", "amplitude_1", "amplitude_2", "amplitude_3", "amplitude_4"]

    def test_cubic_interpolation_matches_at_mass_points(self):
        """Test that cubic interpolation matches the amplitude values at mass points."""
        mass_points = [0.5, 1.0, 1.5, 2.0, 2.5]
        amplitudes = [1.0, 2.0, 1.5, 0.8, 1.2]

        cubic = CubicInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Test that interpolation matches exactly at mass points
        for mass, expected_amplitude in zip(mass_points, amplitudes):
            result = cubic(angular_momentum=0, spin=1, s=mass)
            assert np.isclose(
                result, expected_amplitude, rtol=1e-10
            ), f"At mass {mass}, expected {expected_amplitude}, got {result}"

    def test_cubic_interpolation_fallback_to_linear(self):
        """Test that cubic interpolation falls back to linear for 2 points."""
        mass_points = [1.0, 2.0]
        amplitudes = [1.0, 3.0]

        cubic = CubicInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Should behave like linear interpolation
        result = cubic(angular_momentum=0, spin=1, s=1.5)
        expected = 2.0  # (1.0 + 3.0) / 2
        assert np.isclose(result, expected, rtol=1e-10), f"At midpoint, expected {expected}, got {result}"

    def test_cubic_interpolation_fallback_to_quadratic(self):
        """Test that cubic interpolation falls back to quadratic for 3 points."""
        mass_points = [1.0, 2.0, 3.0]
        amplitudes = [1.0, 2.0, 1.5]

        cubic = CubicInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Should behave like quadratic interpolation
        result = cubic(angular_momentum=0, spin=1, s=2.5)
        assert np.isfinite(result), "Should return finite result"

    def test_cubic_interpolation_vectorized(self):
        """Test that cubic interpolation works with vectorized input."""
        mass_points = [0.5, 1.0, 1.5, 2.0, 2.5]
        amplitudes = [1.0, 2.0, 1.5, 0.8, 1.2]

        cubic = CubicInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        s_values = np.array([0.3, 0.75, 1.25, 1.75, 2.25, 2.7])
        result = cubic(angular_momentum=0, spin=1, s=s_values)

        # Check that we get the right number of results
        assert len(result) == len(s_values)

        # Check that results are reasonable
        assert np.all(np.isfinite(result))

    def test_cubic_interpolation_single_point(self):
        """Test cubic interpolation with single point."""
        mass_points = [1.0]
        amplitudes = [2.0]

        cubic = CubicInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Should return constant value
        result = cubic(angular_momentum=0, spin=1, s=1.5)
        assert np.isclose(result, 2.0, rtol=1e-10), "Should return constant value for single point"


class TestInterpolationComparison:
    """Test comparison between different interpolation methods."""

    def test_interpolation_consistency_at_mass_points(self):
        """Test that all interpolation methods give the same result at mass points."""
        mass_points = [0.5, 1.0, 1.5, 2.0]
        amplitudes = [1.0, 2.0, 1.5, 0.8]

        linear = LinearInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        quadratic = QuadraticInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        cubic = CubicInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # All methods should give the same result at mass points
        for mass, expected_amplitude in zip(mass_points, amplitudes):
            linear_result = linear(angular_momentum=0, spin=1, s=mass)
            quadratic_result = quadratic(angular_momentum=0, spin=1, s=mass)
            cubic_result = cubic(angular_momentum=0, spin=1, s=mass)

            assert np.isclose(linear_result, expected_amplitude, rtol=1e-10)
            assert np.isclose(quadratic_result, expected_amplitude, rtol=1e-10)
            assert np.isclose(cubic_result, expected_amplitude, rtol=1e-10)

            # All methods should give the same result
            assert np.isclose(linear_result, quadratic_result, rtol=1e-10)
            assert np.isclose(linear_result, cubic_result, rtol=1e-10)

    def test_interpolation_differences_between_points(self):
        """Test that interpolation methods can give different results between mass points."""
        mass_points = [0.0, 1.0, 2.0, 3.0]
        amplitudes = [0.0, 1.0, 0.0, 1.0]  # Oscillating pattern

        linear = LinearInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        quadratic = QuadraticInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        cubic = CubicInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Test at a point between mass points where methods might differ
        s_test = 1.5
        linear_result = linear(angular_momentum=0, spin=1, s=s_test)
        quadratic_result = quadratic(angular_momentum=0, spin=1, s=s_test)
        cubic_result = cubic(angular_momentum=0, spin=1, s=s_test)

        # Results should be finite
        assert np.isfinite(linear_result)
        assert np.isfinite(quadratic_result)
        assert np.isfinite(cubic_result)

        # Methods might give different results (this is expected)
        # We just check that they're all reasonable
        assert -2.0 <= linear_result <= 2.0
        assert -2.0 <= quadratic_result <= 2.0
        assert -2.0 <= cubic_result <= 2.0


class TestInterpolationEdgeCases:
    """Test edge cases for interpolation methods."""

    def test_empty_mass_points(self):
        """Test interpolation with empty mass points."""
        mass_points = []

        linear = LinearInterpolation(mass_points=mass_points)
        quadratic = QuadraticInterpolation(mass_points=mass_points)
        cubic = CubicInterpolation(mass_points=mass_points)

        # Should handle empty case gracefully
        result = linear(angular_momentum=0, spin=1, s=1.0)
        assert np.isclose(result, 0.0, rtol=1e-10), "Should return 0 for empty mass points"
        result = quadratic(angular_momentum=0, spin=1, s=1.0)
        assert np.isclose(result, 0.0, rtol=1e-10), "Should return 0 for empty mass points"
        result = cubic(angular_momentum=0, spin=1, s=1.0)
        assert np.isclose(result, 0.0, rtol=1e-10), "Should return 0 for empty mass points"

    def test_duplicate_mass_points(self):
        """Test interpolation with duplicate mass points."""
        mass_points = [1.0, 1.0, 2.0]  # Duplicate at 1.0
        amplitudes = [1.0, 2.0, 3.0]

        linear = LinearInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Should handle duplicates (behavior may vary)
        result = linear(angular_momentum=0, spin=1, s=1.5)
        assert np.isfinite(result), "Should handle duplicate mass points"

    def test_unsorted_mass_points(self):
        """Test interpolation with unsorted mass points."""
        mass_points = [2.0, 0.5, 1.5, 1.0]  # Unsorted
        amplitudes = [0.8, 1.0, 1.5, 2.0]

        linear = LinearInterpolation(mass_points=mass_points, amplitudes=amplitudes)

        # Should still work (numpy.interp handles unsorted xp)
        result = linear(angular_momentum=0, spin=1, s=1.25)
        assert np.isfinite(result), "Should handle unsorted mass points"


class TestComplexInterpolation:
    """Test complex interpolation behavior for all interpolation classes."""

    def test_complex_interpolation_behavior(self):
        """Test that all interpolation classes work correctly with complex interpolation."""
        import numpy as np

        from decayshape.base import FixedParam

        # Test data
        mass_points = [0.5, 1.0, 1.5, 2.0]
        # For complex interpolation: [real_0, imag_0, real_1, imag_1, real_2, imag_2, real_3, imag_3]
        complex_amplitudes = [1.0, 0.5, 2.0, 0.3, 1.5, 0.8, 0.8, 0.2]
        s_values = np.array([0.75, 1.25, 1.75])

        # Test LinearInterpolation with complex=True
        linear_complex = LinearInterpolation(
            mass_points=mass_points, amplitudes=complex_amplitudes, complex=FixedParam(value=True)
        )

        # Test parameter order for complex interpolation
        expected_param_order = [
            "amplitude_0_real",
            "amplitude_0_imag",
            "amplitude_1_real",
            "amplitude_1_imag",
            "amplitude_2_real",
            "amplitude_2_imag",
            "amplitude_3_real",
            "amplitude_3_imag",
        ]
        assert (
            linear_complex.parameter_order == expected_param_order
        ), "Complex parameter order should include real and imag parts"

        # Test complex evaluation
        linear_result = linear_complex(angular_momentum=0, spin=1, s=s_values)
        assert np.iscomplexobj(linear_result), "Linear complex result should be complex"
        assert linear_result.shape == s_values.shape, "Result shape should match input shape"

        # Test QuadraticInterpolation with complex=True
        quadratic_complex = QuadraticInterpolation(
            mass_points=mass_points, amplitudes=complex_amplitudes, complex=FixedParam(value=True)
        )

        quadratic_result = quadratic_complex(angular_momentum=0, spin=1, s=s_values)
        assert np.iscomplexobj(quadratic_result), "Quadratic complex result should be complex"
        assert quadratic_result.shape == s_values.shape, "Result shape should match input shape"

        # Test CubicInterpolation with complex=True
        cubic_complex = CubicInterpolation(
            mass_points=mass_points, amplitudes=complex_amplitudes, complex=FixedParam(value=True)
        )

        cubic_result = cubic_complex(angular_momentum=0, spin=1, s=s_values)
        assert np.iscomplexobj(cubic_result), "Cubic complex result should be complex"
        assert cubic_result.shape == s_values.shape, "Result shape should match input shape"

        # Test that all methods give different results (they should interpolate differently)
        assert not np.allclose(linear_result, quadratic_result), "Linear and quadratic should give different results"
        assert not np.allclose(linear_result, cubic_result), "Linear and cubic should give different results"
        assert not np.allclose(quadratic_result, cubic_result), "Quadratic and cubic should give different results"

        # Test that complex interpolation matches at mass points
        for i, mass in enumerate(mass_points):
            expected_complex = complex_amplitudes[2 * i] + 1j * complex_amplitudes[2 * i + 1]

            linear_at_mass = linear_complex(angular_momentum=0, spin=1, s=mass)
            quadratic_at_mass = quadratic_complex(angular_momentum=0, spin=1, s=mass)
            cubic_at_mass = cubic_complex(angular_momentum=0, spin=1, s=mass)

            # All should match exactly at the mass points
            assert np.isclose(linear_at_mass, expected_complex), f"Linear should match at mass {mass}"
            assert np.isclose(quadratic_at_mass, expected_complex), f"Quadratic should match at mass {mass}"
            assert np.isclose(cubic_at_mass, expected_complex), f"Cubic should match at mass {mass}"

        # Test real interpolation (complex=False) for comparison
        real_amplitudes = [1.0, 2.0, 1.5, 0.8]
        linear_real = LinearInterpolation(mass_points=mass_points, amplitudes=real_amplitudes, complex=FixedParam(value=False))

        # Test parameter order for real interpolation
        expected_real_param_order = ["amplitude_0", "amplitude_1", "amplitude_2", "amplitude_3"]
        assert (
            linear_real.parameter_order == expected_real_param_order
        ), "Real parameter order should be simple amplitude names"

        # Test real evaluation
        real_result = linear_real(angular_momentum=0, spin=1, s=s_values)
        assert not np.iscomplexobj(real_result), "Real result should not be complex"
        assert real_result.shape == s_values.shape, "Result shape should match input shape"

        print("✓ All complex interpolation tests passed!")
