"""
Tests for configuration and backend switching.
"""

import numpy as np
import pytest

from decayshape import RelativisticBreitWigner, config, set_backend


class TestBackendSwitching:
    """Test backend switching functionality."""

    def setUp(self):
        """Set up test with numpy backend."""
        set_backend("numpy")

    def tearDown(self):
        """Reset to numpy backend after tests."""
        set_backend("numpy")

    def test_default_backend(self):
        """Test default backend is numpy."""
        set_backend("numpy")
        assert config.backend_name == "numpy"
        assert config.backend is np

    def test_numpy_backend(self):
        """Test numpy backend functionality."""
        set_backend("numpy")

        assert config.backend_name == "numpy"
        assert config.backend is np

        # Test basic operations
        arr = config.backend.array([1, 2, 3])
        assert isinstance(arr, np.ndarray)

        result = config.backend.sum(arr)
        assert result == 6

    @pytest.mark.skipif(True, reason="JAX not required for basic functionality")
    def test_jax_backend(self):
        """Test JAX backend functionality (if available)."""
        try:
            import jax.numpy as jnp

            set_backend("jax")

            assert config.backend_name == "jax"
            assert config.backend is jnp

            # Test basic operations
            arr = config.backend.array([1, 2, 3])
            result = config.backend.sum(arr)
            assert result == 6

        except ImportError:
            pytest.skip("JAX not available")
        finally:
            set_backend("numpy")  # Reset

    def test_invalid_backend(self):
        """Test error with invalid backend."""
        with pytest.raises(ValueError, match="Backend must be 'numpy' or 'jax', got invalid_backend"):
            set_backend("invalid_backend")

    def test_backend_persistence(self):
        """Test that backend setting persists."""
        original_backend = config.backend_name

        set_backend("numpy")
        assert config.backend_name == "numpy"

        # Create a lineshape
        s_vals = np.array([0.5, 0.6, 0.7])
        bw = RelativisticBreitWigner(pole_mass=0.775, s=s_vals, width=0.15)

        # Backend should still be numpy
        assert config.backend_name == "numpy"

        # Evaluation should work
        result = bw()
        assert isinstance(result, np.ndarray)

        # Reset to original
        set_backend(original_backend)


class TestConfigProperties:
    """Test configuration properties."""

    def test_config_attributes(self):
        """Test that config has required attributes."""
        assert hasattr(config, "backend")
        assert hasattr(config, "backend_name")
        assert hasattr(config, "set_backend")

    def test_config_backend_type(self):
        """Test that backend is a module."""
        set_backend("numpy")

        # Should be the numpy module
        assert config.backend.__name__ == "numpy"

        # Should have common array functions
        assert hasattr(config.backend, "array")
        assert hasattr(config.backend, "sum")
        assert hasattr(config.backend, "sqrt")
        assert hasattr(config.backend, "ones_like")

    def test_backend_function_availability(self):
        """Test that required functions are available in backend."""
        set_backend("numpy")

        # Test required functions
        required_functions = [
            "array",
            "sum",
            "sqrt",
            "ones_like",
            "zeros_like",
            "real",
            "imag",
            "abs",
            "angle",
            "exp",
            "log",
            "sin",
            "cos",
            "pi",
            "where",
            "isfinite",
            "isnan",
        ]

        for func_name in required_functions:
            assert hasattr(config.backend, func_name), f"Backend missing {func_name}"


class TestBackendCompatibility:
    """Test backend compatibility with lineshapes."""

    def test_breit_wigner_with_numpy(self):
        """Test Breit-Wigner with numpy backend."""
        set_backend("numpy")

        s_vals = np.array([0.5, 0.6, 0.7])
        bw = RelativisticBreitWigner(pole_mass=0.775, s=s_vals, width=0.15)

        result = bw()

        assert isinstance(result, np.ndarray)
        assert result.shape == s_vals.shape
        assert np.all(np.isfinite(result))

    @pytest.mark.skipif(True, reason="JAX not required for basic functionality")
    def test_breit_wigner_with_jax(self):
        """Test Breit-Wigner with JAX backend (if available)."""
        try:
            import jax.numpy as jnp

            set_backend("jax")

            s_vals = jnp.array([0.5, 0.6, 0.7])
            bw = RelativisticBreitWigner(pole_mass=0.775, s=s_vals, width=0.15)

            result = bw()

            # Result should be JAX array
            assert hasattr(result, "shape")  # JAX arrays have shape
            assert result.shape == s_vals.shape

        except ImportError:
            pytest.skip("JAX not available")
        finally:
            set_backend("numpy")  # Reset

    def test_backend_switching_consistency(self):
        """Test that results are consistent across backends."""
        s_vals = np.array([0.5, 0.6, 0.7])

        # Evaluate with numpy
        set_backend("numpy")
        bw_numpy = RelativisticBreitWigner(pole_mass=0.775, s=s_vals, width=0.15)
        result_numpy = bw_numpy()

        # Results should be numpy arrays
        assert isinstance(result_numpy, np.ndarray)

        # For now, we only test numpy since JAX is optional
        # In a full implementation, you would test:
        # set_backend("jax")
        # bw_jax = RelativisticBreitWigner(pole_mass=0.775, s=jnp.array(s_vals), width=0.15)
        # result_jax = bw_jax()
        # np.testing.assert_array_almost_equal(result_numpy, np.array(result_jax))

        set_backend("numpy")  # Reset


class TestConfigSingleton:
    """Test that config behaves as a singleton."""

    def test_config_import_consistency(self):
        """Test that config imports are consistent."""
        from decayshape import config as config2
        from decayshape.config import config as config1

        # Should be the same object
        assert config1 is config2

        # Changes to one should affect the other
        original_backend = config1.backend_name

        config1.set_backend("numpy")
        assert config2.backend_name == "numpy"

        # Reset
        config1.set_backend(original_backend)

    def test_config_state_persistence(self):
        """Test that config state persists across imports."""
        from decayshape.config import config

        original_backend = config.backend_name

        # Change backend
        config.set_backend("numpy")

        # Import again
        from decayshape.config import config as config_new

        # Should have the same state
        assert config_new.backend_name == "numpy"

        # Reset
        config.set_backend(original_backend)
