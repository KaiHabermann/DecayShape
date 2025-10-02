"""
Tests for FixedParam Pydantic model.
"""

import pytest
import numpy as np
from pydantic import ValidationError
from decayshape import FixedParam
from decayshape.particles import CommonParticles, Particle


class TestFixedParamBasic:
    """Test basic FixedParam functionality."""
    
    def test_create_with_scalar(self):
        """Test creating FixedParam with scalar value."""
        fp = FixedParam(value=3.14)
        assert fp.value == 3.14
        assert isinstance(fp, FixedParam)
    
    def test_create_with_numpy_array(self, sample_numpy_array):
        """Test creating FixedParam with numpy array."""
        fp = FixedParam(value=sample_numpy_array)
        np.testing.assert_array_equal(fp.value, sample_numpy_array)
        assert isinstance(fp, FixedParam)
    
    def test_create_with_particle(self, sample_particle):
        """Test creating FixedParam with Particle."""
        fp = FixedParam(value=sample_particle)
        assert fp.value == sample_particle
        assert isinstance(fp, FixedParam)
    
    def test_pydantic_validation(self):
        """Test Pydantic validation works correctly."""
        # Valid creation
        fp = FixedParam(value=42)
        assert fp.value == 42
        
        # Test that we need the value parameter
        with pytest.raises(ValidationError):
            FixedParam()  # Missing required 'value' parameter


class TestFixedParamAttributeForwarding:
    """Test attribute forwarding functionality."""
    
    def test_numpy_array_attributes(self, sample_fixed_param_array):
        """Test forwarding numpy array attributes."""
        fp = sample_fixed_param_array
        
        # Test shape attribute
        assert fp.shape == fp.value.shape
        assert fp.shape == (5,)
        
        # Test dtype attribute
        assert fp.dtype == fp.value.dtype
        
        # Test ndim attribute
        assert fp.ndim == fp.value.ndim
        assert fp.ndim == 1
    
    def test_particle_attributes(self, sample_fixed_param_particle):
        """Test forwarding particle attributes."""
        fp = sample_fixed_param_particle
        
        # Test mass attribute
        assert fp.mass == fp.value.mass
        assert fp.mass == CommonParticles.PI_PLUS.mass
        
        # Test spin attribute
        assert fp.spin == fp.value.spin
        assert fp.spin == CommonParticles.PI_PLUS.spin
        
        # Test parity attribute
        assert fp.parity == fp.value.parity
        assert fp.parity == CommonParticles.PI_PLUS.parity
    
    def test_nonexistent_attribute(self, sample_fixed_param_array):
        """Test that nonexistent attributes raise AttributeError."""
        fp = sample_fixed_param_array
        
        with pytest.raises(AttributeError):
            _ = fp.nonexistent_attribute
    
    def test_private_attributes_not_forwarded(self, sample_fixed_param_array):
        """Test that private attributes are not forwarded."""
        fp = sample_fixed_param_array
        
        with pytest.raises(AttributeError):
            _ = fp._private_attr
    
    def test_pydantic_internals_not_forwarded(self, sample_fixed_param_array):
        """Test that Pydantic internals are not forwarded."""
        fp = sample_fixed_param_array
        
        # These should work (accessing Pydantic internals directly)
        assert hasattr(fp, 'value')
        assert hasattr(fp, 'model_fields')
        
        # But they shouldn't be forwarded through __getattr__
        # (they should be accessed directly, not through forwarding)


class TestFixedParamIndexing:
    """Test indexing functionality."""
    
    def test_indexing_numpy_array(self, sample_fixed_param_array):
        """Test indexing with numpy array."""
        fp = sample_fixed_param_array
        
        # Test single index
        assert fp[0] == fp.value[0]
        assert fp[0] == 1.0
        
        # Test slice
        np.testing.assert_array_equal(fp[1:3], fp.value[1:3])
        np.testing.assert_array_equal(fp[1:3], np.array([2.0, 3.0]))
    
    def test_indexing_scalar_fails(self):
        """Test that indexing scalar values fails appropriately."""
        fp = FixedParam(value=3.14)
        
        with pytest.raises(TypeError):
            _ = fp[0]


class TestFixedParamSerialization:
    """Test Pydantic serialization functionality."""
    
    def test_model_dump_scalar(self):
        """Test serialization of scalar FixedParam."""
        fp = FixedParam(value=3.14)
        data = fp.model_dump()
        
        assert data == {'value': 3.14}
    
    def test_model_dump_numpy_array(self, sample_numpy_array):
        """Test serialization of numpy array FixedParam."""
        fp = FixedParam(value=sample_numpy_array)
        data = fp.model_dump()
        
        # Check structure
        assert 'value' in data
        # Numpy arrays should be serialized (may be as arrays or lists)
        if isinstance(data['value'], list):
            assert data['value'] == sample_numpy_array.tolist()
        else:
            # If still numpy array, should be equal
            np.testing.assert_array_equal(data['value'], sample_numpy_array)
    
    def test_model_validate_scalar(self):
        """Test deserialization of scalar FixedParam."""
        data = {'value': 3.14}
        fp = FixedParam.model_validate(data)
        
        assert fp.value == 3.14
        assert isinstance(fp, FixedParam)
    
    def test_model_validate_list(self):
        """Test deserialization of list to FixedParam."""
        data = {'value': [1.0, 2.0, 3.0]}
        fp = FixedParam.model_validate(data)
        
        assert fp.value == [1.0, 2.0, 3.0]
        assert isinstance(fp, FixedParam)


class TestFixedParamMethods:
    """Test method forwarding."""
    
    def test_numpy_methods(self, sample_fixed_param_array):
        """Test forwarding numpy array methods."""
        fp = sample_fixed_param_array
        
        # Test sum method
        assert fp.sum() == fp.value.sum()
        assert fp.sum() == 15.0
        
        # Test mean method
        assert fp.mean() == fp.value.mean()
        assert fp.mean() == 3.0
        
        # Test max method
        assert fp.max() == fp.value.max()
        assert fp.max() == 5.0
    
    def test_string_methods(self):
        """Test forwarding string methods."""
        fp = FixedParam(value="hello world")
        
        # Test upper method
        assert fp.upper() == fp.value.upper()
        assert fp.upper() == "HELLO WORLD"
        
        # Test split method
        assert fp.split() == fp.value.split()
        assert fp.split() == ["hello", "world"]


class TestFixedParamTypeHints:
    """Test type hint functionality."""
    
    def test_generic_typing(self):
        """Test that generic typing works correctly."""
        # These should work without type errors
        fp_int: FixedParam[int] = FixedParam(value=42)
        fp_float: FixedParam[float] = FixedParam(value=3.14)
        fp_array: FixedParam[np.ndarray] = FixedParam(value=np.array([1, 2, 3]))
        
        assert isinstance(fp_int, FixedParam)
        assert isinstance(fp_float, FixedParam)
        assert isinstance(fp_array, FixedParam)
    
    def test_particle_typing(self):
        """Test typing with Particle objects."""
        fp_particle: FixedParam[Particle] = FixedParam(value=CommonParticles.PI_PLUS)
        
        assert isinstance(fp_particle, FixedParam)
        assert isinstance(fp_particle.value, Particle)
