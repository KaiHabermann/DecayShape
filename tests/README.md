# DecayShape Test Suite

This directory contains comprehensive tests for the DecayShape package.

## Test Structure

### Core Tests
- **`test_fixedparam.py`** - Tests for the FixedParam Pydantic model
  - Basic functionality (creation, validation)
  - Attribute forwarding (`__getattr__`)
  - Indexing support (`__getitem__`)
  - Pydantic serialization/deserialization
  - Type hints and generic typing

- **`test_particles.py`** - Tests for Particle and Channel classes
  - Particle creation and validation
  - CommonParticles definitions
  - Channel threshold calculations
  - Phase space factors and momentum calculations

- **`test_lineshapes.py`** - Tests for lineshape implementations
  - RelativisticBreitWigner functionality
  - Flatte lineshape for coupled channels
  - Parameter override mechanisms
  - Resonance behavior validation

### Advanced Tests
- **`test_kmatrix.py`** - Tests for K-matrix functionality
  - Single and multi-channel K-matrices
  - Threshold effects and unitarity
  - Helper function tests
  - Physics validation

- **`test_utils.py`** - Tests for utility functions
  - Blatt-Weiskopf form factors
  - Angular momentum barrier factors
  - Breit-Wigner denominators
  - K-matrix elements

### System Tests
- **`test_config.py`** - Tests for configuration and backend switching
  - Backend switching (numpy/JAX)
  - Configuration persistence
  - Compatibility testing

- **`test_integration.py`** - Integration and end-to-end tests
  - Real-world usage scenarios
  - Multi-component workflows
  - Performance characteristics
  - Error handling

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Files
```bash
python -m pytest tests/test_fixedparam.py -v
python -m pytest tests/test_lineshapes.py -v
```

### Run Specific Test Classes
```bash
python -m pytest tests/test_fixedparam.py::TestFixedParamBasic -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=decayshape --cov-report=html
```

### Using the Test Runner Script
```bash
# Run all tests
python run_tests.py

# Run specific test file
python run_tests.py fixedparam
python run_tests.py lineshapes
```

## Test Configuration

- **`conftest.py`** - Pytest fixtures and configuration
- **`pytest.ini`** - Pytest settings and markers

## Key Features Tested

### FixedParam Pydantic Model
- ✅ Pydantic BaseModel inheritance
- ✅ Generic typing support
- ✅ Attribute forwarding via `__getattr__`
- ✅ Indexing support via `__getitem__`
- ✅ Serialization/deserialization
- ✅ Auto-wrapping in LineshapeBase

### Lineshape Functionality
- ✅ Parameter override mechanisms
- ✅ Positional and keyword argument handling
- ✅ Physics validation (resonance behavior)
- ✅ Backend compatibility

### K-matrix Physics
- ✅ Multi-channel coupling
- ✅ Threshold effects
- ✅ Unitarity properties
- ✅ Analytic continuation

### Integration
- ✅ End-to-end workflows
- ✅ Real-world scenarios
- ✅ Performance with large arrays
- ✅ Error handling and validation

## Test Coverage

The test suite provides comprehensive coverage of:
- All public APIs
- Edge cases and error conditions
- Physics validation
- Performance characteristics
- Integration scenarios

## Notes

- Some tests are marked as `slow` for performance testing
- JAX backend tests are optional (skipped if JAX not available)
- Integration tests cover real-world amplitude analysis scenarios
- All tests maintain backward compatibility with existing code
