# Contributing to DecayShape

Thank you for your interest in contributing to DecayShape! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/DecayShape.git
   cd DecayShape
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e .
   pip install pytest pytest-cov black isort flake8 mypy
   pip install matplotlib pandas  # For benchmarks and examples
   ```

4. **Install JAX (optional, for backend testing)**
   ```bash
   pip install jax jaxlib
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=decayshape --cov-report=html

# Run specific test file
pytest tests/test_lineshapes.py

# Run tests for specific backend
pytest -k "not jax"  # Skip JAX tests
```

## Code Style

We use several tools to maintain code quality:

```bash
# Format code
black decayshape/
isort decayshape/

# Check linting
flake8 decayshape/

# Type checking
mypy decayshape/
```

## Adding New Lineshapes

When adding a new lineshape:

1. **Inherit from the `Lineshape` base class**
2. **Use Pydantic fields with proper type annotations**
3. **Separate fixed parameters (use `FixedParam[Type]`) from optimization parameters**
4. **Implement the `parameter_order` property**
5. **Add comprehensive docstrings**
6. **Include physics references in docstrings**

Example:
```python
class MyLineshape(Lineshape):
    """
    My custom lineshape.
    
    References:
        - Author et al., "Paper Title", Journal (Year)
    """
    
    # Fixed parameters
    channel_mass: FixedParam[float] = Field(..., description="Channel mass")
    
    # Optimization parameters  
    pole_mass: float = Field(..., description="Pole mass")
    width: float = Field(..., description="Width")
    
    @property
    def parameter_order(self) -> List[str]:
        return ["pole_mass", "width"]
    
    def __call__(self, *args, **kwargs) -> Union[float, Any]:
        params = self._get_parameters(*args, **kwargs)
        np = config.backend  # Always get backend dynamically
        
        # Implementation here
        return result
```

## Testing Guidelines

1. **Write tests for all new functionality**
2. **Test with both NumPy and JAX backends**
3. **Include physics validation tests**
4. **Test parameter override functionality**
5. **Test serialization/deserialization**

Example test structure:
```python
class TestMyLineshape:
    def test_basic_functionality(self):
        # Test basic lineshape evaluation
        pass
    
    def test_parameter_override(self):
        # Test parameter override at call time
        pass
    
    def test_backend_switching(self):
        # Test that backend switching works
        pass
    
    def test_serialization(self):
        # Test Pydantic serialization
        pass
```

## Physics Validation

When implementing physics lineshapes:

1. **Compare with literature values**
2. **Test limiting cases**
3. **Verify units and conventions**
4. **Include references to original papers**
5. **Test threshold behavior**

## Documentation

1. **Write clear docstrings** following NumPy style
2. **Include parameter descriptions**
3. **Add usage examples**
4. **Reference physics literature**
5. **Update README.md** if adding major features

## Submitting Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes**
   - Follow the code style guidelines
   - Add tests
   - Update documentation

3. **Run the test suite**
   ```bash
   pytest
   black decayshape/
   isort decayshape/
   flake8 decayshape/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add new lineshape: MyLineshape"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/my-new-feature
   ```

## Pull Request Guidelines

- **Use descriptive titles and descriptions**
- **Reference any related issues**
- **Include physics validation results**
- **Ensure all tests pass**
- **Update documentation as needed**

## Questions?

Feel free to open an issue for:
- Questions about implementation
- Physics discussions
- Feature requests
- Bug reports

Thank you for contributing to DecayShape!
