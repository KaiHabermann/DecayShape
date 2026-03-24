# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DecayShape** is a Python package providing resonance lineshapes for hadron physics amplitude and partial wave analysis. It supports NumPy and JAX backends and is designed for use in particle physics parameter optimization workflows.

## Commands

**Install for development:**
```bash
pip install -e ".[dev]"
```

**Run all tests:**
```bash
pytest tests/
```

**Run a single test file:**
```bash
pytest tests/test_lineshapes.py
```

**Run a single test:**
```bash
pytest tests/test_lineshapes.py::TestRelativisticBreitWigner::test_create_breit_wigner
```

**Format and lint:**
```bash
black decayshape/
isort decayshape/
flake8 decayshape/
```

**Run all pre-commit checks:**
```bash
pre-commit run --all-files
```

Line length is 127 for black, isort, and flake8.

## Architecture

### Core Abstractions (`decayshape/base.py`)

Three key classes form the foundation:

- **`FixedParam[T]`**: A Pydantic generic model wrapping values that are fixed during optimization (e.g., channel masses, input `s` arrays). Fields typed as `FixedParam[T]` are *automatically wrapped* by a model validator — callers do not manually construct `FixedParam()`.
- **`JsonSchemaMixin`**: Mixin that adds `to_json_schema()` and `to_json_string()` methods, distinguishing `fixed_parameters` from `optimization_parameters` in the output schema. Used for frontend/API integration.
- **`Lineshape`**: Abstract base class (inherits `BaseModel`, `JsonSchemaMixin`). Subclasses must implement `__call__` and `parameter_order`. The `parameter_order` property defines positional override order for optimization.

### Parameter Override Pattern

All lineshapes support both positional and keyword overrides of optimization parameters at call time:
```python
lineshape = RelativisticBreitWigner(s=s_values, channel=ch, pole_mass=0.77, ...)
amplitude = lineshape()                    # use stored parameters
amplitude = lineshape(0.05, 1.0, 0)        # positional: width=0.05, r=1.0, L=0
amplitude = lineshape(width=0.05, r=1.0)   # keyword override
```

### Backend Switching (`decayshape/config.py`)

All math uses `config.backend` (aliased as `np`-like API). Switch with:
```python
import decayshape as ds
ds.set_backend("jax")   # or "numpy" (default)
```

### Lineshapes (`decayshape/lineshapes.py`)

All lineshapes are Pydantic models. Key implementations:
- **`RelativisticBreitWigner`**: Standard resonance with Blatt-Weisskopf form factors and mass-dependent width
- **`GounarisSakurai`**: Variant for ρ(770), with optional ρ-ω interference
- **`Flatté`**: Coupled-channel resonance (e.g., f₀(980) → ππ/KK̄); two independent channels
- **`Interpolation` classes** (`LinearInterpolation`, `QuadraticInterpolation`, `CubicInterpolation`): Amplitude interpolated at fixed mass points; the mass grid is `FixedParam`, the complex amplitudes are optimization parameters
- **`KMatrixAdvanced`** (`decayshape/kmatrix_advanced.py`): Multi-pole, multi-channel K-matrix

### Particles (`decayshape/particles.py`)

- **`Particle`**: mass, spin, parity
- **`Channel`**: Two-body decay channel. Key methods: `momentum(s)`, `phase_space_factor(s)`, `n(s, s0, r, L)` (barrier × form factor). Pre-built instances in `CommonParticles`.

### Physics Utilities (`decayshape/utils.py`)

Standalone functions: `blatt_weiskopf_form_factor`, `angular_momentum_barrier_factor`, `phase_space_factor`, `mass_dependent_width`.

### Schema Generation (`decayshape/schema_utils.py`)

`export_schemas_to_file(path)` generates `lineshape_schemas.json` used in CI/CD for documentation and frontend consumption.

## Testing Notes

Test markers available: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.unit`.

Key shared fixtures in `tests/conftest.py`: `sample_s_values`, `rho_parameters`, `f0_980_parameters`, `sample_particle`.
