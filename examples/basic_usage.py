"""
Basic usage examples for the DecayShape package.
"""

import numpy as np
import matplotlib.pyplot as plt
import decayshape as ds

def example_breit_wigner():
    """Example of using Relativistic Breit-Wigner."""
    print("=== Relativistic Breit-Wigner Example ===")
    
    # Evaluate over a range of s values
    s_values = np.linspace(0.3, 1.2, 100)
    
    # Create a Breit-Wigner for the rho(770) meson
    rho = ds.RelativisticBreitWigner(
        mass=0.775,    # GeV
        s=s_values,    # Mandelstam variable s
        width=0.15,    # GeV
        r=1.0,         # GeV^-1
        L=1            # P-wave decay
    )
    
    # Evaluate with default parameters
    amplitude = rho()
    
    print(f"Rho mass: {rho.mass} GeV")
    print(f"Rho width: {rho.width} GeV")
    print(f"Amplitude range: {np.min(np.abs(amplitude)):.3f} to {np.max(np.abs(amplitude)):.3f}")
    
    return s_values, amplitude

def example_backend_switching():
    """Example of switching between backends."""
    print("\n=== Backend Switching Example ===")
    
    # Start with numpy
    print(f"Initial backend: {ds.config.backend_name}")
    
    # Create a lineshape with s values
    s_test = np.array([0.5, 0.6, 0.7])
    bw = ds.RelativisticBreitWigner(mass=0.775, s=s_test, width=0.15)
    
    # Evaluate with numpy
    result_numpy = bw()
    print(f"NumPy result: {result_numpy}")
    
    # Switch to JAX
    ds.set_backend("jax")
    print(f"Switched to: {ds.config.backend_name}")
    
    # Evaluate with JAX
    import jax.numpy as jnp
    s_jax = jnp.array([0.5, 0.6, 0.7])
    bw_jax = ds.RelativisticBreitWigner(mass=0.775, s=s_jax, width=0.15)
    result_jax = bw_jax()
    print(f"JAX result: {result_jax}")
    
    # Switch back
    ds.set_backend("numpy")
    print(f"Back to: {ds.config.backend_name}")

def example_parameter_optimization():
    """Example of parameter optimization usage."""
    print("\n=== Parameter Optimization Example ===")
    
    # Test s values
    s_values = np.linspace(0.5, 1.0, 5)
    
    # Create a lineshape for optimization
    bw = ds.RelativisticBreitWigner(mass=0.775, s=s_values, width=0.15, r=1.0, L=1)
    print(f"Parameter order: {bw.parameter_order}")
    
    # Simulate optimization with different parameter sets
    test_params = [
        [0.1, 0.8, 0],  # width, r, L
        [0.2, 1.2, 1],  # width, r, L
        [0.15, 1.0, 2], # width, r, L
    ]
    
    print("Testing different parameter combinations:")
    for i, params in enumerate(test_params):
        # Use positional arguments for optimization
        result = bw(*params)
        print(f"  Params {i+1}: {params} -> amplitude range: {np.min(np.abs(result)):.3f} to {np.max(np.abs(result)):.3f}")
    
    # Test mixed positional and keyword arguments
    print("\nTesting mixed arguments:")
    result_mixed = bw(0.2, 1.5, L=0)  # width=0.2, r=1.5, L=0
    print(f"Mixed args result: amplitude range: {np.min(np.abs(result_mixed)):.3f} to {np.max(np.abs(result_mixed)):.3f}")

def example_utility_functions():
    """Example of using utility functions."""
    print("\n=== Utility Functions Example ===")
    
    # Test Blatt-Weiskopf form factor
    q = np.array([0.1, 0.2, 0.3])
    q0 = 0.2
    r = 1.0
    
    for L in [0, 1, 2]:
        F = ds.blatt_weiskopf_form_factor(q, q0, r, L)
        B = ds.angular_momentum_barrier_factor(q, q0, L)
        print(f"L={L}: F={F}, B={B}")

if __name__ == "__main__":
    # Run examples
    example_breit_wigner()
    example_backend_switching()
    example_parameter_optimization()
    example_utility_functions()
    
    print("\n=== All examples completed! ===")
