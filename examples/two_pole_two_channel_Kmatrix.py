"""
Example: Two-channel, two-pole K-matrix with threshold effects.

This example demonstrates a K-matrix with:
- Two poles: one below ππ threshold, one above KK threshold
- Two channels: ππ and KK
- Threshold effects when channels open/close
"""

import numpy as np
import matplotlib.pyplot as plt
import decayshape as ds
from decayshape.particles import CommonParticles

def create_threshold_kmatrix():
    """
    Create a K-matrix that demonstrates threshold effects.
    
    Setup:
    - Pole 1: 0.6 GeV (below ππ threshold ~0.28 GeV)
    - Pole 2: 1.0 GeV (above KK threshold ~0.98 GeV)
    - Channel 1: ππ (threshold ~0.28 GeV)
    - Channel 2: KK (threshold ~0.98 GeV)
    """
    
    # Create s values from below ππ threshold to above KK threshold
    s_values = np.linspace(0.1, 1.5, 200)**2
    
    # Create channels
    pipi_channel = ds.Channel(
        particle1=CommonParticles.PI_PLUS,
        particle2=CommonParticles.PI_MINUS
    )
    
    kk_channel = ds.Channel(
        particle1=CommonParticles.K_PLUS,
        particle2=CommonParticles.K_MINUS
    )
    
    # Create K-matrix with two poles and two channels
    kmat = ds.KMatrixAdvanced(
        s=s_values,
        channels=[pipi_channel, kk_channel],
        pole_masses=[0.6, 1.0],  # First pole below ππ threshold, second above KK threshold
        production_couplings=[1.0, 0.8],  # Different production strengths
        decay_couplings=[1.0, 0.5, 0.3, 0.7],  # 2 poles × 2 channels = 4 couplings
        output_channel=0  # Will be overridden for different channels
    )
    
    return kmat, s_values

def plot_kmatrix_channels():
    """Plot the K-matrix for both channels showing threshold effects."""
    
    # Create the K-matrix
    kmat, s_values = create_threshold_kmatrix()
    mass_values = s_values**0.5
    # Calculate for both channels
    print("Calculating K-matrix for both channels...")
    
    # Channel 0 (ππ)
    kmat_0 = ds.KMatrixAdvanced(
        s=s_values,
        channels=kmat.channels,
        pole_masses=kmat.pole_masses,
        production_couplings=kmat.production_couplings,
        decay_couplings=kmat.decay_couplings,
        output_channel=0
    )
    result_0 = kmat_0()
    
    # Channel 1 (KK)
    kmat_1 = ds.KMatrixAdvanced(
        s=s_values,
        channels=kmat.channels,
        pole_masses=kmat.pole_masses,
        production_couplings=kmat.production_couplings,
        decay_couplings=kmat.decay_couplings,
        output_channel=1
    )
    result_1 = kmat_1()
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Magnitude of both channels
    ax1.plot(mass_values, np.abs(result_0) * kmat.channels.value[0].phase_space_factor(s_values), 'b-', linewidth=2, label='ππ channel')
    ax1.plot(mass_values, np.abs(result_1) * kmat.channels.value[1].phase_space_factor(s_values), 'r-', linewidth=2, label='KK channel')
    ax1.axvline(0.28, color='b', linestyle='--', alpha=0.7, label='ππ threshold')
    ax1.axvline(0.98, color='r', linestyle='--', alpha=0.7, label='KK threshold')
    ax1.axvline(0.6, color='g', linestyle=':', alpha=0.7, label='Pole 1 (0.6 GeV)')
    ax1.axvline(1.0, color='orange', linestyle=':', alpha=0.7, label='Pole 2 (1.0 GeV)')
    ax1.set_xlabel('s (GeV²)')
    ax1.set_ylabel('|F|')
    ax1.set_title('K-matrix Magnitude - Both Channels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # ax1.set_xlim(0.1, 1.5)
    
    # Plot 2: Real part of both channels
    ax2.plot(mass_values, np.real(result_0), 'b-', linewidth=2, label='ππ channel')
    ax2.plot(mass_values, np.real(result_1), 'r-', linewidth=2, label='KK channel')
    ax2.axvline(0.28, color='b', linestyle='--', alpha=0.7, label='ππ threshold')
    ax2.axvline(0.98, color='r', linestyle='--', alpha=0.7, label='KK threshold')
    ax2.axvline(0.6, color='g', linestyle=':', alpha=0.7, label='Pole 1 (0.6 GeV)')
    ax2.axvline(1.0, color='orange', linestyle=':', alpha=0.7, label='Pole 2 (1.0 GeV)')
    ax2.set_xlabel('s (GeV²)')
    ax2.set_ylabel('Re(F)')
    ax2.set_title('K-matrix Real Part - Both Channels')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # ax2.set_xlim(0.1, 1.5)
    
    # Plot 3: Imaginary part of both channels
    ax3.plot(mass_values, np.imag(result_0), 'b-', linewidth=2, label='ππ channel')
    ax3.plot(mass_values, np.imag(result_1), 'r-', linewidth=2, label='KK channel')
    ax3.axvline(0.28, color='b', linestyle='--', alpha=0.7, label='ππ threshold')
    ax3.axvline(0.98, color='r', linestyle='--', alpha=0.7, label='KK threshold')
    ax3.axvline(0.6, color='g', linestyle=':', alpha=0.7, label='Pole 1 (0.6 GeV)')
    ax3.axvline(1.0, color='orange', linestyle=':', alpha=0.7, label='Pole 2 (1.0 GeV)')
    ax3.set_xlabel('s (GeV²)')
    ax3.set_ylabel('Im(F)')
    ax3.set_title('K-matrix Imaginary Part - Both Channels')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # ax3.set_xlim(0.1, 1.5)
    
    # Plot 4: Phase of both channels
    phase_0 = np.angle(result_0)
    phase_1 = np.angle(result_1)
    ax4.plot(mass_values, phase_0, 'b-', linewidth=2, label='ππ channel')
    ax4.plot(mass_values, phase_1, 'r-', linewidth=2, label='KK channel')
    ax4.axvline(0.28, color='b', linestyle='--', alpha=0.7, label='ππ threshold')
    ax4.axvline(0.98, color='r', linestyle='--', alpha=0.7, label='KK threshold')
    ax4.axvline(0.6, color='g', linestyle=':', alpha=0.7, label='Pole 1 (0.6 GeV)')
    ax4.axvline(1.0, color='orange', linestyle=':', alpha=0.7, label='Pole 2 (1.0 GeV)')
    ax4.set_xlabel('s (GeV²)')
    ax4.set_ylabel('Phase (rad)')
    ax4.set_title('K-matrix Phase - Both Channels')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.1, 1.5)
    
    plt.tight_layout()
    plt.show()
    
    return result_0, result_1, s_values

def analyze_threshold_effects(result_0, result_1, s_values):
    """Analyze the threshold effects in the K-matrix."""
    
    print("\n=== Threshold Effects Analysis ===")
    
    # Find indices for key points
    pipi_threshold = 0.28
    kk_threshold = 0.98
    pole_1 = 0.6
    pole_2 = 1.0
    
    # Find closest indices
    idx_pipi = np.argmin(np.abs(s_values - pipi_threshold))
    idx_kk = np.argmin(np.abs(s_values - kk_threshold))
    idx_pole_1 = np.argmin(np.abs(s_values - pole_1))
    idx_pole_2 = np.argmin(np.abs(s_values - pole_2))
    
    print(f"ππ threshold (s = {pipi_threshold:.2f} GeV²):")
    print(f"  ππ channel: |F| = {np.abs(result_0[idx_pipi]):.4f}, phase = {np.angle(result_0[idx_pipi]):.4f} rad")
    print(f"  KK channel: |F| = {np.abs(result_1[idx_pipi]):.4f}, phase = {np.angle(result_1[idx_pipi]):.4f} rad")
    
    print(f"\nKK threshold (s = {kk_threshold:.2f} GeV²):")
    print(f"  ππ channel: |F| = {np.abs(result_0[idx_kk]):.4f}, phase = {np.angle(result_0[idx_kk]):.4f} rad")
    print(f"  KK channel: |F| = {np.abs(result_1[idx_kk]):.4f}, phase = {np.angle(result_1[idx_kk]):.4f} rad")
    
    print(f"\nPole 1 (s = {pole_1:.2f} GeV²):")
    print(f"  ππ channel: |F| = {np.abs(result_0[idx_pole_1]):.4f}, phase = {np.angle(result_0[idx_pole_1]):.4f} rad")
    print(f"  KK channel: |F| = {np.abs(result_1[idx_pole_1]):.4f}, phase = {np.angle(result_1[idx_pole_1]):.4f} rad")
    
    print(f"\nPole 2 (s = {pole_2:.2f} GeV²):")
    print(f"  ππ channel: |F| = {np.abs(result_0[idx_pole_2]):.4f}, phase = {np.angle(result_0[idx_pole_2]):.4f} rad")
    print(f"  KK channel: |F| = {np.abs(result_1[idx_pole_2]):.4f}, phase = {np.angle(result_1[idx_pole_2]):.4f} rad")
    
    # Analyze behavior around thresholds
    print(f"\n=== Channel Behavior Analysis ===")
    
    # Below ππ threshold
    below_pipi = s_values < pipi_threshold
    print(f"Below ππ threshold (s < {pipi_threshold:.2f}):")
    print(f"  ππ channel: complex = {np.any(np.imag(result_0[below_pipi]) != 0)}")
    print(f"  KK channel: complex = {np.any(np.imag(result_1[below_pipi]) != 0)}")
    
    # Between thresholds
    between_thresholds = (s_values >= pipi_threshold) & (s_values < kk_threshold)
    print(f"Between thresholds ({pipi_threshold:.2f} ≤ s < {kk_threshold:.2f}):")
    print(f"  ππ channel: complex = {np.any(np.imag(result_0[between_thresholds]) != 0)}")
    print(f"  KK channel: complex = {np.any(np.imag(result_1[between_thresholds]) != 0)}")
    
    # Above KK threshold
    above_kk = s_values >= kk_threshold
    print(f"Above KK threshold (s ≥ {kk_threshold:.2f}):")
    print(f"  ππ channel: complex = {np.any(np.imag(result_0[above_kk]) != 0)}")
    print(f"  KK channel: complex = {np.any(np.imag(result_1[above_kk]) != 0)}")

def main():
    """Main function to run the example."""
    print("=== Two-Channel, Two-Pole K-Matrix Example ===")
    print("Demonstrating threshold effects in coupled-channel analysis")
    print()
    
    # Create and plot the K-matrix
    result_0, result_1, s_values = plot_kmatrix_channels()
    
    # Analyze the threshold effects
    analyze_threshold_effects(result_0, result_1, s_values)
    
    print("\n=== Example completed! ===")
    print("The plots show:")
    print("1. Magnitude: How the K-matrix amplitude changes with energy")
    print("2. Real part: Dispersion relation effects")
    print("3. Imaginary part: Absorption effects")
    print("4. Phase: Phase shifts due to resonances and thresholds")
    print()
    print("Key features to observe:")
    print("- Threshold effects when channels open/close")
    print("- Resonance behavior near poles")
    print("- Different behavior in different channels")
    print("- Complex behavior below thresholds")

if __name__ == "__main__":
    main()