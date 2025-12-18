import matplotlib.pyplot as plt
import numpy as np

from decayshape.base import FixedParam
from decayshape.lineshapes import Flatte, RelativisticBreitWigner
from decayshape.particles import Channel, CommonParticles


def main():
    # Define mass range (0.6 to 1.4 GeV covering f0(980))
    # We want to see the cusp at the KK threshold (~0.99 GeV)
    mass = np.linspace(0.6, 1.4, 1000)
    s = mass**2

    # Define channels
    # Channel 1: pi+ pi-
    pipi = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)

    # Channel 2: K+ K-
    kk = Channel(particle1=CommonParticles.K_PLUS, particle2=CommonParticles.K_MINUS)

    print(f"pi+pi- threshold: {pipi.threshold:.4f} GeV")
    print(f"K+K- threshold: {kk.threshold:.4f} GeV")

    # Instantiate Flatté lineshape for f0(980)
    # f0(980) couples strongly to KK, even though it's near/below threshold
    # L=0 (S-wave) for both channels since f0 is scalar (J=0) decaying to pseudoscalars

    # Parameters for f0(980) - illustrative values
    m0 = 0.970
    g_pi = 0.050  # Coupling to pipi
    g_k = 0.050  # Coupling to KK (usually large)

    flatte = Flatte(
        s=s,
        channel1=FixedParam(value=pipi),
        channel2=FixedParam(value=kk),
        pole_mass=m0,
        width1=g_pi,
        width2=g_k,
        r1=1.0,
        r2=1.0,
    )

    # Instantiate Relativistic Breit-Wigner for comparison
    # We try to match the peak and width of the Flatté roughly with a simple BW
    # The simple BW only knows about the pipi channel
    rbw = RelativisticBreitWigner(
        s=s,
        channel=FixedParam(value=pipi),
        pole_mass=m0,
        width=(g_pi + g_k) / 2,  # Crude estimate of effective width
        r=1.0,
    )

    # Evaluate amplitudes
    # f0(980) -> pi pi: J=0, L=0
    # In decayshape convention: angular_momentum = 2 * L = 0, spin = 2 * J = 0

    amp_flatte = flatte(angular_momentum=0, spin=0)
    amp_rbw = rbw(angular_momentum=0, spin=0)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot 1: Intensity
    plt.subplot(1, 2, 1)
    plt.plot(mass, np.abs(amp_flatte) ** 2 / np.sum(np.abs(amp_flatte) ** 2), label="Flatté ($f_0(980)$)", linewidth=2)
    plt.plot(mass, np.abs(amp_rbw) ** 2 / np.sum(np.abs(amp_rbw) ** 2), label="Breit-Wigner", linestyle="--", linewidth=2)

    # Add vertical line for KK threshold
    plt.axvline(x=kk.threshold, color="red", linestyle=":", label="$K^+K^-$ threshold", alpha=0.7)

    plt.xlabel("Mass [GeV/$c^2$]")
    plt.ylabel("Intensity $|A|^2$")
    plt.title("Lineshape Intensity Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Phase (Argand diagram components or just phase vs mass)
    plt.subplot(1, 2, 2)
    plt.plot(mass, np.angle(amp_flatte), label="Flatté", linewidth=2)
    plt.plot(mass, np.angle(amp_rbw), label="Breit-Wigner", linestyle="--", linewidth=2)
    plt.axvline(x=kk.threshold, color="red", linestyle=":", label="$K^+K^-$ threshold", alpha=0.7)

    plt.xlabel("Mass [GeV/$c^2$]")
    plt.ylabel("Phase [radians]")
    plt.title("Phase Motion")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("flatte_vs_breitwigner.png", dpi=150)
    print("Plot saved to flatte_vs_breitwigner.png")


if __name__ == "__main__":
    main()
