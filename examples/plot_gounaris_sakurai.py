import matplotlib.pyplot as plt
import numpy as np

from decayshape.base import FixedParam
from decayshape.lineshapes import GounarisSakurai, RelativisticBreitWigner
from decayshape.particles import Channel, CommonParticles

# Define mass range (0.3 to 1.2 GeV covering rho(770))
mass = np.linspace(0.3, 1.2, 500)
s = mass**2

# Define channel (pi+ pi-)
pipi = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)

# Instantiate Gounaris-Sakurai lineshape
# Note: GS is specifically designed for P-wave (L=1) resonances like rho
gs = GounarisSakurai(s=s, channel=FixedParam(value=pipi), pole_mass=0.775, width=0.150)

# Instantiate Relativistic Breit-Wigner for comparison
rbw = RelativisticBreitWigner(
    s=s,
    channel=FixedParam(value=pipi),
    pole_mass=0.775,
    width=0.150,
    r=1.0,  # Standard interaction radius ~1 fm ~ 5 GeV^-1, but using default 1.0 from code
)

# Evaluate amplitudes
# For rho(770) -> pi pi:
# Angular momentum L=1 (P-wave). In decayshape convention, angular_momentum = 2 * L = 2
# Spin J=1. In decayshape convention, spin = 2 * J = 2

amp_gs = gs(angular_momentum=2, spin=2)
amp_rbw = rbw(angular_momentum=2, spin=2)

# Plot intensity |A|^2
plt.figure(figsize=(10, 6))

# Plot absolute squared amplitudes (intensity)
plt.plot(mass, np.abs(amp_gs) ** 2, label="Gounaris-Sakurai", linewidth=2)
plt.plot(mass, np.abs(amp_rbw) ** 2, label="Relativistic Breit-Wigner", linestyle="--", linewidth=2)

plt.xlabel("Mass [GeV/$c^2$]")
plt.ylabel("Intensity $|A|^2$")
plt.title(r"$\rho(770) \to \pi^+ \pi^-$ Lineshape Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0.3, 1.2)

plt.tight_layout()
plt.savefig("gounaris_sakurai_rho.png", dpi=150)
print("Plot saved to gounaris_sakurai_rho.png")
