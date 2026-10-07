"""
K-matrix Divergence Analysis

Follow-up to benchmark/normalization_scan.py: that script showed the K-matrix's
normalization integral still grows with certain couplings even after the
coupling_normalization bolt-on. This script uses the sympy reference implementation
(decayshape._lineshapes_sympy) to pin down *why*, then verifies the mechanism
numerically.

Part 1 - exact single-pole closed form
---------------------------------------
For a single K-matrix pole with any number of channels, Sherman-Morrison-Woodbury
(K is rank-1 in the couplings: K_ij = g_i g_j / (m^2 - s)) collapses the whole
matrix construction to an exact multi-channel Breit-Wigner:

    A_out(s) = beta * g_out / (m^2 - s - i * sum_c g_c^2 * rho_tilde_c(s))

This is verified symbolically below by comparing kmatrix_amplitude_raw's output
(which goes through the full 2x2 matrix inversion) against this closed form.

It shows the coupling_normalization heuristic's effective width,
Gamma_R = sum_c g_Rc^2 * rho_c(m_R^2) / m_R, is not a rough guess for a single
pole - it is exactly the pole width, evaluated at the pole instead of general s.

Part 2 - where the divergence actually lives
----------------------------------------------
rho_tilde_c(s) is identically 0 below channel c's threshold (no phase space, no
unitarity damping). So in any mass region below the *output* channel's own
threshold, the amplitude reduces to a pure, undamped pole ~ beta * g_out / (m^2 - s)
- unboundedly growing with g_out, with nothing to stop it. Above threshold, the
same g_out instead broadens (and pointwise suppresses) the resonance. This part
splits the intensity integral at each relevant threshold and tracks the two
pieces separately as a coupling is scaled up, to show the divergence is
overwhelmingly sourced from the below-threshold piece.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decayshape as ds  # noqa: E402
import decayshape._lineshapes_sympy as lss  # noqa: E402
from decayshape.kmatrix_advanced import KMatrixAdvanced  # noqa: E402
from decayshape.particles import Channel, CommonParticles  # noqa: E402

ds.set_backend("numpy")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PIPI = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
KK = Channel(particle1=CommonParticles.K_PLUS, particle2=CommonParticles.K_MINUS)
PIPI_MASSES = (CommonParticles.PI_PLUS.mass, CommonParticles.PI_MINUS.mass)
KK_MASSES = (CommonParticles.K_PLUS.mass, CommonParticles.K_MINUS.mass)


def verify_single_pole_closed_form():
    """Symbolically check A_out(s) = beta*g_out / (m^2 - s - i*sum_c g_c^2*rho_tilde_c(s))
    against the general (matrix-inversion-based) kmatrix_amplitude_raw, for a single pole,
    2 channels, at a handful of numeric (s, couplings) points."""
    g0, g1, beta, m = sp.symbols("g0 g1 beta m", positive=True)
    r = 1.0

    print("Part 1: verifying the exact single-pole closed form")
    print("-" * 60)
    max_rel_diff = 0.0
    for s_val in [0.5, 1.0, 2.0]:
        raw = lss.kmatrix_amplitude_raw(
            sp.Float(s_val),
            [m],
            [beta],
            [g0, g1],
            channel_masses=[PIPI_MASSES, KK_MASSES],
            r=r,
            output_channel=0,
            angular_momentum=0,
        )
        rho_tilde_0 = (
            lss.phase_space_factor(sp.Float(s_val), *PIPI_MASSES)
            * lss.channel_n_factor(sp.Float(s_val), m**2, *PIPI_MASSES, r, 0) ** 2
        )
        rho_tilde_1 = (
            lss.phase_space_factor(sp.Float(s_val), *KK_MASSES)
            * lss.channel_n_factor(sp.Float(s_val), m**2, *KK_MASSES, r, 0) ** 2
        )
        closed_form = beta * g0 / (m**2 - s_val - sp.I * (g0**2 * rho_tilde_0 + g1**2 * rho_tilde_1))

        for g0_val, g1_val, beta_val, m_val in [(0.3, 0.9, 0.5, 0.775), (2.0, 0.1, 1.5, 1.2), (5.0, 5.0, 0.7, 0.9)]:
            subs = {g0: g0_val, g1: g1_val, beta: beta_val, m: m_val}
            lhs = complex(raw.subs(subs).evalf())
            rhs = complex(closed_form.subs(subs).evalf())
            rel_diff = abs(lhs - rhs) / (abs(rhs) + 1e-30)
            max_rel_diff = max(max_rel_diff, rel_diff)

    print(f"Max relative difference between general and closed-form amplitude: {max_rel_diff:.2e}")
    print("(should be ~0 - confirms the closed form is exact, not approximate)\n")


def threshold_split_scan(param_name, values, output_channel, threshold_sq, xlabel):
    """
    Build the benchmark's 2-pole/2-channel K-matrix, override `param_name` over `values`,
    and integrate the output channel's intensity separately below/above `threshold_sq`.
    """
    mass_grid = np.linspace(PIPI.threshold + 1e-3, 14.0, 40000)
    s_grid = mass_grid**2
    below = s_grid < threshold_sq
    above = ~below

    base = KMatrixAdvanced(
        s=s_grid,
        channels=[PIPI, KK],
        pole_masses=[0.6, 1.1],
        production_couplings=[1.0, 1.0],
        decay_couplings=[1.0, 0.5, 0.5, 1.0],
        r=1.0,
        output_channel=output_channel,
    )

    below_totals = np.empty(len(values))
    above_totals = np.empty(len(values))
    for i, v in enumerate(values):
        amp = base(0, 0, **{param_name: v})  # L=0 to isolate the effect from barrier-factor growth
        intensity = np.abs(np.asarray(amp)) ** 2
        below_totals[i] = np.trapezoid(intensity[below], mass_grid[below])
        above_totals[i] = np.trapezoid(intensity[above], mass_grid[above])

    return below_totals, above_totals


def plot_threshold_split(title, filename, param_name, values, xlabel, output_channel, threshold_sq, split_label="split point"):
    below_totals, above_totals = threshold_split_scan(param_name, values, output_channel, threshold_sq, xlabel)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(values, below_totals, "o-", label=f"below {split_label}")
    ax.plot(values, above_totals, "o-", label=f"above {split_label}")
    ax.plot(values, below_totals + above_totals, "k--", alpha=0.6, label="total")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\int |A(s)|^2\,dm$ (partial)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    verify_single_pole_closed_form()

    print("Part 2: below/above-threshold split of the divergent scans")
    print("-" * 60)

    # decay_coupling_1_1: pole 1's own coupling to KK, output channel = KK (channel 1).
    # Divergence should live below the KK threshold.
    plot_threshold_split(
        "decay_coupling_1_1 (pole1->KK), output=KK: below vs above KK threshold",
        "kmatrix_divergence_decay_coupling_1_1.png",
        "decay_coupling_1_1",
        np.geomspace(0.05, 100.0, 20),
        "decay_coupling_1_1",
        output_channel=1,
        threshold_sq=KK.threshold**2,
        split_label="KK threshold",
    )

    # decay_coupling_0_0: pole 0's own coupling to pipi, output channel = pipi (channel 0).
    # The mass grid never goes below the pipi threshold (it's the grid's own lower bound),
    # so rho_tilde_pipi is never exactly 0 here - unlike the KK case above. Split instead at
    # pole 0's own mass (0.6 GeV): rho_tilde_pipi is still small (but nonzero) just above
    # threshold, so this checks whether the divergence instead concentrates in that
    # low-(but not zero)-phase-space region rather than needing an exact threshold cutoff.
    plot_threshold_split(
        "decay_coupling_0_0 (pole0->pipi), output=pipi: below vs above pole0's mass",
        "kmatrix_divergence_decay_coupling_0_0.png",
        "decay_coupling_0_0",
        np.geomspace(0.05, 100.0, 20),
        "decay_coupling_0_0",
        output_channel=0,
        threshold_sq=0.6**2,
        split_label="pole0 mass (0.6 GeV)",
    )

    # production_coupling_0: never appears in K at all, so nothing should regulate it -
    # expect it to diverge in BOTH regions (unlike the two cases above), since the
    # divergence mechanism here isn't threshold-related at all.
    plot_threshold_split(
        "production_coupling_0, output=pipi: below vs above pole0 mass",
        "kmatrix_divergence_production_coupling_0.png",
        "production_coupling_0",
        np.geomspace(0.05, 100.0, 20),
        "production_coupling_0",
        output_channel=0,
        threshold_sq=0.6**2,
        split_label="pole0 mass (0.6 GeV)",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
