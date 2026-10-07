"""
Normalization Benchmark

Numerically integrates the intensity |A(s)|^2 over s for each lineshape while
varying its physical parameters one at a time (pole mass, width, radius,
couplings, ...). Plots integral vs. parameter value on a log y-axis so that
normalization sensitivities (e.g. to pole position or coupling strength) are
easy to spot.

For the K-matrix, a 2-pole, 2-channel setup is used and the pole positions
and couplings are scanned explicitly.

Usage:
    python benchmark/normalization_scan.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decayshape as ds  # noqa: E402
from decayshape.kmatrix_advanced import KMatrixAdvanced  # noqa: E402
from decayshape.lineshapes import Flatte, GounarisSakurai, RelativisticBreitWigner  # noqa: E402
from decayshape.particles import Channel, CommonParticles  # noqa: E402

ds.set_backend("numpy")

ANGULAR_MOMENTUM = 2  # doubled convention -> L = 1
SPIN = 2

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PIPI = Channel(particle1=CommonParticles.PI_PLUS, particle2=CommonParticles.PI_MINUS)
KK = Channel(particle1=CommonParticles.K_PLUS, particle2=CommonParticles.K_MINUS)

# Common integration grid: mass from just above the pi-pi threshold to well above
# the highest pole mass / barrier-factor tail scanned below, evaluated on a fine
# mass grid (not s grid) so narrow peaks near threshold are resolved as well as
# peaks at higher mass. The upper bound must be wide enough that the barrier-factor
# tails scanned in `r` (which grow with q, i.e. with s) are not artificially cut off.
MASS_MIN = PIPI.threshold + 1e-3
MASS_MAX = 30.0
N_POINTS = 40000
MASS_GRID = np.linspace(MASS_MIN, MASS_MAX, N_POINTS)
S_GRID = MASS_GRID**2


def integrate_intensity(
    lineshape, s_grid=S_GRID, mass_grid=MASS_GRID, angular_momentum=ANGULAR_MOMENTUM, spin=SPIN, **overrides
):
    """
    Integrate |amplitude|^2 over the invariant mass, with optional parameter overrides.

    Integrating over mass (dm) rather than s (ds) matters here: the relativistic
    Breit-Wigner normalization (see relativistic_breit_wigner_normalization) is
    defined so that the *mass* distribution integrates to a constant, following the
    Wikipedia convention (https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution).
    Since s = m^2, ds = 2m dm, so integrating over ds instead of dm reintroduces a
    spurious factor of ~2m (i.e. ~pole_mass) that the amplitude normalization does not
    (and is not meant to) cancel.
    """
    amplitude = lineshape(angular_momentum, spin, s=s_grid, **overrides)
    intensity = np.abs(np.asarray(amplitude)) ** 2
    return np.trapezoid(intensity, mass_grid)


def integrate_intensity_above_threshold(
    lineshape, threshold_sq, s_grid=S_GRID, mass_grid=MASS_GRID, angular_momentum=ANGULAR_MOMENTUM, spin=SPIN, **overrides
):
    """
    Like integrate_intensity, but restricted to s >= threshold_sq.

    For a K-matrix output channel, the amplitude below that channel's own threshold is
    not physically meaningful (no phase space, no unitarity constraint - see
    benchmark/kmatrix_divergence_analysis.py) - integrating through it anyway is exactly
    what manufactured the earlier "exploding" coupling scans. This is the correct fix for
    that case, not a coupling-dependent rescaling.
    """
    amplitude = lineshape(angular_momentum, spin, s=s_grid, **overrides)
    intensity = np.abs(np.asarray(amplitude)) ** 2
    mask = s_grid >= threshold_sq
    return np.trapezoid(intensity[mask], mass_grid[mask])


def run_scan(lineshape, param_name, values):
    """Evaluate the integral for a single parameter swept over `values`."""
    return np.array([integrate_intensity(lineshape, **{param_name: v}) for v in values])


def run_scan_above_threshold(lineshape, param_name, values, threshold_sq):
    """Like run_scan, but integrating only over s >= threshold_sq (see integrate_intensity_above_threshold)."""
    return np.array([integrate_intensity_above_threshold(lineshape, threshold_sq, **{param_name: v}) for v in values])


def plot_lineshape_scans(title, filename, scans):
    """
    scans: list of (label, values, integrals) tuples, one per scanned parameter.
    Produces a figure with one subplot per scanned parameter, log-y.
    """
    n = len(scans)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (label, values, integrals) in zip(axes_flat, scans):
        ax.plot(values, integrals, "o-")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\int |A(s)|^2\,dm$")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def scan_breit_wigner():
    base = RelativisticBreitWigner(s=S_GRID, channel=PIPI, pole_mass=0.775, width=0.15, r=1.0)

    pole_masses = np.linspace(0.4, 1.5, 20)
    widths = np.geomspace(0.01, 0.6, 20)
    radii = np.geomspace(0.2, 8.0, 20)

    scans = [
        ("pole_mass [GeV]", pole_masses, run_scan(base, "pole_mass", pole_masses)),
        ("width [GeV]", widths, run_scan(base, "width", widths)),
        ("r [GeV^-1]", radii, run_scan(base, "r", radii)),
    ]
    plot_lineshape_scans("RelativisticBreitWigner: normalization vs. parameters", "normalization_breit_wigner.png", scans)


def scan_gounaris_sakurai():
    base = GounarisSakurai(s=S_GRID, channel=PIPI, pole_mass=0.775, width=0.15, r=1.0)

    pole_masses = np.linspace(0.4, 1.5, 20)
    widths = np.geomspace(0.01, 0.6, 20)
    radii = np.geomspace(0.2, 8.0, 20)

    scans = [
        ("pole_mass [GeV]", pole_masses, run_scan(base, "pole_mass", pole_masses)),
        ("width [GeV]", widths, run_scan(base, "width", widths)),
        ("r [GeV^-1]", radii, run_scan(base, "r", radii)),
    ]
    plot_lineshape_scans("GounarisSakurai: normalization vs. parameters", "normalization_gounaris_sakurai.png", scans)


def scan_flatte():
    base = Flatte(
        s=S_GRID,
        channel1=PIPI,
        channel2=KK,
        pole_mass=0.98,
        width1=0.2,
        width2=0.6,
        r1=1.0,
        r2=1.0,
    )

    pole_masses = np.linspace(0.4, 1.5, 20)
    widths1 = np.geomspace(0.01, 0.6, 20)
    widths2 = np.geomspace(0.01, 0.6, 20)

    scans = [
        ("pole_mass [GeV]", pole_masses, run_scan(base, "pole_mass", pole_masses)),
        ("width1 (pipi) [GeV]", widths1, run_scan(base, "width1", widths1)),
        ("width2 (KK) [GeV]", widths2, run_scan(base, "width2", widths2)),
    ]
    plot_lineshape_scans("Flatte: normalization vs. parameters", "normalization_flatte.png", scans)


def scan_kmatrix():
    """2-pole, 2-channel K-matrix: scan pole positions and couplings."""
    base = KMatrixAdvanced(
        s=S_GRID,
        channels=[PIPI, KK],
        pole_masses=[0.6, 1.1],
        production_couplings=[1.0, 1.0],
        decay_couplings=[1.0, 0.5, 0.5, 1.0],
        r=1.0,
        output_channel=0,
    )

    # Production couplings are held fixed at O(1) throughout: their overall scale is a
    # redundant degree of freedom (degenerate with an external partial-wave coupling in
    # any real fit amplitude), and their *relative* scale is a genuine, data-constrained
    # physical parameter with no internal bound - neither is a "divergence" for the
    # lineshape itself to fix. See the production_coupling_0 discussion in the
    # conversation this benchmark grew out of.
    output_threshold_sq = PIPI.threshold**2  # output_channel=0 -> integrate only where physical

    pole_mass_0 = np.linspace(0.29, 0.95, 24)  # below/around KK threshold
    pole_mass_1 = np.linspace(0.95, 6.0, 24)  # above KK threshold
    decay_coupling_00 = np.geomspace(0.001, 100.0, 24)  # pole 0 -> pipi
    decay_coupling_11 = np.geomspace(0.001, 100.0, 24)  # pole 1 -> KK

    scans = [
        (
            "pole_mass_0 [GeV]",
            pole_mass_0,
            run_scan_above_threshold(base, "pole_mass_0", pole_mass_0, output_threshold_sq),
        ),
        (
            "pole_mass_1 [GeV]",
            pole_mass_1,
            run_scan_above_threshold(base, "pole_mass_1", pole_mass_1, output_threshold_sq),
        ),
        (
            "decay_coupling_0_0 (pole0->pipi)",
            decay_coupling_00,
            run_scan_above_threshold(base, "decay_coupling_0_0", decay_coupling_00, output_threshold_sq),
        ),
        (
            "decay_coupling_1_1 (pole1->KK)",
            decay_coupling_11,
            run_scan_above_threshold(base, "decay_coupling_1_1", decay_coupling_11, output_threshold_sq),
        ),
    ]
    plot_lineshape_scans(
        "KMatrixAdvanced (2 poles, 2 channels): normalization vs. pole positions & couplings",
        "normalization_kmatrix.png",
        scans,
    )


def main():
    print("Running normalization benchmark...")
    print(f"Integration grid: {N_POINTS} points, mass in [{MASS_MIN:.4f}, {MASS_MAX:.2f}] GeV")

    scan_breit_wigner()
    scan_gounaris_sakurai()
    scan_flatte()
    scan_kmatrix()

    print("Done.")


if __name__ == "__main__":
    main()
