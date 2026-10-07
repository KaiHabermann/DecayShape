"""
Sympy reference implementation of the K-matrix lineshape (internal/experimental).

This mirrors ``decayshape.kmatrix_advanced.KMatrixAdvanced`` term-for-term, but builds a
symbolic expression instead of evaluating numpy/jax arrays. The point is to be able to
manipulate the amplitude analytically (limits, series expansions, residues) to better
understand where its normalization diverges - something that's awkward to do from the
numeric implementation alone.

Not part of the public API yet: only the K-matrix construction is implemented, and it does
not (yet) support the ``background`` / ``production_background`` terms of KMatrixAdvanced.
Its numeric equivalence to KMatrixAdvanced (for the terms it does implement) is checked in
tests/test_kmatrix_sympy.py for up to 3 poles and 3 channels.

Construction (matching KMatrixAdvanced.function / _build_k_matrix / _build_p_vector /
_build_amplitude exactly):

    K_ij(s)        = sum_R g_Ri * g_Rj / (m_R^2 - s)
    P_i(s)         = sum_R beta_R * g_Ri / (m_R^2 - s)
    s_0            = mean(pole_masses)^2
    rho_tilde_i(s) = rho_i(s) * n_i(s, s_0, r)^2
    T(s)           = (I - i * K(s) @ diag(rho_tilde(s)))^-1
    A(s)           = T(s) @ P(s)
    raw(s)         = A_output(s) * B(s)

where B(s) is the outer angular-momentum barrier factor times Blatt-Weiskopf form factor
for the output channel, evaluated at q(s) with reference momentum q(s_0).

``kmatrix_amplitude_raw`` returns exactly ``raw(s)`` above. ``kmatrix_amplitude`` is what
actually matches KMatrixAdvanced's current output: KMatrixAdvanced.function additionally
multiplies by a real, s-independent ``coupling_normalization`` scalar (added to keep the
intensity integral from exploding as couplings are scaled up - see
KMatrixAdvanced._coupling_normalization for the physics rationale), so
``kmatrix_amplitude = kmatrix_amplitude_raw * coupling_normalization``. Both are exposed
separately since ``kmatrix_amplitude_raw`` - the bare physics, without that bolt-on - is
the more useful one for analyzing where the divergent behavior actually originates.
"""

from collections.abc import Sequence
from typing import Optional

import sympy as sp


def channel_momentum(s, m1, m2):
    """Two-body breakup momentum q(s). Matches decayshape.particles.Channel.momentum."""
    s_plus = s - (m1 + m2) ** 2
    s_minus = s - (m1 - m2) ** 2
    return sp.sqrt(s_plus) * sp.sqrt(s_minus) / (2 * sp.sqrt(s))


def phase_space_factor(s, m1, m2):
    """rho(s) = 2q/sqrt(s) above threshold, 0 below. Matches Channel.phase_space_factor."""
    threshold_sq = (m1 + m2) ** 2
    q = channel_momentum(s, m1, m2)
    return sp.Piecewise((sp.Integer(0), s < threshold_sq), (2 * q / sp.sqrt(s), True))


def blatt_weiskopf_form_factor(q, r, angular_momentum: int):
    """Matches decayshape.utils.blatt_weiskopf_form_factor (damps with q, standard convention)."""
    x = q * r
    if angular_momentum == 0:
        return sp.Integer(1)
    elif angular_momentum == 1:
        return 1 / sp.sqrt(1 + x**2)
    elif angular_momentum == 2:
        return 1 / sp.sqrt(9 + 3 * x**2 + x**4)
    elif angular_momentum == 3:
        return 1 / sp.sqrt(225 + 45 * x**2 + 6 * x**4 + x**6)
    elif angular_momentum == 4:
        return 1 / sp.sqrt(11025 + 1575 * x**2 + 135 * x**4 + 10 * x**6 + x**8)
    raise ValueError(f"Blatt-Weiskopf form factor not implemented for L={angular_momentum}")


def angular_momentum_barrier_factor(q, q0, angular_momentum: int):
    """Matches decayshape.utils.angular_momentum_barrier_factor."""
    if angular_momentum == 0:
        return sp.Integer(1)
    return (q / q0) ** angular_momentum


def channel_n_factor(s, s0, m1, m2, r, angular_momentum: int):
    """Matches decayshape.particles.Channel.n (barrier factor * form factor, q0 = q(s0))."""
    q = channel_momentum(s, m1, m2)
    q0 = channel_momentum(s0, m1, m2)
    return angular_momentum_barrier_factor(q, q0, angular_momentum) * blatt_weiskopf_form_factor(q, r, angular_momentum)


def kmatrix_amplitude_raw(
    s,
    pole_masses: Sequence,
    production_couplings: Sequence,
    decay_couplings: Sequence,
    channel_masses: Sequence[tuple],
    r,
    output_channel: int = 0,
    angular_momentum: int = 0,
    channel_angular_momenta: Optional[Sequence[int]] = None,
    q0: Optional[object] = None,
):
    """
    Build the symbolic, un-normalized K-matrix amplitude for ``output_channel``.

    Args:
        s: sympy symbol/expression for the Mandelstam variable (mass squared)
        pole_masses: pole masses, length n_poles
        production_couplings: production couplings beta_R, length n_poles
        decay_couplings: decay couplings g_Rc, flattened as
            [pole0_chan0, pole0_chan1, ..., pole1_chan0, ...] (n_poles * n_channels),
            matching KMatrixAdvanced's flat layout
        channel_masses: [(m1, m2), ...], one (daughter mass) pair per channel
        r: Blatt-Weiskopf radius
        output_channel: which channel's amplitude to return (0-indexed)
        angular_momentum: doubled angular momentum for the *outer* barrier/form factor
            only, matching KMatrixAdvanced's convention (actual L = angular_momentum // 2)
        channel_angular_momenta: doubled angular momentum *per channel*, used for the
            internal unitarization n-factor - this matches ``Channel.n``'s own ``l`` field,
            which is independent of the call-time ``angular_momentum`` and defaults to 0
            for every channel (KMatrixAdvanced never overrides it). Length n_channels;
            defaults to all-zero, matching decayshape.particles.Channel's default ``l=0``.
        q0: reference momentum for the outer barrier factor; computed from
            sqrt(mean(pole_masses)^2) if not given, matching KMatrixAdvanced

    Returns:
        sympy expression for the amplitude, as a function of ``s``
    """
    n_poles = len(pole_masses)
    n_channels = len(channel_masses)

    if len(production_couplings) != n_poles:
        raise ValueError(f"production_couplings must have length {n_poles}, got {len(production_couplings)}")
    if len(decay_couplings) != n_poles * n_channels:
        raise ValueError(f"decay_couplings must have length {n_poles * n_channels}, got {len(decay_couplings)}")
    if not (0 <= output_channel < n_channels):
        raise ValueError(f"output_channel must be between 0 and {n_channels - 1}, got {output_channel}")
    if channel_angular_momenta is None:
        channel_angular_momenta = [0] * n_channels
    elif len(channel_angular_momenta) != n_channels:
        raise ValueError(f"channel_angular_momenta must have length {n_channels}, got {len(channel_angular_momenta)}")

    g = [[decay_couplings[pole_idx * n_channels + c] for c in range(n_channels)] for pole_idx in range(n_poles)]

    # K_ij(s) and P_i(s)
    K = sp.zeros(n_channels, n_channels)
    P = sp.zeros(n_channels, 1)
    for pole_idx in range(n_poles):
        denom = pole_masses[pole_idx] ** 2 - s
        for i in range(n_channels):
            P[i, 0] += production_couplings[pole_idx] * g[pole_idx][i] / denom
            for j in range(n_channels):
                K[i, j] += g[pole_idx][i] * g[pole_idx][j] / denom

    L = angular_momentum // 2
    s_0 = (sum(pole_masses) / n_poles) ** 2

    # rho_tilde_i(s) = rho_i(s) * n_i(s, s_0, r)^2, as a diagonal matrix. n_i uses the
    # channel's OWN angular momentum (channel_angular_momenta[i] // 2), not L.
    rho_tilde = sp.zeros(n_channels, n_channels)
    for i, (m1, m2) in enumerate(channel_masses):
        L_i = channel_angular_momenta[i] // 2
        rho_i = phase_space_factor(s, m1, m2)
        n_i = channel_n_factor(s, s_0, m1, m2, r, L_i)
        rho_tilde[i, i] = rho_i * n_i**2

    identity = sp.eye(n_channels)
    T = (identity - sp.I * K * rho_tilde).inv()
    A = T * P

    m1_out, m2_out = channel_masses[output_channel]
    if q0 is None:
        q0 = channel_momentum(s_0, m1_out, m2_out)
    q_out = channel_momentum(s, m1_out, m2_out)

    B = angular_momentum_barrier_factor(q_out, q0, L) * blatt_weiskopf_form_factor(q_out, r, L)

    return A[output_channel, 0] * B


def relativistic_breit_wigner_normalization(mass, width):
    """Matches decayshape.utils.relativistic_breit_wigner_normalization."""
    gamma = sp.sqrt(mass**2 * (mass**2 + width**2))
    return (2 * sp.sqrt(2) * mass * width * gamma) / (sp.pi * sp.sqrt(mass**2 + gamma))


def coupling_normalization(
    pole_masses: Sequence,
    production_couplings: Sequence,
    decay_couplings: Sequence,
    channel_masses: Sequence[tuple],
    output_channel: int = 0,
):
    """
    Matches KMatrixAdvanced._coupling_normalization exactly: a real, s-independent scalar
    (so it is a function of the pole masses / couplings only, never of ``s``).
    """
    n_poles = len(pole_masses)
    n_channels = len(channel_masses)
    g = [[decay_couplings[pole_idx * n_channels + c] for c in range(n_channels)] for pole_idx in range(n_poles)]

    epsilon = sp.Float("1e-12")
    factors = []
    for pole_idx in range(n_poles):
        m_r = pole_masses[pole_idx]

        gamma_r = sp.Integer(0)
        for channel_idx, (m1, m2) in enumerate(channel_masses):
            rho_c = phase_space_factor(m_r**2, m1, m2)
            gamma_r += g[pole_idx][channel_idx] ** 2 * rho_c
        gamma_r = gamma_r / m_r
        gamma_r = sp.Max(gamma_r, epsilon)

        norm_r = relativistic_breit_wigner_normalization(m_r, gamma_r)
        coupling_scale = sp.Abs(production_couplings[pole_idx] * g[pole_idx][output_channel]) + epsilon
        factors.append(sp.sqrt(norm_r) / coupling_scale)

    product = sp.Integer(1)
    for factor in factors:
        product *= factor
    return product ** (sp.Rational(1, n_poles))


def kmatrix_amplitude(
    s,
    pole_masses: Sequence,
    production_couplings: Sequence,
    decay_couplings: Sequence,
    channel_masses: Sequence[tuple],
    r,
    output_channel: int = 0,
    angular_momentum: int = 0,
    channel_angular_momenta: Optional[Sequence[int]] = None,
    q0: Optional[object] = None,
):
    """
    KMatrixAdvanced's actual current output: ``kmatrix_amplitude_raw`` rescaled by
    ``coupling_normalization`` (see module docstring). Arguments match
    ``kmatrix_amplitude_raw``.
    """
    raw = kmatrix_amplitude_raw(
        s,
        pole_masses=pole_masses,
        production_couplings=production_couplings,
        decay_couplings=decay_couplings,
        channel_masses=channel_masses,
        r=r,
        output_channel=output_channel,
        channel_angular_momenta=channel_angular_momenta,
        angular_momentum=angular_momentum,
        q0=q0,
    )
    norm = coupling_normalization(pole_masses, production_couplings, decay_couplings, channel_masses, output_channel)
    return raw * norm
