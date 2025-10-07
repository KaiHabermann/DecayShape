"""
Utility functions for hadron physics lineshapes.

Contains Blatt-Weiskopf form factors and angular momentum barrier factors
commonly used in amplitude analysis.
"""

from typing import Any, Union

from decayshape import config


def blatt_weiskopf_form_factor(q: Union[float, Any], q0: Union[float, Any], r: float, L: int) -> Union[float, Any]:
    """
    Calculate the Blatt-Weiskopf form factor.

    The Blatt-Weiskopf form factor accounts for the finite size of hadrons
    and depends on the angular momentum L of the decay.

    Args:
        q: Momentum in the decay frame
        q0: Reference momentum (typically at resonance mass)
        r: Hadron radius parameter (in GeV^-1)
        L: Angular momentum of the decay

    Returns:
        Blatt-Weiskopf form factor
    """
    np = config.backend  # Get backend dynamically

    if L == 0:
        return np.ones_like(q)
    elif L == 1:
        return np.sqrt((1 + (r * q0) ** 2) / (1 + (r * q) ** 2))
    elif L == 2:
        return np.sqrt((9 + 3 * (r * q0) ** 2 + (r * q0) ** 4) / (9 + 3 * (r * q) ** 2 + (r * q) ** 4))
    elif L == 3:
        return np.sqrt(
            (225 + 45 * (r * q0) ** 2 + 6 * (r * q0) ** 4 + (r * q0) ** 6)
            / (225 + 45 * (r * q) ** 2 + 6 * (r * q) ** 4 + (r * q) ** 6)
        )
    elif L == 4:
        return np.sqrt(
            (11025 + 1575 * (r * q0) ** 2 + 135 * (r * q0) ** 4 + 10 * (r * q0) ** 6 + (r * q0) ** 8)
            / (11025 + 1575 * (r * q) ** 2 + 135 * (r * q) ** 4 + 10 * (r * q) ** 6 + (r * q) ** 8)
        )
    else:
        raise ValueError(f"Blatt-Weiskopf form factor not implemented for L={L}")


def angular_momentum_barrier_factor(q: Union[float, Any], q0: Union[float, Any], L: int) -> Union[float, Any]:
    """
    Calculate the angular momentum barrier factor.

    The barrier factor accounts for the angular momentum dependence
    of the decay amplitude.

    Args:
        q: Momentum in the decay frame
        q0: Reference momentum (typically at resonance mass)
        L: Angular momentum of the decay

    Returns:
        Angular momentum barrier factor
    """
    np = config.backend  # Get backend dynamically

    if L == 0:
        return np.ones_like(q)
    else:
        return (q / q0) ** L


def relativistic_breit_wigner_denominator(s: Union[float, Any], mass: float, width: float) -> Union[float, Any]:
    """
    Calculate the denominator of a relativistic Breit-Wigner.

    Args:
        s: Mandelstam variable s (mass squared)
        mass: Resonance mass
        width: Resonance width

    Returns:
        Denominator of the Breit-Wigner
    """
    return s - mass**2 + 1j * mass * width


def two_body_breakup_momentum(s: Union[float, Any], m1: float, m2: float) -> Union[float, Any]:
    """
    Calculate the two-body breakup momentum in the center-of-mass frame.

    This is the momentum of each daughter particle in the center-of-mass frame
    of the parent particle decay.

    Args:
        s: Mandelstam variable s (mass squared of the parent)
        m1: Mass of first daughter particle
        m2: Mass of second daughter particle

    Returns:
        Breakup momentum in GeV/c
    """
    np = config.backend  # Get backend dynamically

    # Two-body breakup momentum formula
    # q = sqrt((s - (m1 + m2)^2) * (s - (m1 - m2)^2)) / (2 * sqrt(s))
    return np.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)) / (2 * np.sqrt(s))
