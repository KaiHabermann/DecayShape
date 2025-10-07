"""
Lineshape implementations for hadron physics.

Contains various lineshapes commonly used in amplitude analysis:
- Relativistic Breit-Wigner
- Flatté
- K-matrix
"""

from typing import Any, Optional, Union

from pydantic import Field

from decayshape import config

from .base import FixedParam, Lineshape
from .particles import Channel
from .utils import angular_momentum_barrier_factor, blatt_weiskopf_form_factor, relativistic_breit_wigner_denominator


class RelativisticBreitWigner(Lineshape):
    """
    Relativistic Breit-Wigner lineshape.

    The most common lineshape for hadron resonances, accounting for
    the finite width and relativistic effects.
    """

    # Fixed parameters (don't change during optimization)
    channel: FixedParam[Channel] = Field(..., description="Decay channel for the resonance")

    # Optimization parameters
    pole_mass: float = Field(default=0.775, description="Pole mass of the resonance")
    width: float = Field(default=0.15, description="Resonance width")
    r: float = Field(default=1.0, description="Hadron radius parameter for Blatt-Weiskopf form factor")
    q0: Optional[float] = Field(default=None, description="Reference momentum (calculated from channel if None)")

    @property
    def parameter_order(self) -> list[str]:
        """Return the order of parameters for positional arguments."""
        return ["pole_mass", "width", "r", "q0"]

    def function(self, angular_momentum, spin, s, *args, **kwargs) -> Union[float, Any]:
        """
        Evaluate the Relativistic Breit-Wigner at given s values.

        Args:
            angular_momentum: Angular momentum parameter (doubled values: 0, 2, 4, ...)
            spin: Spin parameter (doubled values: 1, 3, 5, ...)
            s: Mandelstam variable s (mass squared) or array of s values
            *args: Positional parameter overrides (width, r, q0)
            **kwargs: Keyword parameter overrides

        Returns:
            Breit-Wigner amplitude
        """
        # Get parameters with overrides
        params = self._get_parameters(*args, **kwargs)

        if params["q0"] is None:
            params["q0"] = self.channel.value.momentum(params["pole_mass"] ** 2)

        # Calculate momentum in the decay frame using channel masses
        q = self.channel.momentum(s)

        # Convert doubled angular momentum to actual L value
        L = angular_momentum // 2

        # Blatt-Weiskopf form factor
        F = blatt_weiskopf_form_factor(q, params["q0"], params["r"], L)

        # Angular momentum barrier factor
        B = angular_momentum_barrier_factor(q, params["q0"], L)

        # Breit-Wigner denominator (use optimization parameter pole_mass)
        denominator = relativistic_breit_wigner_denominator(s, params["pole_mass"], params["width"])

        return F * B / denominator

    def __call__(self, angular_momentum, spin, *args, s=None, **kwargs) -> Union[float, Any]:
        # Resolve s: prefer call-time s, else field value
        s_val = s if s is not None else (self.s.value if self.s is not None else None)
        if s_val is None:
            raise ValueError("s must be provided either at construction or call time")
        return self.function(angular_momentum, spin, s_val, *args, **kwargs)


class Flatte(Lineshape):
    """
    Flatté lineshape for coupled-channel resonances.

    Used for resonances that can decay into multiple channels,
    such as the f0(980) which couples to both ππ and KK.
    """

    # Fixed parameters (don't change during optimization)
    channel1: FixedParam[Channel] = Field(..., description="First decay channel")
    channel2: FixedParam[Channel] = Field(..., description="Second decay channel")

    # Optimization parameters
    pole_mass: float = Field(description="Pole mass of the resonance")
    width1: float = Field(description="Width for first channel")
    width2: float = Field(description="Width for second channel")
    r1: float = Field(description="Hadron radius for first channel")
    r2: float = Field(description="Hadron radius for second channel")
    q01: Optional[float] = Field(default=None, description="Reference momentum for first channel")
    q02: Optional[float] = Field(default=None, description="Reference momentum for second channel")

    @property
    def parameter_order(self) -> list[str]:
        """Return the order of parameters for positional arguments."""
        return ["pole_mass", "width1", "width2", "r1", "r2", "q01", "q02"]

    def model_post_init(self, __context):
        """Post-initialization to set q01, q02 if not provided."""
        if self.q01 is None:
            self.q01 = self.pole_mass / 2.0
        if self.q02 is None:
            self.q02 = self.pole_mass / 2.0

    def function(self, angular_momentum, spin, s, *args, **kwargs) -> Union[float, Any]:
        """
        Evaluate the Flatté lineshape at given s values.

        Args:
            angular_momentum: Angular momentum parameter (doubled values: 0, 2, 4, ...)
            spin: Spin parameter (doubled values: 1, 3, 5, ...)
            s: Mandelstam variable s (mass squared) or array of s values
            *args: Positional parameter overrides (width1, width2, r1, r2, q01, q02)
            **kwargs: Keyword parameter overrides

        Returns:
            Flatté amplitude
        """
        # Get parameters with overrides
        params = self._get_parameters(*args, **kwargs)

        np = config.backend  # Get backend dynamically

        # Calculate momenta in both channels using Channel objects
        # Channel 1 momentum
        channel1 = self.channel1.value
        m1_1 = channel1.particle1.value.mass
        m1_2 = channel1.particle2.value.mass
        q1 = np.sqrt((s - (m1_1 + m1_2) ** 2) * (s - (m1_1 - m1_2) ** 2)) / (2 * np.sqrt(s))

        # Channel 2 momentum
        channel2 = self.channel2.value
        m2_1 = channel2.particle1.value.mass
        m2_2 = channel2.particle2.value.mass
        q2 = np.sqrt((s - (m2_1 + m2_2) ** 2) * (s - (m2_1 - m2_2) ** 2)) / (2 * np.sqrt(s))

        # Convert doubled angular momentum to actual L value
        L = angular_momentum // 2

        # Form factors and barrier factors for both channels
        F1 = blatt_weiskopf_form_factor(q1, params["q01"], params["r1"], L)
        F2 = blatt_weiskopf_form_factor(q2, params["q02"], params["r2"], L)
        B1 = angular_momentum_barrier_factor(q1, params["q01"], L)
        B2 = angular_momentum_barrier_factor(q2, params["q02"], L)

        # Total width
        total_width = params["width1"] * F1 * B1 + params["width2"] * F2 * B2

        # Flatté denominator (use optimization parameter pole_mass)
        denominator = s - params["pole_mass"] ** 2 + 1j * params["pole_mass"] * total_width

        return 1.0 / denominator

    def __call__(self, angular_momentum, spin, *args, s=None, **kwargs) -> Union[float, Any]:
        s_val = s if s is not None else (self.s.value if self.s is not None else None)
        if s_val is None:
            raise ValueError("s must be provided either at construction or call time")
        return self.function(angular_momentum, spin, s_val, *args, **kwargs)
