"""
Lineshape implementations for hadron physics.

Contains various lineshapes commonly used in amplitude analysis:
- Relativistic Breit-Wigner
- Flatté
- K-matrix
"""

from typing import Union, Any, Optional, List
from pydantic import Field
from .base import Lineshape, FixedParam
from .config import config
from .utils import (
    blatt_weiskopf_form_factor,
    angular_momentum_barrier_factor,
    relativistic_breit_wigner_denominator
)


class RelativisticBreitWigner(Lineshape):
    """
    Relativistic Breit-Wigner lineshape.
    
    The most common lineshape for hadron resonances, accounting for
    the finite width and relativistic effects.
    """
    
    # All parameters are optimization parameters (no FixedParam used)
    pole_mass: float = Field(default=0.775, description="Pole mass of the resonance")
    width: float = Field(default=0.15, description="Resonance width")
    r: float = Field(default=1.0, description="Hadron radius parameter for Blatt-Weiskopf form factor")
    L: int = Field(default=0, description="Angular momentum of the decay")
    q0: Optional[float] = Field(default=None, description="Reference momentum (calculated from pole_mass if None)")
    
    @property
    def parameter_order(self) -> List[str]:
        """Return the order of parameters for positional arguments."""
        return ["pole_mass", "width", "r", "L", "q0"]
    
    def model_post_init(self, __context):
        """Post-initialization to set q0 if not provided."""
        if self.q0 is None:
            self.q0 = self.pole_mass / 2.0
    
    def __call__(self, *args, **kwargs) -> Union[float, Any]:
        """
        Evaluate the Relativistic Breit-Wigner at the s values from construction.
        
        Args:
            *args: Positional parameter overrides (width, r, L, q0)
            **kwargs: Keyword parameter overrides
            
        Returns:
            Breit-Wigner amplitude
        """
        # Get parameters with overrides
        params = self._get_parameters(*args, **kwargs)
        
        np = config.backend
        
        # Calculate momentum in the decay frame
        q = np.sqrt(self.s.value) / 2.0
        
        # Blatt-Weiskopf form factor
        F = blatt_weiskopf_form_factor(q, params["q0"], params["r"], params["L"])
        
        # Angular momentum barrier factor
        B = angular_momentum_barrier_factor(q, params["q0"], params["L"])
        
        # Breit-Wigner denominator (use optimization parameter pole_mass)
        denominator = relativistic_breit_wigner_denominator(self.s.value, params["pole_mass"], params["width"])
        
        return F * B / denominator


class Flatte(Lineshape):
    """
    Flatté lineshape for coupled-channel resonances.
    
    Used for resonances that can decay into multiple channels,
    such as the f0(980) which couples to both ππ and KK.
    """
    
    # Fixed parameters (don't change during optimization)
    channel1_mass1: FixedParam[float]
    channel1_mass2: FixedParam[float]
    channel2_mass1: FixedParam[float]
    channel2_mass2: FixedParam[float]
    
    # Optimization parameters
    pole_mass: float = Field(description="Pole mass of the resonance")
    width1: float = Field(description="Width for first channel")
    width2: float = Field(description="Width for second channel")
    r1: float = Field(description="Hadron radius for first channel")
    r2: float = Field(description="Hadron radius for second channel")
    L1: int = Field(description="Angular momentum for first channel")
    L2: int = Field(description="Angular momentum for second channel")
    q01: Optional[float] = Field(default=None, description="Reference momentum for first channel")
    q02: Optional[float] = Field(default=None, description="Reference momentum for second channel")
    @property
    def parameter_order(self) -> List[str]:
        """Return the order of parameters for positional arguments."""
        return ["pole_mass", "width1", "width2", "r1", "r2", "L1", "L2", "q01", "q02"]
    
    def model_post_init(self, __context):
        """Post-initialization to set q01, q02 if not provided."""
        if self.q01 is None:
            self.q01 = self.pole_mass / 2.0
        if self.q02 is None:
            self.q02 = self.pole_mass / 2.0
    
    def __call__(self, *args, **kwargs) -> Union[float, Any]:
        """
        Evaluate the Flatté lineshape at the s values from construction.
        
        Args:
            *args: Positional parameter overrides (width1, width2, r1, r2, L1, L2, q01, q02)
            **kwargs: Keyword parameter overrides
            
        Returns:
            Flatté amplitude
        """
        # Get parameters with overrides
        params = self._get_parameters(*args, **kwargs)
        
        np = config.backend
        
        # Calculate momenta in both channels using proper channel masses
        # Channel 1 momentum
        m1_1 = self.channel1_mass1.value
        m1_2 = self.channel1_mass2.value
        q1 = np.sqrt((self.s.value - (m1_1 + m1_2)**2) * (self.s.value - (m1_1 - m1_2)**2)) / (2 * np.sqrt(self.s.value))
        
        # Channel 2 momentum
        m2_1 = self.channel2_mass1.value
        m2_2 = self.channel2_mass2.value
        q2 = np.sqrt((self.s.value - (m2_1 + m2_2)**2) * (self.s.value - (m2_1 - m2_2)**2)) / (2 * np.sqrt(self.s.value))
        
        # Form factors and barrier factors for both channels
        F1 = blatt_weiskopf_form_factor(q1, params["q01"], params["r1"], params["L1"])
        F2 = blatt_weiskopf_form_factor(q2, params["q02"], params["r2"], params["L2"])
        B1 = angular_momentum_barrier_factor(q1, params["q01"], params["L1"])
        B2 = angular_momentum_barrier_factor(q2, params["q02"], params["L2"])
        
        # Total width
        total_width = params["width1"] * F1 * B1 + params["width2"] * F2 * B2
        
        # Flatté denominator (use optimization parameter pole_mass)
        denominator = self.s.value - params["pole_mass"]**2 + 1j * params["pole_mass"] * total_width
        
        return 1.0 / denominator
