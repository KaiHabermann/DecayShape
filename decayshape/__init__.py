"""
DecayShape: Lineshapes for hadron physics amplitude analysis.

This package provides various lineshapes commonly used in hadron physics
for amplitude or partial wave analysis, with support for both numpy and
JAX backends.
"""

from .config import config, set_backend
from .base import FixedParam
from .lineshapes import (
    RelativisticBreitWigner,
    Flatte,
)
from .particles import (
    Particle,
    Channel,
    CommonParticles,
)
from .kmatrix_advanced import (
    KMatrixAdvanced,
)
from .utils import (
    blatt_weiskopf_form_factor,
    angular_momentum_barrier_factor,
)

__version__ = "0.1.0"
__all__ = [
    "config",
    "set_backend",
    "FixedParam",
    "RelativisticBreitWigner",
    "Flatte",
    "Particle",
    "Channel",
    "CommonParticles",
    "KMatrixAdvanced",
    "blatt_weiskopf_form_factor",
    "angular_momentum_barrier_factor",
]
