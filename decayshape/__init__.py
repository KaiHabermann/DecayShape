"""
DecayShape: Lineshapes for hadron physics amplitude analysis.

This package provides various lineshapes commonly used in hadron physics
for amplitude or partial wave analysis, with support for both numpy and
JAX backends.
"""

from .base import FixedParam
from .config import config, set_backend
from .kmatrix_advanced import KMatrixAdvanced
from .lineshapes import Flatte, RelativisticBreitWigner
from .particles import Channel, CommonParticles, Particle
from .schema_utils import (
    export_schemas_to_file,
    get_all_lineshape_schemas,
    get_available_lineshapes,
    get_common_particles_info,
    get_lineshape_schema,
)
from .utils import angular_momentum_barrier_factor, blatt_weiskopf_form_factor

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
    "get_all_lineshape_schemas",
    "get_lineshape_schema",
    "get_available_lineshapes",
    "export_schemas_to_file",
    "get_common_particles_info",
]
