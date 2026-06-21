# reward/ — Person 5 reward-shaping methods (CA-SLOPE, RQ2).
"""P5 reward shaping. See ca_slope.py for Category-Aware SLOPE."""

from .ca_slope import (
    CASlopeShaper,
    CATEGORY_NAMES,
    DEFAULT_CATEGORY_GAINS,
    DEFAULT_GAMMA,
    DEFAULT_GENERIC_GAIN,
    state_from_obs,
)
from .ca_slope_wrapper import CASlopeEnvWrapper

__all__ = [
    "CASlopeShaper",
    "CASlopeEnvWrapper",
    "CATEGORY_NAMES",
    "DEFAULT_CATEGORY_GAINS",
    "DEFAULT_GAMMA",
    "DEFAULT_GENERIC_GAIN",
    "state_from_obs",
]
