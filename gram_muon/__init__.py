"""Gram Newton-Schulz tools for Muon experiments."""

from .coefficients import (
    COEFFICIENT_SETS,
    DAO_POLAR_EXPRESS,
    NEMO_POLAR_EXPRESS,
    POLAR_EXPRESS,
    YOU_COEFFICIENTS,
    get_coefficients,
    iter_coefficients,
)
from .orthogonalization import gram_newton_schulz, orthogonalize, standard_newton_schulz
from .optimizer import GramMuon


def find_best_restarts(*args, **kwargs):
    from .restart_autotune import find_best_restarts as _find_best_restarts

    return _find_best_restarts(*args, **kwargs)

__all__ = [
    "COEFFICIENT_SETS",
    "DAO_POLAR_EXPRESS",
    "NEMO_POLAR_EXPRESS",
    "POLAR_EXPRESS",
    "YOU_COEFFICIENTS",
    "GramMuon",
    "find_best_restarts",
    "get_coefficients",
    "gram_newton_schulz",
    "iter_coefficients",
    "orthogonalize",
    "standard_newton_schulz",
]
