from __future__ import annotations

import numpy as np

from gram_muon.numpy_reference import gram_newton_schulz_np, polar_svd_np, standard_newton_schulz_np
from gram_muon.restart_autotune import find_best_restarts


def test_gram_matches_standard_without_square_fallback() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 64))
    standard = standard_newton_schulz_np(x, coefficients="polar_express")
    gram = gram_newton_schulz_np(
        x,
        coefficients="polar_express",
        restarts_after=(2,),
        fallback_to_standard_on_square=False,
    )
    assert np.max(np.abs(standard - gram)) < 1e-10


def test_transposed_tall_matrix_keeps_shape() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 16))
    y = gram_newton_schulz_np(x, coefficients="you", restarts_after=(2,))
    assert y.shape == x.shape


def test_newton_schulz_is_close_to_svd_polar_for_random_matrix() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 48))
    approx = gram_newton_schulz_np(x, coefficients="polar_express", restarts_after=(2,))
    exact = polar_svd_np(x)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.35


def test_autotune_prefers_restart_after_two_for_polar_express() -> None:
    assert find_best_restarts("polar_express", num_restarts=1, num_grid_points=512) == (2,)
