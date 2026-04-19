"""NumPy reference implementation for algorithm checks without PyTorch."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .coefficients import Coefficient, iter_coefficients


def _prepare_matrix(x: np.ndarray, eps: float, transpose: bool | None) -> tuple[np.ndarray, bool]:
    if x.ndim != 2:
        raise ValueError("The NumPy reference implementation accepts one 2D matrix.")
    if transpose is None:
        transpose = x.shape[-2] > x.shape[-1]
    work = np.asarray(x, dtype=np.float64)
    if transpose:
        work = work.T
    work = work / max(float(np.linalg.norm(work)), eps)
    return work, transpose


def standard_newton_schulz_np(
    x: np.ndarray,
    *,
    steps: int | None = None,
    coefficients: str | Sequence[Coefficient] = "polar_express",
    eps: float = 1e-7,
    transpose: bool | None = None,
) -> np.ndarray:
    """NumPy version of the standard Newton-Schulz loop."""
    work, did_transpose = _prepare_matrix(x, eps, transpose)
    for a, b, c in iter_coefficients(coefficients, steps):
        gram = work @ work.T
        z = b * gram + c * (gram @ gram)
        work = a * work + z @ work
    return work.T if did_transpose else work


def gram_newton_schulz_np(
    x: np.ndarray,
    *,
    steps: int | None = None,
    coefficients: str | Sequence[Coefficient] = "polar_express",
    restarts_after: Sequence[int] = (2,),
    eps: float = 1e-7,
    transpose: bool | None = None,
    fallback_to_standard_on_square: bool = True,
) -> np.ndarray:
    """NumPy version of stabilized Gram Newton-Schulz."""
    work, did_transpose = _prepare_matrix(x, eps, transpose)
    if fallback_to_standard_on_square and work.shape[0] == work.shape[1]:
        result = standard_newton_schulz_np(
            work,
            steps=steps,
            coefficients=coefficients,
            eps=eps,
            transpose=False,
        )
        return result.T if did_transpose else result

    schedule = tuple(iter_coefficients(coefficients, steps))
    restart_points = set(restarts_after)
    total_steps = len(schedule)

    gram = work @ work.T
    q = np.eye(gram.shape[-1], dtype=work.dtype)

    for step, (a, b, c) in enumerate(schedule, start=1):
        if step - 1 in restart_points:
            work = q @ work
            gram = work @ work.T
            q = np.eye(gram.shape[-1], dtype=work.dtype)

        z = b * gram + c * (gram @ gram)
        q = q @ z + a * q

        if step != total_steps and step not in restart_points:
            rz = gram @ z + a * gram
            gram = z @ rz + a * rz

    work = q @ work
    return work.T if did_transpose else work


def polar_svd_np(x: np.ndarray) -> np.ndarray:
    """Exact polar factor by SVD for small diagnostics."""
    u, _, vh = np.linalg.svd(x, full_matrices=False)
    return u @ vh
