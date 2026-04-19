"""PyTorch implementations of standard and Gram Newton-Schulz."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from .coefficients import Coefficient, iter_coefficients

try:  # pragma: no cover - exercised in environments with torch installed
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

Method = Literal["standard", "gram"]


def _require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError(
            "gram_muon.orthogonalization requires PyTorch. Install the package with "
            "`pip install -e .` in an environment where torch is available."
        )
    return torch


def _validate_matrix(x: Any) -> None:
    if x.ndim < 2:
        raise ValueError("Newton-Schulz orthogonalization requires tensors with at least 2 dimensions.")
    if not x.is_floating_point():
        raise TypeError("Newton-Schulz orthogonalization requires a floating-point tensor.")


def _identity_like_gram(r: Any) -> Any:
    t = _require_torch()
    n = r.size(-1)
    eye = t.eye(n, device=r.device, dtype=r.dtype)
    return eye.expand(*r.shape[:-2], n, n)


def _normalize_and_maybe_transpose(
    x: Any,
    eps: float,
    transpose: bool | None,
    compute_dtype: Any | None,
) -> tuple[Any, bool]:
    t = _require_torch()
    _validate_matrix(x)

    if transpose is None:
        transpose = x.size(-2) > x.size(-1)

    work = x.to(t.float32 if x.dtype in (t.float16, t.bfloat16) else x.dtype)
    if transpose:
        work = work.mT

    norm = t.linalg.vector_norm(work, ord=2, dim=(-2, -1), keepdim=True).clamp_min(eps)
    work = work / norm

    if compute_dtype is not None:
        work = work.to(compute_dtype)
    return work, transpose


def standard_newton_schulz(
    x: Any,
    *,
    steps: int | None = None,
    coefficients: str | Sequence[Coefficient] = "polar_express",
    eps: float = 1e-7,
    transpose: bool | None = None,
    compute_dtype: Any | None = None,
    return_dtype: Any | None = None,
) -> Any:
    """Approximate the polar factor with the usual Muon Newton-Schulz loop.

    The input is normalized by Frobenius norm and processed on the smaller
    Gram side, matching common Muon implementations.
    """
    _require_torch()
    original_dtype = x.dtype
    original_shape = x.shape
    x, did_transpose = _normalize_and_maybe_transpose(x, eps, transpose, compute_dtype)

    for a, b, c in iter_coefficients(coefficients, steps):
        gram = x @ x.mT
        update_poly = b * gram + c * (gram @ gram)
        x = a * x + update_poly @ x

    if did_transpose:
        x = x.mT
    return x.to(return_dtype or original_dtype).reshape(original_shape)


def gram_newton_schulz(
    x: Any,
    *,
    steps: int | None = None,
    coefficients: str | Sequence[Coefficient] = "polar_express",
    restarts_after: Sequence[int] = (2,),
    eps: float = 1e-7,
    transpose: bool | None = None,
    compute_dtype: Any | None = None,
    return_dtype: Any | None = None,
    fallback_to_standard_on_square: bool = True,
) -> Any:
    """Approximate the polar factor with stabilized Gram Newton-Schulz.

    ``restarts_after=(2,)`` means reconstructing the Gram matrix after the
    second Newton-Schulz polynomial, the setting recommended for five-step
    Polar Express in the Gram Newton-Schulz release.
    """
    _require_torch()
    original_dtype = x.dtype
    original_shape = x.shape
    x, did_transpose = _normalize_and_maybe_transpose(x, eps, transpose, compute_dtype)

    if fallback_to_standard_on_square and x.size(-2) == x.size(-1):
        for a, b, c in iter_coefficients(coefficients, steps):
            gram = x @ x.mT
            update_poly = b * gram + c * (gram @ gram)
            x = a * x + update_poly @ x
        if did_transpose:
            x = x.mT
        return x.to(return_dtype or original_dtype).reshape(original_shape)

    schedule = tuple(iter_coefficients(coefficients, steps))
    restart_points = set(restarts_after)
    total_steps = len(schedule)

    gram = x @ x.mT
    q = _identity_like_gram(gram)

    for step, (a, b, c) in enumerate(schedule, start=1):
        if step - 1 in restart_points:
            x = q @ x
            gram = x @ x.mT
            q = _identity_like_gram(gram)

        z = b * gram + c * (gram @ gram)
        q = q @ z + a * q

        if step != total_steps and step not in restart_points:
            rz = gram @ z + a * gram
            gram = z @ rz + a * rz

    x = q @ x

    if did_transpose:
        x = x.mT
    return x.to(return_dtype or original_dtype).reshape(original_shape)


def orthogonalize(
    x: Any,
    *,
    method: Method = "gram",
    steps: int | None = None,
    coefficients: str | Sequence[Coefficient] = "polar_express",
    eps: float = 1e-7,
    transpose: bool | None = None,
    compute_dtype: Any | None = None,
    return_dtype: Any | None = None,
    restarts_after: Sequence[int] = (2,),
) -> Any:
    """Dispatch to standard or Gram Newton-Schulz."""
    if method == "standard":
        return standard_newton_schulz(
            x,
            steps=steps,
            coefficients=coefficients,
            eps=eps,
            transpose=transpose,
            compute_dtype=compute_dtype,
            return_dtype=return_dtype,
        )
    if method == "gram":
        return gram_newton_schulz(
            x,
            steps=steps,
            coefficients=coefficients,
            restarts_after=restarts_after,
            eps=eps,
            transpose=transpose,
            compute_dtype=compute_dtype,
            return_dtype=return_dtype,
        )
    raise ValueError("method must be 'standard' or 'gram'.")
