"""Restart placement utilities for Gram Newton-Schulz."""

from __future__ import annotations

import argparse
from itertools import combinations
from typing import Sequence

import numpy as np

from .coefficients import Coefficient, get_coefficients


def simulate_perturbed_gram_newton_schulz(
    x_eigenvalues: np.ndarray,
    coefficients: str | Sequence[Coefficient] = "polar_express",
    *,
    most_negative_gram_eigenvalue: float = -4e-4,
    restarts_after: Sequence[int] = (),
) -> dict[int, np.ndarray]:
    """Simulate scalar Gram NS under a negative Gram eigenvalue perturbation."""
    if most_negative_gram_eigenvalue >= 0:
        raise ValueError("most_negative_gram_eigenvalue should be negative.")

    x_values = np.array(x_eigenvalues, dtype=np.float64, copy=True)
    q_values: dict[int, np.ndarray] = {}
    q = np.ones_like(x_values)
    restart_points = set(restarts_after)

    with np.errstate(over="ignore", invalid="ignore"):
        for step, (a, b, c) in enumerate(get_coefficients(coefficients), start=1):
            if step == 1 or step - 1 in restart_points:
                if step != 1:
                    x_values = x_values * q
                r = x_values**2 + most_negative_gram_eigenvalue
                q = np.ones_like(x_values)

            z = a + r * (b + r * c)
            q = q * z
            r = r * z**2
            q_values[step] = q.copy()

    return q_values


def stability_metric(q_values: dict[int, np.ndarray]) -> float:
    """Maximum condition number of simulated Q values."""
    worst = 0.0
    for values in q_values.values():
        abs_values = np.abs(values)
        min_value = abs_values.min()
        if min_value == 0:
            return float("inf")
        worst = max(worst, float(abs_values.max() / min_value))
    return worst


def find_best_restarts(
    coefficients: str | Sequence[Coefficient] = "polar_express",
    *,
    num_restarts: int = 1,
    most_negative_gram_eigenvalue: float = -4e-4,
    num_grid_points: int = 2048,
    min_singular_value: float = 1e-6,
) -> tuple[int, ...]:
    """Find restart positions that minimize the scalar stability metric."""
    coeffs = get_coefficients(coefficients)
    possible_positions = tuple(range(1, len(coeffs)))
    if num_restarts < 0:
        raise ValueError("num_restarts must be non-negative.")
    if num_restarts > len(possible_positions):
        raise ValueError("Too many restarts for the coefficient schedule.")

    eigenvalues = np.linspace(min_singular_value, 1.0, num_grid_points)
    best_metric = float("inf")
    best: tuple[int, ...] = ()

    for candidate in combinations(possible_positions, num_restarts):
        q_values = simulate_perturbed_gram_newton_schulz(
            eigenvalues,
            coeffs,
            most_negative_gram_eigenvalue=most_negative_gram_eigenvalue,
            restarts_after=candidate,
        )
        metric = stability_metric(q_values)
        if metric < best_metric:
            best_metric = metric
            best = tuple(candidate)

    if not np.isfinite(best_metric):
        raise RuntimeError("All restart choices diverged. Try more restarts.")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Autotune Gram Newton-Schulz restart positions.")
    parser.add_argument("--coefficients", default="polar_express")
    parser.add_argument("--num-restarts", type=int, default=1)
    parser.add_argument("--most-negative-gram-eigenvalue", type=float, default=-4e-4)
    parser.add_argument("--num-grid-points", type=int, default=2048)
    args = parser.parse_args()

    best = find_best_restarts(
        args.coefficients,
        num_restarts=args.num_restarts,
        most_negative_gram_eigenvalue=args.most_negative_gram_eigenvalue,
        num_grid_points=args.num_grid_points,
    )
    print(",".join(str(item) for item in best))


if __name__ == "__main__":
    main()
