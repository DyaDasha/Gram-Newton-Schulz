"""Newton-Schulz coefficient presets used by Muon-style optimizers."""

from __future__ import annotations

from itertools import chain, cycle, islice, repeat
from typing import Iterable, Iterator, Literal, Sequence

Coefficient = tuple[float, float, float]
CoefficientName = Literal[
    "simple",
    "you",
    "quintic",
    "polar_express",
    "polar_express_dao",
    "polar_express_nemo",
    "cans",
    "aol",
]
IterationMode = Literal["auto", "cycle", "repeat_last"]

SIMPLE: tuple[Coefficient, ...] = ((3.4445, -4.7750, 2.0315),)

YOU_COEFFICIENTS: tuple[Coefficient, ...] = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

_UNMODIFIED_POLAR_EXPRESS: tuple[Coefficient, ...] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)

DAO_POLAR_EXPRESS: tuple[Coefficient, ...] = tuple(
    (a / 1.05, b / 1.05**3, c / 1.05**5)
    for a, b, c in _UNMODIFIED_POLAR_EXPRESS
)

NEMO_POLAR_EXPRESS: tuple[Coefficient, ...] = (
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
)

POLAR_EXPRESS = DAO_POLAR_EXPRESS

CANS: tuple[Coefficient, ...] = (
    (8.4703, -25.1081, 18.6293),
    (4.1828, -3.1087, 0.5806),
    (3.9619, -2.9541, 0.5630),
    (3.2866, -2.4647, 0.5074),
    (2.2737, -1.6447, 0.4162),
)

AOL: tuple[Coefficient, ...] = (
    (4.0098, -7.0585, 2.4635),
    (3.4585, -5.5479, 2.5959),
    (2.7573, -3.2939, 1.4254),
    (2.7215, -3.0494, 1.3169),
)

COEFFICIENT_SETS: dict[str, tuple[Coefficient, ...]] = {
    "simple": SIMPLE,
    "you": YOU_COEFFICIENTS,
    "quintic": YOU_COEFFICIENTS,
    "polar_express": POLAR_EXPRESS,
    "polar_express_dao": DAO_POLAR_EXPRESS,
    "polar_express_nemo": NEMO_POLAR_EXPRESS,
    "cans": CANS,
    "aol": AOL,
}

_REPEAT_LAST_BY_DEFAULT = {"polar_express", "polar_express_dao", "polar_express_nemo", "cans"}


def get_coefficients(name: str | Sequence[Coefficient]) -> tuple[Coefficient, ...]:
    """Resolve a named or explicit coefficient schedule."""
    if isinstance(name, str):
        try:
            return COEFFICIENT_SETS[name]
        except KeyError as exc:
            known = ", ".join(sorted(COEFFICIENT_SETS))
            raise ValueError(f"Unknown coefficient set {name!r}. Known sets: {known}") from exc
    if not name:
        raise ValueError("Coefficient sequence must be non-empty.")
    return tuple((float(a), float(b), float(c)) for a, b, c in name)


def iter_coefficients(
    coefficients: str | Sequence[Coefficient],
    steps: int | None = None,
    mode: IterationMode = "auto",
) -> Iterator[Coefficient]:
    """Yield a coefficient schedule with Muon-compatible end behavior."""
    coeffs = get_coefficients(coefficients)
    if steps is None:
        return iter(coeffs)
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    resolved_mode = mode
    if mode == "auto":
        resolved_mode = "repeat_last" if isinstance(coefficients, str) and coefficients in _REPEAT_LAST_BY_DEFAULT else "cycle"

    base: Iterable[Coefficient]
    if resolved_mode == "cycle":
        base = cycle(coeffs)
    elif resolved_mode == "repeat_last":
        base = chain(coeffs, repeat(coeffs[-1]))
    else:
        raise ValueError("mode must be 'auto', 'cycle', or 'repeat_last'.")
    return islice(base, steps)
