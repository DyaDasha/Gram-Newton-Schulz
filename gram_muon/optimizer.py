"""A compact Muon optimizer with Gram Newton-Schulz support."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .coefficients import Coefficient
from .orthogonalization import Method, orthogonalize

try:  # pragma: no cover - exercised in environments with torch installed
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError(
            "GramMuon requires PyTorch. Install torch in your experiment environment first."
        )
    return torch


def _matrix_view(tensor: Any) -> tuple[Any, tuple[int, ...]]:
    shape = tuple(tensor.shape)
    if tensor.ndim == 2:
        return tensor, shape
    return tensor.reshape(tensor.shape[0], -1), shape


def _restore_shape(matrix: Any, shape: tuple[int, ...]) -> Any:
    return matrix.reshape(shape)


def _muon_lr_scale(matrix: Any, mode: str) -> float:
    if mode == "none":
        return 1.0
    rows = matrix.size(-2)
    cols = matrix.size(-1)
    if mode == "sqrt_aspect":
        return max(1.0, rows / cols) ** 0.5
    if mode == "rms_norm":
        return (max(rows, cols) ** 0.5) / max(1.0, min(rows, cols) ** 0.5)
    raise ValueError("adjust_lr must be 'none', 'sqrt_aspect', or 'rms_norm'.")


if torch is not None:

    class GramMuon(torch.optim.Optimizer):
        """Muon optimizer prototype using standard or Gram Newton-Schulz.

        Parameters with fewer than two dimensions are updated with SGD-style
        momentum unless they are handled by an external scalar optimizer.
        """

        def __init__(
            self,
            params: Iterable[Any],
            *,
            lr: float = 3e-3,
            momentum: float = 0.95,
            weight_decay: float = 0.0,
            nesterov: bool = True,
            ns_method: Method = "gram",
            ns_steps: int | None = None,
            ns_coefficients: str | Sequence[Coefficient] = "polar_express",
            gram_restarts_after: Sequence[int] = (2,),
            ns_compute_dtype: Any | None = None,
            adjust_lr: str = "sqrt_aspect",
            scalar_optimizer: Any | None = None,
        ) -> None:
            if lr <= 0:
                raise ValueError("lr must be positive.")
            if not 0 <= momentum < 1:
                raise ValueError("momentum must be in [0, 1).")
            defaults = dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                ns_method=ns_method,
                ns_steps=ns_steps,
                ns_coefficients=ns_coefficients,
                gram_restarts_after=tuple(gram_restarts_after),
                ns_compute_dtype=ns_compute_dtype,
                adjust_lr=adjust_lr,
                split_fn=None,
                recombine_fn=None,
                update_non_matrix=True,
            )
            super().__init__(params, defaults)
            self.scalar_optimizer = scalar_optimizer

        @torch.no_grad()
        def step(self, closure: Callable[[], Any] | None = None) -> Any:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            if self.scalar_optimizer is not None:
                self.scalar_optimizer.step()

            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grad = param.grad
                    if grad.ndim < 2:
                        if group["update_non_matrix"]:
                            self._sgd_like_step(param, grad, group)
                        continue
                    self._muon_step(param, grad, group)
            return loss

        def zero_grad(self, set_to_none: bool = True) -> None:
            super().zero_grad(set_to_none=set_to_none)
            if self.scalar_optimizer is not None:
                self.scalar_optimizer.zero_grad(set_to_none=set_to_none)

        def _sgd_like_step(self, param: Any, grad: Any, group: dict[str, Any]) -> None:
            state = self.state[param]
            if group["weight_decay"]:
                param.mul_(1.0 - group["lr"] * group["weight_decay"])
            if group["momentum"]:
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad)
                    state["momentum_buffer"] = buf
                buf.mul_(group["momentum"]).add_(grad)
                update = grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
            else:
                update = grad
            param.add_(update, alpha=-group["lr"])

        def _muon_step(self, param: Any, grad: Any, group: dict[str, Any]) -> None:
            state = self.state[param]
            matrix_grad, original_shape = _matrix_view(grad)

            buf = state.get("momentum_buffer")
            if buf is None:
                buf = torch.zeros_like(matrix_grad)
                state["momentum_buffer"] = buf
            buf.mul_(group["momentum"]).add_(matrix_grad)
            update = matrix_grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

            split_fn = group.get("split_fn")
            recombine_fn = group.get("recombine_fn")
            if split_fn is not None:
                pieces = split_fn(update)
                ortho_pieces = [self._orthogonalize_piece(piece, group) for piece in pieces]
                if recombine_fn is None:
                    raise ValueError("A param group with split_fn also needs recombine_fn.")
                ortho_update = recombine_fn(ortho_pieces)
            else:
                ortho_update = self._orthogonalize_piece(update, group)

            if group["weight_decay"]:
                param.mul_(1.0 - group["lr"] * group["weight_decay"])
            lr_scale = _muon_lr_scale(update, group["adjust_lr"])
            param.add_(_restore_shape(ortho_update, original_shape), alpha=-group["lr"] * lr_scale)

        def _orthogonalize_piece(self, piece: Any, group: dict[str, Any]) -> Any:
            matrix, shape = _matrix_view(piece)
            ortho = orthogonalize(
                matrix,
                method=group["ns_method"],
                steps=group["ns_steps"],
                coefficients=group["ns_coefficients"],
                compute_dtype=group["ns_compute_dtype"],
                return_dtype=matrix.dtype,
                restarts_after=group["gram_restarts_after"],
            )
            return _restore_shape(ortho, shape)

else:

    class GramMuon:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
