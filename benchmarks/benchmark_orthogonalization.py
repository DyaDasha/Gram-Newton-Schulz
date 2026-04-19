#!/usr/bin/env python3
"""Benchmark standard Newton-Schulz against Gram Newton-Schulz."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("PyTorch is required for this benchmark. Install with `pip install -e .`.")

from gram_muon import gram_newton_schulz, standard_newton_schulz


def parse_shape(value: str) -> tuple[int, int]:
    left, right = value.lower().split("x", maxsplit=1)
    return int(left), int(right)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp64":
        return torch.float64
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError("dtype must be one of fp32, fp64, fp16, bf16.")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn, x: torch.Tensor, warmup: int, iters: int, device: torch.device) -> tuple[float, torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn(x)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(x)
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iters, out


def orthogonality_defect(y: torch.Tensor) -> float:
    work = y if y.size(-2) <= y.size(-1) else y.mT
    gram = work @ work.mT
    eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype)
    return float(torch.linalg.vector_norm(gram - eye) / torch.linalg.vector_norm(eye))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=["128x512", "512x2048", "2048x7168"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp64", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--coefficients", default="polar_express")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else None

    print("shape,standard_ms,gram_ms,speedup,standard_defect,gram_defect,max_abs_diff")
    for shape_text in args.shapes:
        rows, cols = parse_shape(shape_text)
        x = torch.randn(rows, cols, device=device, dtype=torch.float32)
        if dtype in (torch.float16, torch.bfloat16):
            x = x.to(dtype)

        standard = lambda inp: standard_newton_schulz(
            inp,
            coefficients=args.coefficients,
            compute_dtype=compute_dtype,
            return_dtype=torch.float32,
        )
        gram = lambda inp: gram_newton_schulz(
            inp,
            coefficients=args.coefficients,
            compute_dtype=compute_dtype,
            return_dtype=torch.float32,
        )

        standard_ms, standard_out = time_call(standard, x, args.warmup, args.iters, device)
        gram_ms, gram_out = time_call(gram, x, args.warmup, args.iters, device)
        speedup = standard_ms / gram_ms if gram_ms > 0 else float("inf")
        max_abs_diff = float((standard_out - gram_out).abs().max())
        print(
            f"{rows}x{cols},{standard_ms:.4f},{gram_ms:.4f},{speedup:.3f},"
            f"{orthogonality_defect(standard_out):.6f},{orthogonality_defect(gram_out):.6f},"
            f"{max_abs_diff:.6e}"
        )


if __name__ == "__main__":
    main()
