#!/usr/bin/env python3
"""Tiny smoke example for GramMuon."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("PyTorch is required for this example. Install with `pip install -e .`.")

from gram_muon import GramMuon


def main() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 4),
    )
    x = torch.randn(256, 16)
    y = torch.randn(256, 4)

    matrix_params = [p for p in model.parameters() if p.ndim >= 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    scalar_optimizer = torch.optim.AdamW(scalar_params, lr=1e-3, weight_decay=0.01)
    optimizer = GramMuon(
        [{"params": matrix_params, "lr": 3e-3, "weight_decay": 0.01}],
        scalar_optimizer=scalar_optimizer,
        ns_method="gram",
    )

    for step in range(20):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print(f"step={step:02d} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
