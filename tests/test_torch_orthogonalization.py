from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gram_muon import GramMuon, gram_newton_schulz, standard_newton_schulz


def test_torch_gram_matches_standard_in_float64() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 32, dtype=torch.float64)
    standard = standard_newton_schulz(x, coefficients="polar_express", return_dtype=torch.float64)
    gram = gram_newton_schulz(
        x,
        coefficients="polar_express",
        restarts_after=(2,),
        return_dtype=torch.float64,
        fallback_to_standard_on_square=False,
    )
    assert torch.max(torch.abs(standard - gram)) < 1e-10


def test_gram_muon_smoke_step() -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(8, 4)
    optimizer = GramMuon(layer.parameters(), lr=1e-3, ns_method="gram")
    x = torch.randn(16, 8)
    loss = layer(x).square().mean()
    loss.backward()
    before = layer.weight.detach().clone()
    optimizer.step()
    assert not torch.allclose(before, layer.weight)
