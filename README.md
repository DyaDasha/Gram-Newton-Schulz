# Gram Muon

Проект из proposal: минимальная исследовательская база для проверки Gram Newton-Schulz как ускоренной замены Newton-Schulz внутри Muon.

Что уже есть:

- `gram_muon.standard_newton_schulz` — базовый Newton-Schulz из Muon.
- `gram_muon.gram_newton_schulz` — стабилизированный Gram Newton-Schulz с рестартом после 2-й итерации.
- `gram_muon.GramMuon` — компактный PyTorch optimizer для матричных параметров.
- `benchmarks/benchmark_orthogonalization.py` — замер времени standard vs gram на заданных формах матриц.
- `gram_muon.restart_autotune` — подбор позиций рестартов по scalar stability proxy.
- `gram_muon.numpy_reference` — NumPy-reference, чтобы проверять математику даже без установленного Torch.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Если PyTorch ставится отдельно под конкретную CUDA/Metal-сборку, сначала поставь его по инструкции PyTorch, затем выполни `pip install -e ".[dev]"`.

## Быстрая проверка

```bash
pytest
```

В текущем системном окружении Torch может отсутствовать; тогда PyTorch-тесты будут пропущены, а NumPy-тесты всё равно проверят эквивалентность алгоритмов.

## Бенчмарк ортогонализации

```bash
python benchmarks/benchmark_orthogonalization.py \
  --device cuda \
  --dtype bf16 \
  --shapes 128x512 512x2048 2048x7168
```

Вывод CSV показывает время standard NS, время Gram NS, speedup, дефект ортогональности и максимальную разницу между методами.

## Подбор рестартов

```bash
python -m gram_muon.restart_autotune --coefficients polar_express --num-restarts 1
```

Для Polar Express ожидаемая позиция — `2`, то есть рестарт после второй итерации.

## Пример Muon

```python
import torch
from gram_muon import GramMuon

matrix_params = [p for p in model.parameters() if p.ndim >= 2]
scalar_params = [p for p in model.parameters() if p.ndim < 2]

scalar_optimizer = torch.optim.AdamW(scalar_params, lr=1e-3, weight_decay=0.1)
optimizer = GramMuon(
    [{"params": matrix_params, "lr": 3e-3, "weight_decay": 0.1}],
    scalar_optimizer=scalar_optimizer,
    ns_method="gram",
    ns_coefficients="polar_express",
    gram_restarts_after=(2,),
)
```

Для QKV или MLP-весов, которые нужно ортогонализовать по частям, можно передать в param group `split_fn` и `recombine_fn`.

## Источники реализации

- [NVIDIA NeMo Emerging Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/main/emerging_optimizers/orthogonalized_optimizers/muon_utils.py): коэффициенты и baseline-паттерн Newton-Schulz.
- [Dao-AILab Gram Newton-Schulz](https://github.com/Dao-AILab/gram-newton-schulz): идея работы на матрице Грама и рестарт после второй итерации для устойчивости.
- [DAO Lab blog post](https://dao-lab.ai/blog/2026/gram-newton-schulz/): численная мотивация рестартов и план экспериментов.
- Proposal: фокус на сравнении Muon, GramMuon и AdamW по качеству обучения и стоимости ортогонализации.
