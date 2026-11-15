"""Tagger-eval metrics: the AP/F1 math against hand-computed values."""

from __future__ import annotations

import math

import torch

from anime_tools.tagger.cli.eval_metrics import (
    micro_f1,
    per_tag_average_precision,
    per_tag_prf,
)


def _logits(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


# ── Metric math ───────────────────────────────────────────────────────────


def test_per_tag_prf_hand_values():
    pred = torch.tensor([[1, 0], [1, 1], [0, 0]], dtype=torch.float32)
    tgt = torch.tensor([[1, 1], [0, 1], [0, 0]], dtype=torch.float32)
    prec, rec, f1, support = per_tag_prf(pred, tgt)
    # tag0: tp=1 fp=1 fn=0 → p=0.5 r=1 f1=2/3 ; tag1: tp=1 fp=0 fn=1 → p=1 r=0.5 f1=2/3
    assert torch.allclose(prec, torch.tensor([0.5, 1.0]))
    assert torch.allclose(rec, torch.tensor([1.0, 0.5]))
    assert torch.allclose(f1, torch.tensor([2 / 3, 2 / 3]))
    assert torch.equal(support, torch.tensor([1.0, 2.0]))


def test_average_precision_hand_values():
    # scores rank samples [0.9, 0.7, 0.4, 0.2]; positives at ranks 1 and 3
    # → AP = (1/1 + 2/3) / 2 = 5/6. Second tag has no positives → NaN.
    scores = torch.tensor([[0.9, 0.1], [0.7, 0.2], [0.4, 0.3], [0.2, 0.4]])
    tgt = torch.tensor([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=torch.float32)
    ap = per_tag_average_precision(scores, tgt)
    assert abs(ap[0].item() - 5 / 6) < 1e-6
    assert math.isnan(ap[1].item())


def test_micro_f1_pools_cells():
    pred = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    tgt = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
    # tp=1 fp=1 fn=1 → micro F1 = 2*1 / (2*1 + 1 + 1) = 0.5
    assert abs(micro_f1(pred, tgt) - 0.5) < 1e-6
