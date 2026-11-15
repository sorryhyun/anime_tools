"""Tagger eval metrics — per-tag precision / recall / F1, AP, micro-F1.

Pure helpers over collected ``[N, n_tags]`` score / multi-hot tensors: no model,
no loader, no disk. The sidecar trainer scores its validation split with them.
"""

from __future__ import annotations

import torch

# Per-tag metrics.


def per_tag_prf(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(precision, recall, f1, support)`` per tag from boolean/0-1 tensors.

    A tag with no predictions AND no positives gets precision/recall 0
    (denominators clamped), not 1, so unsupported tags don't inflate macro
    averages. Callers should mask by ``support > 0``.
    """
    pred_f = pred.float()
    target_f = target.float()
    tp = (pred_f * target_f).sum(dim=0)
    fp = (pred_f * (1 - target_f)).sum(dim=0)
    fn = ((1 - pred_f) * target_f).sum(dim=0)
    prec = tp / (tp + fp).clamp_min(1.0)
    rec = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)
    return prec, rec, f1, target_f.sum(dim=0)


def per_tag_average_precision(
    scores: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Threshold-free AP per tag; NaN where the split has no positives.

    Works for softmax-group tags too, since no threshold enters.
    """
    N = scores.shape[0]
    order = scores.argsort(dim=0, descending=True)
    tgt_sorted = target.float().gather(0, order)
    ranks = torch.arange(1, N + 1, dtype=torch.float32, device=scores.device)
    prec_at_k = tgt_sorted.cumsum(dim=0) / ranks.unsqueeze(1)
    n_pos = tgt_sorted.sum(dim=0)
    ap = (prec_at_k * tgt_sorted).sum(dim=0) / n_pos.clamp_min(1.0)
    return torch.where(n_pos > 0, ap, torch.full_like(ap, float("nan")))


def micro_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pooled-cell F1 — stable on small splits where per-tag support is thin."""
    pred_f = pred.float()
    target_f = target.float()
    tp = (pred_f * target_f).sum()
    fp = (pred_f * (1 - target_f)).sum()
    fn = ((1 - pred_f) * target_f).sum()
    denom = (2 * tp + fp + fn).clamp_min(1e-8)
    return float((2 * tp / denom).item())
