"""Per-tag F1-optimal threshold sweep on the val split.

A global 0.5 threshold under-fires rare tags and over-fires common ones, so this
sweeps [0.05, 0.95] step 0.05 per tag and picks the F1-maximizing threshold.
Tags with no positive val examples or zero achievable F1 keep ``default=0.5``:
they can't be calibrated, but the floor keeps the head well-formed.
"""

from __future__ import annotations

import torch

DEFAULT_SWEEP = torch.arange(0.05, 0.951, 0.05)


def calibrate_thresholds(
    scores: torch.Tensor,  # [N, n_tags] sigmoid probabilities
    targets: torch.Tensor,  # [N, n_tags] multi-hot
    sweep: torch.Tensor,  # [K] candidate thresholds
    default: float = 0.5,
    skip_indices: torch.Tensor
    | None = None,  # LongTensor of tag indices to leave at default
    min_support: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tag F1-optimal threshold sweep.

    Returns ``(thresholds[n_tags], best_f1[n_tags])``. Tags with fewer than
    ``min_support`` positives in the val split keep ``default``: off a handful
    of positives the sweep lands on hair-trigger values that over-fire at
    inference. Same fallback for tags whose best achievable F1 is 0.

    ``skip_indices`` names tags that belong to a softmax group and shouldn't be
    sigmoid-thresholded (inference uses argmax); those keep ``default`` and
    ``best_f1=0``.
    """
    n_tags = scores.shape[1]
    K = sweep.shape[0]
    best_thresh = torch.full((n_tags,), default)
    best_f1 = torch.zeros(n_tags)
    pos_count = targets.sum(dim=0)  # [n_tags]
    has_pos = pos_count >= max(1, min_support)
    if skip_indices is not None and skip_indices.numel() > 0:
        skip_mask = torch.zeros(n_tags, dtype=torch.bool)
        skip_mask[skip_indices.cpu()] = True
        has_pos = has_pos & ~skip_mask
    # Block over tags: the dense [N, n_tags, K] tensor would be ~1.1B floats.
    block_size = 256
    for start in range(0, n_tags, block_size):
        end = min(start + block_size, n_tags)
        s = scores[:, start:end]  # [N, b]
        t = targets[:, start:end]
        # [N, b, K] boolean
        pred = s.unsqueeze(-1) > sweep.view(1, 1, K)
        pred_f = pred.float()
        tp = (pred_f * t.unsqueeze(-1)).sum(dim=0)  # [b, K]
        fp = (pred_f * (1 - t).unsqueeze(-1)).sum(dim=0)
        fn = ((1 - pred_f) * t.unsqueeze(-1)).sum(dim=0)
        prec = tp / (tp + fp).clamp_min(1e-8)
        rec = tp / (tp + fn).clamp_min(1e-8)
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)  # [b, K]
        f1_best, k_best = f1.max(dim=-1)  # [b]
        thresh_best = sweep[k_best]  # [b]
        local_has_pos = has_pos[start:end]
        keep = local_has_pos & (f1_best > 0)
        best_f1[start:end] = torch.where(keep, f1_best, best_f1[start:end])
        best_thresh[start:end] = torch.where(keep, thresh_best, best_thresh[start:end])
    return best_thresh, best_f1
