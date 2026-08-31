"""Per-tag F1-optimal threshold sweep on the val split.

Library half of the old ``--mode calibrate`` (its PE-head CLI driver was
archived 2026-08-27 → ``_archive/anima_tagger_training/scripts/calibrate.py``).
``calibrate_thresholds`` is what ``train_sidecar.py`` and
``bench/tagger_external/calibration_check.py`` call.

A global 0.5 threshold under-fires rare tags and over-fires common ones.
This sweeps thresholds in [0.05, 0.95] step 0.05 per tag and picks the
F1-maximizing one. Tags with no positive val examples or zero
achievable F1 keep ``default=0.5`` — they can't be calibrated and the F1
sweep is degenerate, but the floor keeps the head well-formed for inference.
"""

from __future__ import annotations

import torch

# The sweep the docstring describes, in one place: both callers used to spell
# `torch.arange(0.05, 0.951, 0.05)` themselves, one of them under a comment
# saying "same sweep as calibrate.py".
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
    ``min_support`` positives in the val split keep ``default`` — with a
    handful of positives the F1-optimal threshold is noise, and the sweep
    routinely lands on hair-trigger values (measured on the v4 checkpoint:
    62% of the vocab has <5 val positives and ~300 of those swept to ≤0.3,
    e.g. `shaded face` at 0.20 from a single positive — which then over-fires
    at inference). Same fallback for tags whose best achievable F1 is 0
    (model never predicts them at any threshold).

    ``skip_indices`` is the trainer-side hint that some tags belong to a
    softmax group and shouldn't be sigmoid-thresholded (inference uses
    argmax). Those keep ``default`` and ``best_f1=0``.
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
    # Block over tags to keep memory bounded — the dense [N, n_tags, K] tensor
    # would be ~12k × 5k × 19 ≈ 1.1B floats.
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
