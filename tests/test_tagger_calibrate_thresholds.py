"""``calibrate_thresholds(min_support=…)``: a tag with too few val positives keeps
the 0.5 default rather than trusting a degenerate F1 sweep."""

from __future__ import annotations

import torch

from anime_tools.tagger.cli.calibrate import calibrate_thresholds


def _sweep():
    return torch.linspace(0.05, 0.95, 19)


def test_calibrate_min_support_keeps_default_for_thin_tags():
    # Tag 0: 2 positives, perfectly separable at a low threshold.
    # Tag 1: 8 positives, separable at a high threshold.
    scores = torch.zeros(20, 2)
    targets = torch.zeros(20, 2)
    scores[:2, 0] = 0.30
    targets[:2, 0] = 1.0
    scores[2:, 0] = 0.05
    scores[:8, 1] = 0.90
    targets[:8, 1] = 1.0
    scores[8:, 1] = 0.10

    thresh, f1 = calibrate_thresholds(scores, targets, _sweep(), min_support=5)
    assert thresh[0].item() == 0.5  # 2 < 5 positives → default, sweep untrusted
    assert f1[0].item() == 0.0
    assert 0.10 < thresh[1].item() < 0.90  # 8 ≥ 5 → swept normally
    assert f1[1].item() > 0.99

    # min_support=1 restores the old trust-any-positive behaviour.
    thresh_old, _ = calibrate_thresholds(scores, targets, _sweep(), min_support=1)
    assert thresh_old[0].item() < 0.30
