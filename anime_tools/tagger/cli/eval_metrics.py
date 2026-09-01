"""Stratified tagger eval — per-tag metrics, inference-rule prediction, KB slices.

Pure helpers over collected ``[N, n_tags]`` logit / multi-hot tensors: no model,
no loader, no disk. The trainer's ``macro_f1`` excludes softmax-group tags
(their sigmoid scores are argmax-only at inference), so
:func:`predict_with_inference_rule` mirrors the real ``AnimaTagger.predict``
decision procedure and every tag can be scored end-to-end.

Slice keys: general tags map to their danbooru taxonomy ``소분류`` (KB plus
``derive_groups``' ``_EN_KEY`` naming), other categories to ``cat:<category>``,
KB misses to ``general:unmatched``. Frequency tiers cut on the vocab ``freq``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .derive_groups import _EN_KEY, _slug

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


# Inference-rule prediction (mirrors AnimaTagger.predict group refinement).


def predict_with_inference_rule(
    tag_logits: torch.Tensor,
    thresholds: torch.Tensor,
    router,
) -> torch.Tensor:
    """``[N, n_tags] bool`` predictions via the real inference decision rule:
    per-tag sigmoid threshold, then group argmax refinement exactly as
    ``AnimaTagger.predict``.

    Solo-ness and escape both come from the *thresholded predictions* — at
    inference there is no GT — and where a softmax group is applicable its tags
    are cleared and the argmax winner emitted even if none passed threshold.
    ``router`` is a ``GroupRouter``; only its index sets are read.
    """
    pred = tag_logits.sigmoid() > thresholds.to(tag_logits.device)
    if not router.softmax_groups:
        return pred

    N = pred.shape[0]
    if router.solo_indices is not None:
        has_single = pred[:, router.solo_indices].any(dim=1)
        if router.multi_indices is not None:
            has_multi = pred[:, router.multi_indices].any(dim=1)
        else:
            has_multi = torch.zeros_like(has_single)
        is_solo = has_single & ~has_multi
    else:
        is_solo = torch.ones(N, dtype=torch.bool, device=pred.device)

    for g in router.softmax_groups:
        if g.escape_indices.numel() > 0:
            escape_fired = pred[:, g.escape_indices].any(dim=1)
        else:
            escape_fired = torch.zeros_like(is_solo)
        if g.mode == "softmax_when_solo":
            applicable = is_solo & ~escape_fired
        else:  # "softmax"
            applicable = ~escape_fired
        rows = applicable.nonzero(as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            continue
        winner_local = tag_logits[:, g.tag_indices].argmax(dim=1)
        pred[rows[:, None], g.tag_indices[None, :]] = False
        pred[rows, g.tag_indices[winner_local[rows]]] = True
    if getattr(router, "sentinel_indices", None) is not None:
        # Sentinel slots are never emitted: not as argmax winners (cleared
        # here) and not via the sigmoid-threshold path (their calibrated
        # threshold is skipped like all softmax members).
        pred[:, router.sentinel_indices] = False
    return pred


# Slice assignment — KB taxonomy + frequency tiers.

UNMATCHED_SLICE = "general:unmatched"


@dataclass
class TagSlices:
    """Per-tag slice keys plus the derived slice → tag-index map."""

    kb_slice: list[str]  # [n_tags] taxonomy / category slice key
    freq_tier: list[str]  # [n_tags] "head" | "mid" | "tail"

    def indices_by(self, which: str) -> dict[str, list[int]]:
        keys = self.kb_slice if which == "kb" else self.freq_tier
        out: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            out.setdefault(k, []).append(i)
        return out


def assign_slices(
    vocab: dict,
    kb,
    rename_recovery: dict[str, str] | None = None,
    freq_cuts: tuple[int, int] = (1000, 200),
) -> TagSlices:
    """Map every vocab tag to a KB-taxonomy slice and a frequency tier.

    ``rename_recovery`` maps our (post-rule) tag name → original danbooru
    name, consulted when the KB misses the current name (same mechanism as
    ``derive_groups``). ``freq_cuts = (head_min, mid_min)`` on the vocab's
    train ``freq``.
    """
    rename_recovery = rename_recovery or {}
    head_min, mid_min = freq_cuts
    kb_slice: list[str] = []
    freq_tier: list[str] = []
    for t in vocab["tags"]:
        name = t["name"]
        cat = str(t.get("category", "general"))
        if cat != "general":
            kb_slice.append(f"cat:{cat}")
        else:
            info = kb.describe(name) if kb is not None else None
            if info is None and name in rename_recovery and kb is not None:
                info = kb.describe(rename_recovery[name])
            if info is None or not info.category_path:
                kb_slice.append(UNMATCHED_SLICE)
            else:
                path = info.category_path
                kb_slice.append(_EN_KEY.get(path) or _slug(path))
        freq = int(t.get("freq", 0))
        freq_tier.append(
            "head" if freq >= head_min else ("mid" if freq >= mid_min else "tail")
        )
    return TagSlices(kb_slice=kb_slice, freq_tier=freq_tier)


# Aggregation.


def aggregate_slices(
    slice_indices: dict[str, list[int]],
    pred: torch.Tensor,
    target: torch.Tensor,
    f1: torch.Tensor,
    ap: torch.Tensor,
    support: torch.Tensor,
) -> list[dict]:
    """One summary row per slice, sorted by descending tag count.

    ``macro_f1`` / ``mean_ap`` average only over *supported* tags (≥1 positive
    in the split); ``micro_f1`` pools every cell in the slice. ``n_tags`` vs
    ``n_supported`` makes thin-support slices visible.
    """
    rows: list[dict] = []
    for key, idxs in slice_indices.items():
        idx_t = torch.tensor(idxs, dtype=torch.long, device=pred.device)
        sup = support[idx_t]
        supported = sup > 0
        n_supported = int(supported.sum().item())
        row = {
            "slice": key,
            "n_tags": len(idxs),
            "n_supported": n_supported,
            "support_sum": int(sup.sum().item()),
            "macro_f1": float(f1[idx_t][supported].mean().item())
            if n_supported
            else float("nan"),
            "mean_ap": float(ap[idx_t][supported].nanmean().item())
            if n_supported
            else float("nan"),
            "micro_f1": micro_f1(pred[:, idx_t], target[:, idx_t]),
        }
        rows.append(row)
    rows.sort(key=lambda r: -r["n_tags"])
    return rows
