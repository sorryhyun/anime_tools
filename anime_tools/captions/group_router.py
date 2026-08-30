"""Typed tag-group routing for the Anima Tagger — ``GroupRouter`` + grouped loss.

Promoted out of the archived PE-head trainer (``_archive/anima_tagger_training/
scripts/train_common.py``, 2026-08-27) because the router outlives it: the
dbv4 backend's inference rule (``anime_tools/tagger/cli/eval_metrics.py``), the
threshold calibrator, and ``bench/tagger_external`` all resolve softmax groups
through it. ``compute_grouped_loss`` stays with it so the sentinel / escape /
inactive-negative semantics remain testable (``tests/test_tagger_sentinel_groups.py``,
``tests/test_grouped_loss_negweight.py``) even though no shipped trainer calls
it today.

A vocab built without ``--groups`` produces an empty router: BCE applies
everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from anime_tools.captions.taxonomy import _COUNT_RE

__all__ = [
    "GroupRouter",
    "compute_grouped_loss",
    "maxsup_term",
    "pos_weight_sqrt",
]


def maxsup_term(logits: torch.Tensor) -> torch.Tensor:
    """MaxSup regularizer (arXiv:2502.15798): batch-mean of ``z_max − mean(z)``.

    Added as ``ε · maxsup_term(logits)`` on top of *hard* CE, in place of
    ``label_smoothing=ε`` — same regularization pressure as LS on correct
    predictions, without LS's error-amplification term on misclassified ones.
    """
    return (logits.max(dim=1).values - logits.mean(dim=1)).mean()


def pos_weight_sqrt(multi_hot: torch.Tensor) -> torch.Tensor:
    """``sqrt(n_neg / n_pos)`` per tag — softens BCE long-tail without overshoot."""
    n_pos = multi_hot.sum(dim=0).clamp_min(1.0)
    n_neg = multi_hot.shape[0] - n_pos
    return torch.sqrt(n_neg / n_pos)


@dataclass
class _SoftmaxGroup:
    """One softmax group projected onto trainer-side tensor indices."""

    name: str
    mode: str  # "softmax_when_solo" | "softmax"
    tag_indices: torch.Tensor  # LongTensor [K_g] (includes sentinel when present)
    escape_indices: torch.Tensor  # LongTensor [E_g]
    # Local position of the group's synthetic "none of these" class inside
    # ``tag_indices``, or None for legacy exactly-one groups. With a sentinel,
    # CE fires on every applicable sample (target = sentinel when no member
    # label), instead of only on samples carrying a member label.
    sentinel_local: int | None = None


@dataclass
class GroupRouter:
    """Per-batch loss routing for typed tag groups.

    Built once at trainer init from ``vocab.json[groups]``. Maintains:

    * ``bce_pos_weight`` — full ``[n_tags]`` pos-weight, same as the
      pre-grouping trainer. BCE applies to all tags by default; the
      :func:`compute_grouped_loss` helper masks out (sample, tag)
      positions where CE fires for that sample-group pair.
    * ``softmax_groups`` — per-group ``(mode, tag_indices, escape_indices)``;
      CE applies on these, gated by solo/escape for ``softmax_when_solo``.
    * ``softmax_member_indices`` — union of all softmax-group tag indices.
      Used by the calibrator + the inference rule to skip those tags
      from sigmoid-threshold F1 / threshold sweep (they're argmax-only at
      inference, so per-tag thresholds don't apply).
    * ``solo_indices`` / ``multi_indices`` — vocab indices used to detect
      single-subject samples at runtime from ``multi_hot``.

    A vocab built without ``--groups`` produces an empty router: BCE
    applies everywhere and behavior matches the pre-grouping trainer
    exactly.
    """

    n_tags: int
    bce_pos_weight: torch.Tensor  # FloatTensor [n_tags]
    softmax_groups: list[_SoftmaxGroup] = field(default_factory=list)
    softmax_member_indices: torch.Tensor | None = None  # LongTensor [Σ K_g]
    # All sentinel vocab indices across groups — CE-only slots, masked from
    # BCE unconditionally (their multi_hot column is always 0 by construction).
    sentinel_indices: torch.Tensor | None = None  # LongTensor [S]
    solo_indices: torch.Tensor | None = None  # LongTensor [s]
    multi_indices: torch.Tensor | None = None  # LongTensor [m]
    # Per-tag group id over ALL groups (any mode), for group-conditional
    # negative weighting (:func:`compute_grouped_loss` ``inactive_neg_weight``).
    # Value in ``[0, n_group_slots)``; the sentinel ``n_group_slots`` marks an
    # ungrouped tag (never down-weighted). ``None`` when the vocab has no groups.
    group_of_tag: torch.Tensor | None = None  # LongTensor [n_tags]
    n_group_slots: int = 0

    @classmethod
    def from_vocab(
        cls,
        vocab_dict: dict,
        train_multi_hot: torch.Tensor,
        device: torch.device,
    ) -> GroupRouter:
        """Build the router from a vocab dict + the train split's multi_hot."""
        n_tags = int(train_multi_hot.shape[1])
        groups_raw: list[dict] = list(vocab_dict.get("groups") or [])

        softmax_member: list[int] = []
        softmax_groups: list[_SoftmaxGroup] = []
        sentinel_idx_list: list[int] = []
        for g in groups_raw:
            mode = g["mode"]
            if mode in ("softmax_when_solo", "softmax"):
                idxs = list(g.get("tag_indices") or [])
                esc = list(g.get("escape_indices") or [])
                if not idxs:
                    continue
                sent = g.get("sentinel_index")
                sentinel_local: int | None = None
                if sent is not None:
                    sentinel_local = idxs.index(int(sent))
                    sentinel_idx_list.append(int(sent))
                softmax_member.extend(idxs)
                softmax_groups.append(
                    _SoftmaxGroup(
                        name=str(g["name"]),
                        mode=mode,
                        tag_indices=torch.tensor(idxs, dtype=torch.long, device=device),
                        escape_indices=torch.tensor(
                            esc, dtype=torch.long, device=device
                        ),
                        sentinel_local=sentinel_local,
                    )
                )
            # multilabel groups are documentation-only — they stay in BCE.

        softmax_member_indices = (
            torch.tensor(sorted(set(softmax_member)), dtype=torch.long, device=device)
            if softmax_member
            else None
        )

        # Full-vocab pos-weight (matches pre-grouping trainer); per-batch
        # masking knocks out the (sample, group_tag) positions CE supervises.
        bce_pos_weight = pos_weight_sqrt(train_multi_hot).to(device)

        # ``solo`` is a non-count membership tag — gelcrawl writes it alongside
        # ``1girl``/``1boy`` when there's exactly one figure.
        single_count_names = {"solo", "1girl", "1boy", "1other"}
        solo_idx_list: list[int] = []
        multi_idx_list: list[int] = []
        for t in vocab_dict.get("tags", []):
            name = t["name"]
            idx = int(t["index"])
            if name in single_count_names:
                solo_idx_list.append(idx)
            elif _COUNT_RE.match(name):
                multi_idx_list.append(idx)
        solo_indices = (
            torch.tensor(solo_idx_list, dtype=torch.long, device=device)
            if solo_idx_list
            else None
        )
        multi_indices = (
            torch.tensor(multi_idx_list, dtype=torch.long, device=device)
            if multi_idx_list
            else None
        )

        # Per-tag group id over ALL groups (any mode, incl. multilabel) — drives
        # group-conditional negative weighting. tag_to_group is ≤1 group/tag so
        # no collision. Ungrouped tags get the sentinel ``n_group_slots``.
        n_group_slots = len(groups_raw)
        group_of_tag: torch.Tensor | None = None
        if n_group_slots > 0:
            got = [n_group_slots] * n_tags
            for gid, g in enumerate(groups_raw):
                for ti in g.get("tag_indices") or []:
                    got[int(ti)] = gid
            group_of_tag = torch.tensor(got, dtype=torch.long, device=device)

        return cls(
            n_tags=n_tags,
            bce_pos_weight=bce_pos_weight,
            softmax_groups=softmax_groups,
            softmax_member_indices=softmax_member_indices,
            sentinel_indices=(
                torch.tensor(sorted(sentinel_idx_list), dtype=torch.long, device=device)
                if sentinel_idx_list
                else None
            ),
            solo_indices=solo_indices,
            multi_indices=multi_indices,
            group_of_tag=group_of_tag,
            n_group_slots=n_group_slots,
        )

    def is_active(self) -> bool:
        return bool(self.softmax_groups)

    def solo_mask(self, multi_hot: torch.Tensor) -> torch.Tensor:
        """``[B] bool`` — True when sample is single-subject.

        Single-subject = at least one of ``solo``/``1girl``/``1boy``/``1other``
        fires AND no other count tag (``2+girls``, ``multiple_*``, …)
        fires. When the vocab carries no count tags at all (degenerate),
        every sample is treated as solo so the trainer doesn't silently
        skip every CE update.
        """
        B = multi_hot.shape[0]
        if self.solo_indices is None:
            # No solo signal in the vocab. Be permissive — assume solo.
            return torch.ones(B, dtype=torch.bool, device=multi_hot.device)
        has_single = multi_hot[:, self.solo_indices].any(dim=1)
        if self.multi_indices is None:
            return has_single
        has_multi = multi_hot[:, self.multi_indices].any(dim=1)
        return has_single & ~has_multi


def compute_grouped_loss(
    tag_logits: torch.Tensor,  # [B, n_tags]
    multi_hot: torch.Tensor,  # [B, n_tags]
    router: GroupRouter,
    label_smooth: float = 0.0,
    inactive_neg_weight: float = 1.0,
    ce_maxsup: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``(total_tag_loss, per_group_metrics_for_logging)``.

    BCE applies element-wise across all (sample, tag) positions. For
    each softmax group, samples where (gating allows AND a group label
    fires) get K-way CE on the group's logits — and BCE for those
    (sample, group_tag) positions is masked out so we don't double-count
    supervision. Samples without that gate (multi-subject, escape, no
    in-group label) keep BCE on the group's tags as a fallback.

    Groups carrying a sentinel slot (``sentinel_local`` set) relax the
    exactly-one assumption to at-most-one: CE fires on *every* applicable
    sample, targeting the sentinel class when no member label is present.
    Sentinel cells are excluded from BCE unconditionally — they are CE-only.

    ``label_smooth`` (ε ∈ [0, 1)) applies classic label smoothing as a
    train-time regularizer against the overconfidence that blows up the
    memorized-train / held-out gap on this head. The per-tag BCE targets
    soften symmetrically to ``[ε/2, 1−ε/2]`` (the binary special case of
    redistributing ε mass over the two classes), and the same ε feeds the
    softmax-group ``cross_entropy`` so both tag objectives smooth
    consistently. Pass ``0.0`` (default) on the val/metric path so reported
    loss stays the true unsmoothed objective and is comparable across ε.

    ``ce_maxsup`` swaps the softmax-group regularizer from label smoothing
    to MaxSup (arXiv:2502.15798): hard CE plus ``ε·(z_max − mean(z))`` over
    the group logits. Same regularization term as LS when the prediction is
    correct, but drops LS's error-amplification term (LS penalizes z_gt even
    when a *larger* wrong logit exists, reinforcing confident mistakes —
    common in genuinely ambiguous exclusive groups like hair/eye color).
    BCE targets keep the binary smoothing above: per-tag sigmoid has no
    competing-logit structure, so the MaxSup analysis doesn't apply there
    (its binary degenerate form is a global |z| penalty ≈ Logit Penalty,
    which the paper itself shows harms representations). ε=0 is inert either
    way, so the unsmoothed val/metric path is unaffected by this flag.

    ``inactive_neg_weight`` (λ ∈ (0, 1]) implements group-conditional negative
    weighting: a negative (sample, tag) cell whose group is *inactive* for that
    sample (no tag in the group fired → the annotator likely never attended to
    the category, so the negative may be a missing label) gets its BCE
    contribution scaled by λ. Positives, active-group negatives, and ungrouped
    tags are untouched. λ=1.0 (default) is bit-identical to the un-weighted
    path. Phase-0 gold-check (_archive/bench/tagger_groups) found inactive-group
    negatives only *mildly* less reliable than active-group ones, so a gentle
    λ≈0.6–0.75 is the intended operating range — masking (λ→0) would trade away
    too much precision.

    Returned metrics: ``"bce"`` (mean of unmasked BCE entries) plus
    ``f"ce_{group_name}"`` for each softmax group; loss curves stay
    separable in TensorBoard.
    """
    B, n_tags = tag_logits.shape
    metrics: dict[str, float] = {}

    # Label-smoothed BCE targets: 1 → 1−ε/2, 0 → ε/2 (ε=0 leaves multi_hot
    # untouched — bit-identical to the un-smoothed path).
    bce_target = (
        multi_hot * (1.0 - label_smooth) + 0.5 * label_smooth
        if label_smooth > 0.0
        else multi_hot
    )

    # Element-wise BCE-with-logits — we'll mask and reduce manually.
    bce_per_elem = F.binary_cross_entropy_with_logits(
        tag_logits,
        bce_target,
        pos_weight=router.bce_pos_weight,
        reduction="none",
    )

    # BCE applies everywhere by default; CE-supervised positions get masked off below.
    bce_mask = torch.ones(B, n_tags, dtype=torch.bool, device=tag_logits.device)
    # Sentinel slots are CE-only: their multi_hot column is always 0, so BCE
    # would just push them down everywhere, fighting the CE that raises them
    # on group-inactive samples.
    if router.sentinel_indices is not None:
        bce_mask[:, router.sentinel_indices] = False

    ce_total = tag_logits.new_zeros(())
    if router.softmax_groups:
        solo_mask = router.solo_mask(multi_hot)
        for g in router.softmax_groups:
            if g.escape_indices.numel() > 0:
                has_escape = multi_hot.index_select(1, g.escape_indices).any(dim=1)
            else:
                has_escape = torch.zeros_like(solo_mask)
            if g.mode == "softmax_when_solo":
                applicable = solo_mask & ~has_escape
            else:  # "softmax"
                applicable = ~has_escape

            group_logits = tag_logits.index_select(1, g.tag_indices)  # [B, K_g]
            group_target = multi_hot.index_select(1, g.tag_indices)  # [B, K_g]
            has_label = group_target.sum(dim=1) > 0
            # Sentinel groups supervise every applicable sample — no member
            # label means the CE target IS the sentinel class ("none of
            # these"), which is also what lets decode reject instead of
            # always emitting an argmax winner.
            if g.sentinel_local is not None:
                ce_samples = applicable
            else:
                ce_samples = applicable & has_label
            n_keep = int(ce_samples.sum().item())
            if n_keep == 0:
                metrics[f"ce_{g.name}"] = 0.0
                continue
            sel_logits = group_logits[ce_samples]  # [n_keep, K_g]
            sel_target = group_target[ce_samples].argmax(dim=1)  # [n_keep]
            if g.sentinel_local is not None:
                sel_target = torch.where(
                    has_label[ce_samples],
                    sel_target,
                    torch.full_like(sel_target, g.sentinel_local),
                )
            if ce_maxsup and label_smooth > 0.0:
                l_ce = F.cross_entropy(
                    sel_logits, sel_target
                ) + label_smooth * maxsup_term(sel_logits)
            else:
                l_ce = F.cross_entropy(
                    sel_logits, sel_target, label_smoothing=label_smooth
                )
            ce_total = ce_total + l_ce
            metrics[f"ce_{g.name}"] = float(l_ce.detach().item())

            # Mask BCE for the supervised (sample, group_tag) cells; broadcast
            # indexing touches the cartesian product.
            ce_idx = ce_samples.nonzero(as_tuple=False).squeeze(1)
            bce_mask[ce_idx[:, None], g.tag_indices[None, :]] = False

    # Weighted mean over surviving (un-CE-masked) cells. ``cell_w`` is 1.0
    # everywhere except inactive-group negatives, which get λ — so both the
    # numerator and the denominator shrink consistently (a proper weighted
    # mean, not a magnitude rescale).
    cell_w = bce_mask.float()
    if inactive_neg_weight != 1.0 and router.group_of_tag is not None:
        G = router.n_group_slots
        # Per-group positive count this batch via scatter-add over tag→group.
        gpc = multi_hot.new_zeros(B, G + 1)
        gpc.index_add_(1, router.group_of_tag, multi_hot)
        active = gpc > 0  # [B, G+1]; column G is the ungrouped sentinel bucket
        per_tag_active = active.gather(
            1, router.group_of_tag.unsqueeze(0).expand(B, -1)
        )  # [B, n_tags]
        grouped = (router.group_of_tag != G).unsqueeze(0)  # [1, n_tags]
        is_neg = multi_hot == 0
        downweight = is_neg & grouped & ~per_tag_active  # [B, n_tags]
        cell_w = torch.where(downweight, cell_w * inactive_neg_weight, cell_w)
    bce_denom = cell_w.sum().clamp_min(1.0)
    l_bce = (bce_per_elem * cell_w).sum() / bce_denom
    metrics["bce"] = float(l_bce.detach().item())

    return l_bce + ce_total, metrics
