"""Sentinel-augmented softmax groups + MaxSup regularization.

Sentinel groups relax the softmax-group contract from "exactly one member
present" to "at most one": a synthetic ``<none:group>`` vocab slot becomes the
CE target on applicable samples with no member label, and decode emits nothing
when it wins the argmax.

1. Resolution folds the sentinel slot into ``tag_indices`` (last) and records
   ``sentinel_index``; a vocab built without the slot degrades to legacy.
2. ``$category:<cat>`` member markers expand against the vocab.
3. CE supervises every applicable sample of a sentinel group (target =
   sentinel when label-less); legacy groups keep the has-label gate.
4. Sentinel cells are BCE-inert — the BCE term is invariant to their logits.
5. Decode never emits a sentinel: not as an argmax winner (group emits
   nothing), not via the sigmoid-threshold path.
6. ``maxsup_term`` matches the paper formula: batch-mean of z_max − mean(z)
   (arXiv:2502.15798 Eq. 8) — used by ``--ce_maxsup`` for the softmax groups
   and the rating / people heads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from anime_tools.captions import tag_groups as tg
from anime_tools.captions.group_router import (
    GroupRouter,
    compute_grouped_loss,
    maxsup_term,
)
from anime_tools.tagger.cli.eval_metrics import predict_with_inference_rule

# --------------------------------------------------------------------------- #
# Resolution layer
# --------------------------------------------------------------------------- #


def _groups_yaml_dict(sentinel: bool) -> dict:
    return {
        "version": 1,
        "medium": {
            "mode": "softmax",
            "tags": ["watercolor", "oil painting"],
            **({"sentinel": True} if sentinel else {}),
        },
    }


def test_resolve_folds_sentinel_last_and_records_index():
    groups = tg.from_dict(_groups_yaml_dict(sentinel=True))
    vocab_idx = {
        "watercolor": 0,
        "oil painting": 1,
        tg.sentinel_tag_name("medium"): 2,
    }
    resolved, dropped = tg.resolve_groups(groups, vocab_idx)
    (g,) = resolved
    assert g.sentinel_index == 2
    assert g.tag_indices == (0, 1, 2)
    assert g.tag_names[-1] == tg.sentinel_tag_name("medium")
    assert not dropped


def test_resolve_without_slot_degrades_to_legacy():
    groups = tg.from_dict(_groups_yaml_dict(sentinel=True))
    vocab_idx = {"watercolor": 0, "oil painting": 1}  # pre-sentinel vocab
    resolved, dropped = tg.resolve_groups(groups, vocab_idx)
    (g,) = resolved
    assert g.sentinel_index is None
    assert g.tag_indices == (0, 1)
    assert tg.sentinel_tag_name("medium") in dropped


def test_expand_category_members():
    groups = tg.from_dict(
        {
            "version": 1,
            "artist": {
                "mode": "softmax",
                "sentinel": True,
                "tags": ["$category:artist"],
            },
        }
    )
    name_to_cat = {"@mignon": "artist", "@sincos": "artist", "blush": "general"}
    out = tg.expand_category_members(groups, name_to_cat)
    (g,) = out.groups
    assert g.tags == ("@mignon", "@sincos")
    assert g.sentinel


def test_sentinel_name_roundtrip():
    name = tg.sentinel_tag_name("body_shape")
    assert tg.is_sentinel_name(name)
    assert not tg.is_sentinel_name("body_shape")
    assert not tg.is_sentinel_name("<3")


# --------------------------------------------------------------------------- #
# Loss layer
# --------------------------------------------------------------------------- #

# Vocab layout used below — 6 tags:
#   0,1  : members of softmax group G   2: G's sentinel slot
#   3    : "solo" (single-count)        4: "2girls" (multi-count)
#   5    : ungrouped free tag
_SENT = 2


def _router(mh: torch.Tensor, sentinel: bool) -> GroupRouter:
    names = ["red", "blue", tg.sentinel_tag_name("G"), "solo", "2girls", "free"]
    g: dict = {
        "name": "G",
        "mode": "softmax_when_solo",
        "tag_indices": [0, 1] + ([_SENT] if sentinel else []),
        "escape_indices": [],
    }
    if sentinel:
        g["sentinel_index"] = _SENT
    vocab = {
        "tags": [{"name": n, "index": i} for i, n in enumerate(names)],
        "groups": [g],
    }
    return GroupRouter.from_vocab(vocab, mh, device=torch.device("cpu"))


def _mh() -> torch.Tensor:
    # img0: solo, member 0 fires        → CE target = member 0
    # img1: solo, no member fires       → CE target = sentinel (sentinel mode)
    # img2: multi-subject, member fires → not applicable, BCE fallback
    return torch.tensor(
        [
            [1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )


def test_sentinel_ce_covers_labelless_applicable_samples():
    mh = _mh()
    torch.manual_seed(0)
    logits = torch.randn(3, 6)
    router = _router(mh, sentinel=True)
    (g,) = router.softmax_groups
    assert g.sentinel_local == 2  # slot folded last

    _, metrics = compute_grouped_loss(logits, mh, router)
    # Manual: CE over rows {0, 1}; row0 → member 0, row1 → sentinel (local 2).
    sel = logits[[0, 1]][:, [0, 1, _SENT]]
    tgt = torch.tensor([0, 2])
    expected = F.cross_entropy(sel, tgt)
    assert abs(metrics["ce_G"] - expected.item()) < 1e-6


def test_legacy_group_keeps_haslabel_gate():
    mh = _mh()
    torch.manual_seed(0)
    logits = torch.randn(3, 6)
    router = _router(mh, sentinel=False)
    _, metrics = compute_grouped_loss(logits, mh, router)
    # Only row0 (solo AND member label) gets CE.
    expected = F.cross_entropy(logits[[0]][:, [0, 1]], torch.tensor([0]))
    assert abs(metrics["ce_G"] - expected.item()) < 1e-6


def test_sentinel_cells_are_bce_inert():
    mh = _mh()
    torch.manual_seed(0)
    logits = torch.randn(3, 6)
    router = _router(mh, sentinel=True)
    _, m_a = compute_grouped_loss(logits, mh, router)
    wild = logits.clone()
    wild[:, _SENT] += 1e3  # blow up every sentinel logit
    _, m_b = compute_grouped_loss(wild, mh, router)
    assert abs(m_a["bce"] - m_b["bce"]) < 1e-6  # BCE never sees the slot
    assert m_a["ce_G"] != m_b["ce_G"]  # CE does


def test_sentinel_composes_with_maxsup():
    mh = _mh()
    torch.manual_seed(0)
    logits = torch.randn(3, 6)
    router = _router(mh, sentinel=True)
    _, m = compute_grouped_loss(logits, mh, router, label_smooth=0.05, ce_maxsup=True)
    sel = logits[[0, 1]][:, [0, 1, _SENT]]
    expected = F.cross_entropy(sel, torch.tensor([0, 2])) + 0.05 * maxsup_term(sel)
    assert abs(m["ce_G"] - expected.item()) < 1e-6


# --------------------------------------------------------------------------- #
# Decode layer
# --------------------------------------------------------------------------- #


def test_decode_sentinel_winner_emits_nothing():
    mh = _mh()
    router = _router(mh, sentinel=True)
    thresholds = torch.full((6,), 0.5)
    # Row is solo; sentinel logit dominates the group; member 1 would have
    # passed the sigmoid threshold — the group verdict must override it.
    logits = torch.tensor([[-4.0, 2.0, 6.0, 4.0, -4.0, -4.0]])
    pred = predict_with_inference_rule(logits, thresholds, router)
    assert not pred[0, 0] and not pred[0, 1]  # members cleared
    assert not pred[0, _SENT]  # sentinel never emitted
    assert pred[0, 3]  # unrelated tags untouched


def test_decode_member_winner_still_emits():
    mh = _mh()
    router = _router(mh, sentinel=True)
    thresholds = torch.full((6,), 0.5)
    logits = torch.tensor([[-4.0, 2.0, -6.0, 4.0, -4.0, -4.0]])
    pred = predict_with_inference_rule(logits, thresholds, router)
    assert pred[0, 1] and not pred[0, 0]
    assert not pred[0, _SENT]


# --------------------------------------------------------------------------- #
# MaxSup formula
# --------------------------------------------------------------------------- #


def test_maxsup_term_matches_paper_gradient():
    torch.manual_seed(0)
    z = torch.randn(4, 8, requires_grad=True)
    (g,) = torch.autograd.grad(maxsup_term(z), z)
    g = g * 4  # undo batch mean
    top1 = z.argmax(dim=1)
    K = z.shape[1]
    for i in range(4):
        for k in range(K):
            want = (1 - 1 / K) if k == top1[i] else -1 / K
            assert abs(g[i, k].item() - want) < 1e-6
