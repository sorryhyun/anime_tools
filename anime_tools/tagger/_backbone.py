"""What the two tagger backbones share — torch, but backbone-agnostic.

:mod:`~anime_tools.tagger.dbv4_backend` and
:mod:`~anime_tools.tagger.pixai_backend` both answer a :class:`BackboneOutput`
(``probs`` over their own columns + the ``hidden`` feature the sidecar trains
on), both have their columns joined onto our vocab by :func:`align_vocab`, and
both carry the same optional :class:`SidecarHead`. These lived in
``dbv4_backend`` while it was the only backbone; it still re-exports them, and
``Dbv4Output`` is kept as an alias of :class:`BackboneOutput`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save
from torch import nn

from anime_tools._json import read_json, write_json

__all__ = [
    "SIDECAR_META",
    "SIDECAR_WEIGHTS",
    "UNSUPPORTED_LOGIT",
    "BackboneOutput",
    "Card",
    "Dbv4Output",
    "SidecarHead",
    "VocabAlignment",
    "align_vocab",
    "default_dtype",
    "probs_to_logits",
    "rename_recovery_from_rules",
]

# Sigmoid probability floor/ceiling when converting projected probs back to
# logits for the group-argmax path; tags the backbone cannot emit get ``UNSUPPORTED_LOGIT``.
_PROB_EPS = 1e-6
UNSUPPORTED_LOGIT = -30.0


def default_dtype(device: torch.device | str) -> torch.dtype:
    """bfloat16 on CUDA, float32 everywhere else.

    bf16 is a win only where the hardware has kernels for it. On CPU torch emulates
    it: measured on caformer_b36 at 384px, 2.21 s/img against float32's 1.54 — 43%
    slower *and* 2e-2 off the float32 scores, which is the larger error of the two
    the tagger can make. (``bench/tagger_external/probe_position_rescore.py`` has
    picked per device this way all along; this is that rule, in the one place every
    caller goes through.)

    MPS is the same story and stays on the ``cuda``-only side of the test: measured
    2026-09-09 at batch 4, float32 87 ms/img against bfloat16's 102 and 1.3e-02 off.
    float16 is the one that would be faster there (77 ms), but at 2.4e-03 on the
    scores for a 12 % saving — not a trade this stage wants made for it silently.
    """
    return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32


SIDECAR_WEIGHTS = "sidecar.safetensors"
SIDECAR_META = "sidecar.json"


class Card(Protocol):
    """What :func:`align_vocab` and the tagger read off a backbone's card
    (``Dbv4Card`` / ``PixaiCard``)."""

    name_to_col: dict[str, int]
    rating_cols: dict[str, int]


@dataclass
class BackboneOutput:
    probs: torch.Tensor
    """``[B, n_classes]`` sigmoid probabilities over the backbone's columns (float32, cpu)."""
    hidden: torch.Tensor
    """``[B, d_hidden]`` the feature the sidecar trains on (dbv4: the MLP-head
    hidden, post fc1→act→norm; pixai: the attention-pool output; float32, cpu)."""


Dbv4Output = BackboneOutput
"""The name from when dbv4 was the only backbone."""


@dataclass
class VocabAlignment:
    ours_idx: torch.Tensor
    """LongTensor — our vocab indices the backbone can emit."""
    ext_idx: torch.Tensor
    """LongTensor — the matching backbone columns, same order as ``ours_idx``."""
    unmatched_by_category: dict[str, int]
    unmatched: list[tuple[int, str, str]]
    """``(our_index, name, category)`` for every tag the backbone cannot emit."""

    def supported_mask(self, n_tags: int) -> torch.Tensor:
        m = torch.zeros(n_tags, dtype=torch.bool)
        m[self.ours_idx] = True
        return m


def align_vocab(
    vocab_tags: Sequence[Mapping[str, object]],
    card: Card,
    rename_recovery: Mapping[str, str] | None = None,
) -> VocabAlignment:
    """Join our ``vocab.json['tags']`` onto a backbone card by name.

    Our names are space-separated (``rules.yaml`` may have renamed some — the
    recovered original is tried second); the card's ``name_to_col`` is already
    space-normalised (``load_dbv4_card`` / ``load_pixai_card``).
    """
    rename_recovery = rename_recovery or {}
    ours_idx: list[int] = []
    ext_idx: list[int] = []
    unmatched_by_cat: dict[str, int] = {}
    unmatched: list[tuple[int, str, str]] = []
    for t in vocab_tags:
        name = str(t["name"])
        j = card.name_to_col.get(name)
        if j is None and name in rename_recovery:
            j = card.name_to_col.get(rename_recovery[name])
        cat = str(t["category"])
        if j is None:
            unmatched_by_cat[cat] = unmatched_by_cat.get(cat, 0) + 1
            unmatched.append((int(t["index"]), name, cat))
            continue
        ours_idx.append(int(t["index"]))
        ext_idx.append(j)
    return VocabAlignment(
        ours_idx=torch.tensor(ours_idx, dtype=torch.long),
        ext_idx=torch.tensor(ext_idx, dtype=torch.long),
        unmatched_by_category=unmatched_by_cat,
        unmatched=unmatched,
    )


def rename_recovery_from_rules(rules) -> dict[str, str]:
    """``rules.yaml`` replacements are ``src → tgt``; alignment needs ``tgt → src``."""
    return {tgt: src for src, tgt in rules.replacements}


# Sidecar head — what the backbone cannot say.


class SidecarHead(nn.Module):
    """Linear head on the backend's hidden feature.

    Output layout: ``[bce_indices…] ++ [people_count_labels…]``. BCE rows are
    our-vocab tag indices (copyright + dataset-only characters); the trailing
    block is the softmax people-count bucket (empty list = no people head).
    """

    def __init__(
        self,
        d_in: int,
        bce_indices: Sequence[int],
        people_count_labels: Sequence[str] = (),
        feature: str = "mlp_hidden",
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.bce_indices: tuple[int, ...] = tuple(int(i) for i in bce_indices)
        self.people_count_labels: tuple[str, ...] = tuple(people_count_labels)
        self.feature = feature
        n_out = len(self.bce_indices) + len(self.people_count_labels)
        self.fc = nn.Linear(self.d_in, n_out)

    @property
    def n_bce(self) -> int:
        return len(self.bce_indices)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.fc(hidden)
        bce = out[:, : self.n_bce]
        people = out[:, self.n_bce :] if self.people_count_labels else None
        return bce, people

    def meta(self) -> dict[str, object]:
        return {
            "d_in": self.d_in,
            "feature": self.feature,
            "bce_indices": list(self.bce_indices),
            "people_count_labels": list(self.people_count_labels),
        }

    def save(
        self, ckpt_dir: str | Path, extra_meta: Mapping[str, object] | None = None
    ) -> None:
        ckpt_dir = Path(ckpt_dir)
        st_save(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
            str(ckpt_dir / SIDECAR_WEIGHTS),
        )
        meta = self.meta()
        if extra_meta:
            meta.update(dict(extra_meta))
        write_json(ckpt_dir / SIDECAR_META, meta)

    @classmethod
    def load(cls, ckpt_dir: str | Path) -> SidecarHead | None:
        ckpt_dir = Path(ckpt_dir)
        w, m = ckpt_dir / SIDECAR_WEIGHTS, ckpt_dir / SIDECAR_META
        if not (w.exists() and m.exists()):
            return None
        meta = read_json(m)
        head = cls(
            d_in=int(meta["d_in"]),
            bce_indices=meta.get("bce_indices", []),
            people_count_labels=meta.get("people_count_labels", []),
            feature=str(meta.get("feature", "mlp_hidden")),
        )
        head.load_state_dict(st_load(str(w)))
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
        return head


def probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(_PROB_EPS, 1.0 - _PROB_EPS)
    return torch.log(p) - torch.log1p(-p)
