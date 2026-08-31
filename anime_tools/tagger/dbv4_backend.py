"""dbv4 backend for :class:`AnimaTagger` — an off-the-shelf danbooru tagger
behind the Anima tagger contract.

The ``animetimm/*.dbv4-full`` family (caformer_b36 / convnextv2_huge, …) is a
full backbone fine-tuned on all of danbooru (12,476 tags). Measured on our own
held-out split it is 2–2.5× better than the in-house frozen-PE linear probe on
every slice (``bench/tagger_external/``, 2026-08-26), so the tagger's
*inference contract* stays and the model behind it is swapped.

What this module provides:

* :func:`load_dbv4_card` — the repo's ``selected_tags.csv`` (name / category /
  ``best_threshold``) and ``meta.json`` (timm arch kwargs), fetched through
  the repo-wide ``hf_download`` helper. **The weights are GPL-3.0 and gated**:
  they are never vendored or bundled; the user's own HF token (which implies
  accepting the repo terms) is what downloads them.
* :func:`align_vocab` — the single join point between dbv4's snake_case names
  and our space-separated vocab (``rules.yaml`` renames recovered through
  ``rename_recovery``). Everything downstream indexes by *our* vocab index.
* :class:`Dbv4Backend` — image → ``(probs over dbv4 tags, MLP-head hidden
  feature)``. The hidden feature (post ``fc1 → act → norm``, 3072-d on
  caformer_b36) is what the sidecar head consumes.
* :class:`SidecarHead` — a linear head over that hidden feature that emits
  **only** what dbv4 cannot say: copyright, dataset-only characters, and the
  8-way people-count bucket. ``@artist`` is deliberately not part of it
  (2026-08-27 decision: artist attribution is not a tagger goal any more).

Categories in dbv4's card: 0 general, 4 character, 9 rating — no copyright,
no artist. Rating names are danbooru's (``general``/``questionable``) and are
mapped onto Anima's ``safe``/``nsfw`` here.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save
from torch import nn

logger = logging.getLogger(__name__)

from anime_tools._device import resolve_device
from anime_tools._json import read_json, write_json
from anime_tools.tagger.dbv4_meta import (
    DEFAULT_DBV4_ARCH,
    DEFAULT_DBV4_IMG_SIZE,
    DEFAULT_DBV4_REPO,
    gated_hint,
)

# danbooru rating name (dbv4 card, category 9) -> Anima rating name.
DBV4_RATING_MAP: dict[str, str] = {
    "general": "safe",
    "sensitive": "sensitive",
    "questionable": "nsfw",
    "explicit": "explicit",
}

# Sigmoid probability floor/ceiling when converting projected probs back to
# logits for the group-argmax path; tags dbv4 cannot emit get ``UNSUPPORTED_LOGIT``.
_PROB_EPS = 1e-6
UNSUPPORTED_LOGIT = -30.0

SIDECAR_WEIGHTS = "sidecar.safetensors"
SIDECAR_META = "sidecar.json"


# --------------------------------------------------------------------------- #
# Card (tag list + thresholds) and vocab alignment
# --------------------------------------------------------------------------- #


@dataclass
class Dbv4Card:
    repo: str
    rows: list[dict]
    """``selected_tags.csv`` rows in column order (column j == row j)."""
    model_args: dict[str, object]
    """``meta.json['model_args']`` — timm ``create_model`` kwargs (``act_layer`` …)."""
    name_to_col: dict[str, int] = field(default_factory=dict)
    """space-normalised tag name → column, ratings excluded."""
    rating_cols: dict[str, int] = field(default_factory=dict)
    """Anima rating name → column."""

    @property
    def n_classes(self) -> int:
        return len(self.rows)

    def best_thresholds(self) -> torch.Tensor:
        return torch.tensor([float(r["best_threshold"]) for r in self.rows])


def load_dbv4_card(
    repo: str = DEFAULT_DBV4_REPO, revision: str | None = None
) -> Dbv4Card:
    """Fetch + parse ``selected_tags.csv`` and ``meta.json`` for ``repo``."""
    from anime_tools._hf import hf_download

    hint = gated_hint(repo)
    tags_csv = hf_download(
        what=f"dbv4 tag list ({repo})",
        hint=hint,
        repo_id=repo,
        filename="selected_tags.csv",
        revision=revision,
    )
    meta_p = hf_download(
        what=f"dbv4 meta ({repo})",
        hint=hint,
        repo_id=repo,
        filename="meta.json",
        revision=revision,
    )
    meta = read_json(meta_p)
    with open(tags_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    card = Dbv4Card(repo=repo, rows=rows, model_args=dict(meta.get("model_args", {})))
    for j, r in enumerate(rows):
        if int(r["category"]) == 9:
            our = DBV4_RATING_MAP.get(r["name"])
            if our:
                card.rating_cols[our] = j
            continue
        card.name_to_col[r["name"].replace("_", " ")] = j
    return card


@dataclass
class VocabAlignment:
    ours_idx: torch.Tensor
    """LongTensor — our vocab indices that dbv4 can emit."""
    ext_idx: torch.Tensor
    """LongTensor — the matching dbv4 columns, same order as ``ours_idx``."""
    unmatched_by_category: dict[str, int]
    unmatched: list[tuple[int, str, str]]
    """``(our_index, name, category)`` for every tag dbv4 cannot emit."""

    def supported_mask(self, n_tags: int) -> torch.Tensor:
        m = torch.zeros(n_tags, dtype=torch.bool)
        m[self.ours_idx] = True
        return m


def align_vocab(
    vocab_tags: Sequence[Mapping[str, object]],
    card: Dbv4Card,
    rename_recovery: Mapping[str, str] | None = None,
) -> VocabAlignment:
    """Join our ``vocab.json['tags']`` onto the dbv4 card by name.

    Our names are space-separated (``rules.yaml`` may have renamed some — the
    recovered original is tried second); dbv4's are snake_case, normalised in
    :func:`load_dbv4_card`.
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


# --------------------------------------------------------------------------- #
# Image preprocessing (matches the card's preprocess.json: pad-white square →
# bicubic resize → ImageNet normalisation)
# --------------------------------------------------------------------------- #


def pad_square_white(im: Image.Image) -> Image.Image:
    w, h = im.size
    if w == h:
        return im
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def preprocess_dbv4(im: Image.Image, size: int) -> torch.Tensor:
    """PIL → ``[3, size, size]`` float in [0, 1] (normalisation happens in the backend)."""
    im = pad_square_white(im.convert("RGB")).resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #


@dataclass
class Dbv4Output:
    probs: torch.Tensor
    """``[B, n_classes]`` sigmoid probabilities over the dbv4 columns (float32, cpu)."""
    hidden: torch.Tensor
    """``[B, d_hidden]`` MLP-head hidden feature (post fc1→act→norm; float32, cpu)."""


class Dbv4Backend:
    """Lazy-loading timm wrapper around a dbv4 checkpoint.

    ``forward`` splits timm's ``MlpHead`` so the hidden feature the sidecar
    trains on comes out of the same pass as the tag probabilities.
    """

    def __init__(
        self,
        repo: str = DEFAULT_DBV4_REPO,
        arch: str = DEFAULT_DBV4_ARCH,
        img_size: int = DEFAULT_DBV4_IMG_SIZE,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        revision: str | None = None,
        card: Dbv4Card | None = None,
    ):
        self.repo = repo
        self.arch = arch
        self.img_size = int(img_size)
        self.device = torch.device(resolve_device(device))
        self.dtype = dtype
        self.revision = revision
        self._card = card
        self._model: nn.Module | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    @property
    def card(self) -> Dbv4Card:
        if self._card is None:
            self._card = load_dbv4_card(self.repo, revision=self.revision)
        return self._card

    @property
    def d_hidden(self) -> int:
        return int(self.model.head.fc.fc1.out_features)

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> nn.Module:
        import timm

        from anime_tools._hf import hf_download

        card = self.card
        weights = hf_download(
            what=f"dbv4 weights ({self.repo})",
            hint=gated_hint(self.repo),
            repo_id=self.repo,
            filename="model.safetensors",
            revision=self.revision,
        )
        kwargs = {}
        if "act_layer" in card.model_args:
            kwargs["act_layer"] = card.model_args["act_layer"]
        model = timm.create_model(
            self.arch, pretrained=False, num_classes=card.n_classes, **kwargs
        )
        missing, unexpected = model.load_state_dict(st_load(weights), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"dbv4 state_dict mismatch for arch={self.arch}: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        if not hasattr(model.head, "fc") or not hasattr(model.head.fc, "fc1"):
            raise RuntimeError(
                f"dbv4 backend expects a timm MetaFormer MlpHead (head.fc.fc1/fc2); "
                f"arch={self.arch} has {type(model.head).__name__}"
            )
        model.to(self.device, dtype=self.dtype).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        pcfg = model.pretrained_cfg
        self._mean = torch.tensor(pcfg.get("mean", (0.485, 0.456, 0.406))).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor(pcfg.get("std", (0.229, 0.224, 0.225))).view(
            1, 3, 1, 1
        )
        logger.info(
            "Dbv4Backend: %s (%s, %d classes, %dpx) on %s",
            self.repo,
            self.arch,
            card.n_classes,
            self.img_size,
            self.device,
        )
        return model

    @torch.no_grad()
    def forward_tensor(self, x01: torch.Tensor) -> Dbv4Output:
        """``[B, 3, S, S]`` in [0, 1] → probs + hidden."""
        model = self.model
        x = ((x01 - self._mean) / self._std).to(self.device, dtype=self.dtype)
        feats = model.forward_features(x)
        pooled = model.forward_head(feats, pre_logits=True)
        fc = model.head.fc
        hidden = fc.norm(fc.act(fc.fc1(pooled)))
        logits = fc.fc2(hidden)
        return Dbv4Output(
            probs=logits.float().sigmoid().cpu(), hidden=hidden.float().cpu()
        )

    @torch.no_grad()
    def forward(self, images: Sequence[Image.Image]) -> Dbv4Output:
        x = torch.stack([preprocess_dbv4(im, self.img_size) for im in images])
        return self.forward_tensor(x)


# --------------------------------------------------------------------------- #
# Sidecar head — what dbv4 cannot say
# --------------------------------------------------------------------------- #


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
