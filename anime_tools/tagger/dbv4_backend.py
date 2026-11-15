"""dbv4 backend for :class:`AnimaTagger` — the off-the-shelf
``animetimm/*.dbv4-full`` danbooru tagger behind the Anima tagger contract.

* :func:`load_dbv4_card` — the repo's ``selected_tags.csv`` + ``meta.json``.
  The weights are GPL-3.0 and gated: never vendored, only ever fetched under
  the user's own HF token.
* :func:`align_vocab` — joins dbv4's snake_case names onto our space-separated
  vocab (``rules.yaml`` renames recovered through ``rename_recovery``);
  everything downstream indexes by *our* vocab index.
* :class:`Dbv4Backend` — image → ``(probs over dbv4 tags, MLP-head hidden
  feature)``; the hidden feature is what the sidecar consumes.
* :class:`SidecarHead` — a linear head emitting only what dbv4 cannot say:
  copyright, dataset-only characters, the 8-way people-count bucket.

:func:`align_vocab`, :class:`SidecarHead`, ``Dbv4Output`` (now
:class:`BackboneOutput`), :func:`probs_to_logits` and :func:`default_dtype`
serve the pixai backbone too, so they live in
:mod:`anime_tools.tagger._backbone`; they are re-exported here.

dbv4 card categories: 0 general, 4 character, 9 rating — no copyright, no
artist. Its danbooru rating names are mapped onto Anima's here.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as st_load
from torch import nn

from anime_tools._device import resolve_device
from anime_tools._json import read_json

# The backbone-agnostic half lives in ``_backbone``; re-exported here so every
# ``from anime_tools.tagger.dbv4_backend import …`` written before the split
# still resolves.
from anime_tools.tagger._backbone import (
    SIDECAR_META,
    SIDECAR_WEIGHTS,
    UNSUPPORTED_LOGIT,
    BackboneOutput,
    Dbv4Output,
    SidecarHead,
    VocabAlignment,
    align_vocab,
    default_dtype,
    probs_to_logits,
    rename_recovery_from_rules,
)
from anime_tools.tagger.dbv4_meta import (
    DEFAULT_DBV4_ARCH,
    DEFAULT_DBV4_IMG_SIZE,
    DEFAULT_DBV4_REPO,
    Backbone,
    gated_hint,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DBV4_RATING_MAP",
    "SIDECAR_META",
    "SIDECAR_WEIGHTS",
    "UNSUPPORTED_LOGIT",
    "BackboneOutput",
    "Dbv4Backend",
    "Dbv4Card",
    "Dbv4Output",
    "SidecarHead",
    "VocabAlignment",
    "align_vocab",
    "default_dtype",
    "load_dbv4_card",
    "pad_square_white",
    "preprocess_dbv4",
    "probs_to_logits",
    "rename_recovery_from_rules",
]

# danbooru rating name (dbv4 card, category 9) -> Anima rating name.
DBV4_RATING_MAP: dict[str, str] = {
    "general": "safe",
    "sensitive": "sensitive",
    "questionable": "nsfw",
    "explicit": "explicit",
}

# Card (tag list + thresholds) and vocab alignment.


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


# Image preprocessing — matches the card's preprocess.json: pad-white square →
# bicubic resize → ImageNet normalisation.


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


# Backend.


class Dbv4Backend:
    """Lazy-loading timm wrapper around a dbv4 checkpoint.

    ``forward`` splits timm's ``MlpHead`` so the hidden feature the sidecar
    trains on comes out of the same pass as the tag probabilities.
    """

    kind = "dbv4"
    feature = "mlp_hidden"
    """What ``hidden`` is — stamped into the feature cache and the sidecar."""
    preprocess_image = staticmethod(preprocess_dbv4)
    """``(PIL, size) → [3, S, S]`` — on the class so a DataLoader worker can
    take it without the instance (and its model)."""

    def __init__(
        self,
        repo: str = DEFAULT_DBV4_REPO,
        arch: str = DEFAULT_DBV4_ARCH,
        img_size: int = DEFAULT_DBV4_IMG_SIZE,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        revision: str | None = None,
        card: Dbv4Card | None = None,
    ):
        self.repo = repo
        self.arch = arch
        self.img_size = int(img_size)
        self.device = torch.device(resolve_device(device))
        self.dtype = dtype if dtype is not None else default_dtype(self.device)
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
    def backbone(self) -> Backbone:
        return Backbone("dbv4", self.repo, self.revision)

    @property
    def d_hidden(self) -> int:
        return int(self.model.head.fc.fc1.out_features)

    def preprocess(self, im: Image.Image) -> torch.Tensor:
        return preprocess_dbv4(im, self.img_size)

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
        # On the device, because ``forward_tensor`` normalises there: doing it
        # on the CPU costs a float32 pass over the whole batch before the
        # transfer that has to happen anyway.
        self._mean = (
            torch.tensor(pcfg.get("mean", (0.485, 0.456, 0.406)))
            .view(1, 3, 1, 1)
            .to(self.device)
        )
        self._std = (
            torch.tensor(pcfg.get("std", (0.229, 0.224, 0.225)))
            .view(1, 3, 1, 1)
            .to(self.device)
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
    def forward_tensor(self, x01: torch.Tensor) -> BackboneOutput:
        """``[B, 3, S, S]`` in [0, 1] → probs + hidden.

        The transfer comes first and the normalisation runs on the device: the
        arithmetic is one pass over ``B x 3 x S x S`` floats either way, and the
        CPU is the slower place to do it (and the one holding up the decode of
        the next batch).
        """
        model = self.model
        x = x01.to(self.device, dtype=torch.float32)
        x = ((x - self._mean) / self._std).to(self.dtype)
        feats = model.forward_features(x)
        pooled = model.forward_head(feats, pre_logits=True)
        fc = model.head.fc
        hidden = fc.norm(fc.act(fc.fc1(pooled)))
        logits = fc.fc2(hidden)
        return BackboneOutput(
            probs=logits.float().sigmoid().cpu(), hidden=hidden.float().cpu()
        )

    @torch.no_grad()
    def forward(self, images: Sequence[Image.Image]) -> BackboneOutput:
        x = torch.stack([self.preprocess(im) for im in images])
        return self.forward_tensor(x)
