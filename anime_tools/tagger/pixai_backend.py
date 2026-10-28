"""pixai backend for :class:`AnimaTagger` — ``pixai-labs/pixai-tagger-v1.0``
behind the Anima tagger contract.

* :func:`load_pixai_card` — the repo's ``config.json``: 30,877 tags in
  category blocks (general / character / copyright / style / meta / rating)
  and one card threshold per category.
* :class:`PixaiBackend` — image → ``(probs over pixai tags, pooled feature)``.
  The model is a SAM3-initialised ViTDet (1008², patch 14) whose head is an
  attention pool then a linear layer; the 1024-d pool output is the hidden
  feature the sidecar trains on. The model code is the repo's own
  ``tagger_pipeline.py``, loaded through transformers' remote code at the
  pinned ``revision``.

The card duck-types :class:`~anime_tools.tagger.dbv4_backend.Dbv4Card` where
:func:`~anime_tools.tagger.dbv4_backend.align_vocab` and the tagger read it
(``name_to_col``, ``rating_cols``, ``n_classes``, ``best_thresholds``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from anime_tools._device import resolve_device
from anime_tools._json import read_json
from anime_tools.tagger.dbv4_backend import Dbv4Output, default_dtype
from anime_tools.tagger.dbv4_meta import (
    DEFAULT_PIXAI_IMG_SIZE,
    DEFAULT_PIXAI_REPO,
    DEFAULT_PIXAI_REVISION,
    Backbone,
    fetch_hint,
)

logger = logging.getLogger(__name__)

# pixai rating tag -> Anima rating name.
PIXAI_RATING_MAP: dict[str, str] = {
    "rating:g": "safe",
    "rating:s": "sensitive",
    "rating:q": "nsfw",
    "rating:e": "explicit",
}


@dataclass
class PixaiCard:
    repo: str
    tags: list[str]
    """Tag names in column order, as the repo spells them (snake_case)."""
    categories: list[str]
    """Category of each column (``tags_split`` expanded)."""
    category_thresholds: dict[str, float]
    name_to_col: dict[str, int] = field(default_factory=dict)
    """space-normalised tag name → column, ratings excluded."""
    rating_cols: dict[str, int] = field(default_factory=dict)
    """Anima rating name → column."""

    @property
    def n_classes(self) -> int:
        return len(self.tags)

    def best_thresholds(self) -> torch.Tensor:
        return torch.tensor(
            [float(self.category_thresholds[c]) for c in self.categories]
        )


def card_from_config(repo: str, cfg: dict) -> PixaiCard:
    tags = list(cfg["tags"])
    cats: list[str] = []
    for cat, n in cfg["tags_split"]:
        cats += [str(cat)] * int(n)
    if len(cats) != len(tags):
        raise ValueError(
            f"{repo} config.json: tags_split covers {len(cats)} columns, tags has {len(tags)}"
        )
    card = PixaiCard(
        repo=repo,
        tags=tags,
        categories=cats,
        category_thresholds=dict(cfg.get("category_best_threshold") or {}),
    )
    for j, (name, cat) in enumerate(zip(tags, cats, strict=True)):
        if cat == "rating":
            if name in PIXAI_RATING_MAP:
                card.rating_cols[PIXAI_RATING_MAP[name]] = j
            continue
        card.name_to_col.setdefault(name.replace("_", " "), j)
    for cat in set(cats):
        card.category_thresholds.setdefault(cat, 0.2)
    return card


def _fetch_snapshot(backbone: Backbone) -> Path:
    """Every backbone file into the hub cache; returns the snapshot dir."""
    from anime_tools._hf import hf_download

    path = None
    for fname in backbone.spec.files:
        path = hf_download(
            what=f"pixai tagger ({backbone.repo})",
            hint=fetch_hint(backbone),
            repo_id=backbone.repo,
            filename=fname,
            revision=backbone.revision,
        )
    return Path(path).parent


def load_pixai_card(
    repo: str = DEFAULT_PIXAI_REPO, revision: str | None = DEFAULT_PIXAI_REVISION
) -> PixaiCard:
    from anime_tools._hf import hf_download

    backbone = Backbone("pixai", repo, revision)
    cfg_path = hf_download(
        what=f"pixai tagger config ({repo})",
        hint=fetch_hint(backbone),
        repo_id=repo,
        filename="config.json",
        revision=revision,
    )
    return card_from_config(repo, read_json(Path(cfg_path)))


# Image preprocessing — the repo's RescalePadProcessor: alpha onto white,
# longer side to ``size`` (bilinear, antialiased), centre-pad with 0. The
# [-1, 1] normalisation runs in the backend, on the device.


def preprocess_pixai(im: Image.Image, size: int) -> torch.Tensor:
    """PIL → ``[3, size, size]`` float in [0, 1], black-padded."""
    import torchvision.transforms.functional as vF

    if im.mode != "RGB":
        im = im.convert("RGBA")
        canvas = Image.new("RGBA", im.size, (255, 255, 255))
        canvas.alpha_composite(im)
        im = canvas.convert("RGB")
    x = vF.to_tensor(im)
    h, w = x.shape[-2:]
    if h != size or w != size:
        r = min(size / h, size / w)
        nh, nw = int(h * r), int(w * r)
        left, top = (size - nw) // 2, (size - nh) // 2
        x = vF.resize(x, [nh, nw])
        x = vF.pad(x, [left, top, size - nw - left, size - nh - top], 0)
    return x


class PixaiBackend:
    """Lazy-loading wrapper around the pixai tagger.

    Same surface as :class:`~anime_tools.tagger.dbv4_backend.Dbv4Backend`:
    ``card``, ``d_hidden``, ``img_size``, ``preprocess``, ``forward`` /
    ``forward_tensor`` → :class:`Dbv4Output`.
    """

    kind = "pixai"

    def __init__(
        self,
        repo: str = DEFAULT_PIXAI_REPO,
        revision: str | None = DEFAULT_PIXAI_REVISION,
        img_size: int = DEFAULT_PIXAI_IMG_SIZE,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        card: PixaiCard | None = None,
    ):
        self.repo = repo
        self.revision = revision
        self.arch = "pixai-vitdet"
        self.img_size = int(img_size)
        self.device = torch.device(resolve_device(device))
        self.dtype = dtype if dtype is not None else default_dtype(self.device)
        self._card = card
        self._model: nn.Module | None = None

    @property
    def backbone(self) -> Backbone:
        return Backbone("pixai", self.repo, self.revision)

    @property
    def card(self) -> PixaiCard:
        if self._card is None:
            self._card = load_pixai_card(self.repo, self.revision)
        return self._card

    @property
    def d_hidden(self) -> int:
        return int(self.model.head.in_features)

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def preprocess(self, im: Image.Image) -> torch.Tensor:
        return preprocess_pixai(im, self.img_size)

    def _load_model(self) -> nn.Module:
        from transformers import AutoModel

        snap = _fetch_snapshot(self.backbone)
        model = AutoModel.from_pretrained(
            str(snap), trust_remote_code=True, dtype=self.dtype
        )
        if not (hasattr(model, "head_pool") and hasattr(model, "forward_feature")):
            raise RuntimeError(
                f"pixai backend expects ViTDetCls (forward_feature / head_pool / head); "
                f"{self.repo}@{self.revision} loaded {type(model).__name__}"
            )
        model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        logger.info(
            "PixaiBackend: %s@%s (%d classes, %dpx) on %s",
            self.repo,
            (self.revision or "main")[:8],
            self.card.n_classes,
            self.img_size,
            self.device,
        )
        return model

    @torch.no_grad()
    def forward_tensor(self, x01: torch.Tensor) -> Dbv4Output:
        """``[B, 3, S, S]`` in [0, 1] → probs + pooled feature."""
        model = self.model
        x = x01.to(self.device, dtype=torch.float32)
        x = ((x - 0.5) / 0.5).to(self.dtype)
        feats = model.forward_feature(x)[-1]
        hidden = model.head_pool(feats.flatten(2).transpose(1, 2))
        logits = model.head(hidden)
        return Dbv4Output(
            probs=logits.float().sigmoid().cpu(), hidden=hidden.float().cpu()
        )

    @torch.no_grad()
    def forward(self, images) -> Dbv4Output:
        x = torch.stack([self.preprocess(im) for im in images])
        return self.forward_tensor(x)
