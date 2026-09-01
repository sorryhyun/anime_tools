"""Tag read-back — the Anima Tagger read backwards as a caption-adherence score.

After *Read It Back* (arXiv 2607.11886): the "prompt" is an Anima caption and
the "likelihood" is the tagger's own per-tag confidence:

    readback(x, caption) := mean over content tags t of  log σ(tag_logit_t(x))

Two constraints it enforces:

* **Content tags only.** ``artist`` / ``metadata`` / ``deprecated`` and the
  softmax-group sentinels are masked; rating and people-count are separate
  heads and are never folded into the tag mean.
* **Group-relative use only.** Absolute values across *different* captions carry
  a per-caption base-rate term; only comparisons holding the caption fixed (one
  caption, N images) cancel it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from anime_tools.captions import tag_groups as tg
from anime_tools.tagger.tagger import DEFAULT_TAGGER_DIR, AnimaTagger

logger = logging.getLogger(__name__)

CONTENT_CATEGORIES: tuple[str, ...] = ("general", "character", "copyright", "count")
MASKED_CATEGORIES: tuple[str, ...] = ("artist", "metadata", "deprecated")

AGG_LOGSIGMOID = "logsigmoid"  # mean log σ(logit) — dense, the paper's likelihood
AGG_RECALL = "recall"  # fraction of caption tags fired above the calibrated threshold
AGGREGATIONS: tuple[str, ...] = (AGG_LOGSIGMOID, AGG_RECALL)


class TagReadback:
    """Scores caption adherence by reading the frozen Anima Tagger backwards.

    Wraps an :class:`AnimaTagger` for its vocab, category typing and calibrated
    thresholds; never trains.
    """

    def __init__(
        self,
        tagger: AnimaTagger | None = None,
        ckpt_dir: str | Path = DEFAULT_TAGGER_DIR,
        device: torch.device | str | None = None,
        content_categories: Sequence[str] = CONTENT_CATEGORIES,
    ):
        self.tagger = (
            tagger if tagger is not None else AnimaTagger(ckpt_dir, device=device)
        )
        self.device = self.tagger.device
        self.n_tags = int(self.tagger.cfg.n_tags)
        self.content_categories = tuple(content_categories)
        keep = set(self.content_categories)

        # Per-index arrays (vocab index == logit index == column in tag_logits).
        self.name_by_idx: list[str] = [""] * self.n_tags
        self.category_by_idx: list[str] = ["general"] * self.n_tags
        self.name_to_idx: dict[str, int] = {}
        for e in self.tagger.tag_entries:
            if not (0 <= e.index < self.n_tags):
                continue
            self.name_by_idx[e.index] = e.name
            self.category_by_idx[e.index] = e.category
            self.name_to_idx[e.name] = e.index

        # Content mask: keep-category AND not an internal softmax-group sentinel.
        mask = torch.zeros(self.n_tags, dtype=torch.bool)
        for i in range(self.n_tags):
            name = self.name_by_idx[i]
            if self.category_by_idx[i] in keep and not tg.is_sentinel_name(name):
                mask[i] = True
        self.content_mask = mask  # [n_tags] cpu bool
        self.thresholds = self.tagger.thresholds.float().cpu()  # [n_tags]
        logger.info(
            "TagReadback: %d content tags of %d (kept categories=%s)",
            int(mask.sum()),
            self.n_tags,
            self.content_categories,
        )

    # -- caption → content-tag column mask ------------------------------------

    def caption_indices(self, tags: Iterable[str]) -> list[int]:
        """Map caption tag strings → content-tag logit indices.

        Caption tags are space-form (``long hair``), vocab keys underscore-form,
        so both spellings are tried. Unknown / masked tags drop out silently.
        """
        out: list[int] = []
        for t in tags:
            t = t.strip()
            if not t:
                continue
            idx = self.name_to_idx.get(t)
            if idx is None:
                idx = self.name_to_idx.get(t.replace(" ", "_"))
            if idx is None:
                continue
            if bool(self.content_mask[idx]):
                out.append(idx)
        return out

    def caption_mask(self, tags: Iterable[str]) -> torch.Tensor:
        """Caption tag strings → a ``[n_tags]`` bool column mask (content only)."""
        m = torch.zeros(self.n_tags, dtype=torch.bool)
        idxs = self.caption_indices(tags)
        if idxs:
            m[torch.tensor(idxs, dtype=torch.long)] = True
        return m

    def content_multi_hot(self, multi_hot: torch.Tensor) -> torch.Tensor:
        """Restrict a ground-truth ``[..., n_tags]`` multi-hot to content tags."""
        return multi_hot.bool() & self.content_mask.to(multi_hot.device)

    # -- scoring --------------------------------------------------------------

    def readback_from_logits(
        self,
        tag_logits: torch.Tensor,
        caption_masks: torch.Tensor,
        agg: str = AGG_LOGSIGMOID,
    ) -> torch.Tensor:
        """Read-back score for every (image, caption) pair.

        Args:
            tag_logits: ``[N, n_tags]`` raw tag logits (from ``tagger.model``).
            caption_masks: ``[M, n_tags]`` bool, content-restricted (each row a
                caption's content-tag set). Use :meth:`caption_mask` /
                :meth:`content_multi_hot` to build these.
            agg: ``"logsigmoid"`` (mean log σ over the caption's tags — dense,
                the paper's likelihood) or ``"recall"`` (fraction of the
                caption's tags fired above the per-tag calibrated threshold).

        Returns:
            ``[N, M]`` score matrix. ``score[i, j]`` = readback(image_i,
            caption_j). Captions with no content tags score ``nan``.
        """
        if agg not in AGGREGATIONS:
            raise ValueError(f"unknown agg {agg!r}; expected one of {AGGREGATIONS}")
        logits = tag_logits.float()
        cm = caption_masks.to(logits.device).float()  # [M, n_tags]
        counts = cm.sum(dim=-1)  # [M]
        if agg == AGG_LOGSIGMOID:
            per_tag = F.logsigmoid(logits)  # [N, n_tags]
        else:  # recall against calibrated thresholds
            fired = (logits.sigmoid() >= self.thresholds.to(logits.device)).float()
            per_tag = fired
        summed = per_tag @ cm.t()  # [N, M]
        denom = counts.clamp(min=1.0)
        out = summed / denom
        out[:, counts == 0] = float("nan")
        return out

    @torch.no_grad()
    def image_logits(self, pil_img) -> torch.Tensor:
        """Encode one PIL image through the frozen tagger → ``[n_tags]`` logits."""
        return self.tagger.tag_logits(pil_img)

    @torch.no_grad()
    def readback_images(
        self, pil_images: Sequence, tags: Iterable[str], agg: str = AGG_LOGSIGMOID
    ) -> torch.Tensor:
        """Group-relative read-back for N images against ONE caption → ``[N]``."""
        cm = self.caption_mask(tags).unsqueeze(0)  # [1, n_tags]
        logits = torch.stack([self.image_logits(im) for im in pil_images], dim=0)
        return self.readback_from_logits(logits, cm, agg=agg)[:, 0]
