"""Tag read-back — the Anima Tagger read backwards as a caption-adherence score.

The *Read It Back* idea (arXiv 2607.11886): an image-conditioned prompt
log-likelihood is a training-free text-to-image reward. Here the "prompt" is an
Anima caption (an atomic tag set — already a decomposition, so no lossy VQA
decomposer) and the "likelihood" is the tagger's own per-tag confidence:

    readback(x, caption) := mean over content tags t in caption of  log σ(tag_logit_t(x))

This module is the durable primitive. It owns *only* the scoring math + the
content-tag masking; it does not touch the dataset/feature caches (a caller that
has cached logits scores with :meth:`TagReadback.readback_from_logits`; a caller
holding arbitrary PIL images uses :meth:`TagReadback.readback_images`, which
re-encodes through :class:`AnimaTagger`). The Phase-0a validity bench
(``bench/readback/run_bench.py``) drives the cached-feature path; the eventual
consumers (``dave_mod_bestofn`` ``q_tag``, soup ingredient gating, seed
selection, RWR self-captioning) share the same call.

Design constraints inherited from the proposal (``docs/proposal/tag_readback_reward.md``):

* **Content tags only.** Artist tags are masked (RWR *learns* the artist —
  scoring it is circular); ``metadata`` / ``deprecated`` and the internal
  softmax-group sentinels are masked. Rating and people-count live in separate
  heads and are exposed as extra atomic checks, never folded into the tag mean.
* **Group-relative use only.** Absolute readback values across *different*
  captions carry a per-caption base-rate/language-prior term; only comparisons
  that hold the caption fixed (same caption, N images) cancel it cleanly. The
  transpose (same image, N captions — the shuffled-caption control) is the
  *harder* axis precisely because that term does not cancel there, which is why
  it is the make-or-break validity test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from anime_tools.captions import tag_groups as tg
from anime_tools.tagger.tagger import DEFAULT_TAGGER_DIR, AnimaTagger

logger = logging.getLogger(__name__)

# Categories that count as "content" for read-back. ``artist`` is excluded
# (circular for the RWR consumer); ``metadata`` / ``deprecated`` are prior-driven
# noise. Rating and people-count are separate heads, scored via the dedicated
# helpers rather than the tag mean.
CONTENT_CATEGORIES: tuple[str, ...] = ("general", "character", "copyright", "count")
MASKED_CATEGORIES: tuple[str, ...] = ("artist", "metadata", "deprecated")

# Aggregation modes for turning per-tag logits into one scalar per (image, caption).
AGG_LOGSIGMOID = "logsigmoid"  # mean log σ(logit) — dense, the paper's likelihood
AGG_RECALL = "recall"  # fraction of caption tags fired above the calibrated threshold
AGGREGATIONS: tuple[str, ...] = (AGG_LOGSIGMOID, AGG_RECALL)


class TagReadback:
    """Scores caption adherence by reading the frozen Anima Tagger backwards.

    Wraps an :class:`AnimaTagger` (loaded lazily if not supplied) purely to reuse
    its vocab / category typing / calibrated thresholds / dual-encoder head — the
    judge stays frozen throughout, this class never trains anything.
    """

    def __init__(
        self,
        tagger: Optional[AnimaTagger] = None,
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
        self.name_by_idx: List[str] = [""] * self.n_tags
        self.category_by_idx: List[str] = ["general"] * self.n_tags
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

    def caption_indices(self, tags: Iterable[str]) -> List[int]:
        """Map caption tag strings → content-tag logit indices.

        Caption tags are emitted space-form (``long hair``); the vocab keys are
        underscore-form (``long_hair``). Normalize, look up, and keep only
        indices that survive the content mask. Unknown / masked tags drop out
        silently — a caption may carry artist / meta tags this score ignores.
        """
        out: List[int] = []
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
        """Encode one PIL image through the frozen tagger → ``[n_tags]`` logits.

        The live path for arbitrary renders (no cached feature); backend-
        agnostic via :meth:`AnimaTagger.tag_logits` (PE head or dbv4).
        """
        return self.tagger.tag_logits(pil_img)

    @torch.no_grad()
    def readback_images(
        self, pil_images: Sequence, tags: Iterable[str], agg: str = AGG_LOGSIGMOID
    ) -> torch.Tensor:
        """Group-relative read-back for N images against ONE caption.

        The canonical language-prior-cancelling form: fix the caption, score the
        N images, compare within the group. Returns ``[N]``.
        """
        cm = self.caption_mask(tags).unsqueeze(0)  # [1, n_tags]
        logits = torch.stack([self.image_logits(im) for im in pil_images], dim=0)
        return self.readback_from_logits(logits, cm, agg=agg)[:, 0]
