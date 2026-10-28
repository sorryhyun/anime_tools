"""The backbone hidden-state cache — where it lives and how it is read.

One backbone forward per ``dataset.json`` image, kept as
``workspace/anima_tagger/<backend>/<arch>_hidden.safetensors``: the frozen
backbone's hidden feature plus its probabilities projected onto our vocab, so
the sidecar head and the thresholds can be refit without a GPU. The stem list the cache was built for rides in
the safetensors metadata under ``stems`` — a cache built for a different
manifest is misaligned row-for-row, so every reader checks it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file as st_load

from anime_tools import workspace as WS

__all__ = [
    "BACKBONE_CACHE_ROOT",
    "backbone_cache_path",
    "backbone_cache_stems",
    "load_backbone_cache",
    "multi_hot_from_manifest",
]

# Dataset-derived, not checkpoint-derived: shared across every checkpoint built
# over the same corpus.
BACKBONE_CACHE_ROOT = Path(WS.WORKSPACE) / "anima_tagger"


def backbone_cache_path(
    kind: str, arch: str, explicit: str | Path | None = None
) -> Path:
    """Cache file for backend ``kind`` / backbone ``arch``; ``explicit`` is the
    ``--feature_cache`` override, passed straight through."""
    if explicit:
        return Path(explicit)
    return BACKBONE_CACHE_ROOT / kind / f"{arch}_hidden.safetensors"


def backbone_cache_stems(path: str | Path) -> list[str]:
    """The stem list ``path`` was built for, from its safetensors metadata.

    Metadata only — no tensor is read.
    """
    with safe_open(str(path), "pt") as f:
        return json.loads(f.metadata()["stems"])


def load_backbone_cache(path: str | Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    """``(tensors, stems)`` — ``hidden`` / ``probs`` / ``ok``, row-aligned to ``stems``."""
    return st_load(str(path)), backbone_cache_stems(path)


def multi_hot_from_manifest(
    tag_indices: Sequence[Sequence[int]],
    n_cols: int,
    *,
    col_of: Mapping[int, int] | None = None,
) -> torch.Tensor:
    """``[len(tag_indices), n_cols]`` float multi-hot from a manifest's tag lists.

    ``col_of`` maps a vocab index to a column for a *projected* label space (the
    sidecar trains on its own subset of the vocab, so its columns are not vocab
    indices); an index absent from the map contributes nothing. Without it the
    vocab index is the column, and any index ``>= n_cols`` is dropped rather
    than raising.
    """
    out = torch.zeros(len(tag_indices), n_cols)
    for row, idxs in enumerate(tag_indices):
        for t in idxs:
            col = col_of.get(int(t)) if col_of is not None else int(t)
            if col is not None and 0 <= col < n_cols:
                out[row, col] = 1.0
    return out
