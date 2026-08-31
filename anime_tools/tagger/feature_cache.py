"""The dbv4 hidden-state cache — where it lives and how it is read.

One dbv4 forward per ``dataset.json`` image, kept as
``workspace/anima_tagger/dbv4/<arch>_hidden.safetensors``: the frozen
backbone's penultimate features plus its own probabilities, so the sidecar head
can be retrained (and the calibration probe re-run) without touching a GPU
again. The stem list the cache was built for rides in the safetensors metadata
under ``stems`` — a cache built for a different manifest is not merely stale,
it is misaligned row-for-row, so every reader checks it.

The module used to hold only path helpers for the archived PE dual-encoder
token caches (``tokens-<encoder>/`` — reclaimed 2026-08-27 with that trainer)
while claiming "change the layout here, propagate everywhere"; the two live
readers each hardcoded the dbv4 path template instead. Now they don't.
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
    "DBV4_CACHE_DIR",
    "dbv4_cache_path",
    "dbv4_cache_stems",
    "load_dbv4_cache",
    "multi_hot_from_manifest",
]

# Decoupled from the checkpoint dir on purpose: the cache is dataset-derived and
# bulky, so it lives with the other dataset caches and is shared across
# checkpoints built over the same corpus. Dataset-derived means the workspace:
# it is something the tools produced, not something Export ships.
DBV4_CACHE_DIR = Path(WS.WORKSPACE) / "anima_tagger" / "dbv4"


def dbv4_cache_path(arch: str, explicit: str | Path | None = None) -> Path:
    """Cache file for backbone ``arch``; ``explicit`` is the ``--feature_cache``
    override, passed straight through."""
    return Path(explicit) if explicit else DBV4_CACHE_DIR / f"{arch}_hidden.safetensors"


def dbv4_cache_stems(path: str | Path) -> list[str]:
    """The stem list ``path`` was built for, from its safetensors metadata.

    Metadata only — no tensor is read, so this is cheap enough to gate "rebuild
    or reuse?" on.
    """
    with safe_open(str(path), "pt") as f:
        return json.loads(f.metadata()["stems"])


def load_dbv4_cache(path: str | Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    """``(tensors, stems)`` — ``hidden`` / ``probs`` / ``ok``, row-aligned to ``stems``."""
    return st_load(str(path)), dbv4_cache_stems(path)


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
    than raising — a manifest built against a wider vocab is a stale-input
    problem its own reader reports, not an IndexError from here.
    """
    out = torch.zeros(len(tag_indices), n_cols)
    for row, idxs in enumerate(tag_indices):
        for t in idxs:
            col = col_of.get(int(t)) if col_of is not None else int(t)
            if col is not None and 0 <= col < n_cols:
                out[row, col] = 1.0
    return out
