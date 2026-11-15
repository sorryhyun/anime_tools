"""The tolerant image Dataset both frozen-tower passes feed a DataLoader from.

The grouping embedder (:mod:`anime_tools.grouping.features`) and the tagger's
backbone-feature cache (:mod:`anime_tools.tagger.cli.train_sidecar`) each walk a
tree of images through a tower once; one unreadable file must cost its row, not
the pass. Each item is ``(index, tensor, ok)``: a decode that raises one of
``errors`` answers a zero ``[3, size, size]`` tensor with ``ok=False``, and the
caller drops the row.

``load`` runs on the DataLoader's workers, so it must pickle: a module-level
function or a :func:`functools.partial` over one, never a lambda or a bound
method of a loaded model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

__all__ = ["DecodeDataset", "collate"]


class DecodeDataset(Dataset):
    """``load(path)`` per item on the workers; a failed decode is a zero
    tensor flagged ``ok=False`` instead of a crashed pass."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        load: Callable[[str | Path], torch.Tensor],
        size: int,
        errors: tuple[type[BaseException], ...] = (Exception,),
    ):
        self.paths = list(paths)
        self.load = load
        self.size = int(size)
        self.errors = errors

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[int, torch.Tensor, bool]:
        try:
            return i, self.load(self.paths[i]), True
        except self.errors:
            return i, torch.zeros(3, self.size, self.size), False


def collate(
    batch: Sequence[tuple[int, torch.Tensor, bool]],
) -> tuple[list[int], torch.Tensor, list[bool]]:
    """``(indices, stacked [B, 3, S, S], oks)`` — indices and flags stay plain
    Python so a caller can key a dict or a list on them."""
    return (
        [b[0] for b in batch],
        torch.stack([b[1] for b in batch]),
        [b[2] for b in batch],
    )
