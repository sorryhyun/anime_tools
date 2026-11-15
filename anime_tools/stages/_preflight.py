"""The "run Resize first" preflight every stage but Resize borrows.

Its own leaf rather than a name on :mod:`.resize`, which imports PIL at the top:
the text-only stages (Correct, Tag groups, OCR's walk) check the tree exists
before anything else and should not pay for the resampler to do it.
:mod:`.resize` re-exports both names.
"""

from __future__ import annotations

from pathlib import Path

from anime_tools._env import resolve_path

__all__ = ["RESIZE_FIRST", "resized_tree"]

RESIZE_FIRST = "run `make preprocess-resize` (or the Resize stage) first"
"""What a stage says when the tree it reads has not been filled yet."""


def resized_tree(dst: str) -> Path:
    """``dst`` resolved against the curation home, if the tree is there.

    Every stage but Resize reads ``workspace/resized/`` rather than filling
    it, so a missing tree is one preflight failure with one answer.
    """
    resized_dir = resolve_path(dst)
    if not resized_dir.exists():
        raise FileNotFoundError(
            f"resized dir not found: {resized_dir} — {RESIZE_FIRST}"
        )
    return resized_dir
