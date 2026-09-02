"""Which caption file a stage reads for a resized image.

The **revised** caption (``workspace/resized/<rel>.txt``) is authoritative when
it exists; the hand-written **master** is the read-only fallback. Revised-first
is what makes ``is_candidate`` / ``is_audit_target`` skip an image a previous
``--apply`` already rewrote — reading the master would re-propose clauses on
every run. ``autotag`` calls :func:`resolve_caption` directly rather than walking
here (its walk is over images, not captions); ``ab_position_captions``
deliberately reads the master only and does not go through here at all.

Torch-free.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple, Protocol

from ._caption_io import read_caption


class CaptionStats(Protocol):
    """The slice of a stage's stats object :func:`iter_captions` fills in."""

    seen: int

    def skip(self, reason: str) -> None: ...


class CaptionItem(NamedTuple):
    """One walked image and the caption that speaks for it.

    ``dst_caption`` is where a rewrite would be *written* (always the revised
    tree), not necessarily where ``caption`` was read from.
    """

    image_path: Path
    rel: Path
    dst_caption: Path
    caption: str


def resolve_caption(resized_dir: Path, source_dir: Path, rel: Path) -> Path | None:
    """The caption file that speaks for ``rel``, or ``None`` if there is none.

    Revised first, master as the read-only fallback.
    """
    dst_caption = resized_dir / rel
    if dst_caption.exists():
        return dst_caption
    src_caption = source_dir / rel
    return src_caption if src_caption.exists() else None


def iter_captions(
    resized_dir: Path,
    source_dir: Path,
    path_pattern: str | None,
    stats: CaptionStats,
    progress: Callable[[int, int, str], None] | None = None,
) -> Iterator[CaptionItem]:
    """Walk the resized tree, yielding every image that has a caption.

    Sets ``stats.seen``, reports ``stats.skip("no-caption")`` for an image with
    neither caption, and calls ``progress`` once per walked image (captioned or
    not, so the bar tracks the walk rather than the yield).
    """
    from anime_tools._walk import walk_images

    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)
    for index, image_path in enumerate(images, 1):
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        if progress is not None:
            progress(index, len(images), str(rel))
        caption_path = resolve_caption(resized_dir, source_dir, rel)
        if caption_path is None:
            stats.skip("no-caption")
            continue
        yield CaptionItem(
            image_path, rel, resized_dir / rel, read_caption(caption_path)
        )
