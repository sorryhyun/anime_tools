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
    tree); ``caption_path`` is where ``caption`` was read *from*, which is the
    master when the revised caption does not exist yet. The two differ exactly
    when a stage is mirroring an image's master for the first time, so a caller
    counts that case by comparing them rather than by a second ``exists()``.
    """

    image_path: Path
    rel: Path
    dst_caption: Path
    caption: str
    caption_path: Path


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
    *,
    recursive: bool = True,
    missing: Callable[[Path, Path], None] | None = None,
) -> Iterator[CaptionItem]:
    """Walk the resized tree, yielding every image that has a caption.

    Sets ``stats.seen``, reports ``stats.skip("no-caption")`` for an image with
    neither caption, and calls ``progress`` once per walked image (captioned or
    not, so the bar tracks the walk rather than the yield).

    ``missing`` is called with the ``(image_path, rel)`` of each image that has
    no caption of either kind, for the one stage that has to *act* on that case
    rather than only count it: the corrector, whose variant sidecar is an orphan
    once the caption it was drawn from is gone.
    """
    from anime_tools._walk import walk_images

    images = walk_images(resized_dir, recursive=recursive, pattern=path_pattern)
    stats.seen = len(images)
    for index, image_path in enumerate(images, 1):
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        if progress is not None:
            progress(index, len(images), str(rel))
        caption_path = resolve_caption(resized_dir, source_dir, rel)
        if caption_path is None:
            stats.skip("no-caption")
            if missing is not None:
                missing(image_path, rel)
            continue
        yield CaptionItem(
            image_path,
            rel,
            resized_dir / rel,
            read_caption(caption_path),
            caption_path,
        )
