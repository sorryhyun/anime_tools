"""Which caption file a stage reads for a resized image — once, for all of them.

The rule is one line and easy to retype slightly wrong: the **derived** caption
(``post_image_dataset/resized/<rel>.txt``) is authoritative when it exists, and
the hand-written **master** (``image_dataset/<rel>.txt``) is the read-only
fallback for an image the mirror pass has not reached yet. Derived-first matters
in both directions — the derived text is already order-corrected, and it carries
an earlier run's position clauses, which is exactly what makes
``is_candidate`` / ``is_audit_target`` skip an image a previous ``--apply``
already rewrote. Reading the master instead would re-propose clauses on every
run.

:func:`iter_captions` is that rule plus the walk around it: ``stats.seen``, the
``no-caption`` skip, and a progress callback over *every* walked image rather
than only the captioned ones. It is shared by the clause rewrite, its flatten
twin, and the multiview audit — which used to hand-roll the same resolution
under a comment claiming it matched.

Stages that deliberately read one specific tree do **not** go through here:
autotag reads (and writes) the master, and ``ab_position_captions`` reads the
master on purpose, because after one ``--apply`` the derived side already
carries clauses and every rule in it would skip. Those call
:func:`read_caption` directly.

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

    ``dst_caption`` is where a rewrite would be *written* (always the derived
    tree), which is not necessarily where ``caption`` was *read* from.
    """

    image_path: Path
    rel: Path
    dst_caption: Path
    caption: str


def resolve_caption(resized_dir: Path, source_dir: Path, rel: Path) -> Path | None:
    """The caption file that speaks for ``rel``, or ``None`` if there is none.

    Derived first, master as the fallback — see the module docstring.
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

    Sets ``stats.seen`` to the walked count, reports ``stats.skip("no-caption")``
    for an image with neither a derived nor a master caption, and calls
    ``progress`` once per walked image (captioned or not, so the bar tracks the
    walk rather than the yield).
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
