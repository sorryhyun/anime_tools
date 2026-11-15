"""Which caption file a stage reads for a resized image — the caption ladder.

The **revised** caption (``workspace/resized/<rel>.txt``) is authoritative when
it exists; the **master** is the read-only fallback, and the master is itself
two files: the overlay (``workspace/master/<rel>.txt``, where the GUI's caption
editor writes a master edit) shadowing the hand-written original under ``src``.
So the ladder is revised → overlay → src, and it is spelled once, here:
:func:`resolve_caption` for the stages, Export's ``_caption_source`` and the GUI
(``gui.dataset.caption_paths``, through :func:`master_caption`) all read it.

Revised-first is what makes ``is_candidate`` / ``is_audit_target`` skip an image
a previous ``--apply`` already rewrote — reading the master would re-propose
clauses on every run. ``autotag`` walks with ``uncaptioned=True`` (it tags the
images no caption speaks for); ``ab_position_captions`` deliberately reads the
master only, which is :func:`master_caption`.

Torch-free, PIL-free.
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


def read_if_file(path: Path) -> str:
    """:func:`read_caption`, or ``""`` for a caption that is not there — the
    write target's own text, which an empty string says does not exist yet."""
    return read_caption(path) if path.is_file() else ""


class CaptionItem(NamedTuple):
    """One walked image and the caption that speaks for it.

    ``dst_caption`` is where a rewrite would be *written* (always the revised
    tree); ``caption_path`` is where ``caption`` was read *from* — the revised
    caption, or the master (overlay or hand-written) when no revised caption
    exists yet, or ``None`` for an image no caption speaks for (only walked
    with ``uncaptioned=True``). The first two differ exactly when a stage is
    mirroring an image's master for the first time, so a caller counts that case
    by comparing them (:attr:`from_master`) rather than by a second ``exists()``.
    """

    image_path: Path
    rel: Path
    dst_caption: Path
    caption: str
    caption_path: Path | None

    @property
    def from_master(self) -> bool:
        """Whether ``caption`` came from the master rather than the revised
        caption — the write would create the revised caption."""
        return self.caption_path is not None and self.caption_path != self.dst_caption

    @property
    def target_before(self) -> str:
        """What the write target holds now, ``""`` when it does not exist yet —
        every row's ``target_before``, the replay's drift baseline
        (``replay.apply_one``). Read at access, so take it before the write."""
        if self.caption_path == self.dst_caption:
            return self.caption
        return read_if_file(self.dst_caption)


def master_caption(master_dir: Path | None, source_dir: Path, rel: Path) -> Path:
    """The master rung's file for ``rel``: the overlay when it holds one, the
    hand-written ``source_dir`` file otherwise — which need not exist.

    ``master_dir=None`` is a caller with no overlay (a library call that
    predates it); the master is then the hand-written file alone.
    """
    if master_dir is not None:
        over = master_dir / rel
        if over.is_file():
            return over
    return source_dir / rel


def resolve_caption(
    resized_dir: Path, source_dir: Path, rel: Path, *, master_dir: Path | None = None
) -> Path | None:
    """The caption file that speaks for ``rel``, or ``None`` if there is none.

    The ladder: revised, then the master overlay, then the hand-written master.
    ``rel`` is the caption's path (``.txt``) relative to each root.
    """
    dst_caption = resized_dir / rel
    if dst_caption.is_file():
        return dst_caption
    master = master_caption(master_dir, source_dir, rel)
    return master if master.is_file() else None


def iter_captions(
    resized_dir: Path,
    source_dir: Path,
    path_pattern: str | None,
    stats: CaptionStats,
    progress: Callable[[int, int, str], None] | None = None,
    *,
    master_dir: Path | None = None,
    recursive: bool = True,
    uncaptioned: bool = False,
    missing: Callable[[Path, Path], None] | None = None,
) -> Iterator[CaptionItem]:
    """Walk the resized tree, yielding every image that has a caption.

    Sets ``stats.seen``, reports ``stats.skip("no-caption")`` for an image with
    no caption on any rung, and calls ``progress`` once per walked image
    (captioned or not, so the bar tracks the walk rather than the yield).

    ``uncaptioned`` yields such an image too, with ``caption=""`` and
    ``caption_path=None``, instead of skipping it — for the one stage whose
    work *is* that case (autotag's ``missing`` mode).

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
            progress(index, len(images), rel.as_posix())
        dst_caption = resized_dir / rel
        caption_path = resolve_caption(
            resized_dir, source_dir, rel, master_dir=master_dir
        )
        if caption_path is None:
            if missing is not None:
                missing(image_path, rel)
            if uncaptioned:
                yield CaptionItem(image_path, rel, dst_caption, "", None)
            else:
                stats.skip("no-caption")
            continue
        yield CaptionItem(
            image_path, rel, dst_caption, read_caption(caption_path), caption_path
        )
