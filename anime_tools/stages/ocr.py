"""Batch OCR: image → PP-OCRv6 → a text sidecar in the OCR tree.

Walks the resized tree and answers one question per image — *what does the
picture say?* — into ``{stem}.ocr.txt`` under :data:`anime_tools.workspace.OCR`,
mirroring the resized tree's layout. Every recognized line with its box and
confidence (:mod:`anime_tools.captions.ocr_sidecar`).

**It reads no caption and writes no caption.** An earlier version of this stage
proposed a Danbooru script tag — ``english text`` / ``chinese text`` — from the
language of what it recognized, and a later one still guessed a language to
filter on. Both are gone. A pass over one 350-image artist folder proposed 98
such tags and **74 of them rested on evidence two characters or shorter**: a
body-writing tally read as ``正正``, a logo read as ``M``, a vertical Japanese
column read sideways into ``个``. A language guessed off characters is confident
about a fragment and has no opinion on whether the fragment was text at all, so
it is the wrong thing to hang either a caption tag or a drop on. What the run
records is the string it read, and the score floor is the only filter.

So this stage sits outside the caption ladder entirely: it needs no Apply gate
in front of it, invalidates no TE cache, and an ``--apply`` here never means a
caption changed. **Dry-run is still the default** (the caller passes ``apply``),
and a dry run reports every line it would have written, so a sidecar can be
reviewed before it exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from anime_tools._walk import walk_images
from anime_tools.captions.ocr_sidecar import OcrLine, write_ocr_for

CHUNK = 32
"""Images handed to the reader at once. Small enough that progress still moves
and an ``--apply`` streams its writes, large enough that a chunk's line crops
make a batch worth batching. The engine chunks internally too; feeding it runs
rather than the whole corpus is what keeps both properties."""


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


@dataclass
class OcrProposal:
    """One image's OCR and where its sidecar goes."""

    image: str = ""
    sidecar: str = ""
    lines: tuple[OcrLine, ...] = ()
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_row(self) -> dict[str, object]:
        """The report row. ``lines`` are expanded here rather than held as
        dataclasses so a dry run's report *is* the sidecar it would write."""
        return {
            "image": self.image,
            "sidecar": self.sidecar,
            "lines": [line.to_dict() for line in self.lines],
            "status": self.status,
        }


@dataclass
class OcrStats:
    seen: int = 0
    with_text: int = 0
    lines: int = 0
    sidecars: int = 0
    skipped: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1


def number_lines(lines: Sequence[OcrLine]) -> tuple[OcrLine, ...]:
    """The lines, numbered from 1 in the order they arrived.

    Every recognized line is kept. There used to be a ``--lang`` allowlist here
    that dropped a line whose script was not asked for, and it went the way the
    caption-side script tag went: the language was guessed back off the
    characters, so a line reading ``01R`` was "English", a lone ``心`` was
    "Chinese", and ``!?`` was neither and was deleted. That is a language
    classifier being used as a text/not-text detector, which is the one thing it
    cannot do. The score floor is the filter; the sidecar is a readout.
    """
    return tuple(
        OcrLine(seq=i, box=ln.box, score=ln.score, text=ln.text)
        for i, ln in enumerate(lines, 1)
    )


def run_ocr(
    *,
    resized_dir: Path,
    ocr_dir: Path,
    read_fn: Callable[[Path], list[OcrLine]],
    read_many_fn: Callable[[list[Path]], list[list[OcrLine]]] | None = None,
    path_pattern: str | None = None,
    apply: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[OcrProposal], OcrStats]:
    """Read every image in the resized tree and (with ``apply``) write its sidecar.

    Every image is a candidate — unlike the caption stages, this one has nothing
    to look a caption up for, so an uncaptioned image is read like any other.

    The write is unconditional for a candidate rather than conditional on finding
    text, because :func:`~anime_tools.captions.ocr_sidecar.write_ocr` deletes the
    sidecar when a pass finds nothing. That is what makes a re-run over re-cropped
    pixels self-correcting: the record of text that is no longer in the image goes
    away rather than lingering as a claim about the old crop.

    ``read_many_fn`` reads a run of images at once, which is how the engine gets a
    batch worth batching; ``read_fn`` is the one-image fallback, and the two must
    answer the same thing for the same image.
    """
    stats = OcrStats()
    rows: list[OcrProposal] = []

    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)
    read_all = read_many_fn or (lambda paths: [read_fn(p) for p in paths])

    index = 0
    for chunk in _chunks(images, CHUNK):
        for image_path, raw in zip(chunk, read_all(list(chunk)), strict=True):
            rel = image_path.relative_to(resized_dir)
            index += 1
            if progress is not None:
                progress(index, len(images), str(rel))

            lines = number_lines(raw)
            stats.lines += len(lines)
            if lines:
                stats.with_text += 1
            else:
                stats.skip("no-text")

            proposal = OcrProposal(
                image=str(rel),
                sidecar=str(rel.with_suffix(".ocr.txt")),
                lines=lines,
                status="ok" if lines else "no-text",
            )
            rows.append(proposal)

            if apply:
                write_ocr_for(ocr_dir, rel.with_suffix(".txt"), lines)
                stats.sidecars += 1

    return rows, stats
