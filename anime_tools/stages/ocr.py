"""Batch OCR: image → PP-OCRv6 → a text sidecar in the OCR tree.

Walks the resized tree and writes what each picture says into ``{stem}.ocr.txt``
under :data:`anime_tools.workspace.OCR`, mirroring its layout: every recognized
line with box and confidence (:mod:`anime_tools.captions.ocr_sidecar`).

It reads no caption and writes no caption, so it sits outside the caption ladder
and invalidates no TE cache. **Dry-run is the default** (the caller passes
``apply``), and a dry run reports every line it would have written.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from anime_tools._walk import walk_images
from anime_tools.captions.ocr_sidecar import OcrLine, write_ocr_for

CHUNK = 32
"""Images handed to the reader at once: small enough that progress still moves
and an ``--apply`` streams its writes, large enough for a worthwhile batch."""


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
        """The report row, with ``lines`` expanded so a dry run's report *is*
        the sidecar it would write."""
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

    Every recognized line is kept; the score floor is the only filter.
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

    Every image is a candidate; no caption is consulted. The write is
    unconditional, because
    :func:`~anime_tools.captions.ocr_sidecar.write_ocr` deletes the sidecar when
    a pass finds nothing — so a re-run over re-cropped pixels drops the record
    of text no longer in the image.

    ``read_many_fn`` reads a run of images at once; ``read_fn`` is the one-image
    fallback, and the two must answer the same thing for the same image.
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
