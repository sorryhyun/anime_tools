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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from anime_tools._walk import walk_images
from anime_tools.captions.ocr_sidecar import OcrLine, write_ocr_for


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

    The stage filters nothing: the score floor, the CJK join and the
    ``min_chars`` / ``skip_en`` drops all belong to the reader
    (:mod:`anime_tools.ocr._text`), which knows the pixels each line came from.
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
    read_iter_fn: Callable[[list[Path]], Iterable[list[OcrLine]]] | None = None,
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

    ``read_iter_fn`` reads the *whole* run, yielding one result per image as it
    lands; ``read_fn`` is the one-image fallback, and the two must answer the same
    thing for the same image. The batching is entirely the reader's
    (:meth:`~anime_tools.ocr.OcrEngine.read_iter`) — handing it the run in slices
    would only starve its prefetch, and there is nothing to gain by it, since the
    loop below already writes and reports per image.
    """
    stats = OcrStats()
    rows: list[OcrProposal] = []

    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)
    stream = (
        read_iter_fn(images)
        if read_iter_fn is not None
        else (read_fn(p) for p in images)
    )

    for index, (image_path, raw) in enumerate(zip(images, stream, strict=True), 1):
        rel = image_path.relative_to(resized_dir)
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
