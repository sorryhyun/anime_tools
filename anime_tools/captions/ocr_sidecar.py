"""The text an image *contains* — the ``{stem}.ocr.txt`` sidecar.

A record names words that are in the picture, not a caption, so it lives in
:data:`anime_tools.workspace.OCR` — a tree of its own mirroring the resized tree
— rather than beside the caption. An OCR pass writes only there: no caption is
read or rewritten, so it needs no Apply gate and invalidates no TE cache.

A record is ``seq ⇥ box ⇥ score ⇥ text``, in reading order; the text is last so
it may contain tabs. There is no language column — PP-OCRv6 returns a string,
never a language.

Torch-free, stdlib-only and import-light.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from anime_tools.captions._sidecar import (
    read_rows,
    sidecar_header,
    sidecar_path,
    write_rows,
)

OCR_SIDECAR_SUFFIX = ".ocr.txt"
OCR_FIELDS = 4
"""``seq ⇥ box ⇥ score ⇥ text`` — and the text may hold tabs, being last."""


@dataclass(frozen=True)
class OcrLine:
    """One recognized line: where it is, how sure, and what it says.

    ``box`` is the axis-aligned ``(x0, y0, x1, y1)`` bound of the detector's
    rotated quad, in the pixels of the image that was read; the quad itself is
    not kept. ``score`` is the recognizer's mean per-character confidence.
    """

    seq: int
    box: tuple[int, int, int, int]
    score: float
    text: str

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    def as_row(self) -> tuple[str, str, str, str]:
        """The record as the sidecar spells it; ``score`` is rounded to three
        places for a human reader."""
        return (
            str(self.seq),
            ",".join(str(int(v)) for v in self.box),
            f"{self.score:.3f}",
            self.text,
        )

    def to_dict(self) -> dict[str, object]:
        """The row shape a stage report carries, so a dry run can show every
        line it would have written."""
        return {
            "seq": self.seq,
            "box": list(self.box),
            "score": round(self.score, 4),
            "text": self.text,
        }


def ocr_sidecar_path(path: Path) -> Path:
    """``{stem}.ocr.txt`` for a caption or image path, in that path's directory.

    Callers pass a path *inside the OCR tree* (``ocr_dir / rel``), not the
    caption itself; being a plain function of a path lets the GUI resolve a
    sidecar without knowing which root wrote it. A multi-dot stem survives.
    """
    return sidecar_path(path, OCR_SIDECAR_SUFFIX)


def read_ocr(path: Path) -> list[OcrLine]:
    """Parse an OCR sidecar into its lines, in file order.

    ``path`` is the *sidecar*; a missing one is an image with no text found in
    it and answers ``[]``. Records arrive with the format's own tolerance, plus
    one more: a record whose sequence, box or score will not parse is dropped
    like a malformed line.
    """
    if not path.is_file():
        return []
    out: list[OcrLine] = []
    for seq, box, score, text in read_rows(path, OCR_FIELDS):
        try:
            n = int(seq)
            x0, y0, x1, y1 = (int(v) for v in box.split(","))
            conf = float(score)
        except ValueError:
            continue
        out.append(OcrLine(seq=n, box=(x0, y0, x1, y1), score=conf, text=text))
    return out


def write_ocr(path: Path, lines: Sequence[OcrLine]) -> None:
    """Write the sidecar, or delete it when nothing was found.

    An image with no text has no OCR, and an empty file would be
    indistinguishable from a crashed run. It also makes a re-run over changed
    pixels self-correcting: the sidecar of an image whose text was cropped away
    goes away rather than lingering as a claim about the old crop.
    """
    if not lines:
        if path.is_file():
            path.unlink()
        return
    write_rows(path, sidecar_header("ocr"), (line.as_row() for line in lines))


def write_ocr_for(ocr_dir: Path, rel: Path, lines: Iterable[OcrLine]) -> Path:
    """Write ``rel``'s sidecar under ``ocr_dir``, returning where it went.

    ``rel`` is the image's path relative to the resized tree, so the OCR tree
    mirrors it and the two join by the same key every other root does.
    """
    sidecar = ocr_sidecar_path(ocr_dir / rel)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    write_ocr(sidecar, list(lines))
    return sidecar
