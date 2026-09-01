"""The text an image *contains* — the ``{stem}.ocr.txt`` sidecar.

The third vocabulary over :mod:`anime_tools.captions._sidecar`'s one format, and
the one that is not a caption. ``variants`` names captions the TE step will
encode and ``history`` names captions that stopped being true; a record here
names **words that are in the picture**, which is a different kind of fact and is
why it is a sidecar rather than a rung of the caption ladder.

Which is also why it lives in :data:`anime_tools.workspace.OCR`, a tree of its
own mirroring the resized tree, rather than beside the caption the other two sit
beside. An OCR pass writes *only* there: no caption is read, rewritten or
tagged by it, so a run needs no Apply gate, invalidates no TE cache, and can be
deleted or regenerated wholesale without touching the caption ladder at all.

A record is ``seq ⇥ box ⇥ score ⇥ text``, in reading order. The text is last so
it may contain tabs, the same reason it is last in the other two.

There is deliberately no language column. PP-OCRv6 is one model over fifty
languages and returns a string, never a language, so any language here would be
guessed back off the characters — and a guess made from a two-character fragment
is confident and worthless, which is the same reason this stage no longer
proposes an ``english text`` caption tag. What the sidecar records is what was
read; deciding what language it is, or whether it was text at all, is the
person's job and the string is what they need to do it.

Torch-free, stdlib-only and import-light, like the two sidecar vocabularies
beside it.
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
    rotated quad, in the pixels of the image that was read. The quad itself is
    not kept: it exists to cut a good crop for the recognizer, and once the text
    is out of it the only question anyone asks of a line is *where on the image*,
    which a bounding box answers and a rotated one answers no better.

    ``score`` is the recognizer's mean per-character confidence.
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
        """The record as the sidecar spells it. ``score`` is rounded to three
        places: it is read by a person deciding whether a line is junk, and the
        float's last twelve digits are noise in that judgement."""
        return (
            str(self.seq),
            ",".join(str(int(v)) for v in self.box),
            f"{self.score:.3f}",
            self.text,
        )

    def to_dict(self) -> dict[str, object]:
        """The row shape a stage report carries, so a dry run shows every line
        it would have written without the sidecar existing yet."""
        return {
            "seq": self.seq,
            "box": list(self.box),
            "score": round(self.score, 4),
            "text": self.text,
        }


def ocr_sidecar_path(path: Path) -> Path:
    """``{stem}.ocr.txt`` for a caption or image path, in that path's directory.

    Callers pass a path *inside the OCR tree* (``ocr_dir / rel``), not the
    caption itself — the naming rule is the same either way, and keeping it a
    plain function of a path is what lets the GUI resolve a sidecar without
    knowing which root it was written from.

    A multi-dot stem survives; the rule and the reason are
    :func:`anime_tools.captions._sidecar.sidecar_path`'s.
    """
    return sidecar_path(path, OCR_SIDECAR_SUFFIX)


def read_ocr(path: Path) -> list[OcrLine]:
    """Parse an OCR sidecar into its lines, in file order.

    ``path`` is the *sidecar*, and a missing one is simply an image with no text
    found in it — the common case in this corpus, and not a condition worth
    raising over, so it answers ``[]`` the way
    :func:`anime_tools.captions.history.read_history` does. Past that, records
    arrive with the format's own tolerance, plus one of its own: a record whose
    sequence, box or score will not parse is dropped like a malformed line,
    since every field but the text is read as a number by somebody.
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

    The delete is this vocabulary's own, and history's reason for it applies
    twice over: an image with no text in it has no OCR, and an empty file
    claiming otherwise would be indistinguishable from a run that crashed before
    it wrote. It also makes a re-run over a corpus whose pixels changed
    self-correcting — the sidecar of an image whose text was cropped away goes
    away rather than lingering as a claim about the old crop.
    """
    if not lines:
        if path.is_file():
            path.unlink()
        return
    write_rows(path, sidecar_header("ocr"), (line.as_row() for line in lines))


def write_ocr_for(ocr_dir: Path, rel: Path, lines: Iterable[OcrLine]) -> Path:
    """Write ``rel``'s sidecar under ``ocr_dir``, returning where it went.

    The pairing every caller wants, so no stage has to build the path itself —
    the reason :func:`anime_tools.stages._caption_io.write_caption` is a function
    of one path, said once more here. ``rel`` is the image's path relative to the
    resized tree, so the OCR tree mirrors it and the two join by the same key
    every other root does.
    """
    sidecar = ocr_sidecar_path(ocr_dir / rel)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    write_ocr(sidecar, list(lines))
    return sidecar
