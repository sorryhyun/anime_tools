"""The text an image *contains* — the ``{stem}.ocr.txt`` sidecar.

The third vocabulary over :mod:`anime_tools.captions._sidecar`'s one format, and
the one that is not a caption. ``variants`` names captions the TE step will
encode and ``history`` names captions that stopped being true; a record here
names **words that are in the picture**, which is a different kind of fact and is
why it is a sidecar rather than a rung of the caption ladder. Nothing downstream
encodes it, the trainer never reads it, and it is not in ``docs/contract.md``:
like the near-twin feature cache it is curation-private, a readout you consult
when a caption says ``english text`` and you want to know which English.

What the *caption* gets from a run is at most a tag — ``english text``,
``chinese text``, ``korean text``, ``bilingual text`` — never the recognized
string, because the caption grammar has no clause that could hold one and
inventing it is a two-repo change to a frozen contract. So the two halves of an
OCR pass land in two places on purpose: the tag, which is grammar, goes in the
caption; the text, which is evidence, stays here. See
:mod:`anime_tools.ocr.script` for why Japanese earns no tag and still fills a
sidecar.

A record is ``seq ⇥ box ⇥ lang ⇥ score ⇥ text``, in reading order. The text is
last so it may contain tabs, the same reason it is last in the other two.

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
OCR_FIELDS = 5
"""``seq ⇥ box ⇥ lang ⇥ score ⇥ text`` — and the text may hold tabs, being last."""


@dataclass(frozen=True)
class OcrLine:
    """One recognized line: where it is, what language, how sure, what it says.

    ``box`` is the axis-aligned ``(x0, y0, x1, y1)`` bound of the detector's
    rotated quad, in the pixels of the image that was read. The quad itself is
    not kept: it exists to cut a good crop for the recognizer, and once the text
    is out of it the only question anyone asks of a line is *where on the image*,
    which a bounding box answers and a rotated one answers no better.

    ``lang`` is :func:`anime_tools.ocr.script.script_of`'s answer, and ``score``
    the recognizer's mean per-character confidence.
    """

    seq: int
    box: tuple[int, int, int, int]
    lang: str
    score: float
    text: str

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    def as_row(self) -> tuple[str, str, str, str, str]:
        """The record as the sidecar spells it. ``score`` is rounded to three
        places: it is read by a person deciding whether a line is junk, and the
        float's last twelve digits are noise in that judgement."""
        return (
            str(self.seq),
            ",".join(str(int(v)) for v in self.box),
            self.lang,
            f"{self.score:.3f}",
            self.text,
        )

    def to_dict(self) -> dict[str, object]:
        """The row shape a stage report carries, so a dry run shows every line
        it would have written without the sidecar existing yet."""
        return {
            "seq": self.seq,
            "box": list(self.box),
            "lang": self.lang,
            "score": round(self.score, 4),
            "text": self.text,
        }


def ocr_sidecar_path(caption_path: Path) -> Path:
    """``{stem}.ocr.txt`` beside a caption (or its image).

    A multi-dot stem survives; the rule and the reason are
    :func:`anime_tools.captions._sidecar.sidecar_path`'s.
    """
    return sidecar_path(caption_path, OCR_SIDECAR_SUFFIX)


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
    for seq, box, lang, score, text in read_rows(path, OCR_FIELDS):
        try:
            n = int(seq)
            x0, y0, x1, y1 = (int(v) for v in box.split(","))
            conf = float(score)
        except ValueError:
            continue
        out.append(
            OcrLine(
                seq=n, box=(x0, y0, x1, y1), lang=lang.strip(), score=conf, text=text
            )
        )
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


def write_ocr_for(caption_path: Path, lines: Iterable[OcrLine]) -> Path:
    """Write the sidecar belonging to ``caption_path``, returning where it went.

    The pairing every caller wants, so no stage has to hold both the caption path
    and the sidecar path — the reason
    :func:`anime_tools.stages._caption_io.write_caption` is a function of one
    path, said once more here.
    """
    sidecar = ocr_sidecar_path(caption_path)
    write_ocr(sidecar, list(lines))
    return sidecar
