"""The text an image *contains* — the ``{stem}.ocr.txt`` sidecar.

A record names words that are in the picture, not a caption, so it lives in
:data:`anime_tools.workspace.OCR` — a tree of its own mirroring the resized tree
— rather than beside the caption. An OCR pass writes only there: no caption is
read or rewritten, so it needs no Apply gate and invalidates no TE cache.

A record is ``seq ⇥ box ⇥ det ⇥ score ⇥ text``, in reading order; the text is
last so it may contain tabs. ``det`` is the detector's confidence in the box,
``score`` the reader's confidence in the text (both ``0.000`` when nothing
stood behind them). There is no language column — the reader returns a string,
never a language. A pre-``det`` record (``seq ⇥ box ⇥ score ⇥ text``, four
fields) still reads, with ``det`` as ``0.0``.

The one place the record meets a caption is :func:`with_ocr_clause`, which
Export's ``--combine_ocr`` uses to publish the caption with the lines attached
as a text clause (``Japanese text reads as "…", "…"``): the workspace caption
is never rewritten, so the combine is a property of the published tree.

Torch-free, stdlib-only and import-light.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from anime_tools.captions._sidecar import (
    sidecar_header,
    sidecar_path,
    write_rows,
)
from anime_tools.captions.position_clauses import (
    compose_caption,
    parse_caption,
    text_clause,
)

OCR_SIDECAR_SUFFIX = ".ocr.txt"
OCR_FIELDS = 5
"""``seq ⇥ box ⇥ det ⇥ score ⇥ text`` — and the text may hold tabs, being last."""
LEGACY_OCR_FIELDS = 4
"""``seq ⇥ box ⇥ score ⇥ text``: the record before the detector's ``det`` column
(2026-09-07). Read, never written."""


@dataclass(frozen=True)
class OcrLine:
    """One recognized line: where it is, how sure, and what it says.

    ``box`` is the axis-aligned ``(x0, y0, x1, y1)`` bound of the detector's
    rotated quad, in the pixels of the image that was read; the quad itself is
    not kept. ``score`` is the reader's mean per-token confidence in ``text``
    (the VL reader's greedy-decode probability, averaged; ``0.0`` when no
    reader stood behind the line). ``det`` is the detector's confidence in the
    box itself (YOLO objectness; ``0.0`` for a box no detector scored, such as a
    text-mask component).
    """

    seq: int
    box: tuple[int, int, int, int]
    score: float
    text: str
    det: float = 0.0

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
            f"{self.det:.3f}",
            f"{self.score:.3f}",
            self.text,
        )

    def to_dict(self) -> dict[str, object]:
        """The row shape a stage report carries, so a dry run can show every
        line it would have written."""
        return {
            "seq": self.seq,
            "box": list(self.box),
            "det": round(self.det, 4),
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
    one more: a record whose sequence, box or scores will not parse is dropped
    like a malformed line. A four-field record (:data:`LEGACY_OCR_FIELDS`) is
    read as ``det = 0.0``.
    """
    if not path.is_file():
        return []
    out: list[OcrLine] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\r")
        if not line or line.lstrip().startswith("#"):
            continue
        parsed = _parse_record(line)
        if parsed is not None:
            out.append(parsed)
    return out


def _parse_record(line: str) -> OcrLine | None:
    """One sidecar line as an :class:`OcrLine`, trying the five-field record
    first and the legacy four-field one after it; ``None`` when neither fits."""
    parts = line.split("\t", OCR_FIELDS - 1)
    if len(parts) == OCR_FIELDS:
        seq, box, det, score, text = parts
        try:
            return OcrLine(
                seq=int(seq),
                box=_parse_box(box),
                score=float(score),
                text=text,
                det=float(det),
            )
        except ValueError:
            pass  # a legacy record whose text holds a tab lands here
    parts = line.split("\t", LEGACY_OCR_FIELDS - 1)
    if len(parts) != LEGACY_OCR_FIELDS:
        return None
    seq, box, score, text = parts
    try:
        return OcrLine(seq=int(seq), box=_parse_box(box), score=float(score), text=text)
    except ValueError:
        return None


def _parse_box(box: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in box.split(","))
    return (x0, y0, x1, y1)


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


def with_ocr_clause(caption: str, lines: Sequence[OcrLine]) -> str:
    """``caption`` with ``lines`` attached as its text clause, in reading order.

    Any text clause the caption already carries is replaced, so combining twice
    says each line once, and combining with no lines *removes* the clause — a
    re-run over re-cropped pixels that found no text takes the old claim back
    with it. Position clauses and the flat bag are untouched. An empty caption
    with lines becomes the clause alone.
    """
    parsed = parse_caption(caption)
    clauses = list(parsed.position_clauses)
    texts = [line.text for line in lines if line.text]
    if texts:
        clauses.append(text_clause(texts))
    return compose_caption(parsed.flat_tags, clauses)
