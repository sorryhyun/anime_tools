"""The VL pass over a page: every detected box read by the manga reader, plus
what the text mask boxed that no detector did.

:mod:`anime_tools.ocr.engine` finds the boxes and reads nothing;
:mod:`anime_tools.ocr.reader` reads a crop — hearts, small kana, hand-lettered
onomatopoeia — but detects nothing. This module is the seam between them, and
what the OCR stage runs:

* :func:`reread_lines` — one page in, one page out. Each box is cut with the
  reader's padding and read; the read *is* the line, so a read the decode guard
  rejects (``None``, the guard is inside :meth:`~anime_tools.ocr.reader.OcrReader.read`)
  drops its box, and a read that passes must still clear the confidence floors
  (``min_det`` / ``min_score``), lose its pictographs (``strip_symbols``), clear
  the line floors (``min_chars`` / ``skip_en``) and carry a letter
  (:func:`has_script`). With a
  text mask (the MIT ``{stem}_mask.png`` :mod:`anime_tools.masking` writes) its
  connected components that no box already covers become crops too, held to the
  same floors. Reading order is settled afterwards, since a component may sit
  anywhere.
* :class:`RereadEngine` — the :class:`~anime_tools.ocr.engine.OcrEngine` shape
  (``read`` / ``read_iter``) over an engine and a reader, so the stage runs it
  without knowing.

The boxes are never joined: the AnimeText detector answers a balloon as a block
and its columns (nesting settled by :func:`~anime_tools.ocr.animetext.denest`),
and joining those measured a loss (sincos, 2026-09-06: manga-ocr best-match
0.844 → 0.803 over 95 joins — SFX beside a balloon gets pulled in). What
``denest`` leaves is the *single* nesting — the detector seeing ``ぱん♡`` whole
and ``ぱん`` again inside it, or a balloon block with one of its lines — and
that is settled here, after the reads: a line whose box sits inside another
line's box (:data:`NESTED_CONTAINMENT`) is dropped, but only once the outer
read survived the guard, so a runaway on the block never costs its line
(:func:`drop_nested`; sincos 2026-09-07: 83 such pairs on 52 pages, the two
reads a wash against the hand labels — 29 / 25 / 29 — and 5 pairs where only
the inner read lived).

Measured on the sincos shard (the trainer's ``project/cjk_aware_anima_dit``,
2026-09-06): the masked-but-no-line floor 23 → 8 pages, manga-ocr best-match
0.786 → 0.810, hearts back on the speech lines; the regressions are digits
(``91`` → ``9``) and two-glyph crops. A VL line's ``score`` is the reader's
mean greedy token probability (:meth:`~anime_tools.ocr.reader.OcrReader.read_boxes_scored`);
a reader answering bare strings leaves :data:`NO_SCORE`. The detector's box
confidence rides through as ``det`` (a mask component has none). The two-prompt
v7 preview answers a list per crop (:data:`KindedReads`): each read becomes a line
carrying its ``kind``, so one box can give a speech line and an SFX line.

Torch-free to import; cv2 loads inside the two functions that need it.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from anime_tools import _boxes
from anime_tools.captions.ocr_sidecar import OcrLine
from anime_tools.masking._mask_paths import find_mask
from anime_tools.ocr._io import imread
from anime_tools.ocr._text import drop_symbols, keep_line, reading_order
from anime_tools.ocr.animetext import NEST_TH

Box = tuple[int, int, int, int]
Read = str | tuple[str, float] | None
"""One crop's read: the text with the reader's confidence, a bare text (no
confidence — :data:`NO_SCORE`), or ``None`` for a read the guard rejected."""
KindedReads = list[tuple[str, float, str]]
"""One crop's reads from the two-prompt reader: ``(text, confidence, kind)`` per
prompt that found text (``kind`` ``"speech"`` / ``"sfx"``); empty when none did."""
ReadBoxes = Callable[[object, Sequence[Box]], list[Read | KindedReads]]
"""``(bgr, boxes) → reads``: :meth:`~anime_tools.ocr.reader.OcrReader.read_boxes_scored`."""

NO_SCORE = 0.0
"""The ``score`` (or ``det``) of a line nothing stood behind — a read with no
confidence, a box no detector scored — and ``0.000`` in the sidecar says so."""

LINE_MIN_DET = 0.6
"""The detector confidence a box needs for its read to be written
(:func:`reread_lines`). A mask component has no ``det`` and is not held to it.
Stricter than the export floor
(:data:`~anime_tools.captions.ocr_sidecar.DEFAULT_MIN_DET`), which still guards
sidecars written before this one."""
LINE_MIN_SCORE = 0.6
"""The reader's mean token confidence a read needs to be written. A read with
no confidence (:data:`NO_SCORE`) is not held to it."""

NESTED_CONTAINMENT = NEST_TH
"""A kept line's box at least this far inside another kept line's box is the
same text read twice (:func:`drop_nested`) — the detector's block-and-line
doubling, so the same share as :data:`~anime_tools.ocr.animetext.NEST_TH`."""

CLOSE_FRAC = 0.025
"""Closing kernel as a fraction of the page width: merges the glyphs of one
column (and usually the columns of one block) into one mask component."""
FULLPAGE_FRAC = 0.4
"""A component over this share of the page is the mask's idea of a text-heavy
spread, not a line."""
COVERED_IOU = 0.3
COVERED_CONTAINMENT = 0.5
"""A component a line already boxes — IoU or either-way containment past these
— needs no second crop."""


def overlap(a: Box, b: Box) -> tuple[float, float]:
    """``(IoU, containment)`` of two boxes; containment is the intersection over
    the smaller area (:func:`anime_tools._boxes.containment`), so a box inside
    another reads as 1."""
    return _boxes.iou(a, b), _boxes.containment(a, b)


def covered(box: Box, boxes: Sequence[Box]) -> bool:
    return any(
        iou >= COVERED_IOU or cont >= COVERED_CONTAINMENT
        for iou, cont in (overlap(box, b) for b in boxes)
    )


def mask_components(
    mask, *, min_side: int, close_frac: float = CLOSE_FRAC
) -> list[Box]:
    """Axis-aligned boxes of the text-pixel components of an ignore mask
    (``0`` = text, ``255`` = trained on; a ``(H, W)`` ``uint8`` array), largest
    first. A component with a side under ``min_side`` is screentone; one over
    :data:`FULLPAGE_FRAC` of the page is not a line."""
    import cv2
    import numpy as np

    text = (np.asarray(mask) == 0).astype(np.uint8)
    k = max(3, round(close_frac * text.shape[1]) | 1)
    closed = cv2.morphologyEx(
        text, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    page_area = text.shape[0] * text.shape[1]
    boxes: list[Box] = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if min(w, h) < min_side or w * h >= FULLPAGE_FRAC * page_area:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def drop_nested(lines: Sequence[OcrLine]) -> list[OcrLine]:
    """``lines`` minus every one whose box sits at least
    :data:`NESTED_CONTAINMENT` inside a *larger* line's box of the same
    ``kind``, order kept.

    Both lines are reads that survived, so the outer one already says what
    the inner one says (``ぱん♡`` over ``ぱん``, a balloon over one of its
    lines); the inner is the detector doubling it. Equal boxes are left alone.
    A line the reader kinded is only doubled by a line of its kind: an SFX
    read inside a block whose ``OCR:`` read is speech is not in that read.
    """

    area = _boxes.area
    out = []
    for ln in lines:
        inside = any(
            o is not ln
            and o.kind == ln.kind
            and area(o.box) > area(ln.box)
            and _boxes.containment(ln.box, o.box) >= NESTED_CONTAINMENT
            for o in lines
        )
        if not inside:
            out.append(ln)
    return out


KANA_MARKS = frozenset("ーっッゝゞヽヾ")
"""Kana that only modify a neighbour — the long-vowel bar, the sokuon, the
iteration marks. Unicode files them as letters, but ``ー・・・ッ`` or ``っっっ``
has no syllable in it: it is the reader filling a box of motion lines."""


def has_script(text: str) -> bool:
    """Whether a read carries a letter at all — ``♡`` or ``…`` alone is a
    decoration the mask caught, not a line, and a string of
    :data:`KANA_MARKS` alone (``ーーー``, ``ー・・・ッ``) is not one either."""
    return any(
        unicodedata.category(ch).startswith("L") and ch not in KANA_MARKS for ch in text
    )


def reread_lines(
    bgr,
    lines: Sequence[OcrLine],
    read_boxes: ReadBoxes,
    *,
    mask=None,
    comp_min_side: int = 32,
    comp_max: int = 16,
    min_chars: int = 2,
    skip_en: bool = True,
    min_det: float = LINE_MIN_DET,
    min_score: float = LINE_MIN_SCORE,
    strip_symbols: bool = True,
) -> list[OcrLine]:
    """One page through the VL reader: the detected boxes read, the uncovered
    mask components read, reading order and numbering settled afterwards.

    ``read_boxes`` is called once with every crop (the lines' boxes first, then
    the components), so the reader batches the page. A box lives or dies by its
    read: ``None`` (the guard rejected it) drops it; a box under ``min_det`` or
    a read under ``min_score`` drops (an unscored one is not held to either);
    with ``strip_symbols`` the text loses its pictographs (:func:`drop_symbols`)
    before the rest is judged; a text must carry a letter (:func:`has_script`)
    and pass the line floors (``min_chars`` / ``skip_en``) to become a line, and
    a line read inside another surviving line is the same text twice and drops
    (:func:`drop_nested`). The reader's
    confidence is the line's ``score`` (:data:`NO_SCORE` for a bare-string
    read); the detector's ``det`` is kept on its line, and a component has
    none. Whatever text a line arrived with is not consulted — the engine hands
    over none.
    """
    boxes: list[Box] = [tuple(int(v) for v in ln.box) for ln in lines]
    dets: list[float] = [float(ln.det) for ln in lines]
    comps: list[Box] = []
    if mask is not None and comp_max > 0:
        comps = [
            c
            for c in mask_components(mask, min_side=comp_min_side)
            if not covered(c, boxes)
        ][:comp_max]
    if not boxes and not comps:
        return []
    reads = [split_reads(r) for r in read_boxes(bgr, boxes + comps)]
    if strip_symbols:
        reads = [
            [(t if t is None else drop_symbols(t), c, k) for t, c, k in rs]
            for rs in reads
        ]

    def keeps(text: str | None) -> bool:
        return (
            bool(text)
            and has_script(text)
            and keep_line(text, min_chars=min_chars, skip_en=skip_en)
        )

    def sure(det: float, conf: float) -> bool:
        return (det == NO_SCORE or det >= min_det) and (
            conf == NO_SCORE or conf >= min_score
        )

    out: list[OcrLine] = drop_nested(
        [
            OcrLine(seq=0, box=box, score=conf, text=text, det=det, kind=kind)
            for box, det, rs in zip(boxes, dets, reads[: len(boxes)], strict=True)
            for text, conf, kind in rs
            if keeps(text) and sure(det, conf)
        ]
    )
    seen = list(boxes)
    for box, rs in zip(comps, reads[len(boxes) :], strict=True):
        if covered(box, seen):
            continue
        lines_here = [
            OcrLine(seq=0, box=box, score=conf, text=text, det=NO_SCORE, kind=kind)
            for text, conf, kind in rs
            if keeps(text) and sure(NO_SCORE, conf)
        ]
        if lines_here:
            seen.append(box)
            out.extend(lines_here)
    return [replace(ln, seq=i) for i, ln in enumerate(reading_order(out), 1)]


def split_read(read: Read) -> tuple[str | None, float]:
    """A :data:`Read` as ``(text, confidence)``: a bare string carries
    :data:`NO_SCORE`, ``None`` stays ``None``."""
    if read is None:
        return None, NO_SCORE
    if isinstance(read, str):
        return read, NO_SCORE
    text, conf = read
    return text, float(conf)


def split_reads(read: Read | KindedReads) -> list[tuple[str | None, float, str]]:
    """One crop's reads as ``(text, confidence, kind)`` rows: a two-prompt
    reader's list as is, a one-prompt :data:`Read` as one row of kind ``""``."""
    if isinstance(read, list):
        return [(t, float(c), k) for t, c, k in read]
    return [(*split_read(read), "")]


def mask_for(mask_dir: Path, rel: Path) -> Path | None:
    """``<mask_dir>/<rel dir>/{stem}_mask.png``, or the legacy flat one, or
    ``None`` — :func:`anime_tools.masking._mask_paths.find_mask`."""
    return find_mask(mask_dir, rel)


_read_image = imread
"""The fallback page decode (a module name the tests patch)."""


@dataclass
class RereadEngine:
    """An :class:`~anime_tools.ocr.engine.OcrEngine` with the VL pass behind it.

    ``read`` / ``read_iter`` answer the engine's boxes, each page then run
    through :func:`reread_lines`. The pixels come from the engine's own pass
    (``read_iter(..., with_pixels=True)``) rather than a second decode: they are
    alive for the length of its chunk anyway, and a page is decoded once.
    ``masks`` is the mask tree, resolved per image against ``resized_dir`` by
    :func:`mask_for`; ``None`` reads no components.
    """

    engine: object
    read_boxes: ReadBoxes
    resized_dir: Path
    masks: Path | None = None
    comp_min_side: int = 32
    comp_max: int = 16
    min_chars: int = 2
    skip_en: bool = True
    min_det: float = LINE_MIN_DET
    min_score: float = LINE_MIN_SCORE
    strip_symbols: bool = True

    def _page(self, path: Path, lines: list[OcrLine], bgr=None) -> list[OcrLine]:
        mask = None
        if self.masks is not None:
            try:
                rel = path.relative_to(self.resized_dir)
            except ValueError:
                rel = Path(path.name)
            mp = mask_for(self.masks, rel)
            if mp is not None:
                mask = imread(mp, gray=True)
        if not lines and mask is None:
            return []
        # ``bgr`` is the detector pass's own decode; only a caller that has none
        # (or one the engine could not decode) pays for a read here.
        if bgr is None:
            bgr = _read_image(path)
        if bgr is None:
            return list(lines)
        if mask is not None and mask.shape[:2] != bgr.shape[:2]:
            import cv2

            mask = cv2.resize(
                mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        return reread_lines(
            bgr,
            lines,
            self.read_boxes,
            mask=mask,
            comp_min_side=self.comp_min_side,
            comp_max=self.comp_max,
            min_chars=self.min_chars,
            skip_en=self.skip_en,
            min_det=self.min_det,
            min_score=self.min_score,
            strip_symbols=self.strip_symbols,
        )

    def read(self, image_path: Path) -> list[OcrLine]:
        return next(iter(self.read_iter([image_path])))

    def read_iter(self, image_paths: Sequence[Path]) -> Iterator[list[OcrLine]]:
        paths = list(image_paths)
        pages = self.engine.read_iter(paths, with_pixels=True)
        for path, (lines, bgr) in zip(paths, pages, strict=True):
            yield self._page(path, lines, bgr)


__all__ = [
    "KANA_MARKS",
    "LINE_MIN_DET",
    "LINE_MIN_SCORE",
    "NESTED_CONTAINMENT",
    "NO_SCORE",
    "KindedReads",
    "Read",
    "RereadEngine",
    "covered",
    "drop_nested",
    "has_script",
    "mask_components",
    "mask_for",
    "overlap",
    "reread_lines",
    "split_read",
    "split_reads",
]
