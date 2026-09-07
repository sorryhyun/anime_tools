"""The VL pass over a page: every detected box read by the manga reader, plus
what the text mask boxed that no detector did.

:mod:`anime_tools.ocr._onnx` finds the boxes and reads nothing;
:mod:`anime_tools.ocr.sfx` reads a crop — hearts, small kana, hand-lettered
onomatopoeia — but detects nothing. This module is the seam between them, and
what the OCR stage runs:

* :func:`reread_lines` — one page in, one page out. Each box is cut with the
  reader's padding and read; the read *is* the line, so a read the decode guard
  rejects (``None``, the guard is inside :meth:`~anime_tools.ocr.sfx.SfxReader.read`)
  drops its box, and a read that passes must still clear the line floors
  (``min_chars`` / ``skip_en``) and carry a letter (:func:`has_script`). With a
  text mask (the MIT ``{stem}_mask.png`` :mod:`anime_tools.masking` writes) its
  connected components that no box already covers become crops too, held to the
  same floors. Reading order is settled afterwards, since a component may sit
  anywhere.
* :class:`RereadEngine` — the :class:`~anime_tools.ocr._onnx.OcrEngine` shape
  (``read`` / ``read_iter``) over an engine and a reader, so the stage runs it
  without knowing.

The boxes are never joined: the AnimeText detector answers a balloon as a block
and its columns (nesting settled by :func:`~anime_tools.ocr.animetext.denest`),
and joining those measured a loss (sincos, 2026-09-06: manga-ocr best-match
0.844 → 0.803 over 95 joins — SFX beside a balloon gets pulled in).

Measured on the sincos shard (the trainer's ``project/cjk_aware_anima_dit``,
2026-09-06): the masked-but-no-line floor 23 → 8 pages, manga-ocr best-match
0.786 → 0.810, hearts back on the speech lines; the regressions are digits
(``91`` → ``9``) and two-glyph crops. A VL line's ``score`` is the reader's
mean greedy token probability (:meth:`~anime_tools.ocr.sfx.SfxReader.read_boxes_scored`);
a reader answering bare strings leaves :data:`NO_SCORE`. The detector's box
confidence rides through as ``det`` (a mask component has none).

Torch-free to import; cv2 loads inside the two functions that need it.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from anime_tools.captions.ocr_sidecar import OcrLine
from anime_tools.ocr._text import keep_line, reading_order

Box = tuple[int, int, int, int]
Read = str | tuple[str, float] | None
"""One crop's read: the text with the reader's confidence, a bare text (no
confidence — :data:`NO_SCORE`), or ``None`` for a read the guard rejected."""
ReadBoxes = Callable[[object, Sequence[Box]], list[Read]]
"""``(bgr, boxes) → reads``: :meth:`~anime_tools.ocr.sfx.SfxReader.read_boxes_scored`."""

NO_SCORE = 0.0
"""The ``score`` (or ``det``) of a line nothing stood behind — a read with no
confidence, a box no detector scored — and ``0.000`` in the sidecar says so."""

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
    the smaller area, so a box inside another reads as 1."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0, 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter), inter / max(1, min(area_a, area_b))


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


def has_script(text: str) -> bool:
    """Whether a read carries a letter at all — ``♡`` or ``…`` alone is a
    decoration the mask caught, not a line."""
    return any(unicodedata.category(ch).startswith("L") for ch in text)


def reread_lines(
    bgr,
    lines: Sequence[OcrLine],
    read_boxes: ReadBoxes,
    *,
    mask=None,
    comp_min_side: int = 32,
    comp_max: int = 16,
    min_chars: int = 3,
    skip_en: bool = True,
) -> list[OcrLine]:
    """One page through the VL reader: the detected boxes read, the uncovered
    mask components read, reading order and numbering settled afterwards.

    ``read_boxes`` is called once with every crop (the lines' boxes first, then
    the components), so the reader batches the page. A box lives or dies by its
    read: ``None`` (the guard rejected it) drops it, and a text must carry a
    letter (:func:`has_script`) and pass the line floors (``min_chars`` /
    ``skip_en``) to become a line. The reader's confidence is the line's
    ``score`` (:data:`NO_SCORE` for a bare-string read); the detector's ``det``
    is kept on its line, and a component has none. Whatever text a line arrived
    with is not consulted — the engine hands over none.
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
    reads = [split_read(r) for r in read_boxes(bgr, boxes + comps)]

    def keeps(text: str | None) -> bool:
        return (
            bool(text)
            and has_script(text)
            and keep_line(text, min_chars=min_chars, skip_en=skip_en)
        )

    out: list[OcrLine] = [
        OcrLine(seq=0, box=box, score=conf, text=text, det=det)
        for box, det, (text, conf) in zip(boxes, dets, reads[: len(boxes)], strict=True)
        if keeps(text)
    ]
    seen = list(boxes)
    for box, (text, conf) in zip(comps, reads[len(boxes) :], strict=True):
        if not keeps(text) or covered(box, seen):
            continue
        seen.append(box)
        out.append(OcrLine(seq=0, box=box, score=conf, text=text, det=NO_SCORE))
    return [
        OcrLine(seq=i, box=ln.box, score=ln.score, text=ln.text, det=ln.det)
        for i, ln in enumerate(reading_order(out), 1)
    ]


def split_read(read: Read) -> tuple[str | None, float]:
    """A :data:`Read` as ``(text, confidence)``: a bare string carries
    :data:`NO_SCORE`, ``None`` stays ``None``."""
    if read is None:
        return None, NO_SCORE
    if isinstance(read, str):
        return read, NO_SCORE
    text, conf = read
    return text, float(conf)


def mask_for(mask_dir: Path, rel: Path) -> Path | None:
    """``<mask_dir>/<rel dir>/{stem}_mask.png``, or the legacy flat one
    (:func:`anime_tools.masking._masks.mask_name`), or ``None``."""
    name = f"{rel.stem}_mask.png"
    nested = mask_dir / rel.parent / name
    if nested.is_file():
        return nested
    flat = mask_dir / name
    return flat if flat.is_file() else None


def _read_image(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _read_mask(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


@dataclass
class RereadEngine:
    """An :class:`~anime_tools.ocr._onnx.OcrEngine` with the VL pass behind it.

    ``read`` / ``read_iter`` answer the engine's boxes, each page then run
    through :func:`reread_lines`. The page is decoded a second time here (the
    engine keeps no pixels past its chunk); against a 1.9 B-parameter read per
    crop that is noise. ``masks`` is the mask tree, resolved per image against
    ``resized_dir`` by :func:`mask_for`; ``None`` reads no components.
    """

    engine: object
    read_boxes: ReadBoxes
    resized_dir: Path
    masks: Path | None = None
    comp_min_side: int = 32
    comp_max: int = 16
    min_chars: int = 2
    skip_en: bool = True

    def _page(self, path: Path, lines: list[OcrLine]) -> list[OcrLine]:
        mask = None
        if self.masks is not None:
            try:
                rel = path.relative_to(self.resized_dir)
            except ValueError:
                rel = Path(path.name)
            mp = mask_for(self.masks, rel)
            if mp is not None:
                mask = _read_mask(mp)
        if not lines and mask is None:
            return []
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
        )

    def read(self, image_path: Path) -> list[OcrLine]:
        return self._page(image_path, self.engine.read(image_path))

    def read_iter(self, image_paths: Sequence[Path]) -> Iterator[list[OcrLine]]:
        paths = list(image_paths)
        for path, lines in zip(paths, self.engine.read_iter(paths), strict=True):
            yield self._page(path, lines)


__all__ = [
    "NO_SCORE",
    "Read",
    "RereadEngine",
    "covered",
    "has_script",
    "mask_components",
    "mask_for",
    "overlap",
    "reread_lines",
    "split_read",
]
