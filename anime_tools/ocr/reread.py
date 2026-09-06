"""The VL pass over a page: every PP-OCRv6 line re-read by the manga reader,
plus what the text mask boxed that no detector did.

:mod:`anime_tools.ocr._onnx` finds and reads the lines; :mod:`anime_tools.ocr.sfx`
reads a crop better — hearts, small kana, hand-lettered onomatopoeia — but
detects nothing. This module is the seam between them, and what the OCR stage
runs under ``--reader vl``:

* :func:`reread_lines` — one page in, one page out. Each line's box is cut with
  the reader's padding and re-read; a read that passes the decode guard replaces
  the text, a rejected one leaves the PP-OCRv6 text alone (the guard is inside
  :meth:`~anime_tools.ocr.sfx.SfxReader.read`). With a text mask (the MIT
  ``{stem}_mask.png`` :mod:`anime_tools.masking` writes) its connected
  components that no line already covers become crops too, and a read that
  passes the guard and the line floors becomes a line of its own. Reading order
  is settled afterwards, since a new line may sit anywhere.
* :class:`RereadEngine` — the :class:`~anime_tools.ocr._onnx.OcrEngine` shape
  (``read`` / ``read_iter``) over an engine and a reader, so the stage swaps
  it in without knowing.

Measured on the sincos shard (the trainer's ``project/cjk_aware_anima_dit``,
2026-09-06): the masked-but-no-line floor 23 → 8 pages, manga-ocr best-match
0.786 → 0.810, hearts back on the speech lines; the regressions are digits
(``91`` → ``9``) and two-glyph crops. A VL-only line — a mask component — has
no recognizer confidence, so its ``score`` is :data:`NO_SCORE`.

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
ReadBoxes = Callable[[object, Sequence[Box]], list[str | None]]
"""``(bgr, boxes) → texts``: :meth:`~anime_tools.ocr.sfx.SfxReader.read_boxes`,
``None`` for a read the guard rejected."""

NO_SCORE = 0.0
"""The ``score`` of a line only the VL reader produced: no recognizer stood
behind it, and ``0.000`` in the sidecar says so."""

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
    """One page through the VL reader: the lines re-read, the uncovered mask
    components read, reading order and numbering settled afterwards.

    ``read_boxes`` is called once with every crop (the lines' boxes first, then
    the components), so the reader batches the page. A line whose read is
    ``None`` keeps its text and score; a component's read must pass the line
    floors (``min_chars`` / ``skip_en``, the same ones the engine applied) and
    :func:`has_script` to become a line, with :data:`NO_SCORE`.
    """
    boxes: list[Box] = [tuple(int(v) for v in ln.box) for ln in lines]
    comps: list[Box] = []
    if mask is not None and comp_max > 0:
        comps = [
            c
            for c in mask_components(mask, min_side=comp_min_side)
            if not covered(c, boxes)
        ][:comp_max]
    if not boxes and not comps:
        return []
    reads = read_boxes(bgr, boxes + comps)
    out: list[OcrLine] = []
    for ln, text in zip(lines, reads[: len(boxes)], strict=True):
        if text is None or text == ln.text:
            out.append(ln)
        else:
            out.append(OcrLine(seq=ln.seq, box=ln.box, score=ln.score, text=text))
    seen = list(boxes)
    for box, text in zip(comps, reads[len(boxes) :], strict=True):
        if not text or not has_script(text):
            continue
        if not keep_line(text, min_chars=min_chars, skip_en=skip_en):
            continue
        if covered(box, seen):
            continue
        seen.append(box)
        out.append(OcrLine(seq=0, box=box, score=NO_SCORE, text=text))
    return [
        OcrLine(seq=i, box=ln.box, score=ln.score, text=ln.text)
        for i, ln in enumerate(reading_order(out), 1)
    ]


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

    ``read`` / ``read_iter`` answer what the engine's do, each page then run
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
    min_chars: int = 3
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
    "RereadEngine",
    "covered",
    "has_script",
    "mask_components",
    "mask_for",
    "overlap",
    "reread_lines",
]
