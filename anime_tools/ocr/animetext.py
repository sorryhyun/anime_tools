"""AnimeText text-block detection: ``deepghs/AnimeText_yolo`` in front of a reader.

PP-OCRv6's DB head (:class:`anime_tools.ocr._onnx.TextDetector`) finds balloon
lines and misses most of what is drawn onto the artwork; the OCR stage grew two
more layers behind it (VL Spotting, the text mask's components) to reach the
sound effects. This module is the one detector that replaces all three: a
YOLO12 trained on AnimeText (735k anime / manga pages, one class,
``text_block``), run from its ONNX export on onnxruntime. Measured on the
sincos shard (the trainer's ``project/cjk_aware_anima_dit``, 2026-09-06): every
one of PP-OCRv6's 237 lines covered, 98 % of the hand-labelled SFX boxed, the
masked-but-no-box floor 38 → 3 pages, 26 ms a page on the CUDA provider.

What it emits is an **axis-aligned box per text block**, not a line quad. A
vertical balloon comes back as the block *and* each of its columns — nested
boxes — and :func:`denest` settles that: ``inner`` (the default) drops a box
that holds two or more others, keeping the columns, since a block read matches
no balloon line; ``outer`` keeps the block; ``raw`` keeps both. A block box is
multi-column, which PP-OCRv6's line recognizer garbles, so this detector pairs
with the manga VL reader (:mod:`anime_tools.ocr.sfx`) — the OCR stage's
``--detector animetext --reader vl``. ``--reader ppocr`` on its boxes is
allowed, not a default.

Weights are a catalog row (``animetext_det``, :mod:`anime_tools.downloads`),
fetched on first use and **never bundled**: the model card is GPL-3.0 and the
dataset CC-BY-NC-SA-4.0, neither of which this MIT package may carry. Torch-free;
``cv2`` and ``numpy`` load inside the functions that need them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anime_tools._onnx import make_session
from anime_tools.downloads import ANIMETEXT_ONNX, default_animetext_dir

DEFAULT_IMGSZ = 640
"""Letterbox side. The card exports at 640; measured against 1024 and native
on every column of the sincos probe it is a wash, at a twelfth of the wall."""

DEFAULT_CONF = 0.25
"""Score floor. The card's F1 threshold is 0.426 (``threshold.json``); 0.25
adds ~15 % more boxes on a manga page, nearly all of them real (the hand
SFX rows go 98 % → 100 % boxed) at a handful more guard rejections."""

DEFAULT_NMS = 0.5
NEST_TH = 0.85
"""A box ≥ this share inside another is *nested* in it (:func:`denest`)."""

NEST_POLICIES = ("inner", "outer", "raw")
PAD_VALUE = 114
"""Ultralytics' letterbox grey."""

Box = tuple[int, int, int, int]
Scored = tuple[int, int, int, int, float]


def letterbox(bgr, imgsz: int = DEFAULT_IMGSZ):
    """Top-left letterbox onto an ``imgsz`` square (or the native size padded
    to a multiple of 32 when ``imgsz`` is 0); returns the canvas and the scale
    applied to the image."""
    import cv2
    import numpy as np

    h0, w0 = bgr.shape[:2]
    if imgsz:
        r = min(imgsz / h0, imgsz / w0)
        nw, nh = max(1, round(w0 * r)), max(1, round(h0 * r))
        cw = ch = imgsz
    else:
        r, nw, nh = 1.0, w0, h0
        cw, ch = (w0 + 31) // 32 * 32, (h0 + 31) // 32 * 32
    canvas = np.full((ch, cw, 3), PAD_VALUE, np.uint8)
    canvas[:nh, :nw] = (
        cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA) if r != 1.0 else bgr
    )
    return canvas, r


def to_tensor(canvas):
    """BGR ``uint8`` canvas → the ``(1, 3, H, W)`` float RGB tensor the export
    takes (0-1, no mean/std — YOLO's own convention)."""
    import numpy as np

    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    x *= 1.0 / 255
    return np.ascontiguousarray(x)


def decode(
    out,
    r: float,
    w0: int,
    h0: int,
    *,
    conf: float = DEFAULT_CONF,
    nms: float = DEFAULT_NMS,
) -> list[Scored]:
    """One image's ``(5, N)`` head — ``cx cy w h score`` per anchor, canvas
    coordinates — to ``(x0, y0, x1, y1, score)`` boxes in image coordinates,
    greedy-NMS'd at ``nms`` IoU, clipped to the image, empty ones dropped."""
    import cv2
    import numpy as np

    pred = np.asarray(out).T  # (N, 5)
    keep = pred[:, 4] >= conf
    if not keep.any():
        return []
    p = pred[keep]
    xywh = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 2], p[:, 3]], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), p[:, 4].tolist(), conf, nms)
    boxes: list[Scored] = []
    for i in np.asarray(idx).flatten():
        x, y, w, h = xywh[i] / r
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + w), w0), min(int(y + h), h0)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1, float(p[i, 4])))
    return boxes


def containment(inner: Sequence[int], outer: Sequence[int]) -> float:
    """Share of ``inner``'s area inside ``outer``."""
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy / max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))


def denest(boxes: Sequence, policy: str = "inner", th: float = NEST_TH) -> list:
    """Settle the block-and-its-columns nesting YOLO emits for a balloon.

    ``inner`` drops a box holding ≥ 2 smaller boxes at ≥ ``th`` (keeps the
    columns); ``outer`` drops a box ≥ ``th`` inside a larger one (keeps the
    block); ``raw`` keeps both. Boxes may carry a trailing score; order is kept.

    ``outer`` looks tidier and is wrong for balloon lines: a block read matches
    no line (manga-ocr best-match 0.844 → 0.694 on the sincos probe), while
    ``inner`` costs nothing measurable.
    """
    if policy not in NEST_POLICIES:
        raise ValueError(f"nest policy must be one of {list(NEST_POLICIES)}")
    if policy == "raw" or len(boxes) < 2:
        return list(boxes)

    def area(b) -> int:
        return (b[2] - b[0]) * (b[3] - b[1])

    keep = []
    for i, b in enumerate(boxes):
        others = [o for j, o in enumerate(boxes) if j != i]
        if policy == "outer":
            if any(area(o) > area(b) and containment(b, o) >= th for o in others):
                continue
        elif sum(area(o) < area(b) and containment(o, b) >= th for o in others) >= 2:
            continue
        keep.append(b)
    return keep


def as_quad(box: Sequence[int]):
    """An axis-aligned box as the ``(4, 2)`` TL-TR-BR-BL quad the engine's crop
    and size filters take — what :class:`~anime_tools.ocr._onnx.TextDetector`
    emits, so both detectors speak one shape downstream."""
    import numpy as np

    x0, y0, x1, y1 = (float(v) for v in box[:4])
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class AnimeTextWeightsMissing(RuntimeError):
    """The catalog row has not filled the model directory yet."""

    def __init__(self, model_dir: Path) -> None:
        super().__init__(
            f"AnimeText detector weights not found in {model_dir} — run "
            "`python -m anime_tools.downloads animetext_det` "
            "(or ⚙ Settings › Models in the GUI)"
        )


@dataclass
class AnimeTextDetector:
    """YOLO12 text blocks: a page in, axis-aligned boxes out.

    Speaks the engine's detector protocol (:meth:`prepare` / :meth:`forward_batch`
    / :meth:`boxes`) so :class:`~anime_tools.ocr._onnx.OcrEngine` runs it in
    the DB detector's place, one image per forward at one canvas shape.
    """

    session: Any
    imgsz: int = DEFAULT_IMGSZ
    conf: float = DEFAULT_CONF
    nms: float = DEFAULT_NMS
    nest: str = "inner"

    @classmethod
    def load(
        cls,
        model_dir: Path | None = None,
        *,
        device: str = "cpu",
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
        nms: float = DEFAULT_NMS,
        nest: str = "inner",
        fetch: bool = True,
    ) -> AnimeTextDetector:
        """The session on ``device``. ``model_dir`` defaults to the catalog
        row's directory, fetched when missing (``fetch=False`` raises
        :class:`AnimeTextWeightsMissing` instead); a dir passed explicitly is
        used as is."""
        if nest not in NEST_POLICIES:
            raise ValueError(f"nest policy must be one of {list(NEST_POLICIES)}")
        if model_dir is None:
            model_dir = _ensure(fetch)
        onnx = Path(model_dir) / ANIMETEXT_ONNX
        if not onnx.is_file():
            raise AnimeTextWeightsMissing(onnx.parent)
        session = make_session(onnx, device, what="the AnimeText detector")
        return cls(session=session, imgsz=int(imgsz), conf=conf, nms=nms, nest=nest)

    # ---- the engine's detector protocol ---------------------------------

    def prepare(self, bgr):
        """The tensor for one image plus what maps its boxes back: the
        letterbox scale. Pure CPU, pool-safe."""
        canvas, r = letterbox(bgr, self.imgsz)
        return to_tensor(canvas), r

    def forward_batch(self, prepared: Sequence[tuple]) -> list:
        """The raw ``(5, N)`` head per prepared image — forward passes only,
        on the calling thread. One image per forward: every canvas is the
        same shape, so the session never re-plans."""
        name = self.session.get_inputs()[0].name
        return [self.session.run(None, {name: x})[0][0] for x, _ in prepared]

    def boxes(self, raw, prepared: tuple, shape: tuple[int, int]) -> list:
        """One image's boxes from its head, as quads, nesting settled — the
        CPU tail, pool-safe. ``shape`` is the image's ``(h, w)``."""
        _, r = prepared
        scored = decode(raw, r, shape[1], shape[0], conf=self.conf, nms=self.nms)
        return [as_quad(b) for b in denest(scored, self.nest)]

    # ---- the plain calls ---------------------------------------------

    def detect_scored(self, bgr) -> list[Scored]:
        """Every text block in ``bgr`` with its score, nesting settled."""
        prepared = self.prepare(bgr)
        raw = self.forward_batch([prepared])[0]
        _, r = prepared
        h, w = bgr.shape[:2]
        return denest(decode(raw, r, w, h, conf=self.conf, nms=self.nms), self.nest)

    def detect(self, bgr) -> list[Box]:
        """Every text block in ``bgr`` as ``(x0, y0, x1, y1)``."""
        return [b[:4] for b in self.detect_scored(bgr)]

    def detect_batch(self, bgrs: Sequence) -> list[list[Box]]:
        """:meth:`detect` over many images, in order."""
        return [self.detect(bgr) for bgr in bgrs]


def _ensure(fetch: bool) -> Path:
    from anime_tools.downloads import by_id

    row = by_id()["animetext_det"]
    if row.missing():
        if not fetch:
            raise AnimeTextWeightsMissing(row.dest or default_animetext_dir())
        row.fetch()
    assert row.dest is not None
    return row.dest


__all__ = [
    "DEFAULT_CONF",
    "DEFAULT_IMGSZ",
    "DEFAULT_NMS",
    "NEST_POLICIES",
    "NEST_TH",
    "AnimeTextDetector",
    "AnimeTextWeightsMissing",
    "as_quad",
    "containment",
    "decode",
    "denest",
    "letterbox",
    "to_tensor",
]
