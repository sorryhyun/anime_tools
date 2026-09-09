"""AnimeText text-block detection: ``deepghs/AnimeText_yolo`` in front of a reader.

The OCR stage's one detector: a YOLO12 trained on AnimeText (735k anime / manga
pages, one class, ``text_block``), run on torch through the vendored graph in
:mod:`anime_tools.vision.yolo12`. It finds balloon lines and the sound effects drawn onto the artwork alike, so the
extra layers the stage once needed to reach the SFX (VL Spotting, the text
mask's components) are optional at most. Measured on the sincos shard (the
trainer's ``project/cjk_aware_anima_dit``, 2026-09-06): every one of the
previous line detector's 237 lines covered, 98 % of the hand-labelled SFX boxed,
the masked-but-no-box floor 38 → 3 pages. The line-detector-plus-CTC-recognizer
stack it replaced was retired outright 2026-09-07.

It ran on onnxruntime until 2026-09-09. Torch answers the same boxes — the two
graphs agree to 3.0e-03 on a box coordinate in canvas pixels at 640², and CPU and
MPS agree exactly on the boxes a page yields — at the same speed on a CPU (392 ms
a page against 357) and 4.8x faster wherever there is a GPU: 75 ms on an Apple MPS
device, where onnxruntime had no provider at all.

What it emits is an **axis-aligned box per text block**, not a line quad. A
vertical balloon comes back as the block *and* each of its columns — nested
boxes — and :func:`denest` settles that: ``inner`` (the default) drops a box
that holds two or more others, keeping the columns, since a block read matches
no balloon line; ``outer`` keeps the block; ``raw`` keeps both. A block box is
multi-column, which only a reader that sees the whole crop reads right, so the
detector pairs with the manga VL reader (:mod:`anime_tools.ocr.sfx`) through
:mod:`anime_tools.ocr.reread` — the OCR stage's only path.

Weights are a catalog row (``animetext_det``, :mod:`anime_tools.downloads`),
fetched on first use and **never bundled**: the model card is GPL-3.0 and the
dataset CC-BY-NC-SA-4.0, neither of which this MIT package may carry. The pre- and
post-processing here is torch-free — ``cv2``, ``numpy`` and ``torch`` load inside
the functions that need them, so importing this module stays cheap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anime_tools.downloads import ANIMETEXT_WEIGHTS, default_animetext_dir

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
    """An axis-aligned box as the ``(4, 2)`` TL-TR-BR-BL quad the engine's size
    filters and :func:`~anime_tools.ocr.engine.crop_quad` take — the
    :class:`~anime_tools.ocr.engine.Detector` protocol's one box shape."""
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
    / :meth:`boxes`) so :class:`~anime_tools.ocr.engine.OcrEngine` runs it over a
    chunk of pages at one canvas shape.
    """

    model: Any
    device: str = "cpu"
    imgsz: int = DEFAULT_IMGSZ
    conf: float = DEFAULT_CONF
    nms: float = DEFAULT_NMS
    nest: str = "inner"

    @classmethod
    def load(
        cls,
        model_dir: Path | None = None,
        *,
        device: str | None = None,
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
        nms: float = DEFAULT_NMS,
        nest: str = "inner",
        fetch: bool = True,
    ) -> AnimeTextDetector:
        """The graph on ``device`` (auto when ``None``). ``model_dir`` defaults to
        the catalog row's directory, fetched when missing (``fetch=False`` raises
        :class:`AnimeTextWeightsMissing` instead); a dir passed explicitly is
        used as is."""
        from anime_tools._device import resolve_device
        from anime_tools.vision.yolo12 import load_yolo12

        if nest not in NEST_POLICIES:
            raise ValueError(f"nest policy must be one of {list(NEST_POLICIES)}")
        if model_dir is None:
            model_dir = _ensure(fetch)
        weights = Path(model_dir) / ANIMETEXT_WEIGHTS
        if not weights.is_file():
            raise AnimeTextWeightsMissing(weights.parent)
        dev = resolve_device(device)
        model = load_yolo12(weights, nc=1, device=dev)
        return cls(
            model=model, device=dev, imgsz=int(imgsz), conf=conf, nms=nms, nest=nest
        )

    # ---- the engine's detector protocol ---------------------------------

    def prepare(self, bgr):
        """The array for one image plus what maps its boxes back: the
        letterbox scale. Pure CPU and numpy, pool-safe — torch is not touched
        until :meth:`forward_batch` stacks the chunk."""
        canvas, r = letterbox(bgr, self.imgsz)
        return to_tensor(canvas), r

    def forward_batch(self, prepared: Sequence[tuple]) -> list:
        """The raw ``(5, N)`` head per prepared image — forward passes only, on
        the calling thread.

        One forward over the whole chunk when every canvas agrees on its shape,
        which at a fixed ``imgsz`` is always. ``imgsz=0`` letterboxes each page
        at its own native size, so there the run falls back to a forward per
        shape rather than refusing.
        """
        import numpy as np
        import torch

        out: list = [None] * len(prepared)
        by_shape: dict[tuple, list[int]] = {}
        for i, (x, _) in enumerate(prepared):
            by_shape.setdefault(x.shape[1:], []).append(i)
        with torch.inference_mode():
            for idx in by_shape.values():
                batch = torch.from_numpy(
                    np.concatenate([prepared[i][0] for i in idx], 0)
                ).to(self.device)
                heads = self.model(batch).float().cpu().numpy()
                for slot, head in zip(idx, heads, strict=True):
                    out[slot] = head
        return out

    def boxes(self, raw, prepared: tuple, shape: tuple[int, int]) -> list:
        """One image's boxes from its head, as ``(quad, score)`` pairs, nesting
        settled — the CPU tail, pool-safe. ``shape`` is the image's ``(h, w)``."""
        _, r = prepared
        scored = decode(raw, r, shape[1], shape[0], conf=self.conf, nms=self.nms)
        return [(as_quad(b), float(b[4])) for b in denest(scored, self.nest)]

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
        """:meth:`detect` over many images, in one forward where the canvases
        agree."""
        prepared = [self.prepare(bgr) for bgr in bgrs]
        raws = self.forward_batch(prepared)
        return [
            [
                b[:4]
                for b in denest(
                    decode(
                        raw,
                        p[1],
                        bgr.shape[1],
                        bgr.shape[0],
                        conf=self.conf,
                        nms=self.nms,
                    ),
                    self.nest,
                )
            ]
            for raw, p, bgr in zip(raws, prepared, bgrs, strict=True)
        ]


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
