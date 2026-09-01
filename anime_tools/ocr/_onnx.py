"""The only place a PP-OCRv6 model is constructed — detection and recognition.

:mod:`anime_tools.masking._sam3` seen from the OCR side: one module owns loading
the two ONNX sessions, the pre/post-processing that surrounds them, and the two
paths the download catalog writes to, so a Download button can never put a
checkpoint somewhere the loader will not look. Like the MIT stage's ``--ctd-gate``
net, the weights have **no flag**: a ``--det-dir`` you could point elsewhere is a
Download button aimed at the wrong directory.

Deliberately **not PaddlePaddle**. PP-OCRv6 ships an official ONNX mirror of both
halves, and ``inference.yml`` beside each carries everything the wrapping code
needs — the recognizer's 18,708-character dictionary and the detector's DB
thresholds — so nothing is reverse-engineered and no second deep-learning
framework enters a py3.13 / ``numpy>=2`` / torch stack for one 19M-parameter
model. ``onnxruntime`` and ``opencv`` are already here.

Two upstream details are reproduced rather than corrected, because matching
PaddleOCR is the point and being right about them is not:

* both models are fed **BGR**, and the detector's ImageNet mean/std are applied
  in that order — the constants are RGB ones, and upstream applies them to a
  ``cv2.imread`` array anyway;
* the recognizer is normalized ``(x/255 - 0.5) / 0.5`` and padded, never
  stretched, to the batch's widest aspect ratio.

What is *not* reproduced is upstream's habit of feeding each session whatever
shape the image happens to have. ONNX Runtime's CUDA provider re-plans on every
shape change, which on a bucketed resized tree is most of the run time; both
sessions here are held at one shape instead. :func:`_pad_to` has the numbers.

The one thing done differently is DB's *unclip*, and only because the difference
is invisible: upstream offsets the polygon with ``pyclipper``, which for the
rotated **rectangle** ``minAreaRect`` just produced is exactly "grow both sides by
``2·area·ratio/perimeter``". The corners come out square instead of round and the
very next call takes ``minAreaRect`` of the result, so the two agree — and a
dependency that exists to round four corners nobody reads does not get added.

Every heavy import is deferred into a function, so importing this module stays
cheap and torch-free: :mod:`anime_tools.gui.stages` dumps the stage's argparse in
a child interpreter and expects that to cost about 0.2 seconds.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anime_tools.captions.ocr_sidecar import OcrLine
from anime_tools.downloads import default_ppocr_det_dir, default_ppocr_rec_dir

ONNX_NAME = "inference.onnx"
CONFIG_NAME = "inference.yml"
"""What each Hub repo ships and what the catalog rows fetch: the graph and the
config that describes how to feed it. The two *directories* they land in are
:mod:`anime_tools.downloads`', imported rather than restated for the reason the
PE tower and the CTD net import theirs: a download and a load that spell the
same path twice can drift, and the symptom is a Download button that appears to
do nothing."""

DET_LIMIT_SIDE = 960
"""PaddleOCR's ``DetResizeForTest`` default: the longest side is scaled down to
this before detection, and both sides are then rounded to a multiple of 32. The
``inference.yml`` says ``DetResizeForTest: null``, which *is* this."""

REC_HEIGHT = 48
REC_MIN_WIDTH = 320
"""The recognizer's fixed input height, and the narrowest width a batch is padded
to — ``image_shape: [3, 48, 320]`` in its ``inference.yml``."""

MIN_BOX_SIDE = 3
"""DB's own floor on a candidate's short side, before the unclip."""

DET_STRIDE = 32
"""DB's downsampling factor: every side the detector is fed is a multiple of it."""


def _pad_to(limit_side: int) -> int:
    """The square canvas a GPU detector session is fed, for every image.

    ONNX Runtime's CUDA provider re-plans the graph on **every input-shape
    change** — not once per shape, every change: cycling three sizes costs the
    re-plan three times per round. Measured on the shipped detector, a
    960-limited page is 12 ms of forward behind 70 ms of re-planning, and a
    bucketed resized tree hands it a different ``(H, W)`` for almost every
    image. The same applies to the recognizer, where the crop width *and* the
    batch size move: 8 ms of forward, 108 ms whenever either changes.

    So both sessions are fed **one shape for the whole run** when they are on
    the GPU: the detector pads every image onto this canvas, the recognizer
    quantizes width to a multiple of :data:`REC_MIN_WIDTH` and pads every batch
    to :attr:`TextRecognizer.batch_size`. Padding is not free — a 640x960 page
    becomes 960x960, about 25% more convolution — and it is nowhere near the
    cost it removes.

    The CPU provider has no such penalty (and is ~40x slower regardless), so it
    keeps the native shape and pays nothing for the padding.
    """
    return max(round(limit_side / DET_STRIDE) * DET_STRIDE, DET_STRIDE)


def _is_gpu(session) -> bool:
    """Whether this session actually landed on an accelerator.

    ``device`` says what was *asked* for; :func:`_session` warns and continues on
    the CPU when the CUDA provider is missing, so only the session knows — and
    padding a CPU run to a square canvas would be pure loss.
    """
    return any(p != "CPUExecutionProvider" for p in session.get_providers())


@contextmanager
def _cv2_single_threaded() -> Iterator[None]:
    """Take OpenCV's own thread pool away for the duration.

    :meth:`OcrEngine.read_many` already spreads whole images across a pool, so
    OpenCV splitting each ``resize`` and ``warpPerspective`` inside that is a
    second layer of threads over the same cores — on a 12-core box, four workers
    each fanning out twelve ways. The outer split is the useful one: it has whole
    images to work on and it overlaps the session thread.

    Restored on the way out, because this is a process-wide setting and the GUI
    is not the only caller in the process.
    """
    import cv2

    prior = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(prior)


def _load_config(model_dir: Path) -> dict[str, Any]:
    import yaml

    path = model_dir / CONFIG_NAME
    if not path.is_file():
        raise OcrWeightsMissing(model_dir)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class OcrWeightsMissing(RuntimeError):
    """A model directory the catalog has not filled yet.

    Its own class so the CLI can turn it into the one instruction that fixes it
    rather than a stack trace, the way the tagger's ``from_dir`` names the mode
    that builds a missing checkpoint.
    """

    def __init__(self, model_dir: Path) -> None:
        super().__init__(
            f"PP-OCRv6 weights not found in {model_dir} — run "
            "`python -m anime_tools.downloads ppocr_det ppocr_rec` "
            "(or ⚙ Settings › Models in the GUI)"
        )


def _preload_cuda_libs(ort: Any) -> None:
    """Put the ``nvidia-*`` wheels' CUDA/cuDNN libraries on the loader path.

    Idempotent, so calling it once per session costs nothing, and swallowing
    is the right failure: a preload that raises leaves the provider list
    exactly as it was, which the caller already knows how to warn about. Absent
    on the CPU wheel and on onnxruntime < 1.21, hence the ``getattr``.
    """
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:  # pragma: no cover - depends on the install
        return
    try:
        preload()
    except Exception:  # noqa: BLE001, S110 - a failed preload is just "no GPU"
        pass


def _session(onnx_path: Path, device: str):
    """An ``InferenceSession`` on the GPU when one was asked for and is there.

    The CUDA provider is a separate wheel (``onnxruntime-gpu``), so asking for it
    where only the CPU build is installed must warn and continue rather than
    fail: this stage is worth running slowly. The same stance, and nearly the
    same sentence, as the MIT stage's CTD gate.

    ``preload_dlls()`` first, and it is not optional: the CUDA and cuDNN
    libraries the provider links against ship as their own ``nvidia-*`` wheels,
    and nothing puts them on the loader path. Without the preload the provider
    ``.so`` fails to open and ``get_available_providers()`` reports the *CPU*
    build's list — an onnxruntime-gpu install that silently runs on the CPU.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "onnxruntime is required for OCR but is not installed — "
            "`uv pip install onnxruntime` (or onnxruntime-gpu for CUDA). "
            "It is deliberately unpinned: the CPU and GPU wheels conflict."
        ) from exc
    if not onnx_path.is_file():
        raise OcrWeightsMissing(onnx_path.parent)

    providers = ["CPUExecutionProvider"]
    if device.startswith("cuda"):
        _preload_cuda_libs(ort)
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print(
                "WARNING: onnxruntime CUDAExecutionProvider unavailable — "
                "OCR falls back to CPU (install onnxruntime-gpu)",
                flush=True,
            )
    return ort.InferenceSession(str(onnx_path), providers=providers)


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


@dataclass
class TextDetector:
    """DB text detection: an image in, rotated line quads out."""

    session: Any
    thresh: float
    box_thresh: float
    unclip_ratio: float
    max_candidates: int
    limit_side: int

    pad_to: int = 0
    """Side of the square canvas every image is padded onto, or 0 to feed the
    session each image's own shape. See :func:`_pad_to`."""

    @classmethod
    def load(
        cls,
        model_dir: Path | None = None,
        *,
        device: str = "cpu",
        limit_side: int = DET_LIMIT_SIDE,
    ) -> TextDetector:
        model_dir = model_dir or default_ppocr_det_dir()
        post = _load_config(model_dir).get("PostProcess", {})
        session = _session(model_dir / ONNX_NAME, device)
        return cls(
            session=session,
            thresh=float(post.get("thresh", 0.2)),
            box_thresh=float(post.get("box_thresh", 0.45)),
            unclip_ratio=float(post.get("unclip_ratio", 1.4)),
            max_candidates=int(post.get("max_candidates", 3000)),
            limit_side=int(limit_side),
            pad_to=_pad_to(int(limit_side)) if _is_gpu(session) else 0,
        )

    def _preprocess(self, bgr):
        """Scale the longest side under the limit, round both to a multiple of 32.

        Returns the tensor, the two scale factors, and the *live* region inside
        the tensor. The scale factors are kept apart because the 32-rounding
        makes them differ: a box is mapped back through the axis it was found
        on, not through one average ratio.

        With :attr:`pad_to` the tensor is that live region sitting in the
        top-left of a square canvas, so the session sees one shape all run
        (:func:`_pad_to`). The pad is ``BORDER_REPLICATE`` rather than a
        constant: DB is fully convolutional and is entitled to find a text box
        along any hard edge in its field of view, and a border that continues
        the image is not one. The band is cut off the probability map before the
        contour pass regardless, so nothing found in it can survive.
        """
        import cv2
        import numpy as np

        h, w = bgr.shape[:2]
        ratio = min(1.0, self.limit_side / max(h, w))
        rh = max(round(h * ratio / DET_STRIDE) * DET_STRIDE, DET_STRIDE)
        rw = max(round(w * ratio / DET_STRIDE) * DET_STRIDE, DET_STRIDE)
        resized = cv2.resize(bgr, (rw, rh))
        if self.pad_to:
            side = max(self.pad_to, rh, rw)
            resized = cv2.copyMakeBorder(
                resized, 0, side - rh, 0, side - rw, cv2.BORDER_REPLICATE
            )
        x = resized.astype(np.float32) / 255.0
        x -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        x /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return x.transpose(2, 0, 1)[None], w / rw, h / rh, (rh, rw)

    @staticmethod
    def _mini_box(contour):
        """``minAreaRect`` as four points ordered TL, TR, BR, BL, plus its short side.

        The ordering is upstream's and is load-bearing twice over: the perspective
        crop maps these four onto a rectangle's corners in this order, and the
        unclip re-derives the rect from them.
        """
        import cv2

        rect = cv2.minAreaRect(contour)
        pts = sorted(cv2.boxPoints(rect), key=lambda p: p[0])
        i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
        i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
        return [pts[i1], pts[i2], pts[i3], pts[i4]], min(rect[1])

    @staticmethod
    def _box_score(prob, box) -> float:
        """Mean probability inside the quad — upstream's ``box_score_fast``."""
        import cv2
        import numpy as np

        h, w = prob.shape
        pts = np.array(box, dtype=np.int32)
        x0 = int(np.clip(pts[:, 0].min(), 0, w - 1))
        x1 = int(np.clip(pts[:, 0].max(), 0, w - 1))
        y0 = int(np.clip(pts[:, 1].min(), 0, h - 1))
        y1 = int(np.clip(pts[:, 1].max(), 0, h - 1))
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        pts[:, 0] -= x0
        pts[:, 1] -= y0
        cv2.fillPoly(mask, [pts.reshape(1, -1, 2)], 1)
        return float(cv2.mean(prob[y0 : y1 + 1, x0 : x1 + 1], mask)[0])

    def _unclip(self, box):
        """Grow the rectangle by DB's offset distance, without ``pyclipper``.

        For a rectangle the polygon offset is exactly a rectangle whose sides
        each moved out by ``d``; see the module docstring on why the rounded
        corners upstream would produce do not survive to be compared.
        """
        import cv2
        import numpy as np

        (cx, cy), (w, h), angle = cv2.minAreaRect(np.array(box, dtype=np.float32))
        if w <= 0 or h <= 0:
            return None
        d = (w * h) * self.unclip_ratio / (2.0 * (w + h))
        return cv2.boxPoints(((cx, cy), (w + 2 * d, h + 2 * d), angle))

    def detect(self, bgr) -> list:
        """Every text quad in ``bgr``, in the image's own pixel coordinates."""
        x, sx, sy, live = self._preprocess(bgr)
        prob = self.session.run(None, {self.session.get_inputs()[0].name: x})[0][0, 0]
        return self._boxes(prob, bgr.shape[:2], sx, sy, live)

    def probs_batch(self, prepared: Sequence[tuple]) -> list:
        """The probability map for each prepared image — **forward passes only**.

        This is the one method that must run on the calling thread, because it
        is the one that touches the session. Everything either side of it, the
        normalize before and the contour pass after, is the caller's to hand to
        a pool; returning raw maps rather than boxes is what makes that
        possible.

        ``prepared`` is what :meth:`_preprocess` returned for each image.

        One image per forward, deliberately. The detector's activations are
        full-resolution, so a batch of four is four times the work rather than
        four for the price of one — measured at 64 ms against 4x15 ms — and it
        would put a second moving axis into the input shape, which is the thing
        :func:`_pad_to` exists to hold still.
        """
        name = self.session.get_inputs()[0].name
        return [self.session.run(None, {name: x})[0][0, 0] for x, _, _, _ in prepared]

    def _boxes(
        self,
        prob,
        shape: tuple[int, int],
        sx: float,
        sy: float,
        live: tuple[int, int] | None = None,
    ) -> list:
        """DB's contour pass over one probability map — the CPU half of detection.

        Split out of :meth:`detect` so :meth:`probs_batch`'s callers can run it
        per image off the session thread: it is all OpenCV and NumPy, and it is
        the larger half of what detection costs once the GPU is doing the
        convolutions.

        ``live`` is the ``(h, w)`` the image actually occupies in the map; the
        rest is :meth:`_preprocess`'s replicated pad, and cropping it away here
        is what keeps the canvas invisible to everything downstream.
        """
        import cv2
        import numpy as np

        if live is not None:
            prob = prob[: live[0], : live[1]]
        bitmap = (prob > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        h, w = shape
        boxes = []
        for contour in contours[: self.max_candidates]:
            points, side = self._mini_box(contour)
            if side < MIN_BOX_SIDE:
                continue
            if self._box_score(prob, points) < self.box_thresh:
                continue
            grown = self._unclip(points)
            if grown is None:
                continue
            points, side = self._mini_box(grown)
            if side < MIN_BOX_SIDE + 2:
                continue
            box = np.array(points, dtype=np.float32)
            box[:, 0] = np.clip(np.round(box[:, 0] * sx), 0, w - 1)
            box[:, 1] = np.clip(np.round(box[:, 1] * sy), 0, h - 1)
            boxes.append(box)
        return boxes


# --------------------------------------------------------------------------
# recognition
# --------------------------------------------------------------------------


@dataclass
class TextRecognizer:
    """CTC recognition over 50 languages: line crops in, strings out."""

    session: Any
    vocab: list[str]
    batch_size: int = 8

    fixed_shape: bool = False
    """Quantize each batch's width to a multiple of :data:`REC_MIN_WIDTH` and pad
    every batch to :attr:`batch_size` rows, so the session sees one input shape
    for the whole run. See :func:`_pad_to`."""

    @classmethod
    def load(
        cls, model_dir: Path | None = None, *, device: str = "cpu", batch_size: int = 8
    ) -> TextRecognizer:
        model_dir = model_dir or default_ppocr_rec_dir()
        chars = list(
            _load_config(model_dir).get("PostProcess", {}).get("character_dict", [])
        )
        session = _session(model_dir / ONNX_NAME, device)
        return cls(
            session=session,
            vocab=cls._vocab(chars, session),
            batch_size=batch_size,
            fixed_shape=_is_gpu(session),
        )

    @staticmethod
    def _vocab(chars: list[str], session) -> list[str]:
        """``['blank'] + chars`` plus the space upstream appends, if the graph wants it.

        Which of the two it is, is read off the model's own output width rather
        than assumed: an off-by-one here does not fail, it silently shifts every
        character by one and returns fluent-looking garbage. The shipped medium
        model answers 18,710 against a 18,708-entry dictionary, i.e. blank *and*
        space.
        """
        classes = session.get_outputs()[0].shape[-1]
        if not isinstance(classes, int):
            return ["<blank>", *chars, " "]
        if classes == len(chars) + 2:
            return ["<blank>", *chars, " "]
        if classes == len(chars) + 1:
            return ["<blank>", *chars]
        raise RuntimeError(
            f"PP-OCR recognizer has {classes} classes but its dictionary has "
            f"{len(chars)} characters — the two files are not a pair"
        )

    def _width(self, max_ratio: float) -> int:
        """The width a batch of this aspect ratio is padded to.

        Upstream's is ``max(320, ceil(48 * max_ratio))`` — one width per batch,
        which on the GPU is one re-plan per batch. Rounding it up to the next
        multiple of :data:`REC_MIN_WIDTH` collapses that to a two- or three-rung
        ladder, and a line long enough to reach the second rung is rare enough
        that the padding costs nothing on average.
        """
        need = max(REC_MIN_WIDTH, math.ceil(REC_HEIGHT * max_ratio))
        if not self.fixed_shape:
            return need
        return REC_MIN_WIDTH * math.ceil(need / REC_MIN_WIDTH)

    def _resize(self, crop, width: int):
        """One crop, height-normalized and right-padded to the batch's width."""
        import cv2
        import numpy as np

        h, w = crop.shape[:2]
        keep = min(width, max(1, math.ceil(REC_HEIGHT * w / max(h, 1))))
        resized = cv2.resize(crop, (keep, REC_HEIGHT)).astype(np.float32)
        resized = (resized.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        out = np.zeros((3, REC_HEIGHT, width), dtype=np.float32)
        out[:, :, :keep] = resized
        return out

    def _decode(self, logits) -> list[tuple[str, float]]:
        """Greedy CTC: argmax, collapse repeats, drop blank, index the vocabulary."""
        import numpy as np

        # One reduction, not two: the winning class and its score come from the
        # same argmax. `logits.max(axis=2)` would sweep an (N, T, 18710) field a
        # second time, and that field is the largest array in the pipeline.
        idx = logits.argmax(axis=2)
        prob = np.take_along_axis(logits, idx[:, :, None], axis=2)[:, :, 0]
        out: list[tuple[str, float]] = []
        for row, scores in zip(idx, prob, strict=True):
            keep = np.ones(len(row), dtype=bool)
            keep[1:] = row[1:] != row[:-1]
            keep &= row != 0
            chars = [self.vocab[i] for i in row[keep]]
            picked = scores[keep]
            out.append(("".join(chars), float(picked.mean()) if picked.size else 0.0))
        return out

    def _batch(self, crops: Sequence, chunk: Sequence[int]):
        """The padded tensor for one aspect-sorted run of crops.

        Its own method so the pool builds all of a chunk's batches at once:
        this is ``cv2.resize`` plus a normalize per crop, which is CPU that has
        no business being serialized behind the thread doing the forwards.
        """
        import numpy as np

        max_ratio = max(
            REC_MIN_WIDTH / REC_HEIGHT,
            max(crops[i].shape[1] / max(crops[i].shape[0], 1) for i in chunk),
        )
        width = self._width(max_ratio)
        rows = [self._resize(crops[i], width) for i in chunk]
        # The last chunk of a run is short, and a batch size that moves is an
        # input shape that moves. Padding it out and dropping the tail after the
        # decode is a few blank crops against a re-plan.
        if self.fixed_shape and len(rows) < self.batch_size:
            rows.extend([np.zeros_like(rows[0])] * (self.batch_size - len(rows)))
        return np.stack(rows)

    def recognize(self, crops: Sequence, pool=None) -> list[tuple[str, float]]:
        """Recognize every crop, batched by aspect ratio so padding stays cheap.

        With a ``pool``, the two CPU halves — building each padded batch, and the
        CTC decode over an 18,710-wide logit field — run on it, leaving this
        thread with nothing but ``session.run``.
        """
        if not crops:
            return []
        order = sorted(
            range(len(crops)),
            key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1),
        )
        chunks = [
            order[s : s + self.batch_size]
            for s in range(0, len(order), self.batch_size)
        ]
        mapper = pool.map if pool is not None else map

        batches = list(mapper(lambda c: self._batch(crops, c), chunks))
        name = self.session.get_inputs()[0].name
        logits = [self.session.run(None, {name: b})[0] for b in batches]
        decoded = list(mapper(self._decode, logits))

        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        for chunk, rows in zip(chunks, decoded, strict=True):
            for i, row in zip(chunk, rows[: len(chunk)], strict=True):
                results[i] = row
        return results


# --------------------------------------------------------------------------
# the pass over one image
# --------------------------------------------------------------------------


def crop_quad(bgr, box):
    """The perspective-corrected strip a quad encloses, uprighted if it is tall.

    A box taller than 1.5× its width is rotated a quarter turn, which is
    upstream's rule and is what makes a rotated *sign* readable. It is also this
    pipeline's known weak point: it turns genuinely **vertical** Japanese — glyphs
    stacked down a column, which manga is full of — into a row of glyphs each
    lying on its side, and the recognizer answers junk at a low score. The score
    floor is what keeps that out of the sidecar; reading it properly needs a
    per-glyph column split, which is not here.
    """
    import cv2
    import numpy as np

    box = np.array(box, dtype=np.float32)
    width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    if width < 1 or height < 1:
        return None
    target = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    crop = cv2.warpPerspective(
        bgr,
        cv2.getPerspectiveTransform(box, target),
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if height / width >= 1.5:
        crop = np.rot90(crop).copy()
    return crop


def reading_order(lines: list[OcrLine]) -> list[OcrLine]:
    """Sort top-to-bottom, then left-to-right within a band.

    The band is half the median line height, so two captions side by side on one
    row read across rather than down — the same reason the position stage sorts
    subjects by row before column, at a scale where the tolerance has to be
    derived from the content rather than fixed.
    """
    if not lines:
        return []
    heights = sorted(max(1, line.height) for line in lines)
    band = max(1, heights[len(heights) // 2] // 2)
    return sorted(lines, key=lambda line: (line.box[1] // band, line.box[0]))


@dataclass
class OcrEngine:
    """Detector + recognizer as the one callable a stage needs."""

    detector: TextDetector
    recognizer: TextRecognizer
    min_score: float = 0.6
    min_box_px: int = 12
    max_boxes: int = 64
    chunk_size: int = 32
    """How many images :meth:`read_many` holds in flight at once. It buys the
    recognizer every chunk's crops in one pool of lines instead of one or two at
    a time, at the price of holding that many decoded images — and, while the
    next chunk prefetches, two chunks' worth — in RAM."""

    workers: int = 4
    """Threads for the decode / normalize / contour work around the two
    sessions. Every one of those is OpenCV or NumPy and releases the GIL, so
    they genuinely overlap — with each other and with a ``session.run`` on
    another thread, which is what keeps the GPU fed. Four, because OpenCV's own
    threading is off inside them (:func:`_cv2_single_threaded`) and past four
    the run is waiting on the detector rather than on a core."""

    def read(self, image_path: Path) -> list[OcrLine]:
        """Every line of text in one image, filtered, in reading order."""
        return self.read_many([image_path])[0]

    def read_many(self, image_paths: Sequence[Path]) -> list[list[OcrLine]]:
        """:meth:`read` over many images, batching what is batchable.

        Per image the work splits three ways, and one image is the wrong unit
        for all three: decoding and DB post-processing are CPU that should
        overlap the GPU, detection is one small forward whose cost is dwarfed by
        re-planning if its input shape moves (:func:`_pad_to`), and recognition
        is typically **one or two** line crops — a forward whose fixed cost
        dwarfs its content unless the crops of thirty images queue up together.

        So a chunk is decoded and normalized on the pool, detected one image per
        forward, post-processed on the pool, and then *all* of its crops are
        recognized as one pool of lines. Results are scattered back by index, so
        this returns exactly what a ``read``-per-path loop would have.

        The decode of chunk *n+1* is launched before chunk *n* reaches the
        sessions, which is what finally makes the overlap real: it is the one
        piece of per-image CPU with no dependency on the forward in front of it,
        and it is about as expensive as that forward. It is submitted to a
        thread of its own rather than to ``pool`` — a pool task that waits on the
        same pool deadlocks at ``workers=1``.
        """
        from concurrent.futures import ThreadPoolExecutor

        paths = list(image_paths)
        if not paths:
            return []
        size = max(1, self.chunk_size)
        chunks = [paths[s : s + size] for s in range(0, len(paths), size)]

        out: list[list[OcrLine]] = []
        with (
            _cv2_single_threaded(),
            ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool,
            ThreadPoolExecutor(max_workers=1) as ahead,
        ):
            pending = ahead.submit(self._load_chunk, chunks[0], pool)
            for i, chunk in enumerate(chunks):
                loaded = pending.result()
                if i + 1 < len(chunks):
                    pending = ahead.submit(self._load_chunk, chunks[i + 1], pool)
                out.extend(self._read_chunk(chunk, loaded, pool))
        return out

    def _load_chunk(self, paths: Sequence[Path], pool) -> list:
        """One chunk decoded and normalized, on the pool. The prefetched half."""
        return list(pool.map(self._load, paths))

    def _read_chunk(
        self, paths: Sequence[Path], loaded: list, pool
    ) -> list[list[OcrLine]]:
        # Only the two `session.run` loops below stay on this thread. The DB
        # contour pass and the cropping go to the pool, the decode came off it
        # a chunk ago — they are the bulk of the work, they are OpenCV and
        # NumPy, and leaving them here is what leaves the GPU idle behind one
        # saturated core.
        live = [i for i, item in enumerate(loaded) if item is not None]
        prepared = [loaded[i][1] for i in live]

        probs = self.detector.probs_batch(prepared)
        cropped = list(
            pool.map(
                lambda args: self._crops(*args),
                [
                    (loaded[i][0], p, m)
                    for i, p, m in zip(live, probs, prepared, strict=True)
                ],
            )
        )

        # One flat pool of crops for the whole chunk, and the map back: `owner`
        # says which image each crop came from, so one recognizer call serves
        # every image at once and the strings still land where they belong.
        crops: list = []
        owner: list[int] = []
        kept: list[list] = [[] for _ in paths]
        for i, (boxes, image_crops) in zip(live, cropped, strict=True):
            kept[i] = boxes
            crops.extend(image_crops)
            owner.extend([i] * len(image_crops))

        recognized = self.recognizer.recognize(crops, pool=pool)
        per_image: list[list] = [[] for _ in paths]
        for i, result in zip(owner, recognized, strict=True):
            per_image[i].append(result)
        return [self._lines(kept[i], per_image[i]) for i in range(len(paths))]

    def _load(self, image_path: Path):
        """Decode one image and normalize it for the detector, or ``None``.

        Both halves here so the thread pool does them in one hop, and because
        what the batched detector needs is the tensor rather than the image.
        """
        import cv2
        import numpy as np

        data = np.fromfile(str(image_path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return bgr, self.detector._preprocess(bgr)

    def _crops(self, bgr, prob, prepared) -> tuple[list, list]:
        """One image's kept boxes and their crops, from its probability map.

        The whole CPU tail of detection in one call — DB's contour pass, the
        size filters, and the perspective crops — so the pool runs it per image
        while the session is busy with the next batch.
        """
        _, sx, sy, live = prepared
        boxes, crops = [], []
        for box in self._select(
            self.detector._boxes(prob, bgr.shape[:2], sx, sy, live)
        ):
            crop = crop_quad(bgr, box)
            if crop is not None:
                boxes.append(box)
                crops.append(crop)
        return boxes, crops

    def _select(self, boxes: Sequence) -> list:
        """The boxes worth recognizing: big enough, and the largest few.

        The filters live here rather than in the caller because two of the three
        save work — a box under ``min_box_px`` never becomes a crop, and
        ``max_boxes`` caps what a screentone misread as a wall of text can cost.
        Only the score floor has to wait for the recognizer.
        """

        def extent(b) -> tuple[float, float]:
            return b[:, 0].max() - b[:, 0].min(), b[:, 1].max() - b[:, 1].min()

        big = [b for b in boxes if max(extent(b)) >= self.min_box_px]
        return sorted(big, key=lambda b: extent(b)[0] * extent(b)[1], reverse=True)[
            : self.max_boxes
        ]

    def _lines(self, kept: Sequence, recognized: Sequence[tuple[str, float]]):
        """One image's surviving lines, numbered in reading order."""
        lines: list[OcrLine] = []
        for box, (text, score) in zip(kept, recognized, strict=True):
            text = text.strip()
            if not text or score < self.min_score:
                continue
            lines.append(
                OcrLine(
                    seq=0,
                    box=(
                        int(box[:, 0].min()),
                        int(box[:, 1].min()),
                        int(box[:, 0].max()),
                        int(box[:, 1].max()),
                    ),
                    score=score,
                    text=text,
                )
            )
        # Numbered only once the order is final, so a sidecar's sequence always
        # reads top-to-bottom rather than recording detection order.
        return [
            OcrLine(seq=i, box=ln.box, score=ln.score, text=ln.text)
            for i, ln in enumerate(reading_order(lines), 1)
        ]


def load_ocr(
    *,
    device: str = "cpu",
    min_score: float = 0.6,
    min_box_px: int = 12,
    max_boxes: int = 64,
    limit_side: int = DET_LIMIT_SIDE,
    batch_size: int = 8,
    chunk_size: int = 32,
    workers: int = 4,
) -> OcrEngine:
    """Both halves, on one device. The one entry point a stage calls."""
    return OcrEngine(
        detector=TextDetector.load(device=device, limit_side=limit_side),
        recognizer=TextRecognizer.load(device=device, batch_size=batch_size),
        min_score=min_score,
        min_box_px=min_box_px,
        max_boxes=max_boxes,
        chunk_size=chunk_size,
        workers=workers,
    )
