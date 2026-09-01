"""PP-OCRv6 detection and recognition: the only place the two ONNX sessions are built.

Both models are fed **BGR**, with the ImageNet mean/std applied in that order (matching
upstream), and the recognizer is normalized ``(x/255 - 0.5) / 0.5`` and padded, never
stretched, to the batch's widest aspect ratio. Both sessions are held at a fixed input
shape because ORT's CUDA provider re-plans on every shape change (:func:`_pad_to`). The
weights paths come from :mod:`anime_tools.downloads` and have no CLI flag. Every heavy
import is deferred into a function, so importing this module stays cheap and torch-free.
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
from anime_tools.ocr._text import join_cjk, keep_line

ONNX_NAME = "inference.onnx"
CONFIG_NAME = "inference.yml"
"""What each Hub repo ships: the graph, and the config describing how to feed it."""

DET_LIMIT_SIDE = 960
"""PaddleOCR's ``DetResizeForTest`` default: the longest side is scaled down to
this before detection, and both sides are then rounded to a multiple of 32."""

REC_HEIGHT = 48
REC_MIN_WIDTH = 320
"""The recognizer's fixed input height, and the narrowest width a batch is padded
to — ``image_shape: [3, 48, 320]`` in its ``inference.yml``."""

MIN_BOX_SIDE = 3
"""DB's own floor on a candidate's short side, before the unclip."""

DET_STRIDE = 32
"""DB's downsampling factor: every side the detector is fed is a multiple of it."""

DET_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
DET_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)
"""The ImageNet statistics the detector was trained on, pre-scaled to the 0-255 the
decoder hands over: ``(v / 255 - m) / s`` is ``(v - 255m) / 255s``, which is one pass
over the pixels instead of three. BGR order, matching upstream and :meth:`_preprocess`.
"""


def _pad_to(limit_side: int) -> int:
    """The square canvas a GPU detector session is fed, for every image.

    ORT's CUDA provider re-plans the graph on every input-shape change, and a bucketed
    resized tree hands it a different ``(H, W)`` per image: measured at 12 ms of forward
    behind 70 ms of re-planning for the detector, 8 ms behind 108 ms for the recognizer.
    So on the GPU both sessions are held at one shape — the detector pads every image onto
    this canvas, the recognizer quantizes width to a multiple of :data:`REC_MIN_WIDTH` and
    pads every batch to :attr:`TextRecognizer.batch_size`. The CPU provider has no such
    penalty and keeps the native shape.
    """
    return max(round(limit_side / DET_STRIDE) * DET_STRIDE, DET_STRIDE)


def _is_gpu(session) -> bool:
    """Whether this session actually landed on an accelerator.

    ``device`` says what was *asked* for; :func:`_session` warns and continues on the CPU
    when the CUDA provider is missing, so only the session knows.
    """
    return any(p != "CPUExecutionProvider" for p in session.get_providers())


@contextmanager
def _cv2_single_threaded() -> Iterator[None]:
    """Take OpenCV's own thread pool away for the duration.

    :meth:`OcrEngine.read_iter` already spreads whole images across a pool; OpenCV
    fanning out inside each one is a second layer of threads over the same cores.
    Restored on the way out — this is a process-wide setting.
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

    Its own class so the CLI can turn it into the one instruction that fixes it.
    """

    def __init__(self, model_dir: Path) -> None:
        super().__init__(
            f"PP-OCRv6 weights not found in {model_dir} — run "
            "`python -m anime_tools.downloads ppocr_det ppocr_rec` "
            "(or ⚙ Settings › Models in the GUI)"
        )


def _preload_cuda_libs(ort: Any) -> None:
    """Put the ``nvidia-*`` wheels' CUDA/cuDNN libraries on the loader path.

    Idempotent. A failed preload leaves the provider list exactly as it was, which the
    caller already warns about. Absent on the CPU wheel and on onnxruntime < 1.21, hence
    the ``getattr``.
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

    The CUDA provider is a separate wheel (``onnxruntime-gpu``), so asking for it where
    only the CPU build is installed warns and continues.

    ``preload_dlls()`` first, and it is not optional: the CUDA and cuDNN libraries the
    provider links against ship as their own ``nvidia-*`` wheels and nothing else puts
    them on the loader path. Without it the provider ``.so`` fails to open and
    ``get_available_providers()`` reports the *CPU* build's list.
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

        Returns the tensor, the two scale factors, and the *live* region inside the
        tensor. The scale factors are kept apart because the 32-rounding makes them
        differ: a box is mapped back through the axis it was found on.

        With :attr:`pad_to` the live region sits in the top-left of a square canvas
        (:func:`_pad_to`). The pad is ``BORDER_REPLICATE`` rather than a constant, so DB
        finds no text box along the border's hard edge; the band is cut off the
        probability map before the contour pass regardless.

        The normalize writes each channel straight into the CHW tensor the session is
        fed, with :data:`DET_MEAN` / :data:`DET_STD` folded into one pass. It is the
        whole CPU cost of an image that has no text in it, and the arithmetic chain it
        replaces — ``astype`` then ``/255`` then two broadcasts then a ``transpose``
        copy — swept an 11 MB array five times over for the same numbers.
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
        x = np.empty((1, 3, *resized.shape[:2]), dtype=np.float32)
        for i, channel in enumerate(cv2.split(resized)):
            np.subtract(channel, DET_MEAN[i], out=x[0, i], dtype=np.float32)
            x[0, i] *= 1.0 / DET_STD[i]
        return x, w / rw, h / rh, (rh, rw)

    @staticmethod
    def _mini_box(contour):
        """``minAreaRect`` as four points ordered TL, TR, BR, BL, plus its short side.

        The ordering is upstream's and is load-bearing twice over: the perspective crop
        maps these four onto a rectangle's corners in this order, and the unclip
        re-derives the rect from them.
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
        """Grow the rectangle by DB's offset distance.

        For a rectangle the polygon offset is exactly a rectangle whose sides each moved
        out by ``d``, so it is computed directly rather than through a clipper.
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

        The one method that touches the session, so it must run on the calling thread;
        returning raw maps rather than boxes lets the caller hand the normalize before and
        the contour pass after to a pool. ``prepared`` is what :meth:`_preprocess`
        returned for each image.

        One image per forward: the detector's activations are full-resolution, so a batch
        of four is four times the work (64 ms against 4x15 ms) and would put a second
        moving axis into the input shape (:func:`_pad_to`).
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

        Split out of :meth:`detect` so :meth:`probs_batch`'s callers can run it per image
        off the session thread. ``live`` is the ``(h, w)`` the image actually occupies in
        the map; the rest is :meth:`_preprocess`'s replicated pad, cropped away here.
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

        Read off the model's own output width rather than assumed: an off-by-one here
        does not fail, it silently shifts every character by one. The shipped medium model
        answers 18,710 against a 18,708-entry dictionary, i.e. blank *and* space.
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

    @staticmethod
    def _need(ratio: float) -> int:
        """The narrowest width a crop of this aspect ratio fits in, upstream's
        ``max(320, ceil(48 * ratio))``."""
        return max(REC_MIN_WIDTH, math.ceil(REC_HEIGHT * ratio))

    def _width(self, max_ratio: float) -> int:
        """The width a batch of this aspect ratio is padded to.

        Upstream's is one width per batch, which on the GPU is one re-plan per batch.
        Rounding up to the next multiple of :data:`REC_MIN_WIDTH` collapses that to a
        two- or three-rung ladder.
        """
        need = self._need(max_ratio)
        if not self.fixed_shape:
            return need
        return REC_MIN_WIDTH * math.ceil(need / REC_MIN_WIDTH)

    @classmethod
    def _rung(cls, crop) -> int:
        """The width this crop would be padded to on its own — its batching key.

        **Padding a crop past the width it needs costs accuracy**, not just time: the
        recognizer reads the zeros. Measured on one line, the same crop answers
        ``スリスリ`` at 0.92 padded to 320 and ``ス=入ー`` at 0.37 padded to 1280. So a
        batch is drawn from one rung and every member gets the width it asked for; what
        a crop reads no longer depends on which crops it was queued beside.
        """
        ratio = crop.shape[1] / max(crop.shape[0], 1)
        return REC_MIN_WIDTH * math.ceil(cls._need(ratio) / REC_MIN_WIDTH)

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

        # One reduction, not two: the winning class and its score come from the same
        # argmax. `logits.max(axis=2)` would sweep the (N, T, 18710) field a second time.
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

        Its own method so the pool builds all of a chunk's batches at once — this is
        ``cv2.resize`` plus a normalize per crop, and it should not serialize behind the
        thread doing the forwards.
        """
        import numpy as np

        max_ratio = max(
            REC_MIN_WIDTH / REC_HEIGHT,
            max(crops[i].shape[1] / max(crops[i].shape[0], 1) for i in chunk),
        )
        width = self._width(max_ratio)
        rows = [self._resize(crops[i], width) for i in chunk]
        # A batch size that moves is an input shape that moves, so the short last chunk
        # is padded out and its tail dropped after the decode.
        if self.fixed_shape and len(rows) < self.batch_size:
            rows.extend([np.zeros_like(rows[0])] * (self.batch_size - len(rows)))
        return np.stack(rows)

    def recognize(self, crops: Sequence, pool=None) -> list[tuple[str, float]]:
        """Recognize every crop, batched by width rung so nothing is over-padded.

        A batch holds one rung only (:meth:`_rung`), sorted by ratio inside it so the
        CPU path — which pads to the batch's exact need rather than to the rung — stays
        tight as well. Batching the aspect-sorted list by position instead would let a
        narrow crop ride along at a wide neighbour's width, and the pad is not free:
        it changes what the crop says.

        With a ``pool``, the two CPU halves — building each padded batch, and the CTC
        decode over an 18,710-wide logit field — run on it, leaving this thread with
        nothing but ``session.run``.
        """
        if not crops:
            return []
        rungs: dict[int, list[int]] = {}
        for i, crop in enumerate(crops):
            rungs.setdefault(self._rung(crop), []).append(i)
        for group in rungs.values():
            group.sort(key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1))
        chunks = [
            group[s : s + self.batch_size]
            for _, group in sorted(rungs.items())
            for s in range(0, len(group), self.batch_size)
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

    A box taller than 1.5× its width is rotated a quarter turn (upstream's rule). Known
    weak point: genuinely **vertical** Japanese becomes a row of glyphs each lying on its
    side, and the recognizer answers junk at a low score — the score floor is what keeps
    that out of the sidecar. What survives arrives one column per box, which is what
    :func:`~anime_tools.ocr._text.join_cjk` puts back together.
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

    The band is half the median line height, so two captions side by side on one row read
    across rather than down.
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

    min_chars: int = 3
    skip_en: bool = True
    join_cjk: bool = True
    """The content filters, applied to what came back rather than to what was
    fed in (:mod:`anime_tools.ocr._text`): join a balloon's boxes into one line,
    drop the ASCII-only ones, then drop what is left under ``min_chars``."""

    chunk_size: int = 32
    """How many images :meth:`read_iter` holds in flight at once: the recognizer gets
    every chunk's crops in one pool of lines, at the price of holding that many decoded
    images — two chunks' worth while the next prefetches — in RAM."""

    workers: int = 4
    """Threads for the decode / normalize / contour work around the two sessions. All of
    it is OpenCV or NumPy and releases the GIL, so it overlaps a ``session.run`` on
    another thread. Four, because OpenCV's own threading is off inside them
    (:func:`_cv2_single_threaded`) and past four the run waits on the detector."""

    def read(self, image_path: Path) -> list[OcrLine]:
        """Every line of text in one image, filtered, in reading order."""
        return self.read_many([image_path])[0]

    def read_many(self, image_paths: Sequence[Path]) -> list[list[OcrLine]]:
        """:meth:`read_iter` drained into a list, one entry per path."""
        return list(self.read_iter(image_paths))

    def read_iter(self, image_paths: Sequence[Path]) -> Iterator[list[OcrLine]]:
        """:meth:`read` over many images, batching what is batchable.

        A chunk is decoded and normalized on the pool, detected one image per forward,
        post-processed on the pool, and then *all* of its crops are recognized as one pool
        of lines — recognition is typically one or two crops per image, and its fixed cost
        dwarfs its content unless a whole chunk's crops queue up together. Results are
        scattered back by index, so this yields what a ``read``-per-path loop would have,
        in the order the paths arrived.

        Chunk *n+1* is decoded before chunk *n* reaches the sessions, on a thread of its
        own rather than on ``pool`` — a pool task that waits on the same pool deadlocks at
        ``workers=1``.

        **Yielding image by image is what keeps that prefetch alive.** The caller writes
        and reports as each result lands, so it has no reason to hand the run over in
        slices — and a slice the size of a chunk is a chunk with nothing decoded behind
        it, which parks the GPU for a whole chunk's decode. The whole run goes in one
        call; only the chunking below is a batch boundary.
        """
        from concurrent.futures import ThreadPoolExecutor

        paths = list(image_paths)
        if not paths:
            return
        size = max(1, self.chunk_size)
        chunks = [paths[s : s + size] for s in range(0, len(paths), size)]

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
                yield from self._read_chunk(chunk, loaded, pool)

    def _load_chunk(self, paths: Sequence[Path], pool) -> list:
        """One chunk decoded and normalized, on the pool. The prefetched half."""
        return list(pool.map(self._load, paths))

    def _read_chunk(
        self, paths: Sequence[Path], loaded: list, pool
    ) -> list[list[OcrLine]]:
        # Only the two `session.run` loops below stay on this thread; the DB contour pass
        # and the cropping go to the pool, and the decode came off it a chunk ago.
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

        # One flat pool of crops for the whole chunk: `owner` says which image each crop
        # came from, so one recognizer call serves every image and the strings still land
        # where they belong.
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

        Both halves here so the thread pool does them in one hop.
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

        The whole CPU tail of detection in one call — DB's contour pass, the size filters,
        and the perspective crops — so the pool runs it per image while the session is
        busy with the next batch.
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

        Two of the three filters save work — a box under ``min_box_px`` never becomes a
        crop, and ``max_boxes`` caps what a screentone misread as a wall of text costs.
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
        # Join before filtering: a column of two glyphs is only short until the
        # rest of its balloon is on it, and a merged block sits where neither
        # part did, so reading order is settled afterwards.
        if self.join_cjk:
            lines = join_cjk(lines)
        lines = [
            ln
            for ln in lines
            if keep_line(ln.text, min_chars=self.min_chars, skip_en=self.skip_en)
        ]
        # Numbered only once the order is final, so a sidecar's sequence reads
        # top-to-bottom rather than recording detection order.
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
    min_chars: int = 3,
    skip_en: bool = True,
    join_cjk: bool = True,
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
        min_chars=min_chars,
        skip_en=skip_en,
        join_cjk=join_cjk,
        chunk_size=chunk_size,
        workers=workers,
    )
