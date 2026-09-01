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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anime_tools.captions.ocr_sidecar import OcrLine
from anime_tools.downloads import default_ppocr_det_dir, default_ppocr_rec_dir
from anime_tools.ocr.script import script_of

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
    except Exception:  # noqa: BLE001 - a failed preload is just "no GPU"
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
        return cls(
            session=_session(model_dir / ONNX_NAME, device),
            thresh=float(post.get("thresh", 0.2)),
            box_thresh=float(post.get("box_thresh", 0.45)),
            unclip_ratio=float(post.get("unclip_ratio", 1.4)),
            max_candidates=int(post.get("max_candidates", 3000)),
            limit_side=int(limit_side),
        )

    def _preprocess(self, bgr):
        """Scale the longest side under the limit, round both to a multiple of 32.

        Returns the tensor and the two scale factors, which are kept apart
        because the 32-rounding makes them differ: a box is mapped back through
        the axis it was found on, not through one average ratio.
        """
        import cv2
        import numpy as np

        h, w = bgr.shape[:2]
        ratio = min(1.0, self.limit_side / max(h, w))
        rh = max(round(h * ratio / 32) * 32, 32)
        rw = max(round(w * ratio / 32) * 32, 32)
        resized = cv2.resize(bgr, (rw, rh))
        x = resized.astype(np.float32) / 255.0
        x -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        x /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return x.transpose(2, 0, 1)[None], w / rw, h / rh

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
        import cv2
        import numpy as np

        x, sx, sy = self._preprocess(bgr)
        prob = self.session.run(None, {self.session.get_inputs()[0].name: x})[0][0, 0]
        bitmap = (prob > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        h, w = bgr.shape[:2]
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
            session=session, vocab=cls._vocab(chars, session), batch_size=batch_size
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

    def _resize(self, crop, max_ratio: float):
        """One crop, height-normalized and right-padded to the batch's width."""
        import cv2
        import numpy as np

        width = max(REC_MIN_WIDTH, math.ceil(REC_HEIGHT * max_ratio))
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

        idx = logits.argmax(axis=2)
        prob = logits.max(axis=2)
        out: list[tuple[str, float]] = []
        for row, scores in zip(idx, prob, strict=True):
            keep = np.ones(len(row), dtype=bool)
            keep[1:] = row[1:] != row[:-1]
            keep &= row != 0
            chars = [self.vocab[i] for i in row[keep]]
            picked = scores[keep]
            out.append(("".join(chars), float(picked.mean()) if picked.size else 0.0))
        return out

    def recognize(self, crops: Sequence) -> list[tuple[str, float]]:
        """Recognize every crop, batched by aspect ratio so padding stays cheap."""
        import numpy as np

        if not crops:
            return []
        order = sorted(
            range(len(crops)),
            key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1),
        )
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        name = self.session.get_inputs()[0].name
        for start in range(0, len(order), self.batch_size):
            chunk = order[start : start + self.batch_size]
            max_ratio = max(
                REC_MIN_WIDTH / REC_HEIGHT,
                max(crops[i].shape[1] / max(crops[i].shape[0], 1) for i in chunk),
            )
            batch = np.stack([self._resize(crops[i], max_ratio) for i in chunk])
            logits = self.session.run(None, {name: batch})[0]
            for i, decoded in zip(chunk, self._decode(logits), strict=True):
                results[i] = decoded
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

    def read(self, image_path: Path) -> list[OcrLine]:
        """Every line of text in one image, filtered, in reading order.

        The filters are applied here rather than by the caller because two of the
        three save work: a box under ``min_box_px`` never becomes a crop, and
        ``max_boxes`` caps how many crops a screentone misread as a wall of text
        can cost. Only the score floor has to wait for the recognizer.
        """
        import cv2
        import numpy as np

        data = np.fromfile(str(image_path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return []

        boxes = []
        for box in self.detector.detect(bgr):
            w = box[:, 0].max() - box[:, 0].min()
            h = box[:, 1].max() - box[:, 1].min()
            if max(w, h) < self.min_box_px:
                continue
            boxes.append(box)
        boxes = sorted(
            boxes,
            key=lambda b: (
                (b[:, 0].max() - b[:, 0].min()) * (b[:, 1].max() - b[:, 1].min())
            ),
            reverse=True,
        )[: self.max_boxes]

        crops, kept = [], []
        for box in boxes:
            crop = crop_quad(bgr, box)
            if crop is not None:
                crops.append(crop)
                kept.append(box)

        lines: list[OcrLine] = []
        for box, (text, score) in zip(
            kept, self.recognizer.recognize(crops), strict=True
        ):
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
                    lang=script_of(text),
                    score=score,
                    text=text,
                )
            )
        # Numbered only once the order is final, so a sidecar's sequence always
        # reads top-to-bottom rather than recording detection order.
        return [
            OcrLine(seq=i, box=ln.box, lang=ln.lang, score=ln.score, text=ln.text)
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
) -> OcrEngine:
    """Both halves, on one device. The one entry point a stage calls."""
    return OcrEngine(
        detector=TextDetector.load(device=device, limit_side=limit_side),
        recognizer=TextRecognizer.load(device=device, batch_size=batch_size),
        min_score=min_score,
        min_box_px=min_box_px,
        max_boxes=max_boxes,
    )
