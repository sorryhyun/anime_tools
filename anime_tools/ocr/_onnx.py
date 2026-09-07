"""The OCR engine: the detector's pass over a tree of images, on onnxruntime.

:class:`OcrEngine` runs *a* detector through a small protocol (:class:`Detector`)
— the AnimeText text-block detector (:mod:`anime_tools.ocr.animetext`), the one
the package ships — and is **detect-only**: every box the detector keeps becomes
a line with no text, in reading order, for the re-reader
(:mod:`anime_tools.ocr.reread`, the manga VL reader) to fill. The recognition
half that used to live here (a CTC line recognizer and its DB line detector)
was retired 2026-09-07; the module keeps its name because the engine is still
the package's ONNX session over a page.

What the engine owns is the batching: a chunk of images decoded and letterboxed
on a thread pool, detected one image per forward on the calling thread, the
boxes settled on the pool again, with the next chunk decoding meanwhile. Every
heavy import is deferred into a function, so importing this module stays cheap
and torch-free.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from anime_tools.captions.ocr_sidecar import OcrLine
from anime_tools.ocr._text import reading_order


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


NO_TEXT_SCORE = 0.0
"""The ``score`` of a detect-only line: no recognizer stood behind it. Its
``det`` is the detector's own confidence in the box."""


def _bounds(quad) -> tuple[int, int, int, int]:
    """The axis-aligned ``(x0, y0, x1, y1)`` of a ``(4, 2)`` quad."""
    return (
        int(quad[:, 0].min()),
        int(quad[:, 1].min()),
        int(quad[:, 0].max()),
        int(quad[:, 1].max()),
    )


def crop_quad(bgr, box):
    """The perspective-corrected strip a quad encloses, uprighted if it is tall.

    A box taller than 1.5× its width is rotated a quarter turn. Not on the
    engine's own path — the VL reader cuts its crops from the axis-aligned box
    with its own padding (:meth:`~anime_tools.ocr.sfx.SfxReader.read_boxes`) —
    but the one helper for a caller holding a :class:`Detector`'s quads.
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


class Detector(Protocol):
    """What :class:`OcrEngine` asks of a detector, split the way its threads are.

    :meth:`prepare` and :meth:`boxes` are pure CPU and run on the pool;
    :meth:`forward_batch` is the one call that touches the session and runs on
    the calling thread. ``boxes`` answers ``(quad, score)`` pairs: a ``(4, 2)``
    float quad in image pixels, TL-TR-BR-BL — the shape :func:`crop_quad` and
    the size filters take — and the detector's confidence in it, which the line
    keeps as ``det``.
    """

    def prepare(self, bgr) -> Any:
        """One decoded image → what :meth:`forward_batch` is fed, plus whatever
        :meth:`boxes` needs to map the result back."""

    def forward_batch(self, prepared: Sequence[Any]) -> list:
        """The raw head per prepared image, forward passes only."""

    def boxes(self, raw, prepared: Any, shape: tuple[int, int]) -> list:
        """One image's ``(quad, score)`` pairs from its raw head; ``shape`` is
        its ``(h, w)``."""


@dataclass
class OcrEngine:
    """A detector as the one callable a stage needs — detect-only.

    Every box the detector keeps is a line with empty text, :data:`NO_TEXT_SCORE`
    and the detector's confidence as ``det``, in reading order; there is no text
    to filter on, so
    the content floors belong to the re-reader behind it
    (:class:`~anime_tools.ocr.reread.RereadEngine`). Not a stage on its own.
    """

    detector: Detector
    min_box_px: int = 12
    max_boxes: int = 64

    chunk_size: int = 32
    """How many images :meth:`read_iter` holds in flight at once, at the price of
    holding that many decoded images — two chunks' worth while the next
    prefetches — in RAM."""

    workers: int = 4
    """Threads for the decode / letterbox / box work around the session. All of
    it is OpenCV or NumPy and releases the GIL, so it overlaps a ``session.run``
    on another thread. Four, because OpenCV's own threading is off inside them
    (:func:`_cv2_single_threaded`) and past four the run waits on the detector."""

    def read(self, image_path: Path) -> list[OcrLine]:
        """Every text box in one image, as empty lines in reading order."""
        return self.read_many([image_path])[0]

    def read_many(self, image_paths: Sequence[Path]) -> list[list[OcrLine]]:
        """:meth:`read_iter` drained into a list, one entry per path."""
        return list(self.read_iter(image_paths))

    def read_iter(self, image_paths: Sequence[Path]) -> Iterator[list[OcrLine]]:
        """:meth:`read` over many images, batching what is batchable.

        A chunk is decoded and prepared on the pool, detected one image per
        forward, and post-processed on the pool. Results are scattered back by
        index, so this yields what a ``read``-per-path loop would have, in the
        order the paths arrived.

        Chunk *n+1* is decoded before chunk *n* reaches the session, on a thread
        of its own rather than on ``pool`` — a pool task that waits on the same
        pool deadlocks at ``workers=1``.

        **Yielding image by image is what keeps that prefetch alive.** The caller
        writes and reports as each result lands, so it has no reason to hand the
        run over in slices — and a slice the size of a chunk is a chunk with
        nothing decoded behind it, which parks the GPU for a whole chunk's
        decode. The whole run goes in one call; only the chunking below is a
        batch boundary.
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
        """One chunk decoded and prepared, on the pool. The prefetched half."""
        return list(pool.map(self._load, paths))

    def _read_chunk(
        self, paths: Sequence[Path], loaded: list, pool
    ) -> list[list[OcrLine]]:
        # Only the `session.run` loop stays on this thread; the detector's box
        # pass goes to the pool, and the decode came off it a chunk ago.
        live = [i for i, item in enumerate(loaded) if item is not None]
        prepared = [loaded[i][1] for i in live]

        raws = self.detector.forward_batch(prepared)
        kept: list[list] = [[] for _ in paths]
        for i, boxes in zip(
            live,
            pool.map(
                lambda args: self._select(self.detector.boxes(*args)),
                [
                    (raw, m, loaded[i][0].shape[:2])
                    for i, raw, m in zip(live, raws, prepared, strict=True)
                ],
            ),
            strict=True,
        ):
            kept[i] = boxes
        return [self._lines(kept[i]) for i in range(len(paths))]

    def _load(self, image_path: Path):
        """Decode one image and prepare it for the detector, or ``None``.

        Both halves here so the thread pool does them in one hop.
        """
        import cv2
        import numpy as np

        data = np.fromfile(str(image_path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return bgr, self.detector.prepare(bgr)

    def _select(self, boxes: Sequence) -> list:
        """The ``(quad, score)`` pairs worth reading: big enough, and the largest
        few.

        A box under ``min_box_px`` never becomes a crop, and ``max_boxes`` caps
        what a screentone misread as a wall of text costs the reader.
        """

        def extent(pair) -> tuple[float, float]:
            q = pair[0]
            return q[:, 0].max() - q[:, 0].min(), q[:, 1].max() - q[:, 1].min()

        big = [p for p in boxes if max(extent(p)) >= self.min_box_px]
        return sorted(big, key=lambda p: extent(p)[0] * extent(p)[1], reverse=True)[
            : self.max_boxes
        ]

    def _lines(self, kept: Sequence) -> list[OcrLine]:
        """One image's ``(quad, score)`` pairs as empty lines, numbered in
        reading order.

        Numbered only once the order is final, so a sidecar's sequence reads
        top-to-bottom rather than recording detection order.
        """
        empty = [
            OcrLine(seq=0, box=_bounds(q), score=NO_TEXT_SCORE, text="", det=float(s))
            for q, s in kept
        ]
        return [
            OcrLine(seq=i, box=ln.box, score=ln.score, text=ln.text, det=ln.det)
            for i, ln in enumerate(reading_order(empty), 1)
        ]


def load_ocr(
    *,
    device: str = "cpu",
    min_box_px: int = 12,
    max_boxes: int = 64,
    chunk_size: int = 64,
    workers: int = 4,
    det_conf: float | None = None,
    det_nest: str = "inner",
) -> OcrEngine:
    """The detect-only engine over the AnimeText detector, on one device. The
    one entry point a stage calls.

    ``det_conf`` / ``det_nest`` are the detector's score floor and nesting
    policy (:mod:`anime_tools.ocr.animetext`); the weights are fetched on first
    use.
    """
    from anime_tools.ocr.animetext import DEFAULT_CONF, AnimeTextDetector

    det: Detector = AnimeTextDetector.load(
        device=device,
        conf=DEFAULT_CONF if det_conf is None else det_conf,
        nest=det_nest,
    )
    return OcrEngine(
        detector=det,
        min_box_px=min_box_px,
        max_boxes=max_boxes,
        chunk_size=chunk_size,
        workers=workers,
    )
