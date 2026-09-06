"""PP-OCRv6 text recognition, as a curation-side reader of what an image says.

:func:`load_ocr` builds the detector and the recognizer; :meth:`OcrEngine.read` turns an
image path into :class:`~anime_tools.captions.ocr_sidecar.OcrLine` records in reading
order. A run answers only *what does the picture say*, never what language: PP-OCRv6 is
one model over fifty languages and returns a bare string, so what a line is written in
is read off the string itself in :mod:`anime_tools.ocr._text`, which also joins a
balloon's columns and applies the ``min_chars`` / ``skip_en`` floors.

The AnimeText text-block detector (:mod:`anime_tools.ocr.animetext`,
``AnimeTextDetector``; ``load_ocr(detector="animetext")``) stands in for the DB
head: one detector for balloon lines and the SFX drawn onto the artwork, weights
fetched on first use under a licence this package does not carry.

The manga SFX reader — a fine-tuned PaddleOCR-VL-1.6 that reads the hand-lettered
onomatopoeia PP-OCRv6 garbles — is :mod:`anime_tools.ocr.sfx` (``SfxReader``), imported
explicitly because it pulls torch; it reads crops another detector boxed, never pages.
:mod:`anime_tools.ocr.reread` is the seam: the OCR stage's ``--reader vl`` runs every
PP-OCRv6 line and the text mask's uncovered components through it.
"""

# `resolve_onnx_device` is not OCR's own: the tagger's exported backbone asks the
# same question. It lives in `anime_tools._onnx` and is re-exported here, where
# every caller already looks for it.
from anime_tools._onnx import resolve_onnx_device
from anime_tools.ocr._onnx import (
    DETECTORS,
    Detector,
    OcrEngine,
    OcrWeightsMissing,
    TextDetector,
    TextRecognizer,
    load_ocr,
    reading_order,
)
from anime_tools.ocr.animetext import AnimeTextDetector, AnimeTextWeightsMissing

__all__ = [
    "DETECTORS",
    "AnimeTextDetector",
    "AnimeTextWeightsMissing",
    "Detector",
    "OcrEngine",
    "OcrWeightsMissing",
    "TextDetector",
    "TextRecognizer",
    "load_ocr",
    "reading_order",
    "resolve_onnx_device",
]
