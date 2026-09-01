"""PP-OCRv6 text recognition, as a curation-side reader of what an image says.

:func:`load_ocr` builds the detector and the recognizer; :meth:`OcrEngine.read` turns an
image path into :class:`~anime_tools.captions.ocr_sidecar.OcrLine` records in reading
order. A run answers only *what does the picture say*, never what language: PP-OCRv6 is
one model over fifty languages and returns a bare string, so what a line is written in
is read off the string itself in :mod:`anime_tools.ocr._text`, which also joins a
balloon's columns and applies the ``min_chars`` / ``skip_en`` floors.
"""

from anime_tools.ocr._onnx import (
    OcrEngine,
    OcrWeightsMissing,
    TextDetector,
    TextRecognizer,
    load_ocr,
    reading_order,
)

__all__ = [
    "OcrEngine",
    "OcrWeightsMissing",
    "TextDetector",
    "TextRecognizer",
    "load_ocr",
    "reading_order",
]
