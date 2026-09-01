"""PP-OCRv6 text recognition, as a curation-side reader of what an image says.

Two halves behind one call: :func:`load_ocr` builds the detector and the
recognizer, and :meth:`OcrEngine.read` turns an image path into
:class:`~anime_tools.captions.ocr_sidecar.OcrLine` records in reading order.
The weights are ONNX and the runtime is ``onnxruntime`` — see
:mod:`anime_tools.ocr._onnx` for why that is not PaddlePaddle.

:mod:`anime_tools.ocr.script` is the torch-free, weights-free half: which
language a recognized line is in, and which Danbooru tag (if any) a caption may
say so with. It is importable on its own, which is what lets the caption side
and the tests reason about the tag map without a 139 MB download.
"""

from anime_tools.ocr._onnx import (
    OcrEngine,
    OcrWeightsMissing,
    TextDetector,
    TextRecognizer,
    load_ocr,
    reading_order,
)
from anime_tools.ocr.script import LANGS, parse_langs, script_of, tags_for

__all__ = [
    "LANGS",
    "OcrEngine",
    "OcrWeightsMissing",
    "TextDetector",
    "TextRecognizer",
    "load_ocr",
    "parse_langs",
    "reading_order",
    "script_of",
    "tags_for",
]
