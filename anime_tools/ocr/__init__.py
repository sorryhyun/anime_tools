"""PP-OCRv6 text recognition, as a curation-side reader of what an image says.

Two halves behind one call: :func:`load_ocr` builds the detector and the
recognizer, and :meth:`OcrEngine.read` turns an image path into
:class:`~anime_tools.captions.ocr_sidecar.OcrLine` records in reading order.
The weights are ONNX and the runtime is ``onnxruntime`` — see
:mod:`anime_tools.ocr._onnx` for why that is not PaddlePaddle.

A run answers *what does the picture say*, and only that. It does not answer
what language it says it in: PP-OCRv6 is one model over fifty languages and
returns a string, so a language could only be guessed back off the characters,
and a guess drawn from a two-character fragment is confident and worthless. The
recognized string is what goes in the sidecar, for a person to read.
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
