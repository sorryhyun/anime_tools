"""OCR, as a curation-side reader of what an image says.

Two models, one seam. :func:`load_ocr` builds the detect-only
:class:`OcrEngine` over the AnimeText text-block detector
(:mod:`anime_tools.ocr.animetext`, ``AnimeTextDetector``; YOLO12 on
torch through :mod:`anime_tools.vision.yolo12`, weights fetched on first use
under a licence this package does not carry): one detector for balloon lines and the SFX drawn onto the artwork,
answering every box as an empty :class:`~anime_tools.captions.ocr_sidecar.OcrLine`
in reading order. The manga VL reader — a fine-tuned PaddleOCR-VL-1.6 that reads
hand-lettered onomatopoeia, hearts and small kana natively — is
:mod:`anime_tools.ocr.sfx` (``SfxReader``), imported explicitly because it pulls
torch; it reads crops another detector boxed, never pages.
:mod:`anime_tools.ocr.reread` is the seam: :class:`~anime_tools.ocr.reread.RereadEngine`
runs every box (and, with a text mask, the mask's uncovered components) through
the reader, and what the OCR stage writes is what came back.

A run answers only *what does the picture say*, never what language: the reader
returns a bare string, and the content floors — ``min_chars`` / ``skip_en`` —
are read off the string itself in :mod:`anime_tools.ocr._text`.
"""

from anime_tools.ocr.animetext import AnimeTextDetector, AnimeTextWeightsMissing
from anime_tools.ocr.engine import Detector, OcrEngine, load_ocr, reading_order

__all__ = [
    "AnimeTextDetector",
    "AnimeTextWeightsMissing",
    "Detector",
    "OcrEngine",
    "load_ocr",
    "reading_order",
]
