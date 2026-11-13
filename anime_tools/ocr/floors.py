"""Which OCR lines a MOSS i2i caption boxes: the floors the colorize and swap_text
LoRAs trained on (anima_lora ``library/moss/colorize.py``, ``i2i_tasks/swap_text``),
shared with ComfyUI-MOSS-Toolkit's caption helper so inference boxes the same lines.

The OCR stage's floors (``LINE_MIN_DET`` 0.6, glyph ``DEFAULT_MIN_GLYPH`` 16) are
too tight for scanlation lettering: the detector scores horizontal English balloons
0.3–0.55 while the reader reads them right. :func:`readable` keeps a box down to the
detector's floor when its read scores :data:`LOW_DET_SCORE`, and drops an all-ASCII
line's glyph floor to :data:`LATIN_MIN_GLYPH`. :func:`keep_line` adds two rescues on
top: a Latin line read at :data:`LATIN_MIN_SCORE` whatever its det, and a Hangul line
``readable`` lost to its glyph floor alone (small balloon lettering sits at 12–16 px
and reads at 0.96–1.00).

A line is anything with ``text``, ``det``, ``score`` and ``glyph_px``
(:class:`~anime_tools.captions.ocr_sidecar.OcrLine`). Torch-free;
:func:`load_reread_engine` imports the models when called.
"""

from __future__ import annotations

import re
from typing import Any

from anime_tools.captions.ocr_sidecar import DEFAULT_MIN_GLYPH
from anime_tools.ocr.reread import LINE_MIN_DET
from anime_tools.ocr.reader import is_latin

LOW_DET_SCORE = 0.9
"""Reader score a box under ``LINE_MIN_DET`` needs."""
LATIN_MIN_GLYPH = 12.0
"""Glyph floor of an all-ASCII line."""
LATIN_MIN_SCORE = 0.8
""":func:`keep_line`: a Latin line's reader score, whatever its det."""
HANGUL = re.compile(r"[\uac00-\ud7a3]")
JA_SCRIPT = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]"
)


def readable(ln: Any) -> bool:
    """A read with text, a det at ``LINE_MIN_DET`` or a score at
    :data:`LOW_DET_SCORE`, and a glyph at its script's floor."""
    latin = all(ch.isascii() for ch in ln.text if ch.isalpha())
    floor = LATIN_MIN_GLYPH if latin else DEFAULT_MIN_GLYPH
    sure = ln.det >= LINE_MIN_DET or ln.score >= LOW_DET_SCORE
    return bool(ln.text) and sure and ln.glyph_px >= floor


def keep_line(ln: Any) -> bool:
    """A line :func:`readable` keeps, a Latin line read at
    :data:`LATIN_MIN_SCORE` whatever its det, or a Hangul line (no Japanese
    script) that :func:`readable` lost to its glyph floor alone, at
    :data:`LATIN_MIN_GLYPH` or above."""
    if readable(ln):
        return True
    if ln.glyph_px < LATIN_MIN_GLYPH:
        return False
    if is_latin(ln.text):
        return ln.score >= LATIN_MIN_SCORE
    return (
        bool(HANGUL.search(ln.text))
        and not JA_SCRIPT.search(ln.text)
        and (ln.det >= LINE_MIN_DET or ln.score >= LOW_DET_SCORE)
    )


def load_reread_engine(device: str):
    """The AnimeText detector with the OCR reader re-reading its boxes, kept
    down to the detector's own floor (the floors above decide afterwards)."""
    from pathlib import Path

    from anime_tools.ocr import load_ocr
    from anime_tools.ocr.animetext import DEFAULT_CONF
    from anime_tools.ocr.reread import RereadEngine
    from anime_tools.ocr.reader import OcrReader

    reader = OcrReader.load(device=device)
    return RereadEngine(
        engine=load_ocr(device=device),
        read_boxes=reader.read_boxes_scored,
        resized_dir=Path("/"),
        skip_en=False,
        min_det=DEFAULT_CONF,
    )
