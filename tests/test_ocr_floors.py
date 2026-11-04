"""The OCR lines a MOSS i2i caption boxes (``anime_tools.ocr.floors``): the floors
the colorize and swap_text LoRAs trained on. Weights-free."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from anime_tools.ocr.floors import keep_line, readable


def line(text, det, score, glyph):
    return SimpleNamespace(text=text, det=det, score=score, glyph_px=glyph)


def test_readable_holds_a_low_det_box_to_the_reader_score():
    assert readable(line("こんにちは", 0.70, 0.50, 18.0))
    assert readable(line("こんにちは", 0.30, 0.95, 18.0))
    assert not readable(line("こんにちは", 0.30, 0.85, 18.0))
    assert not readable(line("こんにちは", 0.90, 0.99, 15.0))
    assert readable(line("HELLO", 0.90, 0.99, 12.5))
    assert not readable(line("", 0.90, 0.99, 20.0))


def test_small_hangul_line_clears_the_latin_glyph_floor():
    assert keep_line(line("만화에서 이런 걸 하고 있었어?", 0.90, 0.99, 14.7))
    assert not keep_line(line("만화에서 이런 걸", 0.90, 0.99, 11.9))
    assert not keep_line(line("아앙", 0.42, 0.80, 14.0))
    assert not keep_line(line("ざわっ", 0.74, 0.72, 15.9))


def test_latin_line_read_well_is_kept_whatever_its_det():
    assert keep_line(line("I DUNNO", 0.35, 0.80, 12.2))
    assert not keep_line(line("I DUNNO", 0.35, 0.79, 12.2))


def test_the_module_is_torch_free():
    code = "import sys, anime_tools.ocr.floors; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.stdout.strip() == "False", out.stderr
