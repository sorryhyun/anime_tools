"""The VL pass over a page (``anime_tools.ocr.reread``), weights-free: each
box's read is its line and a rejected read drops the box, the mask's uncovered
components become lines, reading order is settled afterwards, and the request
carries the flags. Nothing here loads a model — ``read_boxes`` is a fake."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from anime_tools.captions.ocr_sidecar import OcrLine
from anime_tools.ocr import reread
from anime_tools.stages.requests import OcrRequest


def line(text, box, *, seq=1, score=0.9):
    return OcrLine(seq=seq, box=box, score=score, text=text)


def page(h=400, w=300):
    return np.full((h, w, 3), 255, np.uint8)


def test_the_module_is_torch_free():
    code = (
        "import sys, anime_tools.ocr.reread as r; r.overlap((0,0,1,1),(0,0,1,1)); "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr


# ---- the re-read -------------------------------------------------------


def test_a_read_is_the_line_and_a_rejected_one_drops_the_box():
    lines = [
        line("", (10, 10, 40, 120), score=0.0),
        line("", (100, 10, 130, 120), seq=2, score=0.0),
    ]
    seen = {}

    def read_boxes(bgr, boxes):
        seen["boxes"] = list(boxes)
        return ["ぱんぱん", None]  # the second read failed the guard

    out = reread.reread_lines(page(), lines, read_boxes)
    assert seen["boxes"] == [(10, 10, 40, 120), (100, 10, 130, 120)]
    # the rejected box is gone, not kept with empty text; the read carries no
    # recognizer confidence
    assert [(ln.seq, ln.text, ln.score) for ln in out] == [
        (1, "ぱんぱん", reread.NO_SCORE),
    ]


def test_a_scored_read_keeps_its_confidence_and_the_box_keeps_its_det():
    lines = [
        OcrLine(seq=1, box=(10, 10, 40, 120), score=0.0, text="", det=0.81),
        OcrLine(seq=2, box=(100, 10, 130, 120), score=0.0, text="", det=0.42),
    ]
    reads = [("ぱんぱん", 0.97), "どきどき"]  # a scored read and a bare one
    out = reread.reread_lines(page(), lines, lambda b, x: reads)
    assert [(ln.text, ln.score, ln.det) for ln in out] == [
        ("どきどき", reread.NO_SCORE, 0.42),  # right column first
        ("ぱんぱん", 0.97, 0.81),
    ]


def test_a_read_must_clear_the_floors_and_carry_a_letter():
    lines = [line("", (b, 10, b + 30, 120), score=0.0) for b in (10, 60, 110, 160)]
    reads = ["♡", "12", "OK", "ぱんぱん"]
    out = reread.reread_lines(page(), lines, lambda b, x: reads, min_chars=3)
    # a decoration, a letterless read and an ASCII-only read all drop
    assert [ln.text for ln in out] == ["ぱんぱん"]
    kept = reread.reread_lines(
        page(), lines, lambda b, x: reads, min_chars=1, skip_en=False
    )
    # the floors off, ASCII stays; a read with no letter in it never does
    assert sorted(ln.text for ln in kept) == ["OK", "ぱんぱん"]


def test_an_empty_page_calls_no_reader():
    def read_boxes(bgr, boxes):
        raise AssertionError("no crops to read")

    assert reread.reread_lines(page(), [], read_boxes) == []


# ---- the mask components ---------------------------------------------


def _mask(h=400, w=300, text_boxes=()):
    """An ignore mask: 255 everywhere, 0 where text is."""
    m = np.full((h, w), 255, np.uint8)
    for x0, y0, x1, y1 in text_boxes:
        m[y0:y1, x0:x1] = 0
    return m


def test_mask_components_are_boxed_largest_first_and_floored():
    m = _mask(text_boxes=[(20, 20, 60, 200), (200, 300, 260, 340), (10, 350, 20, 356)])
    boxes = reread.mask_components(m, min_side=16)
    assert boxes == [
        (20, 20, 60, 200),
        (200, 300, 260, 340),
    ]  # the 10×6 one is screentone


def test_a_full_page_component_is_not_a_line():
    m = _mask(text_boxes=[(0, 0, 300, 300)])
    assert reread.mask_components(m, min_side=16) == []


def test_uncovered_components_become_lines_with_no_score_and_the_floors_apply():
    lines = [line("", (20, 20, 60, 200), score=0.0)]  # the detector's box
    m = _mask(
        text_boxes=[
            (20, 20, 60, 200),  # covered by the detector's box
            (200, 40, 240, 160),  # a real SFX
            (100, 300, 160, 340),  # reads as a lone heart
            (200, 300, 260, 340),  # reads as one glyph, under min_chars
            (30, 300, 90, 340),  # reads as romaji, skip_en
        ]
    )
    reads = {
        (20, 20, 60, 200): "こんにちは",
        (200, 40, 240, 160): "ぱんぱん",
        (100, 300, 160, 340): "♡",
        (200, 300, 260, 340): "ぱ",
        (30, 300, 90, 340): "dokidoki",
    }

    def read_boxes(bgr, boxes):
        return [reads[tuple(b)] for b in boxes]

    out = reread.reread_lines(
        page(), lines, read_boxes, mask=m, comp_min_side=16, min_chars=2
    )
    # the covered component is read once (through the detector's box), and a
    # detector box and a mask component are lines of the same kind
    assert [(ln.text, ln.score, ln.box) for ln in out] == [
        ("ぱんぱん", reread.NO_SCORE, (200, 40, 240, 160)),
        ("こんにちは", reread.NO_SCORE, (20, 20, 60, 200)),
    ]


def test_comp_max_caps_the_components_largest_first():
    m = _mask(text_boxes=[(200, 40, 240, 160), (100, 300, 140, 340)])
    got = {}

    def read_boxes(bgr, boxes):
        got["n"] = len(boxes)
        return ["ぱんぱん"] * len(boxes)

    out = reread.reread_lines(
        page(), [], read_boxes, mask=m, comp_min_side=16, comp_max=1
    )
    assert got["n"] == 1 and [ln.box for ln in out] == [(200, 40, 240, 160)]


def test_overlap_reads_containment_off_the_smaller_box():
    iou, cont = reread.overlap((0, 0, 100, 100), (10, 10, 20, 20))
    assert cont == 1.0 and iou == pytest.approx(0.01)
    assert reread.covered((10, 10, 20, 20), [(0, 0, 100, 100)])
    assert not reread.covered((200, 200, 220, 220), [(0, 0, 100, 100)])


# ---- the engine wrapper and the mask lookup -------------------------------


def test_mask_for_prefers_the_nested_tree_and_falls_back_flat(tmp_path: Path):
    (tmp_path / "artist").mkdir()
    flat = tmp_path / "a_mask.png"
    flat.write_bytes(b"")
    assert reread.mask_for(tmp_path, Path("artist/a.png")) == flat
    nested = tmp_path / "artist" / "a_mask.png"
    nested.write_bytes(b"")
    assert reread.mask_for(tmp_path, Path("artist/a.png")) == nested
    assert reread.mask_for(tmp_path, Path("artist/b.png")) is None


def test_the_engine_wrapper_rereads_each_page_the_engine_yields(tmp_path: Path):
    from PIL import Image

    dst = tmp_path / "resized"
    dst.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (120, 80), "white").save(dst / name)

    class Engine:
        def read(self, p):
            return [line("はんぱん", (10, 10, 40, 60))] if p.name == "a.png" else []

        def read_iter(self, paths):
            for p in paths:
                yield self.read(p)

    calls = []

    def read_boxes(bgr, boxes):
        calls.append((bgr.shape, list(boxes)))
        return ["ぱんぱん"] * len(boxes)

    eng = reread.RereadEngine(engine=Engine(), read_boxes=read_boxes, resized_dir=dst)
    pages = list(eng.read_iter([dst / "a.png", dst / "b.png"]))
    assert [[ln.text for ln in pg] for pg in pages] == [["ぱんぱん"], []]
    # the empty page never decoded its pixels for the reader
    assert calls == [((80, 120, 3), [(10, 10, 40, 60)])]
    assert [ln.text for ln in eng.read(dst / "a.png")] == ["ぱんぱん"]


# ---- the request ----------------------------------------------------------


def test_the_request_round_trips_the_vl_flags():
    req = OcrRequest(mask_dir="m", comp_min_side=24, comp_max=4, vl_batch_size=2)
    argv = req.to_argv()
    assert "--mask_dir" in argv and "--vl_batch_size" in argv
    assert OcrRequest.from_argv(OcrRequest.parser(), argv) == req
    # There is one reader and one detector: neither is a flag any more.
    assert "--reader" not in argv and "--detector" not in argv
    assert not hasattr(OcrRequest(), "reader") and not hasattr(OcrRequest(), "detector")
