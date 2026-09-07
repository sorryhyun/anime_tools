"""The manga SFX reader's weight-free half: the decode guard, the crop geometry,
and the catalog rows its loader reads. Nothing here loads the model."""

from __future__ import annotations

import numpy as np
import pytest

from anime_tools import downloads as DL
from anime_tools.ocr import sfx

# ---- the decode guard ---------------------------------------------------


def test_the_guard_is_torch_free():
    import subprocess
    import sys

    code = (
        "import sys, anime_tools.ocr.sfx as s; s.guard('ぱんぱん', 40, 120); "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "text", ["ぱんぱん", "おおおん", "ばるん", "びく♡", "ぱんッ ぱんッ"]
)
def test_a_doubled_onomatopoeia_is_not_a_runaway(text):
    assert not sfx.is_runaway(text)


@pytest.mark.parametrize(
    "text",
    ["ぐく" + "ー" * 30, "ふくっ" * 10, "ぉ" * 8, "ふくっふくっふくっ"],
)
def test_a_glyph_run_or_a_repeated_trigram_is_a_runaway(text):
    assert sfx.is_runaway(text)


def test_the_length_cap_follows_the_crop_area():
    # A two-glyph crop sits at the floor; a five-glyph column and a balloon
    # block cap well above what they hold (the repetition test owns those).
    assert sfx.length_cap(40, 80) == sfx.MIN_LENGTH_CAP
    assert sfx.length_cap(40, 200) == 31
    assert sfx.length_cap(300, 300) == 352
    assert sfx.length_cap(0, 0) == sfx.MIN_LENGTH_CAP


def test_the_guard_rejects_empty_runaway_and_overlong_reads():
    assert sfx.guard("", 40, 120) is None
    assert sfx.guard("   ", 40, 120) is None
    assert sfx.guard("ぐく" + "ー" * 30, 40, 120) is None
    # 20 distinct characters on a two-glyph crop: no repetition, too long.
    assert sfx.guard("あいうえおかきくけこさしすせそたちつてと", 40, 80) is None
    # The same read on a column that could hold it passes.
    assert sfx.guard("あいうえおかきくけこさしすせそたちつてと", 40, 400) is not None


def test_the_guard_normalises_what_it_keeps():
    # Emoji heart + variation selector → ♥; whitespace runs → one space (the
    # JOIN_SEP boundary); VL's LaTeX measurement wrapping stripped.
    assert sfx.guard("びく❤️", 40, 120) == "びく♥"
    assert sfx.guard("ぱんッ\n\nぱんッ", 80, 120) == "ぱんッ ぱんッ"
    assert sfx.guard("身長: \\( 156 \\, cm \\)", 300, 40) == "身長: 156 cm"
    # A native ♡ is kept as read — the reader emits it itself.
    assert sfx.guard("ぱん♡", 40, 120) == "ぱん♡"


# ---- the crop geometry --------------------------------------------------


def test_pad_box_grows_by_the_longer_side_and_clips():
    assert sfx.pad_box((10, 10, 20, 110), 200, 200, 0.12) == (0, 0, 32, 122)
    assert sfx.pad_box((190, 5, 200, 105), 200, 120, 0.12) == (178, 0, 200, 117)


def test_crop_box_returns_the_padded_window_or_none():
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = sfx.crop_box(bgr, (50, 20, 60, 70), 0.1)
    assert crop.shape == (60, 20, 3)  # 50 tall + 2×5, 10 wide + 2×5
    assert sfx.crop_box(bgr, (50, 20, 50, 20), 0.0) is None


# ---- the catalog --------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    monkeypatch.delenv("ANIME_TOOLS_MODELS", raising=False)
    return tmp_path


def test_the_reader_rows_land_where_the_loader_looks(home):
    by = DL.by_id()
    assert by["vl16_base"].dest == DL.default_vl16_base_dir()
    assert by["sfx_reader"].dest == DL.default_sfx_reader_dir()
    assert by["vl16_base"].dest == home / "models" / "paddleocr_vl_1.6"
    assert by["sfx_reader"].dest == home / "models" / "paddleocr_vl_1.6_manga_lora"
    # The loader checks for exactly the files the row fetches.
    assert set(DL.SFX_READER_FILES) == {
        *DL.SFX_READER_ADAPTER_FILES,
        DL.SFX_READER_TOWER_FILE,
    }
    assert "model.safetensors" in DL.VL16_BASE_FILES
    # The OCR stage reads every box with it (2026-09-07, no other reader), so
    # the stage bar warns for both rows before a run; both sit in the ocr pack.
    assert by["vl16_base"].stages == ("ocr",) and by["sfx_reader"].stages == ("ocr",)
    assert by["vl16_base"].pack == by["sfx_reader"].pack == "ocr"


def test_load_without_weights_names_the_row(home):
    with pytest.raises(sfx.SfxWeightsMissing, match="vl16_base"):
        sfx.SfxReader.load(fetch=False)
