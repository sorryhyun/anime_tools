"""The manga OCR reader's weight-free half: the decode guard, the crop geometry,
and the catalog rows its loader reads. Nothing here loads the model."""

from __future__ import annotations

import numpy as np
import pytest

from anime_tools import downloads as DL
from anime_tools.ocr import reader

# ---- the decode guard ---------------------------------------------------


def test_the_guard_is_torch_free():
    import subprocess
    import sys

    code = (
        "import sys, anime_tools.ocr.reader as s; s.guard('ぱんぱん', 40, 120); "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "text",
    [
        "るにいい~っ" + "♥" * 12,
        "ずぷっぐゅぐゅぐゅぐゅ",
        "ぱこんぱこんっぱこんっ",
        "ドボドボドボドボ",
    ],
)
def test_a_repeated_sound_effect_is_not_a_runaway(text):
    assert not reader.is_runaway(text)


@pytest.mark.parametrize(
    "text", ["ぱんぱん", "おおおん", "ばるん", "びく♡", "ぱんッ ぱんッ"]
)
def test_a_doubled_onomatopoeia_is_not_a_runaway(text):
    assert not reader.is_runaway(text)


@pytest.mark.parametrize(
    "text",
    ["ぐく" + "ー" * 30, "ふくっ" * 10, "ぉ" * 16],
)
def test_a_glyph_run_or_a_back_to_back_loop_is_a_runaway(text):
    assert reader.is_runaway(text)


def test_a_timetable_is_not_a_runaway():
    text = (
        "05:30 기상 06:30 출근 06:50 메이크업 07:30 아침 회의 "
        "08:00 「안녕하세요week」 방송 09:55 「안녕하세요week」 종료 "
        "10:15 「안녕하세요week」 반성회 10:45 사무활동 12:00 중식"
    )
    assert not reader.is_runaway(text)
    assert reader.guard(text, 233, 236) is not None


def test_a_long_loop_is_still_a_runaway():
    assert reader.is_runaway(
        "처음 한 줄은 멀쩡한 대사였습니다 그리고" + "저기 말이야 있잖아 " * 6
    )


@pytest.mark.parametrize(
    "text",
    [
        "NO, NO, NO, NO!",
        "WAIT, WAIT, WAIT.",
        "WHAAAAAAAAT?!",
        "Cock♡ Cock♡ Cock♡ Cock♡",
        "BUT… YOU CAN COME TO MY ROOM, OKAY? YOU KNOW WHERE IT IS, RIGHT? "
        "ANYTIME, EVEN IN THE MIDDLE OF THE NIGHT! OH, BUT I REALLY WANT YOU TO",
    ],
)
def test_latin_lettering_is_not_a_runaway(text):
    assert not reader.is_runaway(text)


@pytest.mark.parametrize(
    "text", ["AHAAH " + "♡ " * 38, "o" * 16, "ahahahahahahahah", "LapLap" * 8]
)
def test_a_latin_loop_is_a_runaway(text):
    assert reader.is_runaway(text)


def test_is_latin_reads_the_letters_only():
    assert reader.is_latin("Mh! Mh!♡ 100")
    assert not reader.is_latin("100♡…")
    assert not reader.is_latin("Kiss たらっ")
    assert not reader.is_latin("괜찮아")


def test_the_length_cap_follows_the_crop_area():
    # A two-glyph crop sits at the floor; a five-glyph column and a balloon
    # block cap well above what they hold (the repetition test owns those).
    assert reader.length_cap(40, 80) == reader.MIN_LENGTH_CAP
    assert reader.length_cap(40, 200) == 31
    assert reader.length_cap(300, 300) == 352
    assert reader.length_cap(0, 0) == reader.MIN_LENGTH_CAP
    # A Latin letter takes half a CJK square.
    assert reader.length_cap(40, 200, latin=True) == 62


def test_the_guard_rejects_empty_runaway_and_overlong_reads():
    assert reader.guard("", 40, 120) is None
    assert reader.guard("   ", 40, 120) is None
    assert reader.guard("ぐく" + "ー" * 30, 40, 120) is None
    # 20 distinct characters on a two-glyph crop: no repetition, too long.
    assert reader.guard("あいうえおかきくけこさしすせそたちつてと", 40, 80) is None
    # The same read on a column that could hold it passes.
    assert reader.guard("あいうえおかきくけこさしすせそたちつてと", 40, 400) is not None
    # A scanlation balloon over the CJK cap of its crop fits the Latin one.
    assert reader.guard("Yesterday's video was amazing.", 60, 64) is not None


def test_the_guard_normalises_what_it_keeps():
    # Every heart → ♡ (the emoji + variation selector, the solid ♥ the reader
    # also emits); whitespace runs inside a line → one space; VL's
    # LaTeX measurement wrapping stripped.
    assert reader.guard("びく❤️", 40, 120) == "びく♡"
    assert reader.guard("あ・・・っ♥", 40, 160) == "あ…っ♡"
    # A line break is kept as one \n, a blank line dropped, a run of spaces
    # inside a line one space.
    assert reader.guard("ぱんッ\n\nぱんッ", 80, 120) == "ぱんッ\nぱんッ"
    assert (
        reader.guard(" 최근  제게 \r\n 보내는 관심이\n", 300, 120)
        == "최근 제게\n보내는 관심이"
    )
    assert reader.guard("身長: \\( 156 \\, cm \\)", 300, 40) == "身長: 156 cm"
    # A native ♡ is kept as read — the reader emits it itself.
    assert reader.guard("ぱん♡", 40, 120) == "ぱん♡"


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("はひ・・・・・・", "はひ…"),
        ("え...な...ない......", "え…な…ない…"),
        ("結ばれた二人は......♡", "結ばれた二人は…♡"),
        ("もう‥‥だめ", "もう…だめ"),
        ("1.5倍", "1.5倍"),  # one dot is not an ellipsis
        ("赤・青", "赤・青"),  # nor is one middle dot
    ],
)
def test_every_dot_run_is_one_ellipsis(raw, norm):
    assert reader.normalize_read(raw) == norm


@pytest.mark.parametrize(
    "text",
    [
        # The two sincos blocks the first guard threw away (2026-09-07): the
        # dots held the trigram ``...`` four times over.
        "結ばれた二人は甘いロゾけを交わすんです......",
        "ふ...ぉ...フゥ......♡これから孕むまで毎日使ってやるからな♡",
        "はひ・・・・・・っ",
        "...あ............いや...寝たフリだな...これ.........",
    ],
)
def test_a_pause_is_not_a_runaway(text):
    assert not reader.is_runaway(text)
    assert reader.guard(text, 130, 300) is not None


def test_a_hundred_dots_alone_collapse_to_one():
    # No longer a runaway — but no letter either, so ``has_script`` drops it.
    from anime_tools.ocr.reread import has_script

    read = reader.guard("・" * 100, 60, 60)
    assert read == reader.ELLIPSIS and not has_script(read)


# ---- the crop geometry --------------------------------------------------


def test_pad_box_grows_by_the_longer_side_and_clips():
    assert reader.pad_box((10, 10, 20, 110), 200, 200, 0.12) == (0, 0, 32, 122)
    assert reader.pad_box((190, 5, 200, 105), 200, 120, 0.12) == (178, 0, 200, 117)


def test_crop_box_returns_the_padded_window_or_none():
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = reader.crop_box(bgr, (50, 20, 60, 70), 0.1)
    assert crop.shape == (60, 20, 3)  # 50 tall + 2×5, 10 wide + 2×5
    assert reader.crop_box(bgr, (50, 20, 50, 20), 0.0) is None


# ---- the catalog --------------------------------------------------------


def test_the_reader_rows_land_where_the_loader_looks(home):
    by = DL.by_id()
    assert by["vl16_base"].dest == DL.default_vl16_base_dir()
    assert by["ocr_reader"].dest == DL.default_ocr_reader_dir()
    assert by["vl16_base"].dest == home / "models" / "paddleocr_vl_1.6"
    assert by["ocr_reader"].dest == home / "models" / "paddleocr_vl_1.6_manga_lora"
    # The loader checks for exactly the files the row fetches.
    assert set(DL.OCR_READER_FILES) == {
        *DL.OCR_READER_ADAPTER_FILES,
        DL.OCR_READER_LM_FILE,
    }
    assert "model.safetensors" in DL.VL16_BASE_FILES
    # The OCR stage reads every box with it (2026-09-07, no other reader), so
    # the stage bar warns for both rows before a run; both sit in the ocr pack.
    assert by["vl16_base"].stages == ("ocr",) and by["ocr_reader"].stages == ("ocr",)
    assert by["vl16_base"].pack == by["ocr_reader"].pack == "ocr"


def test_load_without_weights_names_the_row(home):
    with pytest.raises(reader.OcrReaderWeightsMissing, match="vl16_base"):
        reader.OcrReader.load(fetch=False)


def test_token_confidence_averages_the_kept_tokens_and_stops_at_eos():
    ids, conf = reader.token_confidence(
        [5, 6, 7, 2, 0, 0], [0.9, 0.5, 0.7, 0.99, 1.0, 1.0], {2, 0}
    )
    assert ids == [5, 6, 7]
    assert conf == pytest.approx(0.7)
    # an empty decode is sure of nothing
    assert reader.token_confidence([2, 0], [0.99, 1.0], {2, 0}) == ([], 0.0)
    # a decode the step recorder never saw (no steps) is 0.0, not an error
    assert reader.token_confidence([5, 6], [], {2, 0}) == ([5, 6], 0.0)


def test_vl_notices_drop_and_other_warnings_print(capfd) -> None:
    pytest.importorskip("transformers")
    from transformers.utils import logging as hf_logging

    from anime_tools.ocr.reader import _without_vl_notices

    log = hf_logging.get_logger("transformers.modeling_rope_utils")
    with _without_vl_notices():
        log.warning("Unrecognized keys in `rope_parameters` for this load")
        log.warning("Some weights were not initialized")
    log.warning("Unrecognized keys in `rope_parameters` after the load")
    err = capfd.readouterr().err
    assert "for this load" not in err
    assert "Some weights were not initialized" in err
    assert "after the load" in err
