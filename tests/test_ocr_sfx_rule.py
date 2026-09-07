"""The speech / SFX split and the SFX dedupe the export clause is built on
(:mod:`anime_tools.captions.ocr_sfx`) — text-only, torch-free."""

from __future__ import annotations

import pytest

from anime_tools.captions.ocr_sfx import dedupe_sfx, line_kind, sfx_key, split_lines


@pytest.mark.parametrize(
    "text", ["ぱんぱん", "びくっ", "じゅぽ", "ブルン", "ちゅ", "ゴゴ", "きゅっ"]
)
def test_onomatopoeia_is_sfx(text):
    assert line_kind(text) == "sfx"


def test_a_vowel_initial_is_a_mouth_before_it_is_short_katakana():
    # rule 2 runs before rule 4: ウズ is speech to this rule, ゴゴ is SFX
    assert line_kind("ウズ") == "speech"
    # and a kanji-less voiced-initial sentence is the rule's known blind spot
    assert line_kind("どうしたの") == "sfx"


@pytest.mark.parametrize(
    "text",
    ["あっ…うっ…", "はぁ", "おおおん", "今日は光海祭", "いいわよモデルになってあげる"],
)
def test_a_mouth_or_a_sentence_is_speech(text):
    assert line_kind(text) == "speech"


def test_a_balloon_veto_beats_the_text_rule():
    assert line_kind("カリカリ") == "sfx"
    assert line_kind("カリカリ", in_bubble=True) == "speech"


def test_split_keeps_reading_order_within_each_kind():
    speech, sfx = split_lines(["ぱん", "いいわよ", "ぴちょっ", "動画撮影するんでしょ?"])
    assert speech == ["いいわよ", "動画撮影するんでしょ?"]
    assert sfx == ["ぱん", "ぴちょっ"]


def test_dedupe_folds_a_sound_to_its_unit_and_keeps_the_first():
    assert sfx_key("ぱん♡ぱん♡") == "ぱん"
    assert sfx_key("びくッ") == "びく"
    assert dedupe_sfx(["じゅぽ", "じゅぽ", "じゅぽじゅぽ", "ぱん♡", "ぱんッ"]) == [
        "じゅぽ",
        "ぱん♡",
    ]
