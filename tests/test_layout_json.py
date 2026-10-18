from __future__ import annotations

import json
from pathlib import Path

from anime_tools.captions.correction import load_tag_knowledge_base
from anime_tools.captions.layout_json import caption_to_layout, layout_json


def _kb(tmp_path: Path):
    path = tmp_path / "tags.csv"
    path.write_text(
        """name,category,post_count,description
1girl,0,10,"[인물 > 인원수] count"
2girls,0,10,"[인물 > 인원수] count"
hatsune_miku,4,10,"[캐릭터 > vocaloid] character"
kasane_teto,4,10,"[캐릭터 > utau] character"
vocaloid,3,10,"[작품 > series] copyright"
highres,5,10,"[메타 > 화질] resolution meta"
long_hair,0,10,"[머리카락 > 머리 길이] general"
necktie,0,10,"[의상 > 넥타이] general"
cropped,5,10,"[메타 > 구도] meta"
sincos,1,10,"[작가] artist"
""",
        encoding="utf-8",
    )
    return load_tag_knowledge_base(path)


def test_flat_bag_splits_into_summary_fields(tmp_path):
    layout = caption_to_layout(
        "sensitive, highres, 1girl, hatsune miku, vocaloid, @sincos, long hair, "
        "unknown tag",
        _kb(tmp_path),
    )
    assert layout == {
        "summary": {
            "meta": ["highres", "sensitive"],
            "characters": ["hatsune miku"],
            "series": ["vocaloid"],
            "artists": ["@sincos"],
            "caption": "1girl, long hair, unknown tag",
        }
    }


def test_visual_meta_reads_as_caption(tmp_path):
    summary = caption_to_layout(
        "highres, cropped, 1girl, long hair, sincos", _kb(tmp_path)
    )["summary"]
    assert summary == {
        "meta": ["highres"],
        "artists": ["@sincos"],
        "caption": "1girl, cropped, long hair",
    }


def test_clauses_become_items_in_order(tmp_path):
    layout = caption_to_layout(
        "2girls, long hair. On the left, hatsune miku, necktie. On the right, "
        'kasane teto. Japanese text reads as "満足でしょうか", "「はい、」"',
        _kb(tmp_path),
    )
    assert layout["items"] == [
        {
            "label": "character",
            "position": "left",
            "characters": ["hatsune miku"],
            "caption": "necktie",
        },
        {"label": "character", "position": "right", "characters": ["kasane teto"]},
        {"label": "text", "caption": "Japanese text", "text": "満足でしょうか"},
        {"label": "text", "caption": "Japanese text", "text": "「はい、」"},
    ]


def test_no_artist_sentinel_and_legacy_rating(tmp_path):
    summary = caption_to_layout("general, 1girl, @no-artist", _kb(tmp_path))["summary"]
    assert summary == {"meta": ["safe"], "caption": "1girl"}


def test_serialization_is_compact_and_unescaped(tmp_path):
    text = layout_json('1girl. Japanese SFX reads as "ドン"', _kb(tmp_path))
    assert text == (
        '{"summary":{"caption":"1girl"},"items":[{"label":"text",'
        '"caption":"Japanese sound effect","text":"ドン"}]}'
    )
    assert json.loads(text)["items"][0]["text"] == "ドン"


def test_text_items_take_centred_boxes_and_the_top_level_keys(tmp_path):
    """Only text items take a box, clamped to the canvas; the coordinate keys
    ride along only when one did."""
    text = layout_json(
        '1girl. On the left, necktie. Japanese text reads as "あ", "い"',
        _kb(tmp_path),
        text_boxes={"あ": (100, 50, 140, 210), "う": (0, 0, 1, 1)},
        size=(400, 300),
    )
    layout = json.loads(text)
    assert layout["items"] == [
        {"label": "character", "position": "left", "caption": "necktie"},
        {
            "label": "text",
            "caption": "Japanese text",
            "text": "あ",
            "bbox": [-100, -100, 60, -60],
        },
        {"label": "text", "caption": "Japanese text", "text": "い"},
    ]
    assert list(layout)[-2:] == ["bbox_coord_space", "bbox_format"]
    assert (layout["bbox_coord_space"], layout["bbox_format"]) == (
        "abs_centralized",
        "yxyx",
    )


def test_no_matching_box_keeps_the_plain_shape(tmp_path):
    caption = '1girl. Japanese text reads as "あ"'
    kb = _kb(tmp_path)
    assert layout_json(caption, kb, text_boxes={}, size=(64, 64)) == layout_json(
        caption, kb
    )


def test_centered_box_clamps_to_the_canvas():
    from anime_tools.captions.layout_json import centered_box

    assert centered_box((-5, 10, 950, 400), (944, 1136)) == [-558, -472, -168, 472]


def test_text_boxes_key_lines_as_the_clause_spells_them():
    from anime_tools.captions.ocr_sidecar import OcrLine, text_boxes

    lines = [
        OcrLine(seq=1, box=(0, 0, 60, 60), score=0.9, text='say "hi"'),
        OcrLine(seq=2, box=(0, 100, 60, 160), score=0.9, text='say "hi"'),
        OcrLine(seq=3, box=(0, 0, 2, 2), score=0.9, text="tiny"),
    ]
    assert text_boxes(lines) == {"say ”hi”": (0, 0, 60, 60)}
