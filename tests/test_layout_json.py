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
            "label": "subject",
            "position": "left",
            "characters": ["hatsune miku"],
            "caption": "necktie",
        },
        {"label": "subject", "position": "right", "characters": ["kasane teto"]},
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
