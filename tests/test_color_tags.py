"""Colour tags of a caption (``anime_tools.captions.color_tags``): what a colorize
caption keeps. A regression silently changes what a colorize LoRA steers on."""

from __future__ import annotations

import json

from anime_tools.captions.color_tags import (
    _NON_SERIES_COPYRIGHT,
    filter_to_colors,
    filter_to_colors_and_copyright,
    filter_to_colors_and_protected,
    is_character_colour,
    is_color_tag,
    load_copyright_tags,
)

_COPYRIGHT = frozenset({"genshin impact", "fate/grand order"})


def test_keeps_hair_eye_skin_colors():
    for tag in (
        "blue hair",
        "aqua eyes",
        "light brown hair",
        "grey eyes",
        "dark skin",
        "two-tone hair",
        "heterochromia",
    ):
        assert is_color_tag(tag), tag


def test_keeps_color_noun_tags():
    for tag in (
        "yellow shirt",
        "red ribbon",
        "white background",
        "dark green dress",
        "monochrome",
    ):
        assert is_color_tag(tag), tag


def test_drops_non_color_tags():
    for tag in (
        "1girl",
        "looking at viewer",
        "long hair",
        "closed eyes",
        "open mouth",
        "smile",
        "@eufoniuz",
        "ponytail",
    ):
        assert not is_color_tag(tag), tag


def test_filter_extracts_colors_in_order():
    caption = (
        "explicit, 1boy, 1girl, suou yuki, @eufoniuz, brown eyes, brown hair, "
        "looking at viewer, ponytail, smile, yellow shirt, green shorts"
    )
    assert (
        filter_to_colors(caption)
        == "brown eyes, brown hair, yellow shirt, green shorts"
    )


def test_filter_empty_when_no_color():
    assert filter_to_colors("1girl, solo, looking at viewer, smile") == ""
    assert filter_to_colors("") == ""


def test_copyright_kept_first_then_colors():
    caption = (
        "1girl, nilou (genshin impact), genshin impact, @sincos, blue hair, "
        "looking at viewer, red dress"
    )
    out = filter_to_colors_and_copyright(caption, _COPYRIGHT)
    assert out == "genshin impact, blue hair, red dress"


def test_no_vocabulary_no_series():
    assert filter_to_colors_and_protected("genshin impact, blue hair") == "blue hair"


def test_comic_and_copyright_ordering():
    caption = (
        "1girl, nilou (genshin impact), genshin impact, comic, @sincos, "
        "blue hair, red dress"
    )
    out = filter_to_colors_and_protected(
        caption, _COPYRIGHT, keep_copyright=True, keep_comic=True
    )
    assert out == "genshin impact, comic, blue hair, red dress"


def test_original_excluded_from_loaded_copyright_vocab(tmp_path):
    index = {
        "groups": {
            "copyright": {
                "Genshin Impact": [1],
                "Original": [1, 2],
                "original character": [3],
            }
        }
    }
    p = tmp_path / "caption_index.json"
    p.write_text(json.dumps(index), encoding="utf-8")
    vocab = load_copyright_tags(p)
    assert _NON_SERIES_COPYRIGHT.isdisjoint(vocab)
    assert "genshin impact" in vocab
    assert load_copyright_tags(tmp_path / "missing.json") == frozenset()


def test_character_colour_keeps_hair_eyes_and_clothes():
    keep = ["pink hair", "blue eyes", "red pupils", "blue skirt", "light blue dress"]
    drop = ["white background", "dark skin", "dark-skinned female", "blue sky", "blush"]
    assert all(is_character_colour(t) for t in keep)
    assert not any(is_character_colour(t) for t in drop)


def test_the_module_is_stdlib_only():
    import subprocess
    import sys

    code = "import sys, anime_tools.captions.color_tags; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.stdout.strip() == "False", out.stderr
