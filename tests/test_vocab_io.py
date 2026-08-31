"""One reader for ``vocab.json`` — the four hand parsers now go through it.

Pins the two halves that used to be spelled out per consumer: the
``category -> names`` projection (with the caller's own folding), and the
``resolved_to_dict``/``resolved_from_dict`` round trip that lets the group
router revive a snapshot instead of walking raw dicts.
"""

from __future__ import annotations

import json

import pytest

from anime_tools.captions.tag_groups import (
    ResolvedGroup,
    resolved_from_dict,
    resolved_to_dict,
)
from anime_tools.captions.vocab_io import (
    load_vocab,
    names_by_category,
    names_in_categories,
    resolved_groups,
)

VOCAB = {
    "tags": [
        {"name": "1girl", "index": 0, "category": "count"},
        {"name": "Blue_Hair", "index": 1, "category": "general"},
        {"name": "hatsune miku", "index": 2, "category": "character"},
        {"name": "vocaloid", "index": 3, "category": "copyright"},
    ],
    "ratings": ["general", "explicit"],
}


def _write(tmp_path, vocab=VOCAB, name="vocab.json"):
    p = tmp_path / name
    p.write_text(json.dumps(vocab), encoding="utf-8")
    return p


def test_load_vocab_takes_the_file_or_its_checkpoint_dir(tmp_path):
    _write(tmp_path)
    assert load_vocab(tmp_path) == load_vocab(tmp_path / "vocab.json") == VOCAB


def test_names_by_category_preseeds_every_requested_axis(tmp_path):
    sets = names_by_category(VOCAB, ("character", "copyright", "artist"))
    # "artist" has no rows, but membership tests still want a set to ask.
    assert sets == {
        "character": {"hatsune miku"},
        "copyright": {"vocaloid"},
        "artist": set(),
    }


def test_names_by_category_without_a_filter_returns_every_category():
    assert set(names_by_category(VOCAB)) == {
        "count",
        "general",
        "character",
        "copyright",
    }


def test_key_folds_names_on_the_way_in():
    sets = names_by_category(VOCAB, ("general",), key=lambda n: n.strip().lower())
    assert sets["general"] == {"blue_hair"}


def test_names_in_categories_unions_and_survives_an_empty_ask():
    assert names_in_categories(VOCAB, ("character", "copyright")) == frozenset(
        {"hatsune miku", "vocaloid"}
    )
    assert names_in_categories(VOCAB, ()) == frozenset()


GROUPS = (
    ResolvedGroup(
        name="hair_color",
        mode="softmax_when_solo",
        description="one hair colour",
        tag_indices=(1, 4, 9),
        tag_names=("blue hair", "red hair", "hair_color:none"),
        escape_indices=(7,),
        escape_names=("multicolored hair",),
        sentinel_index=9,
    ),
    ResolvedGroup(
        name="framing",
        mode="multilabel",
        description="",
        tag_indices=(2,),
        tag_names=("cowboy shot",),
        escape_indices=(),
        escape_names=(),
    ),
)


def test_resolved_dict_round_trips():
    assert resolved_from_dict(resolved_to_dict(GROUPS)) == GROUPS


def test_resolved_from_dict_revives_a_pre_sentinel_snapshot():
    """An older build wrote neither ``sentinel_index`` nor the escape pair."""
    (g,) = resolved_from_dict([{"name": "g", "mode": "softmax", "tag_indices": [3, 5]}])
    assert (g.tag_indices, g.escape_indices, g.sentinel_index) == ((3, 5), (), None)


def test_resolved_groups_is_empty_for_a_vocab_built_without_groups():
    assert resolved_groups(VOCAB) == ()
    assert resolved_groups({"groups": None}) == ()


def test_resolved_from_dict_still_demands_the_two_required_keys():
    with pytest.raises(KeyError):
        resolved_from_dict([{"tag_indices": [1]}])
