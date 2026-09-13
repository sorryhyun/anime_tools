"""The sidecar leaves artist OCs out: a character qualified by a vocab
``@artist`` handle is a name that means nothing outside the dataset it came
from, and a head trained on a few dozen positives fires it on look-alikes.
Franchise characters dbv4 lacks are the head's whole point and stay."""

from __future__ import annotations

import pytest

from anime_tools.captions.taxonomy import artist_handles_of, artist_oc_handle

VOCAB = [
    {"name": "@mignon", "category": "artist"},
    {"name": "@pepper0", "category": "artist"},
    {"name": "@ @", "category": "general"},  # booru emoticon, not an artist
    {"name": "original", "category": "copyright"},
    {"name": "wuthering waves", "category": "copyright"},
    {"name": "shiro (mignon)", "category": "character"},
    {"name": "akiyama fumika (pepper0)", "category": "character"},
    {"name": "cartethyia (wuthering waves)", "category": "character"},
    {"name": "shigure ui (young) (vtuber)", "category": "character"},
    {"name": "waguri kaoruko", "category": "character"},
    {"name": "black shoes", "category": "general"},
    {"name": "silver hair", "category": "deprecated"},
    {"name": "@kat (bu-kunn)", "category": "artist"},
    {"name": "kuronuma mayu (kat (bu-kunn))", "category": "general"},  # booru mistype
]
HANDLES = artist_handles_of(t["name"] for t in VOCAB)


def test_artist_handles_drop_the_prefix_and_skip_emoticons():
    assert HANDLES == {"mignon", "pepper0", "kat (bu-kunn)"}


@pytest.mark.parametrize(
    ("tag", "handle"),
    [
        ("shiro (mignon)", "mignon"),
        ("akiyama fumika (pepper0)", "pepper0"),
        ("cartethyia (wuthering waves)", None),  # franchise qualifier
        ("shigure ui (young) (vtuber)", None),  # last parenthetical is not an artist
        ("waguri kaoruko", None),
        ("mignon", None),  # bare handle without the parenthetical is not an OC form
        ("kuronuma mayu (kat (bu-kunn))", "kat (bu-kunn)"),  # nested handle
    ],
)
def test_artist_oc_handle(tag, handle):
    assert artist_oc_handle(tag, HANDLES) == handle


def test_trailing_qualifier_is_balanced():
    from anime_tools.captions.taxonomy import trailing_qualifier

    assert trailing_qualifier("ayaka (genshin impact)") == "genshin impact"
    assert trailing_qualifier("kuronuma mayu (kat (bu-kunn))") == "kat (bu-kunn)"
    assert trailing_qualifier("shigure ui (young) (vtuber)") == "vtuber"
    assert trailing_qualifier("waguri kaoruko") is None
    assert trailing_qualifier("(mignon)") == "mignon"
    assert trailing_qualifier("broken (") is None


def test_select_bce_rows_leaves_artist_ocs_out_by_default():
    from anime_tools.tagger.cli.train_sidecar import (
        DEFAULT_CATEGORIES,
        select_bce_rows,
    )

    assert (
        "deprecated" not in DEFAULT_CATEGORIES
    )  # retired names are aliased, not trained
    unmatched = [(i, t["name"], t["category"]) for i, t in enumerate(VOCAB)]
    cats = set(DEFAULT_CATEGORIES)
    kept, dropped = select_bce_rows(unmatched, cats, VOCAB)
    names = [VOCAB[i]["name"] for i in kept]
    assert dropped == [
        "shiro (mignon)",
        "akiyama fumika (pepper0)",
        "kuronuma mayu (kat (bu-kunn))",  # general-typed OC goes too
    ]
    assert "silver hair" not in names
    assert "cartethyia (wuthering waves)" in names
    assert "waguri kaoruko" in names
    assert "original" in names and "black shoes" in names
    assert not any(n in names for n in dropped)
    assert "@mignon" not in names  # artists are never sidecar rows

    kept_all, dropped_all = select_bce_rows(unmatched, cats, VOCAB, keep_artist_oc=True)
    assert dropped_all == []
    assert set(kept_all) == set(kept) | {5, 6, 13}

    # the category filter still applies first: no character/general rows, nothing dropped
    kept_c, dropped_c = select_bce_rows(unmatched, {"copyright"}, VOCAB)
    assert dropped_c == []
    assert [VOCAB[i]["name"] for i in kept_c] == ["original", "wuthering waves"]
