"""Tests for the shared tag-shape primitives in
``anime_tools.captions.taxonomy`` and the contract that the Anima Tagger vocab
build and the caption-index builder type tag *shape* identically (no drift)."""

from __future__ import annotations

from anime_tools.captions import index as bci
from anime_tools.captions import taxonomy as tx


def test_taxonomy_is_torch_free():
    # Importing the primitives must not drag torch (the caption-index script
    # relies on staying lightweight / method-agnostic). Check in a fresh
    # subprocess so a torch import elsewhere in the suite can't mask a
    # regression here.
    import subprocess
    import sys

    code = (
        "import anime_tools.captions.taxonomy, sys; assert 'torch' not in sys.modules"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, (
        "importing anime_tools.captions.taxonomy pulled in torch:\n" + r.stderr
    )


def test_is_artist_tag():
    assert tx.is_artist_tag("@sincos")
    assert tx.is_artist_tag("@sumiyao (amam)")
    # Booru emoticon `@_@` → `@ @` after normalization is NOT an artist.
    assert not tx.is_artist_tag("@ @")
    assert not tx.is_artist_tag("@")
    assert not tx.is_artist_tag("blue eyes")


def test_strip_artist_prefix():
    assert tx.strip_artist_prefix("@sincos") == "sincos"
    assert tx.strip_artist_prefix("hatsune miku") == "hatsune miku"


def test_is_count_tag():
    for t in [
        "1girl",
        "2girls",
        "1boy",
        "3others",
        "6+girls",
        "multiple girls",
        "multiple_boys",
    ]:
        assert tx.is_count_tag(t), t
    for t in ["no girls", "blue eyes", "1girl_1boy", "original"]:
        assert not tx.is_count_tag(t), t


def test_multi_count_regex_copies_match_taxonomy():
    """The re-typed ``_MULTI_COUNT_RE`` copies must stay byte-identical to the
    canonical taxonomy pattern — a copy that drops ``\\+?`` silently stops
    matching real tags like ``6+girls``. (Folding them into one shared import
    is a planned refactor; until then this pins the strings.)"""
    from pathlib import Path

    from anime_tools.tagger.cli import derive_groups as dg
    from anime_tools.tagger.cli import role_markers as rm

    canonical = tx._COUNT_RE.pattern
    assert dg._MULTI_COUNT_RE.pattern == canonical
    assert rm._MULTI_COUNT_RE.pattern == canonical
    # The inline copy in tagger.py (built lazily inside a method) — pinned by
    # source text, since compiling it requires a loaded checkpoint.
    tagger_src = (Path(tx.__file__).parents[1] / "tagger" / "tagger.py").read_text(
        encoding="utf-8"
    )
    assert canonical in tagger_src
    for pat in (dg._MULTI_COUNT_RE, rm._MULTI_COUNT_RE, tx._COUNT_RE):
        assert pat.match("6+girls")
        assert pat.match("multiple_girls") and pat.match("multiple girls")


def test_caption_ratings_are_the_anima_band():
    # The one rating vocabulary: Anima's 4-class band. The tagger's ordered
    # RATINGS (class index for the rating head) must carry exactly the same
    # members as the unordered CAPTION_RATINGS.
    from anime_tools.tagger.tagger import RATINGS

    assert tx.CAPTION_RATINGS == {"safe", "sensitive", "nsfw", "explicit"}
    assert set(RATINGS) == tx.CAPTION_RATINGS
    assert len(RATINGS) == len(tx.CAPTION_RATINGS)  # no duplicate class index


def test_legacy_booru_ratings_alias_onto_the_anima_band():
    # Raw danbooru captions (and pre-rename vocab.json) still read as ratings.
    assert tx.canonical_rating("general") == "safe"
    assert tx.canonical_rating("questionable") == "nsfw"
    for r in tx.CAPTION_RATINGS:
        assert tx.canonical_rating(r) == r  # canonical spellings are fixpoints
        assert tx.is_rating_tag(r)
    for legacy in ("general", "questionable"):
        assert tx.is_rating_tag(legacy)
    for other in ("blue eyes", "1girl", "@sincos", "safety"):
        assert not tx.is_rating_tag(other)
        assert tx.canonical_rating(other) is None


def test_index_and_tagger_agree_on_tag_shape():
    """The two categorization paths must classify artist/count identically —
    both now key off the single shared primitives."""
    from anime_tools.tagger.cli import vocab as v

    for tag in [
        "@sincos",
        "@ @",
        "1girl",
        "2others",
        "multiple girls",
        "no girls",
        "blue eyes",
    ]:
        tagger_cat = v.categorize(tag, cache={}, overrides=None)
        if bci.is_artist_tag(tag):
            assert tagger_cat == "artist", tag
        elif bci.is_count_tag(tag):
            assert tagger_cat == "count", tag
        else:
            # Neither artist nor count shape → tagger must not call it one.
            assert tagger_cat not in ("artist", "count"), tag


def test_constants_reexports_for_back_compat():
    # group_router imports _COUNT_RE from anime_tools.captions.taxonomy; constants re-exports it.
    from anime_tools.tagger.cli import constants as c

    assert c.is_count_tag is tx.is_count_tag
    assert c._COUNT_RE is tx._COUNT_RE


def test_dedupe_count_tags_keeps_top_score_per_family():
    from anime_tools.tagger.tagger import dedupe_count_tags

    # Contradictory exact girl counts: only the higher-scoring one survives;
    # the booru-implied `multiple_girls` co-tag and other families stay.
    kept = {
        "4girls": 0.62,
        "3girls": 0.55,
        "multiple_girls": 0.91,
        "1boy": 0.70,
        "blue eyes": 0.80,
    }
    dedupe_count_tags(kept)
    assert "3girls" not in kept
    assert set(kept) == {"4girls", "multiple_girls", "1boy", "blue eyes"}

    # Families dedupe independently; open-ended counts are exact too.
    kept = {"6+girls": 0.7, "3girls": 0.6, "2boys": 0.8, "1boy": 0.4}
    dedupe_count_tags(kept)
    assert set(kept) == {"6+girls", "2boys"}

    # Single count per family is a no-op.
    kept = {"1girl": 0.9, "solo": 0.95}
    dedupe_count_tags(kept)
    assert set(kept) == {"1girl", "solo"}
