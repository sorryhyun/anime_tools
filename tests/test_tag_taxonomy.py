"""Shared tag-shape primitives in ``anime_tools.captions.taxonomy``, and that the
tagger vocab build and the caption-index builder type tag shape identically."""

from __future__ import annotations

from pathlib import Path

from anime_tools.captions import index as bci
from anime_tools.captions import taxonomy as tx


def test_taxonomy_is_torch_free():
    # In a fresh subprocess, so a torch import elsewhere in the suite can't mask it.
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


def test_the_count_regex_has_no_copies_left():
    """``taxonomy._COUNT_RE`` is the only copy of the count pattern in the tree."""
    from anime_tools.tagger.cli import derive_groups as dg
    from anime_tools.tagger.cli import role_markers as rm

    assert dg._is_solo is tx.is_solo_names
    assert rm._solo_index_sets is tx.solo_multi_indices

    canonical = tx._COUNT_RE.pattern
    package = Path(tx.__file__).parents[1]
    carriers = sorted(
        p.relative_to(package).as_posix()
        for p in package.rglob("*.py")
        if canonical in p.read_text(encoding="utf-8")
    )
    assert carriers == ["captions/taxonomy.py"]

    assert tx._COUNT_RE.match("6+girls")
    assert tx._COUNT_RE.match("multiple_girls") and tx._COUNT_RE.match("multiple girls")


def test_solo_predicate_agrees_across_names_and_indices():
    """Name side and index side share one rule: single counts are excluded from
    the multi test, not merely uncounted (``1girl`` matches the multi pattern)."""
    assert tx.is_solo_names({"solo", "1girl", "blue eyes"})
    assert tx.is_solo_names({"1boy"})
    assert not tx.is_solo_names({"1girl", "2girls"})
    assert not tx.is_solo_names({"1girl", "multiple_girls"})
    assert not tx.is_solo_names({"blue eyes"})  # no count tag at all

    vocab = [
        {"name": "1girl", "index": 0},
        {"name": "solo", "index": 1},
        {"name": "2girls", "index": 2},
        {"name": "multiple boys", "index": 3},
        {"name": "blue eyes", "index": 4},
    ]
    assert tx.solo_multi_indices(vocab) == ({0, 1}, {2, 3})


def test_count_of_reads_one_number_out_of_the_bag():
    assert tx.count_of(["1girl", "solo"], "girl") == 1
    assert tx.count_of(["3girls", "2girls"], "girl") == 3
    # An exact count wins over the ``multiple_*`` implication booru fires beside it,
    # in either spelling.
    assert tx.count_of(["2girls", "multiple_girls"], "girl") == 2
    assert tx.count_of(["multiple_girls"], "girl") is None
    assert tx.count_of(["multiple girls"], "girl") is None
    # ``6+girls`` is "six or more", not six.
    assert tx.count_of(["6+girls"], "girl") is None
    # 0 = the caption doesn't say, which is not the same as "unknown".
    assert tx.count_of(["blue eyes"], "girl") == 0
    assert tx.count_of(["2girls", "1boy"], "boy") == 1
    assert tx.count_of(["2girls"], "boy") == 0


def test_caption_ratings_are_the_anima_band():
    # The tagger's ordered RATINGS (class index for the rating head) carries
    # exactly the members of the unordered CAPTION_RATINGS.
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
    """The two categorization paths classify artist/count identically."""
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

    # Contradictory exact girl counts: only the higher-scoring one survives.
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


def test_no_stage_compares_tags_with_a_bare_lower():
    """Tags under ``stages/`` are keyed by :func:`normalize_tag`, not a bare
    ``lower()`` — which reads ``speech bubble`` and ``speech_bubble`` as two tags.
    The two exemptions below are not tags: an argparse enum and a CLI "off"
    sentinel, matched against literals this repo spells itself.
    """
    package = Path(tx.__file__).parents[1]
    stages = package / "stages"
    exempt = {
        (
            "resize.py",
            "value = str(crop_anchor or DEFAULT_CROP_ANCHOR).strip().lower()",
        ),
        (
            "instance_detection.py",
            "if spec is None or spec.strip().lower() in _PROMPT_EMBED_OFF:",
        ),
    }
    found = {
        (path.relative_to(stages).as_posix(), line.strip())
        for path in stages.rglob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
        if ".lower()" in line
    }
    assert found == exempt
