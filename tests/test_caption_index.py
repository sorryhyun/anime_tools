"""``build_caption_index._classify``, focused on the danbooru `name (series)`
character-recovery heuristic for characters the tagger vocab predates."""

from __future__ import annotations

import json
from pathlib import Path

from anime_tools.captions import index as bci

# Omits the newer characters so recovery has to carry them. "genshin impact" is a
# known copyright; "miside" is not (only reachable via the same-caption bare tag).
# "general" vetoes copyright promotion of homonym qualifiers (`lily (flower)`).
VSETS = {
    "character": {"hatsune miku"},
    "copyright": {"genshin impact", "vocaloid", "hololive", "pokemon", "original"},
    "count": {"1girl", "1boy"},
    "general": {"flower", "sky", "sex", "school uniform"},
}


def test_recovers_paren_character_via_known_copyright():
    typed = bci._classify("1girl, mualani (genshin impact), genshin impact", VSETS)
    assert "mualani (genshin impact)" in typed["character"]
    assert "genshin impact" in typed["copyright"]
    assert "1girl" in typed["count"]


def test_recovers_via_same_caption_bare_tag_when_copyright_unknown():
    # "miside" is not in the vocab copyright set, but appears as a bare tag.
    typed = bci._classify("cool mita (miside), miside, 1girl", VSETS)
    assert "cool mita (miside)" in typed["character"]
    assert "miside" in typed["copyright"]


def test_generic_qualifier_not_recovered():
    # `X (cosplay)` is a generic qualifier, even though "cosplay" is a bare tag here.
    typed = bci._classify("frieren (cosplay), cosplay, 1girl", VSETS)
    assert typed["character"] == []


def test_unknown_series_not_recovered():
    # Parenthetical whose series is neither a known copyright nor a bare tag.
    typed = bci._classify("someone (obscure game), 1girl", VSETS)
    assert typed["character"] == []


def test_exact_vocab_character_still_classified():
    typed = bci._classify("hatsune miku, vocaloid, 1girl", VSETS)
    assert typed["character"] == ["hatsune miku"]
    assert "vocaloid" in typed["copyright"]


def test_recover_paren_can_be_disabled():
    typed = bci._classify(
        "mualani (genshin impact), genshin impact", VSETS, recover_paren=False
    )
    assert typed["character"] == []
    assert "genshin impact" in typed["copyright"]


def test_general_qualifier_not_promoted_to_copyright():
    # `lily (flower)` is a homonym disambiguator: the vocab types "flower" as
    # general, so the co-tagged-bare fallback must not promote it to copyright.
    typed = bci._classify("1girl, lily (flower), flower, @x", VSETS)
    assert "flower" not in typed["copyright"]
    assert typed["character"] == []


def test_character_qualifier_not_promoted_to_copyright():
    # `bubba (hatsune miku)` is disambiguated by a character, not a series, so the
    # qualifier stays out of copyright and keeps its own character slot.
    typed = bci._classify("1girl, bubba (hatsune miku), hatsune miku, @x", VSETS)
    assert "hatsune miku" not in typed["copyright"]
    assert "hatsune miku" in typed["character"]
    assert "bubba (hatsune miku)" not in typed["character"]


def test_vocab_copyright_qualifier_still_promoted():
    # A vocab-confirmed copyright is trusted even when it is also co-tagged bare.
    typed = bci._classify("1girl, mualani (genshin impact), genshin impact, @x", VSETS)
    assert "genshin impact" in typed["copyright"]
    assert "mualani (genshin impact)" in typed["character"]


def test_artist_prefix_unaffected():
    typed = bci._classify(
        "@some artist, mualani (genshin impact), genshin impact", VSETS
    )
    assert typed["artist"] == ["@some artist"]
    assert "mualani (genshin impact)" in typed["character"]


# ── positional bare-name character recovery ─────────────────────────────────


def test_positional_recovers_bare_name_character():
    # `nakiri ayame` is a bare name (no parens, not in vocab) sitting in the
    # pre-@artist band → recovered as character; "hololive" stays copyright.
    typed = bci._classify(
        "sensitive, 1girl, nakiri ayame, hololive, @drawfag, black hair", VSETS
    )
    assert typed["character"] == ["nakiri ayame"]
    assert "1girl" in typed["count"]


def test_positional_excludes_franchise_subtitle():
    # `pokemon scarlet and violet` shares a word with the known copyright → a
    # franchise sub-title, not a character. Only `hilda` is recovered.
    typed = bci._classify(
        "explicit, 1girl, hilda, pokemon, pokemon scarlet and violet, @x",
        VSETS,
    )
    assert typed["character"] == ["hilda"]
    assert "pokemon scarlet and violet" not in typed["character"]


def test_positional_skips_general_tags_after_artist():
    # Descriptive tags live after @artist and must never be read as characters.
    typed = bci._classify(
        "1girl, yukihana lamy, hololive, @y, blue eyes, smile, looking at viewer",
        VSETS,
    )
    assert typed["character"] == ["yukihana lamy"]
    assert "blue eyes" not in typed["character"]


def test_positional_excludes_count_like_tags():
    # A count tag the vocab missed ("2others") must not become a character.
    typed = bci._classify("sensitive, 1girl, 2others, asaka karin, hololive, @z", VSETS)
    assert "2others" not in typed["character"]
    assert "asaka karin" in typed["character"]


def test_positional_needs_artist_anchor():
    # No @artist → no reliable band boundary → no positional recovery.
    typed = bci._classify("1girl, mystery name, hololive", VSETS)
    assert "mystery name" not in typed["character"]


def test_positional_can_be_disabled():
    typed = bci._classify(
        "1girl, nakiri ayame, hololive, @drawfag", VSETS, recover_positional=False
    )
    assert typed["character"] == []


# ── position clauses parse through the grammar (never split(",")) ────────────


def test_claused_caption_parses_through_the_grammar():
    # Parsed through the grammar: no glued "white socks. on the left" pseudo-tag,
    # no trailing period on "kasane teto.", and a clause-bound character still
    # lands in the index.
    typed = bci._classify(
        "safe, 1girl, hatsune miku, vocaloid, @sincos, white socks. "
        "On the left, hatsune miku, twintails. On the right, kasane teto.",
        {**VSETS, "character": {"hatsune miku", "kasane teto"}},
    )
    assert typed["character"] == ["hatsune miku", "kasane teto"]
    assert typed["copyright"] == ["vocaloid"]
    assert typed["artist"] == ["@sincos"]
    assert typed["count"] == ["1girl"]
    # No glued pseudo-tag reached any axis.
    assert all("." not in t and "on the" not in t for ts in typed.values() for t in ts)


# ── `original` sole-copyright clears characters (OC images) ─────────────────


def test_original_only_clears_character():
    # `original` as sole copyright → an OC image, even with a character tag present.
    typed = bci._classify("1girl, hatsune miku, original, @x", VSETS)
    assert typed["character"] == []
    assert "original" in typed["copyright"]


def test_original_crossover_keeps_character():
    # `original` co-occurring with a real franchise (pokemon) is a crossover —
    # the franchise character survives.
    typed = bci._classify("1girl, dawn (pokemon), pokemon, original, @x", VSETS)
    assert "dawn (pokemon)" in typed["character"]


# ── build_index: cross-folder duplicate stems (nested disambiguation) ────────


def _write_vocab(tmp_path: Path) -> str:
    vocab = {
        "tags": [
            {"name": "hatsune miku", "category": "character"},
            {"name": "vocaloid", "category": "copyright"},
            {"name": "genshin impact", "category": "copyright"},
            {"name": "1girl", "category": "count"},
        ]
    }
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(vocab), encoding="utf-8")
    return str(p)


def test_duplicate_bare_stem_across_folders_kept_distinct(tmp_path):
    # The same bare filename in two subfolders: keys are posix relpaths, so both survive.
    src = tmp_path / "captions"
    (src / "en").mkdir(parents=True)
    (src / "ew").mkdir(parents=True)
    (src / "en" / "1.txt").write_text("1girl, hatsune miku, vocaloid", encoding="utf-8")
    (src / "ew" / "1.txt").write_text("1girl, genshin impact", encoding="utf-8")

    index = bci.build_index(str(src), _write_vocab(tmp_path))

    assert set(index["image_meta"]) == {"en/1", "ew/1"}
    assert index["image_meta"]["en/1"]["path"] == "en/1.txt"
    assert index["image_meta"]["ew/1"]["path"] == "ew/1.txt"
    assert index["meta"]["n_images"] == 2
    # groups reference the disambiguated keys, not a conflated bare "1".
    assert index["groups"]["copyright"]["vocaloid"] == ["en/1"]
    assert index["groups"]["copyright"]["genshin impact"] == ["ew/1"]


def test_flat_layout_key_is_bare_stem(tmp_path):
    # A flat (un-nested) dataset keys by the bare stem.
    src = tmp_path / "captions"
    src.mkdir()
    (src / "pic.txt").write_text("1girl, hatsune miku, vocaloid", encoding="utf-8")

    index = bci.build_index(str(src), _write_vocab(tmp_path))

    assert set(index["image_meta"]) == {"pic"}
    assert index["image_meta"]["pic"]["path"] == "pic.txt"


def test_build_index_with_path_arguments_is_json_serializable(tmp_path, monkeypatch):
    """The CLI resolves ``--vocab`` to a ``Path``; the meta block must still
    serialize (regression: ``TypeError: Object of type PosixPath`` from
    ``write_json`` at the end of every ``captions.index`` run)."""
    from anime_tools._json import write_json

    monkeypatch.setattr(bci, "_load_vocab_sets", lambda _path: VSETS)
    src = tmp_path / "image_dataset"
    src.mkdir()
    (src / "a.txt").write_text("1girl, hatsune miku, vocaloid", encoding="utf-8")
    vocab = tmp_path / "vocab.json"
    vocab.write_text("{}", encoding="utf-8")

    index = bci.build_index(src, vocab)
    out = write_json(tmp_path / "caption_index.json", index)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["meta"]["vocab_path"] == str(vocab)
    assert payload["image_meta"]["a"]["character"] == ["hatsune miku"]
