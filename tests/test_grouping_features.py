"""Grouping feature primitives: caption-grammar tag reads + collision-free keys.

- ``read_tags`` parses through ``position_clauses.parse_caption``, so a clause
  leaks no garbage tags into the grouping tag set.
- ``match_decensored.has_censor_tag`` reads through ``read_tags`` and keys
  through ``normalize_tag``.
- ``embed_members`` keys by rel-posix path (``root=``), since stems are not
  unique tree-wide.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

import anime_tools.grouping.features as F
from anime_tools.grouping.features import Member, embed_members, read_tags
from anime_tools.grouping.groups import build_groups

# ---------------------------------------------------------------------------
# read_tags


def test_read_tags_clause_free(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("1girl, White_Socks,  blue eyes ,", encoding="utf-8")
    assert read_tags(txt) == {"1girl", "white socks", "blue eyes"}


def test_read_tags_excludes_position_clauses(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text(
        "safe, 2girls, white socks. On the left, akita neru, yellow eyes. "
        "On the right, kasane teto.",
        encoding="utf-8",
    )
    tags = read_tags(txt)
    assert tags == {"safe", "2girls", "white socks"}
    assert not any("on the" in t for t in tags)
    assert "akita neru" not in tags  # clause-bound, not in the flat bag


def test_read_tags_empty_and_missing(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert read_tags(empty) == set()
    assert read_tags(tmp_path / "missing.txt") == set()


# ---------------------------------------------------------------------------
# match_decensored.has_censor_tag


def _sincos(monkeypatch, tmp_path, stem: str, caption: str):
    from anime_tools.grouping.cli import match_decensored as M

    monkeypatch.setattr(M, "SINCOS_DIR", tmp_path)
    (tmp_path / f"{stem}.txt").write_text(caption, encoding="utf-8")
    return M


def test_has_censor_tag_ignores_a_clause_header(monkeypatch, tmp_path):
    """A clause header is not a tag, and a censor word inside a clause is not a
    whole-image censor."""
    M = _sincos(
        monkeypatch,
        tmp_path,
        "a",
        "safe, 2girls, white socks. On the left, akita neru, censored bikini.",
    )
    # Whole-image censoring is a flat-bag property.
    assert M.has_censor_tag("a") is False

    _sincos(
        monkeypatch,
        tmp_path,
        "b",
        "safe, 2girls, mosaic censoring. On the left, akita neru.",
    )
    assert M.has_censor_tag("b") is True


def test_has_censor_tag_matches_the_underscore_spelling(monkeypatch, tmp_path):
    """``normalize_tag`` keys ``convenient_hair`` and ``convenient hair`` the same."""
    M = _sincos(monkeypatch, tmp_path, "a", "safe, 1girl, convenient_hair")
    assert M.has_censor_tag("a") is True
    assert M.is_censor_tag("convenient_hair") is True
    assert M.is_censor_tag("Convenient Hair") is True


def test_has_censor_tag_on_uncensored_and_on_a_missing_sidecar(monkeypatch, tmp_path):
    M = _sincos(monkeypatch, tmp_path, "a", "safe, 1girl, uncensored, pussy")
    assert M.has_censor_tag("a") is False
    assert M.has_censor_tag("nosuchstem") is False


# ---------------------------------------------------------------------------
# embed_members / build_groups stem collisions


class FakeEmbedder:
    """Deterministic content-keyed features: same pixels → same grid/cls."""

    device = torch.device("cpu")
    dtype = torch.float32
    name = "fake"

    def __call__(self, batch: torch.Tensor):
        cls_rows, grids = [], []
        for item in batch:
            seed = int(torch.round((item.mean() + 1.0) * 1000).item())
            rng = np.random.default_rng(seed)
            grid = rng.standard_normal((16, 16, 768)).astype(np.float16)
            cls = grid.reshape(-1)[:768].astype(np.float32)
            cls /= np.linalg.norm(cls) + 1e-8
            cls_rows.append(cls)
            grids.append(grid)
        return np.stack(cls_rows), np.stack(grids)


def _write_png(path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (value, value, value)).save(path)


def test_embed_members_same_stem_in_two_subfolders(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "CACHE_ROOT", tmp_path / "cache")
    root = tmp_path / "src"
    _write_png(root / "a" / "1.png", 0)
    _write_png(root / "b" / "1.png", 255)
    members = [
        Member("a", "1", root / "a" / "1.png", root / "a" / "1.txt"),
        Member("b", "1", root / "b" / "1.png", root / "b" / "1.txt"),
    ]

    feats = embed_members(
        FakeEmbedder(), members, batch_size=2, num_workers=0, root=root
    )
    assert set(feats) == {"a/1.png", "b/1.png"}
    assert not np.allclose(feats["a/1.png"].cls, feats["b/1.png"].cls)

    # Second pass is served from the cache — still two distinct entries.
    cached = embed_members(
        FakeEmbedder(), members, batch_size=2, num_workers=0, root=root
    )
    assert set(cached) == {"a/1.png", "b/1.png"}
    np.testing.assert_allclose(cached["a/1.png"].cls, feats["a/1.png"].cls)
    np.testing.assert_allclose(cached["b/1.png"].cls, feats["b/1.png"].cls)


def test_embed_members_legacy_stem_keys_without_root(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "CACHE_ROOT", tmp_path / "cache")
    root = tmp_path / "src"
    _write_png(root / "a" / "1.png", 0)
    members = [Member("a", "1", root / "a" / "1.png", root / "a" / "1.txt")]
    feats = embed_members(FakeEmbedder(), members, batch_size=1, num_workers=0)
    assert set(feats) == {"1"}


def test_build_groups_survives_stem_collision(tmp_path, monkeypatch):
    """Two artists sharing a stem keep separate embeddings; members stay rel-posix."""
    monkeypatch.setattr(F, "CACHE_ROOT", tmp_path / "cache")
    root = tmp_path / "src"
    _write_png(root / "a" / "1.png", 0)
    _write_png(root / "a" / "2.png", 0)  # exact twin of a/1
    _write_png(root / "b" / "1.png", 255)  # same stem as a/1, different image
    out = tmp_path / "groups.json"

    manifest = build_groups(
        root, out, embedder=FakeEmbedder(), batch_size=4, num_workers=0
    )
    assert manifest["n_images"] == 3
    assert manifest["n_groups"] == 1
    [group] = manifest["groups"]
    assert group["artist"] == "a"
    assert group["members"] == ["a/1.png", "a/2.png"]
    # b/1.png did not inherit (or clobber) a/1.png's embedding.
    assert manifest["n_singletons"] == 1


# --------------------------------------------------------------------------- #
# Grouping enumerates images through ``_walk``.
# --------------------------------------------------------------------------- #


def test_image_exts_is_the_curation_walkers_list():
    """``F.IMAGE_EXTS`` is ``_walk.IMAGE_EXTENSIONS``."""
    from anime_tools._walk import IMAGE_EXTENSIONS

    assert set(F.IMAGE_EXTS) == {e.lower() for e in IMAGE_EXTENSIONS}


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(path)


def test_iter_images_sees_what_the_stages_see(tmp_path):
    from anime_tools._walk import walk_images

    _touch(tmp_path / "a" / "one.png")
    _touch(tmp_path / "a" / "two.bmp")
    _touch(tmp_path / "b" / "three.webp")
    (tmp_path / "a" / "one.txt").write_text("1girl", encoding="utf-8")

    assert F.iter_images(tmp_path) == walk_images(tmp_path, recursive=True)
    assert [p.name for p in F.iter_images(tmp_path)] == [
        "one.png",
        "two.bmp",
        "three.webp",
    ]


def test_iter_images_on_a_missing_root_is_empty(tmp_path):
    assert F.iter_images(tmp_path / "nope") == []


def test_gather_members_uses_the_same_glob(tmp_path):
    _touch(tmp_path / "artist_a" / "s1.png")
    _touch(tmp_path / "artist_a" / "s2.bmp")
    (tmp_path / "artist_a" / "notes.txt").write_text("", encoding="utf-8")

    by_artist = F.gather_members([tmp_path], None)
    assert [m.stem for m in by_artist["artist_a"]] == ["s1", "s2"]
    assert by_artist["artist_a"][1].txt_path.name == "s2.txt"


# ---------------------------------------------------------------------------
# cache staleness


def test_cached_feature_is_reused_when_the_source_is_untouched(tmp_path, monkeypatch):
    """An unchanged file is still a cache hit; the stamp must not defeat the cache."""
    monkeypatch.setattr(F, "CACHE_ROOT", tmp_path / "cache")
    root = tmp_path / "src"
    _write_png(root / "a" / "1.png", 0)
    members = [Member("a", "1", root / "a" / "1.png", root / "a" / "1.txt")]

    embed_members(FakeEmbedder(), members, batch_size=1, num_workers=0, root=root)

    class CountingEmbedder(FakeEmbedder):
        """A subclass, since ``embedder(batch)`` resolves ``__call__`` on the type."""

        calls = 0

        def __call__(self, batch):
            type(self).calls += 1
            return super().__call__(batch)

    embed_members(CountingEmbedder(), members, batch_size=1, num_workers=0, root=root)
    assert CountingEmbedder.calls == 0  # served entirely from the cache


def test_rewritten_source_re_embeds_instead_of_a_stale_hit(tmp_path, monkeypatch):
    """The cache key is the image's location, which does not move when ``resize``
    rewrites the file; the ``(size, mtime_ns)`` stamp is what catches that."""
    monkeypatch.setattr(F, "CACHE_ROOT", tmp_path / "cache")
    root = tmp_path / "src"
    path = root / "a" / "1.png"
    _write_png(path, 0)
    members = [Member("a", "1", path, root / "a" / "1.txt")]

    first = embed_members(
        FakeEmbedder(), members, batch_size=1, num_workers=0, root=root
    )

    # Same path, same stem, same parent — different pixels.
    _write_png(path, 255)
    os.utime(path, ns=(0, 0))  # a mtime that cannot collide with the original
    second = embed_members(
        FakeEmbedder(), members, batch_size=1, num_workers=0, root=root
    )
    assert not np.allclose(first["a/1.png"].cls, second["a/1.png"].cls)


def test_pre_stamp_cache_entry_is_a_miss(tmp_path, monkeypatch):
    """A ``.npz`` with no ``ver``/``size`` keys is a recompute, never a crash."""
    monkeypatch.setattr(F, "CACHE_ROOT", tmp_path / "cache")
    root = tmp_path / "src"
    path = root / "a" / "1.png"
    _write_png(path, 0)
    member = Member("a", "1", path, root / "a" / "1.txt")

    legacy = F._cache_path(member)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        legacy,
        cls=np.zeros(4, dtype=np.float32),
        grid16=np.zeros((16, 16, 4), dtype=np.float16),
    )
    assert F._load_feature(legacy, F._source_stamp(path)) is None

    feats = embed_members(
        FakeEmbedder(), [member], batch_size=1, num_workers=0, root=root
    )
    assert not np.allclose(feats["a/1.png"].cls, 0)


def test_unreadable_source_never_takes_a_cached_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "CACHE_ROOT", tmp_path / "cache")
    assert F._source_stamp(tmp_path / "gone.png") == (-1, -1)
