"""Grouping feature primitives: caption-grammar tag reads + collision-free keys.

Two regressions:

- ``read_tags`` used to hand-``split(",")`` the sidecar, so a position clause
  leaked garbage tags (``"white socks. on the left"``) into the grouping tag
  set. It now goes through ``position_clauses.parse_caption``.
- ``embed_members`` used to key its result by bare ``stem`` while nothing
  enforces unique stems tree-wide, so two subfolders' ``1.png`` silently shared
  one embedding. Keyed by rel-posix path (``root=``) now.
"""

from __future__ import annotations

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
    # The old split(",") glued the clause header onto the previous tag.
    assert not any("on the" in t for t in tags)
    assert "akita neru" not in tags  # clause-bound, not in the flat bag


def test_read_tags_empty_and_missing(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert read_tags(empty) == set()
    assert read_tags(tmp_path / "missing.txt") == set()


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

    # Second pass is served from the cache — still two distinct entries (the
    # cache is keyed by parent-dir hash + stem, so it never collided).
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
    """Two artists sharing a stem: each keeps its own embedding, and the twin
    pair inside one artist still groups. Manifest members stay rel-posix."""
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
# One walk: grouping enumerates images through ``_walk``, not its own tuple.
# --------------------------------------------------------------------------- #


def test_image_exts_is_the_curation_walkers_list():
    """The two used to disagree in both directions — grouping had no ``.bmp``
    and claimed ``.jxl``/``.avif`` even with no Pillow plugin to decode them."""
    from anime_tools._walk import IMAGE_EXTENSIONS

    assert set(F.IMAGE_EXTS) == {e.lower() for e in IMAGE_EXTENSIONS}


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(path)


def test_iter_images_sees_what_the_stages_see(tmp_path):
    from anime_tools._walk import walk_images

    _touch(tmp_path / "a" / "one.png")
    _touch(tmp_path / "a" / "two.bmp")  # was invisible to grouping
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
