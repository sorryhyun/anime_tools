"""Excluding an image: what moves, what the ledger says, and that the two
stages downstream of the decision honour it.

The invariant under test is the one the feature exists for — an excluded image
is invisible to every stage. That is enforced in exactly one place (``resize``
adds the ledger to its ``--skip``), because every other stage walks the resized
tree the exclusion emptied.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from anime_tools import exclude as X
from anime_tools import workspace as WS
from anime_tools.stages.export_workspace import ExportPaths, plan_export


def _png(path: Path, size=(8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)


def _txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def trees(tmp_path) -> X.Trees:
    """``char_aki/a`` has every artifact an image can have; ``b`` has pixels and
    a caption only."""
    t = X.Trees(
        resized=tmp_path / "workspace" / "resized",
        masks=tmp_path / "workspace" / "masks",
        ocr=tmp_path / "workspace" / "ocr",
        excluded=tmp_path / "workspace" / WS.EXCLUDED_SUBDIR,
    )
    _png(t.resized / "char_aki" / "a.png")
    _txt(t.resized / "char_aki" / "a.txt", "1girl, solo")
    _txt(t.resized / "char_aki" / "a.variants.txt", "# generated\nv0\t1girl\n")
    _txt(t.resized / "char_aki" / "a.history.txt", "# history\n")
    _png(t.masks / "char_aki" / "a_mask.png")
    _txt(t.ocr / "char_aki" / "a.ocr.txt", "# ocr\n")
    _png(t.resized / "b.png")
    _txt(t.resized / "b.txt", "1boy")
    return t


# ---- what moves ---------------------------------------------------------


def test_excluding_empties_all_three_trees(trees):
    """Every file of the image leaves the live trees for one mirror apiece."""
    result = X.exclude_one(trees, "char_aki/a.jpg", note="duplicate")

    assert result.action == "excluded"
    assert set(result.moved) == {
        "resized/char_aki/a.png",
        "resized/char_aki/a.txt",
        "resized/char_aki/a.variants.txt",
        "resized/char_aki/a.history.txt",
        "masks/char_aki/a_mask.png",
        "ocr/char_aki/a.ocr.txt",
    }
    for slot in result.moved:
        assert (trees.excluded / slot).is_file(), slot
    assert not (trees.resized / "char_aki" / "a.png").exists()
    assert not (trees.masks / "char_aki" / "a_mask.png").exists()
    assert not (trees.ocr / "char_aki" / "a.ocr.txt").exists()
    # The neighbour is untouched.
    assert (trees.resized / "b.png").is_file()


def test_the_ledger_is_keyed_by_the_source_rel(trees):
    """The key is the master's path, extension and all — what ``--skip`` names —
    even though what moved is the resized ``.png``."""
    X.exclude_one(trees, "char_aki/a.jpg")

    assert X.excluded_rels(trees.excluded) == ("char_aki/a.jpg",)
    entry = X.read_entries(trees.excluded)["char_aki/a.jpg"]
    assert entry.rel == "char_aki/a.jpg"
    assert entry.at > 0
    assert "resized/char_aki/a.png" in entry.moved


def test_an_image_with_nothing_in_the_workspace_is_still_recorded(trees):
    """Excluded before it was ever resized: nothing to move, but the ledger is
    the state and the point is that resize never makes those files."""
    result = X.exclude_one(trees, "char_aki/never-resized.jpg")

    assert result.moved == ()
    assert "char_aki/never-resized.jpg" in X.read_entries(trees.excluded)


def test_re_excluding_keeps_the_original_time_and_unions_what_moved(trees):
    first = X.exclude_one(trees, "char_aki/a.jpg", note="dup")
    # A file that appeared after the exclusion is swept up by the second pass.
    _txt(trees.resized / "char_aki" / "a.caption", "late")

    second = X.exclude_one(trees, "char_aki/a.jpg")

    assert second.entry.at == first.entry.at
    assert second.entry.note == "dup"
    assert "resized/char_aki/a.caption" in second.entry.moved
    assert set(first.entry.moved) < set(second.entry.moved)


# ---- putting one back ---------------------------------------------------


def test_restoring_puts_every_file_back_and_clears_the_ledger(trees):
    before = {
        p: p.read_bytes() for p in trees.resized.rglob("char_aki/*") if p.is_file()
    }
    X.exclude_one(trees, "char_aki/a.jpg")

    result = X.restore_one(trees, "char_aki/a.jpg")

    assert result.action == "restored"
    assert X.read_entries(trees.excluded) == {}
    for path, data in before.items():
        assert path.read_bytes() == data
    assert (trees.masks / "char_aki" / "a_mask.png").is_file()
    assert (trees.ocr / "char_aki" / "a.ocr.txt").is_file()
    # Nothing is left behind but the ledger itself.
    assert [p.name for p in trees.excluded.rglob("*") if p.is_file()] == [
        WS.EXCLUDED_MANIFEST
    ]


def test_restoring_never_clobbers_a_live_file(trees):
    """A resize that ran without the ledger has re-made the png. The archived
    copy is the older one, so it stays put and is reported."""
    X.exclude_one(trees, "char_aki/a.jpg")
    _png(trees.resized / "char_aki" / "a.png", size=(16, 16))

    result = X.restore_one(trees, "char_aki/a.jpg")

    assert result.skipped == ("resized/char_aki/a.png",)
    assert (trees.excluded / "resized/char_aki/a.png").is_file()
    assert Image.open(trees.resized / "char_aki" / "a.png").size == (16, 16)
    # The image is no longer excluded either way.
    assert X.read_entries(trees.excluded) == {}


def test_restoring_an_image_that_was_never_excluded_does_nothing(trees):
    assert X.restore_one(trees, "b.png").action == "not-excluded"


# ---- the ledger itself --------------------------------------------------


def test_an_unreadable_ledger_refuses_rather_than_reading_as_empty(trees):
    """ "Nothing is excluded" is the answer that would put every excluded image
    back through the pipeline on the next preflight."""
    X.exclude_one(trees, "char_aki/a.jpg")
    X.manifest_path(trees.excluded).write_text("{not json", encoding="utf-8")

    with pytest.raises(X.ExclusionError):
        X.read_entries(trees.excluded)


@pytest.mark.parametrize("rel", ["/abs/a.png", "../a.png", "a/../../b.png", ""])
def test_a_rel_that_could_name_another_tree_is_refused(trees, rel):
    with pytest.raises(X.ExclusionError):
        X.exclude_one(trees, rel)


def test_the_ledger_round_trips_through_json(trees):
    X.exclude_one(trees, "char_aki/a.jpg", note="why")
    entries = X.read_entries(trees.excluded)

    raw = json.loads(X.manifest_path(trees.excluded).read_text(encoding="utf-8"))
    assert raw["version"] == X.MANIFEST_VERSION
    assert X.read_entries(trees.excluded) == entries


# ---- the two stages downstream ------------------------------------------


def test_resize_skips_every_rel_in_the_ledger(tmp_path, monkeypatch):
    """The one chokepoint: resize is what could put an excluded image back into
    the tree every later stage walks."""
    from anime_tools.stages.requests import ResizeRequest
    from anime_tools.stages.resize import run_resize

    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    monkeypatch.delenv("ANIME_TOOLS_WORKSPACE", raising=False)
    src = tmp_path / "image_dataset"
    _png(src / "char_aki" / "a.jpg", size=(1200, 900))
    _png(src / "b.png", size=(1200, 900))
    trees = X.Trees.build()
    X.exclude_one(trees, "char_aki/a.jpg")

    stats = run_resize(ResizeRequest(workers=1))

    assert stats.skipped_excluded == 1
    assert not (trees.resized / "char_aki").exists()
    assert (trees.resized / "b.png").is_file()


def test_export_republishes_the_excluded_tree_beside_the_trainers(tmp_path):
    """``<out>/_excluded/`` mirrors the workspace tree — the same kinds, so the
    compare and the revert are unchanged, but never into ``<out>/resized``."""
    paths = ExportPaths(
        resized=tmp_path / "workspace" / "resized",
        masks=tmp_path / "workspace" / "masks",
        master=tmp_path / "workspace" / "master",
        index=tmp_path / "workspace" / "captions" / "caption_index.json",
        src=tmp_path / "image_dataset",
        out=tmp_path / "post_image_dataset",
        excluded=tmp_path / "workspace" / WS.EXCLUDED_SUBDIR,
    )
    trees = X.Trees(
        resized=paths.resized,
        masks=paths.masks,
        ocr=tmp_path / "workspace" / "ocr",
        excluded=paths.excluded,
    )
    _png(paths.resized / "char_aki" / "a.png")
    _txt(paths.resized / "char_aki" / "a.txt", "1girl")
    _txt(paths.resized / "char_aki" / "a.variants.txt", "# generated\nv0\t1girl\n")
    _png(paths.masks / "char_aki" / "a_mask.png")
    _png(paths.resized / "b.png")
    X.exclude_one(trees, "char_aki/a.jpg")

    rows = plan_export(paths)

    out = tmp_path / "post_image_dataset"
    excluded = {Path(r.dst) for r in rows if r.excluded}
    assert excluded == {
        out / "_excluded" / "resized" / "char_aki" / "a.png",
        out / "_excluded" / "resized" / "char_aki" / "a.txt",
        out / "_excluded" / "resized" / "char_aki" / "a.variants.txt",
        out / "_excluded" / "masks" / "char_aki" / "a_mask.png",
    }
    assert {r.kind for r in rows if r.excluded} == {
        "image",
        "caption",
        "variants",
        "mask",
    }
    # The live half is only the image that stayed in the pipeline.
    assert {r.rel for r in rows if not r.excluded} == {"b.png"}


def test_a_workspace_with_nothing_excluded_publishes_nothing_extra(tmp_path):
    paths = ExportPaths(
        resized=tmp_path / "workspace" / "resized",
        masks=tmp_path / "workspace" / "masks",
        master=tmp_path / "workspace" / "master",
        index=tmp_path / "workspace" / "captions" / "caption_index.json",
        src=tmp_path / "image_dataset",
        out=tmp_path / "post_image_dataset",
        excluded=tmp_path / "workspace" / WS.EXCLUDED_SUBDIR,
    )
    _png(paths.resized / "b.png")

    assert not any(r.excluded for r in plan_export(paths))
